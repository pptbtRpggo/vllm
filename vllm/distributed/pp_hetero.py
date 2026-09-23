# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Emulate heterogeneous PP devices on homogeneous hardware.

The legacy rank-based configuration drives both ``vllm pp-profile`` and ``vllm serve``:

    VLLM_PP_HETERO=<compute_scales>[/<comm_scales>]

Examples::

    VLLM_PP_HETERO=1,2          # rank1 compute 2x slower
    VLLM_PP_HETERO=1,2/4        # plus hop 0->1 comm 4x slower
    VLLM_PP_HETERO=/4           # comm only

Missing ranks/hops default to ``1.0``. When this is set, engine startup
swaps Ascend ``NPUWorker`` for ``PPAscendWorker`` so serve does not
need ``--worker-cls``.

Live communication slowdown also requires ``VLLM_PP_COMM_BANDWIDTH_GBPS``
(comma-separated per-hop, per-TP-lane baseline rates in decimal Gbit/s).
``VLLM_PP_COMM_LATENCY_MS`` optionally supplies per-hop fixed baseline latency.
The added delay is ``(scale - 1) * (latency + wire_bytes / bandwidth)``;
blocking send/recv wait times are never multiplied. Both endpoints delay
completion of the same transfer, without exchanging additional messages.

For device-pair networks, VLLM_PP_NETWORK supplies upper-triangular bandwidth_gbps
and latency_ms rows, without the diagonal. Both directions share each entry.
Their transfer time is added to real communication.
VLLM_PP_DEVICE_ORDER maps ranks to stable device IDs, including compute factors.
Network matrices and legacy per-hop communication settings are mutually exclusive.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import torch

T = TypeVar("T")


def parse_device_order(text: str | None) -> tuple[int, ...]:
    """Stable device IDs in PP rank order; independent of visible device indices."""
    if not text:
        return ()
    order = tuple(int(v.strip()) for v in text.split(","))
    if min(order) < 0 or len(set(order)) != len(order):
        raise ValueError("VLLM_PP_DEVICE_ORDER needs distinct nonnegative device IDs")
    return order


@dataclass(frozen=True)
class PPNetwork:
    """Symmetric extra network delay stored as a compact upper triangle.

    Delay is added to real transport, not substituted for it. Bandwidth is
    per TP lane. These configured values never become planner costs.
    Row i stores pairs (i, i+1), ..., (i, N-1); the last row is empty.
    """

    bandwidth_gbps: tuple[tuple[float, ...], ...]
    latency_ms: tuple[tuple[float, ...], ...]

    @classmethod
    def from_json(cls, text: str) -> PPNetwork:
        data = json.loads(text)
        if not isinstance(data, dict) or set(data) - {"bandwidth_gbps", "latency_ms"}:
            raise ValueError(
                "PP network accepts bandwidth_gbps and latency_ms matrices"
            )
        bandwidth = data.get("bandwidth_gbps")
        if not isinstance(bandwidth, list) or len(bandwidth) < 2:
            raise ValueError("PP network needs N upper-triangle rows, N >= 2")
        size = len(bandwidth)
        latency = data.get("latency_ms", [[0] * (size - i - 1) for i in range(size)])
        for name, matrix in (("bandwidth", bandwidth), ("latency", latency)):
            if not isinstance(matrix, list) or len(matrix) != size:
                raise ValueError("PP network matrices must have the same dimensions")
            for source, row in enumerate(matrix):
                if not isinstance(row, list) or len(row) != size - source - 1:
                    raise ValueError(
                        "PP network upper-triangle row i needs N-i-1 values "
                        "(no diagonal or lower triangle; last row is empty)"
                    )
                for value in row:
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError("PP network values must be finite numbers")
                    if value < 0 or (name == "bandwidth" and value == 0):
                        raise ValueError("PP bandwidth must be > 0 and latency >= 0")
        return cls(tuple(map(tuple, bandwidth)), tuple(map(tuple, latency)))

    def delay_ms(self, source: int, target: int, payload_bytes: int) -> float:
        if (
            source == target
            or min(source, target) < 0
            or max(source, target) >= len(self.bandwidth_gbps)
        ):
            raise ValueError("invalid PP network device pair")
        if payload_bytes == 0:
            return 0.0
        source, target = sorted((source, target))
        offset = target - source - 1
        return self.latency_ms[source][offset] + 8 * payload_bytes / (
            self.bandwidth_gbps[source][offset] * 1_000_000
        )


def parse_scale_list(text: str | None) -> tuple[float, ...]:
    """Parse ``"1,2,1.5"`` into positive floats. ``None`` / empty → ``()``."""
    if text is None:
        return ()
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return ()
    values: list[float] = []
    for part in parts:
        value = float(part)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"PP hetero scale must be finite and > 0, got {value}")
        values.append(value)
    return tuple(values)


def scale_at(scales: Sequence[float], index: int, default: float = 1.0) -> float:
    """Return ``scales[index]``, or ``default`` if the list is short / index < 0."""
    if index < 0 or index >= len(scales):
        return default
    return float(scales[index])


def parse_hetero_spec(text: str | None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Parse ``compute[/comm]``, e.g. ``1,2/4``, ``1,2``, or ``/4``."""
    if text is None:
        return (), ()
    spec = text.strip()
    if not spec:
        return (), ()
    if "/" in spec:
        left, right = spec.split("/", 1)
        return parse_scale_list(left), parse_scale_list(right)
    return parse_scale_list(spec), ()


def stretch_after(elapsed_ms: float, scale: float) -> float:
    """Sleep so wall time becomes ``scale * elapsed_ms``. Return the new time.

    ``scale <= 1`` is a no-op (does not speed the device up).
    """
    if elapsed_ms < 0.0:
        raise ValueError(f"elapsed_ms must be >= 0, got {elapsed_ms}")
    if scale <= 1.0:
        return elapsed_ms
    extra_s = (elapsed_ms / 1000.0) * (scale - 1.0)
    if extra_s > 0.0:
        time.sleep(extra_s)
    return elapsed_ms * scale


def sync_torch_device(device: torch.device) -> None:
    """Block until queued compute/comm on ``device`` finishes."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()


def time_call(
    fn: Callable[[], T],
    sync: Callable[[], None] | None = None,
) -> tuple[T, float]:
    """Run ``fn``, optional device sync, return ``(result, elapsed_ms)``."""
    t0 = time.perf_counter()
    result = fn()
    if sync is not None:
        sync()
    return result, (time.perf_counter() - t0) * 1000.0


@dataclass(frozen=True)
class PPHeteroConfig:
    """Per-rank compute and per-hop comm stretch factors."""

    compute_scales: tuple[float, ...] = ()
    comm_scales: tuple[float, ...] = ()
    comm_bandwidth_gbps: tuple[float, ...] = ()
    comm_latency_ms: tuple[float, ...] = ()
    network: PPNetwork | None = None
    device_order: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.network is not None and (
            self.comm_scales or self.comm_bandwidth_gbps or self.comm_latency_ms
        ):
            raise ValueError(
                "PP network matrix cannot be combined with "
                "per-hop communication settings"
            )
        if self.device_order:
            if (
                len(set(self.device_order)) != len(self.device_order)
                or min(self.device_order) < 0
            ):
                raise ValueError("PP device IDs must be distinct and nonnegative")
            if self.network and max(self.device_order) >= len(
                self.network.bandwidth_gbps
            ):
                raise ValueError("PP device ID is outside the network matrix")
            if self.device_order != tuple(range(len(self.device_order))) and (
                self.comm_scales or self.comm_bandwidth_gbps or self.comm_latency_ms
            ):
                raise ValueError(
                    "reordered devices require a network matrix, not per-hop settings"
                )
        for value in self.comm_bandwidth_gbps:
            if not math.isfinite(value) or value <= 0:
                raise ValueError("PP communication bandwidth must be finite and > 0")
        for value in self.comm_latency_ms:
            if not math.isfinite(value) or value < 0:
                raise ValueError("PP communication latency must be finite and >= 0")
        for rank, scale in enumerate(self.comm_scales):
            if not math.isfinite(scale) or scale < 1:
                raise ValueError("Live PP communication scale must be finite and >= 1")
            if scale > 1 and rank >= len(self.comm_bandwidth_gbps):
                raise ValueError(
                    f"Hop {rank}->{rank + 1} slowdown requires a baseline in "
                    "VLLM_PP_COMM_BANDWIDTH_GBPS; blocking send/recv time "
                    "includes peer waiting and cannot be used as bandwidth."
                )

    @classmethod
    def from_env(cls) -> PPHeteroConfig:
        import vllm.envs as envs

        compute, comm = parse_hetero_spec(envs.VLLM_PP_HETERO)
        latency = tuple(
            float(value.strip())
            for value in (envs.VLLM_PP_COMM_LATENCY_MS or "").split(",")
            if value.strip()
        )
        return cls(
            compute_scales=compute,
            comm_scales=comm,
            comm_bandwidth_gbps=parse_scale_list(envs.VLLM_PP_COMM_BANDWIDTH_GBPS),
            comm_latency_ms=latency,
            network=PPNetwork.from_json(envs.VLLM_PP_NETWORK)
            if envs.VLLM_PP_NETWORK
            else None,
            device_order=parse_device_order(envs.VLLM_PP_DEVICE_ORDER),
        )

    @property
    def enabled(self) -> bool:
        return (
            self.network is not None
            or any(s != 1.0 for s in self.compute_scales)
            or any(s != 1.0 for s in self.comm_scales)
        )

    def compute_scale(self, pp_rank: int) -> float:
        # Sleep emulation cannot speed up a device. Record the effective
        # factor so trace metadata agrees with stretch_after's behavior.
        return max(1.0, scale_at(self.compute_scales, self.device_id(pp_rank)))

    def device_id(self, pp_rank: int) -> int:
        if not self.device_order:
            return pp_rank
        if pp_rank not in range(len(self.device_order)):
            raise ValueError("PP rank is outside VLLM_PP_DEVICE_ORDER")
        return self.device_order[pp_rank]

    def validate_pp_size(self, pp_size: int) -> None:
        if self.device_order and len(self.device_order) != pp_size:
            raise ValueError("VLLM_PP_DEVICE_ORDER length must match PP size")
        if self.network and any(
            self.device_id(r) >= len(self.network.bandwidth_gbps)
            for r in range(pp_size)
        ):
            raise ValueError("PP device ID is outside the network matrix")

    def comm_scale(self, from_rank: int) -> float:
        """Hop ``from_rank -> from_rank+1``. Last rank has no hop."""
        return scale_at(self.comm_scales, from_rank)

    def stretch_compute(self, pp_rank: int, elapsed_ms: float) -> float:
        return stretch_after(elapsed_ms, self.compute_scale(pp_rank))

    def transfer_ms(self, from_rank: int, payload_bytes: int) -> float | None:
        """Modeled hop service time, independent of producer/consumer waiting.

        ``None`` means no baseline is available. Wire bytes exclude the TP
        replication reconstructed by the receiver's all-gather. Empty payloads
        have no simulated transfer cost.
        """
        if payload_bytes < 0:
            raise ValueError("PP payload_bytes must be >= 0")
        if self.network is not None:
            return self.network.delay_ms(
                self.device_id(from_rank), self.device_id(from_rank + 1), payload_bytes
            )
        if from_rank < 0 or from_rank >= len(self.comm_bandwidth_gbps):
            return None
        if payload_bytes == 0:
            return 0.0
        bandwidth = self.comm_bandwidth_gbps[from_rank]
        latency = scale_at(self.comm_latency_ms, from_rank, default=0.0)
        baseline_ms = latency + 8 * payload_bytes / (bandwidth * 1_000_000)
        return baseline_ms * self.comm_scale(from_rank)

    def _stretch_transfer(
        self, from_rank: int, elapsed_ms: float, payload_bytes: int
    ) -> float:
        if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            raise ValueError("PP elapsed_ms must be finite and >= 0")
        if self.network is not None:
            extra_ms = self.transfer_ms(from_rank, payload_bytes)
            if extra_ms:
                time.sleep(extra_ms / 1000)
            return elapsed_ms + extra_ms
        scale = self.comm_scale(from_rank)
        if scale == 1:
            return elapsed_ms
        transfer_ms = self.transfer_ms(from_rank, payload_bytes)
        assert transfer_ms is not None  # Validated by __post_init__.
        extra_ms = transfer_ms * (1 - 1 / scale)
        if extra_ms > 0:
            time.sleep(extra_ms / 1000)
        return elapsed_ms + extra_ms

    def stretch_send(
        self, pp_rank: int, elapsed_ms: float, *, payload_bytes: int
    ) -> float:
        return self._stretch_transfer(pp_rank, elapsed_ms, payload_bytes)

    def stretch_recv(
        self, pp_rank: int, elapsed_ms: float, *, payload_bytes: int
    ) -> float:
        return self._stretch_transfer(pp_rank - 1, elapsed_ms, payload_bytes)
