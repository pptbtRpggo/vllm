# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Emulate heterogeneous PP devices on homogeneous hardware.

One env var drives both ``vllm pp-profile`` and ``vllm serve``:

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
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import torch

from vllm.logger import init_logger
from vllm.pp_hetero_env import (
    PP_ASCEND_WORKER,
    hetero_env_requested,
    maybe_override_pp_worker,
)

logger = init_logger(__name__)

T = TypeVar("T")


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


def format_scale_list(scales: Sequence[float]) -> str | None:
    if not scales:
        return None
    parts: list[str] = []
    for value in scales:
        if value == int(value):
            parts.append(str(int(value)))
        else:
            parts.append(str(value))
    return ",".join(parts)


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


def format_hetero_spec(
    compute_scales: Sequence[float] = (),
    comm_scales: Sequence[float] = (),
) -> str | None:
    compute = format_scale_list(compute_scales) or ""
    comm = format_scale_list(comm_scales) or ""
    if not compute and not comm:
        return None
    if not comm:
        return compute
    return f"{compute}/{comm}"


def hetero_spec_from_text(
    compute_scale: str | None = None,
    comm_scale: str | None = None,
) -> str | None:
    """Build a ``VLLM_PP_HETERO`` string from the two CLI lists."""
    compute = (compute_scale or "").strip()
    comm = (comm_scale or "").strip()
    if not compute and not comm:
        return None
    if not comm:
        return compute
    return f"{compute}/{comm}"


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

    def __post_init__(self) -> None:
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
        )

    @classmethod
    def from_text(
        cls,
        compute_scale: str | None = None,
        comm_scale: str | None = None,
        *,
        comm_bandwidth_gbps: str | None = None,
        comm_latency_ms: str | None = None,
    ) -> PPHeteroConfig:
        return cls(
            compute_scales=parse_scale_list(compute_scale),
            comm_scales=parse_scale_list(comm_scale),
            comm_bandwidth_gbps=parse_scale_list(comm_bandwidth_gbps),
            comm_latency_ms=tuple(
                float(value.strip())
                for value in (comm_latency_ms or "").split(",")
                if value.strip()
            ),
        )

    @property
    def enabled(self) -> bool:
        return any(s != 1.0 for s in self.compute_scales) or any(
            s != 1.0 for s in self.comm_scales
        )

    def compute_scale(self, pp_rank: int) -> float:
        # Sleep emulation cannot speed up a device. Record the effective
        # factor so replay metadata agrees with stretch_after's behavior.
        return max(1.0, scale_at(self.compute_scales, pp_rank))

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

    def export_env(self) -> None:
        """Write ``VLLM_PP_HETERO`` so profile and serve workers inherit it."""
        import os

        spec = format_hetero_spec(self.compute_scales, self.comm_scales)
        if spec is not None:
            os.environ["VLLM_PP_HETERO"] = spec
        bandwidth = format_scale_list(self.comm_bandwidth_gbps)
        latency = format_scale_list(self.comm_latency_ms)
        if bandwidth is not None:
            os.environ["VLLM_PP_COMM_BANDWIDTH_GBPS"] = bandwidth
        if latency is not None:
            os.environ["VLLM_PP_COMM_LATENCY_MS"] = latency
