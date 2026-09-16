# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Emulate heterogeneous PP devices on homogeneous hardware.

EdgeShard's DP only sees per-rank ``t_layer`` and per-hop ``t_comm``.
On a box of identical NPUs those can be faked by stretching wall-clock
time after each stage op:

* ``VLLM_PP_COMPUTE_SCALE`` — comma list, one factor per PP rank.
  Rank ``r`` sleeps extra after forward so compute looks ``scale[r]``
  times slower.
* ``VLLM_PP_COMM_SCALE`` — comma list, one factor per hop ``r -> r+1``.
  The sender sleeps after send and the receiver sleeps after recv, so
  the next stage cannot start early.

Missing entries default to ``1.0``. Live ``pp-profile`` exports these
into the worker env so traces already contain the stretched times.
``--skip-run`` instead multiplies fitted costs (no extra engine run).

Serve with the same env vars, otherwise the partition and the pipeline
disagree.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import torch

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
        if value <= 0.0:
            raise ValueError(f"PP hetero scale must be > 0, got {value}")
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

    @classmethod
    def from_env(cls) -> PPHeteroConfig:
        import vllm.envs as envs

        return cls(
            compute_scales=parse_scale_list(envs.VLLM_PP_COMPUTE_SCALE),
            comm_scales=parse_scale_list(envs.VLLM_PP_COMM_SCALE),
        )

    @classmethod
    def from_text(
        cls,
        compute_scale: str | None = None,
        comm_scale: str | None = None,
    ) -> PPHeteroConfig:
        return cls(
            compute_scales=parse_scale_list(compute_scale),
            comm_scales=parse_scale_list(comm_scale),
        )

    @property
    def enabled(self) -> bool:
        return any(s != 1.0 for s in self.compute_scales) or any(
            s != 1.0 for s in self.comm_scales
        )

    def compute_scale(self, pp_rank: int) -> float:
        return scale_at(self.compute_scales, pp_rank)

    def comm_scale(self, from_rank: int) -> float:
        """Hop ``from_rank -> from_rank+1``. Last rank has no hop."""
        return scale_at(self.comm_scales, from_rank)

    def stretch_compute(self, pp_rank: int, elapsed_ms: float) -> float:
        return stretch_after(elapsed_ms, self.compute_scale(pp_rank))

    def stretch_send(self, pp_rank: int, elapsed_ms: float) -> float:
        return stretch_after(elapsed_ms, self.comm_scale(pp_rank))

    def stretch_recv(self, pp_rank: int, elapsed_ms: float) -> float:
        return stretch_after(elapsed_ms, self.comm_scale(pp_rank - 1))

    def export_env(self) -> None:
        """Write the scales into ``os.environ`` so PP workers inherit them."""
        import os

        compute = format_scale_list(self.compute_scales)
        comm = format_scale_list(self.comm_scales)
        if compute is not None:
            os.environ["VLLM_PP_COMPUTE_SCALE"] = compute
        if comm is not None:
            os.environ["VLLM_PP_COMM_SCALE"] = comm
