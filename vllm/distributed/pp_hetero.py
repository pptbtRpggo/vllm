# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Emulate heterogeneous PP devices on homogeneous hardware.

One env var drives both ``vllm pp-profile`` and ``vllm serve``:

    VLLM_PP_HETERO=<compute_scales>[/<comm_scales>]

Examples::

    VLLM_PP_HETERO=1,2          # rank1 compute 2x slower
    VLLM_PP_HETERO=1,2/4        # plus hop 0->1 comm 4x slower
    VLLM_PP_HETERO=/4           # comm only

Missing ranks/hops default to ``1.0``. When this (or the legacy
``VLLM_PP_COMPUTE_SCALE`` / ``VLLM_PP_COMM_SCALE`` aliases) is set,
engine startup swaps Ascend ``NPUWorker`` for ``PPAscendWorker`` so
serve does not need ``--worker-cls``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import torch

from vllm.logger import init_logger

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


PP_ASCEND_WORKER = "vllm.v1.worker.pp_ascend_worker.PPAscendWorker"


def hetero_env_requested() -> bool:
    """True when the user asked for PP hetero emulation or stage tracing."""
    import os

    return bool(
        os.environ.get("VLLM_PP_HETERO")
        or os.environ.get("VLLM_PP_COMPUTE_SCALE")
        or os.environ.get("VLLM_PP_COMM_SCALE")
        or os.environ.get("VLLM_PP_STAGE_TRACE")
    )


def maybe_override_pp_worker(parallel_config: Any) -> None:
    """Use ``PPAscendWorker`` when hetero/trace is on and the platform picked NPUWorker.

    CUDA ``gpu_worker.Worker`` already stretches, so it is left alone.
    Must run *after* ``Platform.check_and_update_config``.
    """
    if not hetero_env_requested():
        return
    current = getattr(parallel_config, "worker_cls", None)
    if current == PP_ASCEND_WORKER:
        return
    current_s = "" if current is None else str(current)
    is_auto = current_s in ("", "auto")
    is_npu = any(
        marker in current_s.lower()
        for marker in ("vllm_ascend", "npuworker", "npu_worker")
    )
    if not is_auto and not is_npu:
        return
    if is_auto:
        try:
            import vllm_ascend  # noqa: F401
        except ImportError:
            return
    parallel_config.worker_cls = PP_ASCEND_WORKER
    logger.info("PP hetero/trace enabled: worker_cls=%s", PP_ASCEND_WORKER)


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

        compute, comm = parse_hetero_spec(envs.VLLM_PP_HETERO)
        if envs.VLLM_PP_COMPUTE_SCALE:
            compute = parse_scale_list(envs.VLLM_PP_COMPUTE_SCALE)
        if envs.VLLM_PP_COMM_SCALE:
            comm = parse_scale_list(envs.VLLM_PP_COMM_SCALE)
        return cls(compute_scales=compute, comm_scales=comm)

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
        """Write ``VLLM_PP_HETERO`` so profile and serve workers inherit it."""
        import os

        spec = format_hetero_spec(self.compute_scales, self.comm_scales)
        if spec is not None:
            os.environ["VLLM_PP_HETERO"] = spec
