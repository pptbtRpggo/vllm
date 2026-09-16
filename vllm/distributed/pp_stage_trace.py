# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-PP-rank compute / communication stage tracer.

Enabled by ``VLLM_PP_STAGE_TRACE=/path/to/dir``. Each PP rank writes
``pp_stage_pp{rank}_tp{tp}.jsonl`` with one record per forward step:

* ``compute_ms`` — local stage forward (all layers on this rank, plus
  logits on the last rank). CUDA-event timed when CUDA is available.
* ``recv_ms`` / ``send_ms`` — blocking wait for the previous/next rank's
  intermediate-tensor transfer. Timed with CPU clock + device sync,
  because NCCL runs on its own stream.

When tracing is on, send is waited in the same step so each JSONL line
is self-contained. That disables send/next-step overlap for the traced
run; leave the env unset for production.

This is stage-level (one number per PP rank per step), not per-layer.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, TypeVar

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

T = TypeVar("T")


def tensor_dict_nbytes(tensors: dict[str, Any] | None) -> int:
    """Return the payload size of a PP intermediate-tensor dict."""
    if not tensors:
        return 0
    total = 0
    for value in tensors.values():
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
    return total


def layer_range_from_runner(model_runner: Any) -> tuple[int | None, int | None]:
    """Read ``start_layer`` / ``end_layer`` off a loaded model, if present."""
    model = getattr(model_runner, "model", None)
    if model is None:
        return None, None
    inner = getattr(model, "model", model)
    start = getattr(inner, "start_layer", None)
    end = getattr(inner, "end_layer", None)
    if isinstance(start, int) and isinstance(end, int):
        return start, end
    return None, None


@dataclass
class PPStageTraceRecord:
    step: int
    ts_unix: float
    pp_rank: int
    pp_size: int
    tp_rank: int
    start_layer: int | None
    end_layer: int | None
    num_tokens: int
    num_reqs: int
    num_ctx_requests: int
    num_ctx_tokens: int
    num_generation_requests: int
    num_generation_tokens: int
    recv_ms: float | None
    compute_ms: float
    send_ms: float | None
    recv_bytes: int | None
    send_bytes: int | None


class PPStageTracer:
    """Append-only JSONL tracer for one PP rank."""

    def __init__(
        self,
        dump_dir: str,
        pp_rank: int,
        pp_size: int,
        device: torch.device,
        tp_rank: int = 0,
        use_cuda: bool | None = None,
    ) -> None:
        os.makedirs(dump_dir, exist_ok=True)
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.tp_rank = tp_rank
        self.device = device
        self.use_cuda = (
            torch.cuda.is_available() and device.type == "cuda"
            if use_cuda is None
            else use_cuda
        )
        path = os.path.join(dump_dir, f"pp_stage_pp{pp_rank}_tp{tp_rank}.jsonl")
        self._path = path
        self._fp = open(path, "a", encoding="utf-8")
        self._step = 0
        self._records: list[PPStageTraceRecord] = []
        if self.use_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
        else:
            self._start_event = None
            self._end_event = None
        logger.info(
            "PP stage tracer enabled: %s (pp_rank=%s/%s)",
            path,
            pp_rank,
            pp_size,
        )

    @property
    def path(self) -> str:
        return self._path

    def measure_compute(self, fn: Callable[[], T]) -> tuple[T, float]:
        """Time local GPU compute. Returns ``(result, elapsed_ms)``."""
        if self.use_cuda:
            assert self._start_event is not None and self._end_event is not None
            self._start_event.record()
            result = fn()
            self._end_event.record()
            self._end_event.synchronize()
            return result, float(self._start_event.elapsed_time(self._end_event))
        t0 = time.perf_counter()
        result = fn()
        return result, (time.perf_counter() - t0) * 1000.0

    def measure_comm(self, fn: Callable[[], T]) -> tuple[T, float]:
        """Time a blocking send/recv wait. Returns ``(result, elapsed_ms)``."""
        t0 = time.perf_counter()
        result = fn()
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        return result, (time.perf_counter() - t0) * 1000.0

    def record(
        self,
        *,
        num_tokens: int,
        num_reqs: int,
        num_ctx_requests: int,
        num_ctx_tokens: int,
        num_generation_requests: int,
        num_generation_tokens: int,
        compute_ms: float,
        recv_ms: float | None,
        send_ms: float | None,
        recv_bytes: int | None,
        send_bytes: int | None,
        start_layer: int | None,
        end_layer: int | None,
    ) -> PPStageTraceRecord:
        rec = PPStageTraceRecord(
            step=self._step,
            ts_unix=time.time(),
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            tp_rank=self.tp_rank,
            start_layer=start_layer,
            end_layer=end_layer,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_ctx_requests=num_ctx_requests,
            num_ctx_tokens=num_ctx_tokens,
            num_generation_requests=num_generation_requests,
            num_generation_tokens=num_generation_tokens,
            recv_ms=recv_ms,
            compute_ms=compute_ms,
            send_ms=send_ms,
            recv_bytes=recv_bytes,
            send_bytes=send_bytes,
        )
        self._fp.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        self._fp.flush()
        self._records.append(rec)
        self._step += 1
        return rec

    def close(self) -> None:
        if self._fp.closed:
            return
        if self._records:
            comps = [r.compute_ms for r in self._records]
            recvs = [r.recv_ms for r in self._records if r.recv_ms is not None]
            sends = [r.send_ms for r in self._records if r.send_ms is not None]
            logger.info(
                "PP stage tracer pp_rank=%s: %d steps, "
                "mean compute_ms=%.3f recv_ms=%s send_ms=%s -> %s",
                self.pp_rank,
                len(self._records),
                sum(comps) / len(comps),
                f"{sum(recvs) / len(recvs):.3f}" if recvs else "n/a",
                f"{sum(sends) / len(sends):.3f}" if sends else "n/a",
                self._path,
            )
        self._fp.close()


def maybe_create_pp_stage_tracer(
    device: torch.device,
    dump_dir: str | None = None,
) -> PPStageTracer | None:
    """Create a tracer if ``VLLM_PP_STAGE_TRACE`` (or ``dump_dir``) is set."""
    import vllm.envs as envs
    from vllm.distributed.parallel_state import get_pp_group, get_tp_group

    if dump_dir is None:
        dump_dir = envs.VLLM_PP_STAGE_TRACE
    if not dump_dir:
        return None
    pp_group = get_pp_group()
    if pp_group.world_size <= 1:
        logger.warning(
            "VLLM_PP_STAGE_TRACE=%s is set but pipeline_parallel_size is 1; "
            "tracing disabled.",
            dump_dir,
        )
        return None
    tp_rank = 0
    try:
        tp_rank = get_tp_group().rank_in_group
    except Exception:
        pass
    return PPStageTracer(
        dump_dir=dump_dir,
        pp_rank=pp_group.rank_in_group,
        pp_size=pp_group.world_size,
        device=device,
        tp_rank=tp_rank,
    )
