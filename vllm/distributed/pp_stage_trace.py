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
* ``send_tensor_spec`` — actual outgoing tensor layouts for idle link replay.

Tracing additionally synchronizes the device so each JSONL line is
self-contained. The ordinary NCCL send path in this checkout already inserts
a wait on the compute stream; tracing adds host synchronization overhead.
Leave the env unset for production.

``VLLM_PP_COMPUTE_MODEL=layer-measured`` additionally records every local
decoder layer using device events. The default shape-affine mode is stage-only.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

T = TypeVar("T")


def monotonic_clock_domain() -> str | None:
    """Only assert comparability on a shared Linux boot and time namespace.

    Do not use hostnames or wall-clock offsets to infer synchronized clocks on
    different hosts. Unknown platforms fail closed for cross-process timing.
    """
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        namespace = os.readlink("/proc/self/ns/time")
    except OSError:
        return None
    return f"{boot}/{namespace}/{time.get_clock_info('perf_counter').implementation}"


def tensor_dict_nbytes(
    tensors: dict[str, Any] | None,
    *,
    all_gather_size: int = 1,
    all_gather_tensors: dict[str, bool] | None = None,
) -> int:
    """Return PP wire bytes, accounting for send-slice/receiver-all-gather.

    The default counts the full payload. Pass the TP size and overrides used
    by send_tensor_dict to count the bytes sent by this TP rank.
    """
    if all_gather_size < 1:
        raise ValueError("all_gather_size must be >= 1")
    if not tensors:
        return 0
    total = 0
    for key, value in tensors.items():
        if isinstance(value, torch.Tensor):
            numel = value.numel()
            if all_gather_size > 1:
                gather = numel % all_gather_size == 0
                if all_gather_tensors:
                    gather = all_gather_tensors.get(key, gather)
                if gather:
                    numel //= all_gather_size
            total += numel * value.element_size()
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
    compute_scale: float = 1.0
    comm_scale: float = 1.0
    tp_size: int = 1
    batch_shape: list[dict[str, int]] | None = None
    compute_base_ms: float | None = None
    compute_wall_ms: float | None = None
    compute_delay_ms: float | None = None
    is_warmup: bool = False
    trace_id: str = ""
    send_tensor_spec: list[dict[str, Any]] | None = None
    trace_session: str | None = None
    clock_domain: str | None = None
    batch_id: str | None = None
    send_start_ns: int | None = None
    send_end_ns: int | None = None
    recv_start_ns: int | None = None
    recv_end_ns: int | None = None
    compute_model: str = "shape-affine"
    layer_compute_ms: dict[str, float] | None = None
    non_layer_compute_ms: float | None = None
    embedding_ms: float | None = None
    lm_head_ms: float | None = None
    final_norm_ms: float | None = None
    runner_overhead_ms: float | None = None
    comm_delay_in_window: bool = False


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
        tp_size: int = 1,
        compute_model: str = "shape-affine",
    ) -> None:
        if compute_model not in ("shape-affine", "layer-measured"):
            raise ValueError(f"unknown compute profiling mode: {compute_model}")
        self.compute_model = compute_model
        self._layer_measurement: dict[str, Any] = {}
        self._layer_timer = None
        self._layer_model = None
        os.makedirs(dump_dir, exist_ok=True)
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.device = device
        self.use_cuda = (
            torch.cuda.is_available() and device.type == "cuda"
            if use_cuda is None
            else use_cuda
        )
        path = os.path.join(dump_dir, f"pp_stage_pp{pp_rank}_tp{tp_rank}.jsonl")
        self._path = path
        # Kept open across steps; the worker calls close() on shutdown.
        self._fp = open(path, "a", encoding="utf-8")  # noqa: SIM115
        self._step = 0
        self.trace_id = uuid.uuid4().hex
        self.trace_session = os.environ.get("VLLM_PP_TRACE_SESSION")
        self.clock_domain = monotonic_clock_domain()
        self._comm_windows: dict[str, int] = {}
        self.is_warmup = False
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

    def _sync_device(self) -> None:
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        elif self.device.type == "npu":
            torch.npu.synchronize()

    def measure_compute(self, fn: Callable[[], T]) -> tuple[T, float]:
        """Time local GPU/NPU compute. Returns ``(result, elapsed_ms)``."""
        if self.use_cuda:
            assert self._start_event is not None and self._end_event is not None
            self._start_event.record()
            result = fn()
            self._end_event.record()
            self._end_event.synchronize()
            return result, float(self._start_event.elapsed_time(self._end_event))
        t0 = time.perf_counter()
        result = fn()
        self._sync_device()
        return result, (time.perf_counter() - t0) * 1000.0

    def measure_comm(
        self, fn: Callable[[], T], *, kind: str | None = None
    ) -> tuple[T, float]:
        """Time a blocking send/recv wait. Returns ``(result, elapsed_ms)``."""
        if kind not in (None, "send", "recv"):
            raise ValueError("communication kind must be send or recv")
        start_ns = time.perf_counter_ns()
        t0 = time.perf_counter()
        result = fn()
        self._sync_device()
        end_ns = time.perf_counter_ns()
        if kind:
            self._comm_windows.update(
                {f"{kind}_start_ns": start_ns, f"{kind}_end_ns": end_ns}
            )
        return result, (time.perf_counter() - t0) * 1000.0

    def finish_comm(self, kind: str) -> float:
        """Close the window after runtime mock delay, measuring actual wall time."""
        start = self._comm_windows[f"{kind}_start_ns"]
        end = time.perf_counter_ns()
        self._comm_windows[f"{kind}_end_ns"] = end
        self._comm_windows["comm_delay_in_window"] = True
        return (end - start) / 1e6

    def measure_stretched_compute(
        self,
        fn: Callable[[], T],
        stretch: Callable[[float], float],
        *,
        model_runner: Any = None,
        vllm_config: Any = None,
        compute_scale: float = 1.0,
    ) -> tuple[T, dict[str, float]]:
        """Keep modeled slowdown separate from actual synchronized wall time."""
        timer = None
        self._layer_measurement.clear()
        if self.compute_model == "layer-measured":
            from vllm.distributed.pp_layer_trace import PPLayerTimer

            if not getattr(
                getattr(vllm_config, "model_config", None), "enforce_eager", False
            ):
                raise ValueError("layer-measured requires --enforce-eager")
            compilation = getattr(vllm_config, "compilation_config", None)
            if getattr(compilation, "mode", 0) not in (None, 0):
                raise ValueError("layer-measured requires compilation mode NONE (0)")
            if self.tp_size != 1:
                raise ValueError("layer-measured currently requires TP=1")
            if compute_scale != 1.0:
                raise ValueError(
                    "layer-measured cannot attribute stage-level mock compute delay "
                    "to individual layers; disable compute slowdown"
                )
            model = getattr(model_runner, "model", None)
            if self._layer_timer is None or self._layer_model is not model:
                self._layer_timer = PPLayerTimer(model, self.device)
                self._layer_model = model
            timer = self._layer_timer
        with timer.capture() if timer else nullcontext():
            start = time.perf_counter()
            result, base_ms = self.measure_compute(fn)
            delay_start = time.perf_counter()
            modeled_ms = stretch(base_ms)
            end = time.perf_counter()
        if timer:
            layers = timer.elapsed_ms()
            # Includes endpoint modules and runner/launch work, NOT communication.
            residual = (end - start) * 1000 - sum(layers.values())
            if residual < -0.01:
                raise ValueError("layer timing exceeds synchronized stage wall time")
            endpoints = timer.endpoint_elapsed_ms()
            overhead = residual - sum(v for v in endpoints.values() if v is not None)
            if overhead < -0.01:
                raise ValueError("endpoint timing exceeds non-layer stage wall time")
            self._layer_measurement = dict(
                layer_compute_ms=layers,
                non_layer_compute_ms=max(0.0, residual),
                runner_overhead_ms=max(0.0, overhead),
                **endpoints,
            )
        return result, dict(
            compute_ms=modeled_ms,
            compute_base_ms=base_ms,
            compute_wall_ms=(end - start) * 1000,
            compute_delay_ms=(end - delay_start) * 1000,
        )

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
        compute_scale: float = 1.0,
        comm_scale: float = 1.0,
        batch_shape: list[dict[str, int]] | None = None,
        compute_base_ms: float | None = None,
        compute_wall_ms: float | None = None,
        compute_delay_ms: float | None = None,
        send_tensor_spec: list[dict[str, Any]] | None = None,
        batch_id: str | None = None,
    ) -> PPStageTraceRecord:
        if self.compute_model == "layer-measured" and not self._layer_measurement:
            raise ValueError("layer-measured record requires a fresh layer measurement")
        rec = PPStageTraceRecord(
            step=self._step,
            ts_unix=time.time(),
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            tp_size=self.tp_size,
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
            compute_scale=compute_scale,
            comm_scale=comm_scale,
            batch_shape=batch_shape,
            compute_base_ms=compute_base_ms,
            compute_wall_ms=compute_wall_ms,
            compute_delay_ms=compute_delay_ms,
            is_warmup=self.is_warmup,
            trace_id=self.trace_id,
            send_tensor_spec=send_tensor_spec,
            trace_session=self.trace_session,
            clock_domain=self.clock_domain,
            batch_id=batch_id,
            compute_model=self.compute_model,
            **self._layer_measurement,
            **self._comm_windows,
        )
        self._fp.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        self._fp.flush()
        self._records.append(rec)
        self._step += 1
        self._comm_windows.clear()
        self._layer_measurement.clear()
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
    tp_size = 1
    try:
        tp_rank = get_tp_group().rank_in_group
        tp_size = get_tp_group().world_size
    except Exception:
        pass
    return PPStageTracer(
        dump_dir=dump_dir,
        pp_rank=pp_group.rank_in_group,
        pp_size=pp_group.world_size,
        device=device,
        tp_rank=tp_rank,
        tp_size=tp_size,
        compute_model=envs.VLLM_PP_COMPUTE_MODEL,
    )
