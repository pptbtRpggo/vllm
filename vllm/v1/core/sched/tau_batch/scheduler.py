# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.tau_batch.config import (
    configure_tau_batch,
    validate_tau_batch_config,
    validate_tau_kv_layout,
)
from vllm.v1.core.sched.tau_batch.dispatch import (
    DispatchPhase,
    DispatchPolicy,
    DispatchSlot,
    ListDispatcher,
)
from vllm.v1.core.sched.tau_batch.interfaces import (
    ActiveMicroBatchSnapshot,
    ActiveRequestSnapshot,
    LatencyOracle,
    PipelineSnapshot,
)
from vllm.v1.core.sched.tau_batch.planner import TauBatchPlanner
from vllm.v1.core.sched.tau_batch.trace import JsonlTracer, resolve_trace_path
from vllm.v1.core.sched.tau_batch.types import (
    EosEvent,
    MicroBatchList,
    PackContext,
    TaskFeatures,
    TauRequestSnapshot,
    annotate_request_budget,
    estimate_kv_blocks,
    request_budget_dict,
    wait_ms,
)
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)

# Prototype defaults when sampling_params.extra_args has no SLO.
_DEFAULT_TTFT_SLO_MS = 10_000.0
_DEFAULT_TPOT_SLO_MS = 100.0


@dataclass(frozen=True)
class _InFlightForward:
    slot: DispatchSlot
    wave_id: int
    requests: tuple[Request, ...]
    fwd_id: int | None = None
    features: dict[str, Any] | None = None
    emit_mono_ns: int | None = None


def _resolve_max_microbatches(configured: int) -> int:
    """Return the micro-batch list cap. 0 means pack the whole pool."""
    if configured < 0:
        raise ValueError(f"max_microbatches must be >= 0, got {configured}")
    return configured


def _request_seq_len(request: Request, phase: DispatchPhase) -> int:
    """Length that enters the SCLS affine model for this phase.

    Prefill uses prompt length L. Called after _update_after_schedule,
    Decode uses the planned context length including this forward's input
    token. In-flight work is not necessarily present in device KV yet.
    """
    if phase is DispatchPhase.PREFILL:
        return int(request.num_prompt_tokens)
    return int(request.num_computed_tokens)


def snapshot_from_request(
    request: Request,
    now: float | None = None,
    pp_size: int | None = None,
) -> TauRequestSnapshot:
    """Build a planner snapshot from a vLLM Request without mutating it.

    Copies ``Request.arrival_time`` (recorded at engine intake). When
    ``now`` is given, also fills wait and slack.

    SLOs are read from ``sampling_params.extra_args`` when present.

    Args:
        request: Scheduler request. Must not be modified.
        now: Snapshot clock in seconds. None skips wait/slack.
        pp_size: Pipeline stages M for ``τ_max``.

    Returns:
        A TauRequestSnapshot for plan.
    """
    extra: dict[str, Any] = {}
    if request.sampling_params is not None and request.sampling_params.extra_args:
        extra = request.sampling_params.extra_args
    snap = TauRequestSnapshot(
        request_id=request.request_id,
        arrival_time=request.arrival_time,
        prompt_len=request.num_prompt_tokens,
        ttft_slo_ms=float(extra.get("ttft_slo_ms", _DEFAULT_TTFT_SLO_MS)),
        tpot_slo_ms=float(extra.get("tpot_slo_ms", _DEFAULT_TPOT_SLO_MS)),
        max_new_tokens=request.max_tokens,
    )
    if now is None:
        return snap
    return annotate_request_budget(snap, now, pp_size)


class TauScheduler(Scheduler):
    """List-aware scheduler: one DispatchSlot per schedule() call.

    The planner walks the waiting pool to build a micro-batch-task list.
    This class stamps a wave id when dispatch starts, then allocates that exact
    task and builds SchedulerOutput itself; it does not call Scheduler.schedule().
    If allocate_slots fails for one request, that request is finished with ERROR
    and the next request in the slot is tried.

    on_prefill_complete is recorded in update_from_output, not schedule().
    New arrivals stay in waiting until the active list has no unfinished
    admitted requests and all its issued forwards have returned.

    ``configure_vllm_config`` disables unsupported features before workers
    initialize. ``__init__`` only validates that execution contract.
    """

    configure_vllm_config = staticmethod(configure_tau_batch)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        vllm_config = kwargs.get("vllm_config")
        if vllm_config is None and args:
            vllm_config = args[0]
        if vllm_config is None:
            raise TypeError("TauScheduler requires vllm_config")
        validate_tau_batch_config(vllm_config)
        kv_cache_config = kwargs.get(
            "kv_cache_config", args[1] if len(args) > 1 else None
        )
        block_size = kwargs.get("block_size", args[3] if len(args) > 3 else None)
        if kv_cache_config is not None and block_size is not None:
            validate_tau_kv_layout(kv_cache_config, block_size)
        planner = kwargs.pop("planner", None)
        self.latency_oracle: LatencyOracle | None = kwargs.pop("latency_oracle", None)
        super().__init__(*args, **kwargs)
        self.planner = planner if planner is not None else TauBatchPlanner()
        self.dispatcher = ListDispatcher(DispatchPolicy.OVERLAP)
        self.max_reqs_per_microbatch = (
            self.scheduler_config.tau_batch_max_reqs_per_microbatch
        )
        self.max_microbatches = _resolve_max_microbatches(
            self.scheduler_config.tau_batch_max_microbatches,
        )
        self.min_waiting_to_plan = self.scheduler_config.tau_batch_min_waiting
        self._list: MicroBatchList | None = None
        self._wave_requests: dict[str, Request] = {}
        self._wave_id: int | None = None
        self._next_wave_id = 0
        self._inflight: dict[int, _InFlightForward] = {}
        self._pending_errors: dict[int, list[EngineCoreOutput]] = {}
        trace_path = resolve_trace_path(self.scheduler_config.tau_batch_trace)
        self._tracer: JsonlTracer | None = None
        if trace_path:
            self._tracer = JsonlTracer(trace_path, metadata=self._model_trace_fields())
            logger.warning(
                "TauScheduler JSONL trace: %s (created on first write)",
                trace_path,
            )
        logger.warning(
            "TauScheduler: pack the waiting pool into tasks of size <= %d, "
            "list cap %s (0 = whole pool), KV-unfit skipped. "
            "--max-num-seqs=%d limits each forward, not the whole list.",
            self.max_reqs_per_microbatch,
            self.max_microbatches,
            self.max_num_running_reqs,
        )
        if self.min_waiting_to_plan > 0:
            logger.warning(
                "TauScheduler: packing waits for %d waiting requests "
                "(--tau-batch-min-waiting). Set 0 to pack immediately.",
                self.min_waiting_to_plan,
            )

    def _trace(self, event: str, **fields: Any) -> None:
        if self._tracer is None:
            return
        self._tracer.record(event, **fields)

    def add_request(self, request: Request) -> None:
        # Reject per-request features through the normal terminal-output path,
        # not an exception that could stop the engine's input loop.
        super().add_request(request)
        unsupported = None
        if request.structured_output_request is not None:
            unsupported = "structured output"
        elif request.lora_request is not None:
            unsupported = "LoRA"
        elif request.has_encoder_inputs:
            unsupported = "multimodal input"
        elif request.pooling_params is not None:
            unsupported = "pooling"
        elif request.kv_transfer_params is not None:
            unsupported = "KV transfer"
        if unsupported is not None:
            self._drop_request(request, f"TauScheduler does not support {unsupported}")

    def trace_queue(
        self, action: str, scheduler_output: SchedulerOutput, depth: int
    ) -> None:
        """Record batch_queue push/pop. Called from EngineCore."""
        fwd_id = None
        inflight = self._inflight.get(id(scheduler_output))
        if inflight is not None:
            fwd_id = inflight.fwd_id
        self._trace(action, fwd_id=fwd_id, queue_depth=depth)

    def schedule(self) -> SchedulerOutput:
        self._ensure_active_list()
        if self._list is None:
            return self._empty_output()
        slot, allowed = self._next_slot()
        if slot is None:
            return self._empty_output()
        return self._schedule_slot(slot, allowed)

    def _empty_output(self) -> SchedulerOutput:
        # A cleanup forward must consume finished IDs and update the previous
        # worker-batch state just like a normal forward. A pure wait must not.
        if self.finished_req_ids:
            return self._emit_output([], [], [], {}, {})
        return SchedulerOutput.make_empty()

    def is_idle_output(self, scheduler_output: SchedulerOutput) -> bool:
        return (
            scheduler_output.total_num_scheduled_tokens == 0
            and not scheduler_output.finished_req_ids
            and not scheduler_output.free_encoder_mm_hashes
            and scheduler_output.kv_connector_metadata is None
            and scheduler_output.ec_connector_metadata is None
        )

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        slot = None
        fwd_id = None
        features: dict[str, Any] = {}
        emit_mono_ns: int | None = None
        wave_id = None
        inflight = self._inflight.pop(id(scheduler_output), None)
        if inflight is not None:
            slot = inflight.slot
            wave_id = inflight.wave_id
            fwd_id = inflight.fwd_id
            features = inflight.features or {}
            emit_mono_ns = inflight.emit_mono_ns
            # A cancelled ID can be submitted again while its old forward is
            # still in flight. Never apply those old tokens to the new Request.
            live_ids = {
                req.request_id
                for req in inflight.requests
                if self.requests.get(req.request_id) is req
            }
            if live_ids != scheduler_output.num_scheduled_tokens.keys():
                tokens = {
                    rid: n
                    for rid, n in scheduler_output.num_scheduled_tokens.items()
                    if rid in live_ids
                }
                update_output = replace(
                    scheduler_output,
                    num_scheduled_tokens=tokens,
                    total_num_scheduled_tokens=sum(tokens.values()),
                )
            else:
                update_output = scheduler_output
        else:
            update_output = scheduler_output
        prev_finished = set(self.finished_req_ids)
        result = super().update_from_output(update_output, model_runner_output)
        for client_index, errors in self._pending_errors.items():
            if client_index not in result:
                result[client_index] = EngineCoreOutputs()
            result[client_index].outputs.extend(errors)
        self._pending_errors.clear()
        current_wave = slot is not None and wave_id == self._wave_id
        if current_wave and slot is not None and slot.phase is DispatchPhase.PREFILL:
            self.dispatcher.on_prefill_complete(slot.microbatch_index)
        if slot is not None and self._tracer is not None:
            duration_ms = None
            if emit_mono_ns is not None:
                duration_ms = (time.monotonic_ns() - emit_mono_ns) / 1e6
            self._trace(
                "done",
                fwd_id=fwd_id,
                wave_id=wave_id,
                batch_idx=slot.microbatch_index,
                req_ids=list(scheduler_output.num_scheduled_tokens),
                duration_ms=duration_ms,
                **features,
            )
        if current_wave:
            self._maybe_run_eos_hook(slot, update_output, prev_finished)
        if (
            self._list is not None
            and not self._admitted_unfinished()
            and not self._inflight
        ):
            self._clear_list()
        return result

    def _maybe_run_eos_hook(
        self,
        slot: DispatchSlot | None,
        scheduler_output: SchedulerOutput,
        prev_finished: set[str],
    ) -> None:
        """Run the EOS strategy if this step finished admitted requests.

        The default strategy is a no-op. Called before the list is cleared
        so a later refill can still see remaining admitted ids.
        """
        if self._list is None:
            return
        finished_ids = tuple(
            sorted(
                (self.finished_req_ids - prev_finished)
                & set(scheduler_output.num_scheduled_tokens)
                & self._list.admitted_ids
            )
        )
        if not finished_ids:
            return
        remaining_ids = tuple(
            sorted(
                rid
                for rid in self._list.admitted_ids
                if rid not in finished_ids
                and (req := self.requests.get(rid)) is not None
                and self._wave_requests.get(rid) is req
                and not req.is_finished()
            )
        )
        waiting_ids = tuple(req.request_id for req in self.waiting)
        event = EosEvent(
            finished_ids=finished_ids,
            wave_id=self._wave_id,
            batch_idx=None if slot is None else slot.microbatch_index,
            phase=None if slot is None else slot.phase.value,
            remaining_ids=remaining_ids,
            waiting_ids=waiting_ids,
        )
        self._trace(
            "eos",
            wave_id=event.wave_id,
            batch_idx=event.batch_idx,
            phase=event.phase,
            finished_ids=list(event.finished_ids),
            remaining_ids=list(event.remaining_ids),
            waiting_ids=list(event.waiting_ids),
        )
        self.planner.on_eos(event)

    def _clear_list(self) -> None:
        assert not self._inflight, "cannot replace a wave with forwards in flight"
        if self._wave_id is not None:
            self._trace("wave_end", wave_id=self._wave_id)
        self._list = None
        self._wave_requests = {}
        self._wave_id = None
        self.dispatcher.reset()

    def _ensure_active_list(self) -> None:
        """Keep the unfinished list or replace it from the waiting pool."""
        if self._inflight or (self._list is not None and self._admitted_unfinished()):
            return
        self._clear_list()
        self._drop_unfittable_waiting()
        ctx = self._pack_context()
        snapshots = self._waiting_snapshot(ctx.now, ctx.pp_size)
        if not snapshots:
            return
        if self.min_waiting_to_plan > 0 and len(snapshots) < self.min_waiting_to_plan:
            return
        packed = self.planner.plan(snapshots, ctx)
        if packed is None:
            return
        self._list = packed
        self._wave_requests = {rid: self.requests[rid] for rid in packed.admitted_ids}
        self._wave_id = self._next_wave_id
        self._next_wave_id += 1
        self.dispatcher.start(packed)
        self._trace(
            "wave_plan",
            wave_id=self._wave_id,
            batches=[list(task.req_ids) for task in packed.tasks],
            deferred_ids=sorted(packed.deferred_ids),
            waiting=len(snapshots),
            pp_size=ctx.pp_size,
            policy=self.dispatcher.policy.value,
            requests=[request_budget_dict(req) for req in snapshots],
        )

    def _next_slot(self) -> tuple[DispatchSlot | None, set[str]]:
        assert self._list is not None
        # Retire BEFORE peek: an unissued cancelled Prefill can otherwise
        # block the readiness gate before we get a slot to skip.
        for task in self._list.tasks:
            slot = DispatchSlot(task.index, DispatchPhase.PREFILL)
            if not self._alive_ids(slot):
                self.dispatcher.retire(task.index)
        slot = self.dispatcher.peek_slot()
        return (None, set()) if slot is None else (slot, self._alive_ids(slot))

    def _schedule_slot(self, slot: DispatchSlot, allowed: set[str]) -> SchedulerOutput:
        assert self._list is not None
        reqs = [
            self.requests[rid]
            for rid in self._list.tasks[slot.microbatch_index].req_ids
            if rid in allowed
        ]
        # Validate the complete forward before allocating KV or mutating
        # request state. The packer skips non-fitting requests; a malformed
        # custom plan must not silently turn into a partial Prefill.
        planned_tokens = sum(max(0, self._num_new_tokens(req)) for req in reqs)
        self._validate_forward_requests(len(reqs))
        self._validate_forward_tokens(planned_tokens)
        scheduled_new: list[Request] = []
        scheduled_running: list[Request] = []
        scheduled_resumed: list[Request] = []
        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}

        for req in reqs:
            num_new = self._num_new_tokens(req)
            if num_new <= 0:
                continue
            kind = "running"
            if req.status == RequestStatus.WAITING:
                kind = "new"
            elif req.status == RequestStatus.PREEMPTED:
                kind = "resumed"

            new_blocks = self._allocate_request(req, num_new)
            if new_blocks is None:
                self._drop_request(req, "KV allocate_slots failed")
                continue

            self._accept_allocated(req)
            # New/resumed: replace the worker table (all ids). Running: append
            # only this step's new slots. Alternating B0/B1 re-adds the row;
            # resending the full table grows it until add_row overflows.
            if kind == "running":
                req_to_new_blocks[req.request_id] = new_blocks
            else:
                req_to_new_blocks[req.request_id] = self.kv_cache_manager.get_blocks(
                    req.request_id
                )
            num_scheduled_tokens[req.request_id] = num_new
            if kind == "new":
                scheduled_new.append(req)
            elif kind == "resumed":
                scheduled_resumed.append(req)
            else:
                scheduled_running.append(req)

        if not num_scheduled_tokens:
            return self._empty_output()

        out = self._emit_output(
            scheduled_new,
            scheduled_running,
            scheduled_resumed,
            req_to_new_blocks,
            num_scheduled_tokens,
        )
        self.dispatcher.commit_slot(slot)
        assert self._wave_id is not None
        scheduled = tuple(req for req in reqs if req.request_id in num_scheduled_tokens)
        forward = _InFlightForward(slot, self._wave_id, scheduled)
        if self._tracer is None:
            self._inflight[id(out)] = forward
            return out
        features = self._task_features(
            slot, list(scheduled), int(out.total_num_scheduled_tokens)
        ).as_dict()
        now = time.time()
        features["req_waits"] = [
            {
                "req_id": req.request_id,
                "arrival_time": req.arrival_time,
                "wait_ms": wait_ms(req.arrival_time, now),
                "s": _request_seq_len(req, slot.phase),
            }
            for req in scheduled
        ]
        out.tau_task = {
            k: features[k]
            for k in ("phase", "n", "s_max", "s_sum", "tokens", "pp_size", "seq_lens")
            if k in features
        }
        fwd_id = self._tracer.next_fwd_id()
        out.tau_fwd_id = fwd_id
        self._inflight[id(out)] = replace(
            forward,
            fwd_id=fwd_id,
            features=features,
            emit_mono_ns=time.monotonic_ns(),
        )
        self._trace(
            "emit",
            fwd_id=fwd_id,
            wave_id=self._wave_id,
            batch_idx=slot.microbatch_index,
            req_ids=list(num_scheduled_tokens),
            **features,
        )
        return out

    def _validate_forward_tokens(self, tokens: int) -> None:
        if tokens > self.max_num_scheduled_tokens:
            raise ValueError(
                f"TauScheduler forward has {tokens} tokens, "
                f"max_num_batched_tokens is {self.max_num_scheduled_tokens}; "
                "repack complete requests without chunked prefill"
            )

    def _validate_forward_requests(self, count: int) -> None:
        if count > self.max_num_running_reqs:
            raise ValueError(
                f"TauScheduler forward has {count} requests, "
                f"max_num_seqs is {self.max_num_running_reqs}"
            )

    def _num_new_tokens(self, request: Request) -> int:
        num_new = (
            request.num_tokens
            + request.num_output_placeholders
            - request.num_computed_tokens
        )
        return min(num_new, self.max_model_len - 1 - request.num_computed_tokens)

    def _allocate_request(
        self, request: Request, num_new_tokens: int
    ) -> KVCacheBlocks | None:
        if request.has_encoder_inputs or self.connector is not None:
            return None
        if request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
            computed_blocks, num_local = self.kv_cache_manager.get_computed_blocks(
                request
            )
            return self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_local,
                computed_blocks,
                num_lookahead_tokens=0,
            )
        return self.kv_cache_manager.allocate_slots(
            request,
            num_new_tokens,
            num_lookahead_tokens=self.num_lookahead_tokens,
        )

    def _accept_allocated(self, request: Request) -> None:
        if request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
            if request.status == RequestStatus.WAITING:
                request.num_computed_tokens = 0
            if request.num_cached_tokens < 0:
                request.num_cached_tokens = request.num_computed_tokens
            self.waiting.remove_request(request)
            self.running.append(request)
            request.status = RequestStatus.RUNNING
            if self.log_stats:
                request.record_event(EngineCoreEventType.SCHEDULED, time.monotonic())

    def _drop_request(self, request: Request, reason: str, **trace_fields: Any) -> None:
        if request.is_finished():
            return
        logger.error(
            "TauScheduler dropping request %s: %s",
            request.request_id,
            reason,
        )
        self._trace("drop", req_id=request.request_id, reason=reason, **trace_fields)
        self.finish_requests(request.request_id, RequestStatus.FINISHED_ERROR)
        self._pending_errors.setdefault(request.client_index, []).append(
            EngineCoreOutput(
                request_id=request.request_id,
                new_token_ids=[],
                finish_reason=request.get_finished_reason(),
                events=request.take_events(),
                trace_headers=request.trace_headers,
                num_cached_tokens=request.num_cached_tokens,
            )
        )

    def _drop_unfittable_waiting(self) -> None:
        free = self.kv_cache_manager.block_pool.get_num_free_blocks()
        for req in list(self.waiting):
            need = estimate_kv_blocks(
                req.num_prompt_tokens, req.max_tokens, self.block_size
            )
            if need > free:
                self._drop_request(
                    req,
                    f"reserved {need} KV blocks exceeds free {free}",
                    need_blocks=need,
                    free_blocks=free,
                )

    def _emit_output(
        self,
        scheduled_new: list[Request],
        scheduled_running: list[Request],
        scheduled_resumed: list[Request],
        req_to_new_blocks: dict[str, KVCacheBlocks],
        num_scheduled_tokens: dict[str, int],
    ) -> SchedulerOutput:
        self._validate_forward_requests(len(num_scheduled_tokens))
        self._validate_forward_tokens(sum(num_scheduled_tokens.values()))
        if self.use_v2_model_runner:
            scheduled_new = scheduled_new + scheduled_resumed
            scheduled_resumed = []
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new
            ]
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running,
            scheduled_resumed,
            num_scheduled_tokens,
            {},
            req_to_new_blocks,
        )
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        if self.running:
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    self.running[0].request_id
                )
            )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=num_common_prefix_blocks,
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
        )
        self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _waiting_snapshot(
        self, now: float, pp_size: int | None
    ) -> list[TauRequestSnapshot]:
        return [snapshot_from_request(req, now, pp_size) for req in self.waiting]

    def _task_features(
        self,
        slot: DispatchSlot,
        reqs: list[Request],
        tokens: int,
    ) -> TaskFeatures:
        seq_lens = tuple(_request_seq_len(req, slot.phase) for req in reqs)
        return TaskFeatures(
            phase=slot.phase.value,
            n=len(reqs),
            s_max=max(seq_lens) if seq_lens else 0,
            s_sum=sum(seq_lens),
            tokens=tokens,
            pp_size=max(1, self.parallel_config.pipeline_parallel_size),
            seq_lens=seq_lens,
        )

    def _model_trace_fields(self) -> dict[str, Any]:
        cfg = getattr(self.vllm_config.model_config, "hf_text_config", None)
        if cfg is None:
            cfg = getattr(self.vllm_config.model_config, "hf_config", None)
        fields: dict[str, Any] = {
            "model": self.vllm_config.model_config.model,
            "dtype": str(self.vllm_config.model_config.dtype),
            "pp_size": self.parallel_config.pipeline_parallel_size,
            "tp_size": self.parallel_config.tensor_parallel_size,
            "kv_cache_dtype": self.vllm_config.cache_config.cache_dtype,
            "worker_cls": str(self.parallel_config.worker_cls),
        }
        if cfg is None:
            return fields
        for name in (
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "num_hidden_layers",
            "intermediate_size",
        ):
            val = getattr(cfg, name, None)
            if isinstance(val, int):
                fields[name] = val
        return fields

    def _pack_context(self) -> PackContext:
        pp = max(1, self.parallel_config.pipeline_parallel_size)
        return PackContext(
            now=time.time(),
            max_num_seqs=self.max_num_running_reqs,
            max_microbatches=self.max_microbatches,
            max_reqs_per_microbatch=self.max_reqs_per_microbatch,
            pp_size=pp,
            kv_free_blocks=self.kv_cache_manager.block_pool.get_num_free_blocks(),
            block_size=self.block_size,
            max_num_batched_tokens=self.max_num_scheduled_tokens,
            oracle=self.latency_oracle,
        )

    def pipeline_snapshot(self) -> PipelineSnapshot:
        """Expose the live execution state for future event-driven policies.

        This does not run an online policy or apply refill proposals. The
        default scheduler still drains a fixed list before packing another.
        """
        ctx = self._pack_context()
        active = []
        admitted = self._list.admitted_ids if self._list else frozenset()
        if self._list is not None:
            for task in self._list.tasks:
                state = self.dispatcher.task_state(task.index)
                requests = tuple(
                    ActiveRequestSnapshot(
                        request=snapshot_from_request(req, ctx.now, ctx.pp_size),
                        computed_tokens=req.num_computed_tokens,
                        output_tokens=req.num_output_tokens,
                        kv_block_ids=tuple(
                            tuple(ids)
                            for ids in self.kv_cache_manager.get_block_ids(rid)
                        ),
                    )
                    for rid in task.req_ids
                    if (req := self.requests.get(rid)) is not None
                    and self._wave_requests.get(rid) is req
                    and not req.is_finished()
                )
                active.append(
                    ActiveMicroBatchSnapshot(
                        index=task.index,
                        requests=requests,
                        prefill_dispatched=state.prefill_dispatched,
                        prefill_completed=state.prefill_completed,
                        retired=state.retired or not requests,
                        in_flight=sum(
                            forward.slot.microbatch_index == task.index
                            and forward.wave_id == self._wave_id
                            for forward in self._inflight.values()
                        ),
                    )
                )
        return PipelineSnapshot(
            context=ctx,
            wave_id=self._wave_id,
            active_plan=(
                replace(self._list, extra=deepcopy(dict(self._list.extra)))
                if self._list
                else None
            ),
            active=tuple(active),
            waiting=tuple(
                snapshot_from_request(req, ctx.now, ctx.pp_size)
                for req in self.waiting
                if req.request_id not in admitted
                or self._wave_requests.get(req.request_id) is not req
            ),
        )

    def _alive_ids(self, slot: DispatchSlot) -> set[str]:
        assert self._list is not None
        ids = self._list.tasks[slot.microbatch_index].req_ids
        alive: set[str] = set()
        for rid in ids:
            req = self.requests.get(rid)
            if (
                req is not None
                and self._wave_requests.get(rid) is req
                and not req.is_finished()
            ):
                alive.add(rid)
        return alive

    def _admitted_unfinished(self) -> bool:
        if self._list is None:
            return False
        for rid in self._list.admitted_ids:
            req = self.requests.get(rid)
            if (
                req is not None
                and self._wave_requests.get(rid) is req
                and not req.is_finished()
            ):
                return True
        return False
