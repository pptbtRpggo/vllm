# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Any

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.tau_batch.dispatch import (
    DispatchPhase,
    DispatchSlot,
    WaveDispatcher,
    WaveDispatchPolicy,
)
from vllm.v1.core.sched.tau_batch.planner import TauBatchPlanner
from vllm.v1.core.sched.tau_batch.trace import JsonlTracer, resolve_trace_path
from vllm.v1.core.sched.tau_batch.types import (
    PackContext,
    TauRequestSnapshot,
    WavePlan,
    estimate_kv_blocks,
)
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)

# Prototype defaults when sampling_params.extra_args has no SLO.
_DEFAULT_TTFT_SLO_MS = 10_000.0
_DEFAULT_TPOT_SLO_MS = 100.0


def _resolve_max_microbatches(
    configured: int, max_num_seqs: int, max_reqs_per_microbatch: int
) -> int:
    """Return P. 0 means enough batches to hold one full wave at the
    per-micro-batch size. Per-batch size is never derived from P.
    """
    if configured >= 1:
        return configured
    return max(1, cdiv(max_num_seqs, max_reqs_per_microbatch))


def _enforce_wave_runtime_contract(vllm_config: VllmConfig) -> None:
    """Turn off features that split one Prefill slot across forwards.

    Tau-batch treats one ``schedule()`` as one complete Prefill or Decode
    slot. Chunked prefill, prefix-cache hits, long-prefill caps,
    speculative decode, and async scheduling break that contract. This
    mutates ``vllm_config`` before ``Scheduler.__init__`` so the parent
    KV manager and token budget see the disabled flags.
    """
    disabled: list[str] = []
    sched = vllm_config.scheduler_config
    cache = vllm_config.cache_config

    if sched.enable_chunked_prefill:
        sched.enable_chunked_prefill = False
        disabled.append("enable_chunked_prefill")
    if sched.long_prefill_token_threshold != 0:
        sched.long_prefill_token_threshold = 0
        disabled.append("long_prefill_token_threshold")
    if sched.max_num_partial_prefills > 1:
        sched.max_num_partial_prefills = 1
        disabled.append("max_num_partial_prefills")
    if sched.async_scheduling:
        sched.async_scheduling = False
        disabled.append("async_scheduling")
    if cache.enable_prefix_caching:
        cache.enable_prefix_caching = False
        disabled.append("enable_prefix_caching")
    if vllm_config.speculative_config is not None:
        vllm_config.speculative_config = None
        disabled.append("speculative_config")

    if disabled:
        logger.warning(
            "TauScheduler disabled unsupported features: %s. "
            "These would split a micro-batch Prefill across forwards.",
            ", ".join(disabled),
        )


def snapshot_from_request(request: Request) -> TauRequestSnapshot:
    """Build a planner snapshot from a vLLM Request without mutating it.

    SLOs are read from ``sampling_params.extra_args`` when present.

    Args:
        request: Scheduler request. Must not be modified.

    Returns:
        A TauRequestSnapshot for plan_wave.
    """
    extra: dict[str, Any] = {}
    if request.sampling_params is not None and request.sampling_params.extra_args:
        extra = request.sampling_params.extra_args
    return TauRequestSnapshot(
        request_id=request.request_id,
        arrival_time=request.arrival_time,
        prompt_len=request.num_prompt_tokens,
        ttft_slo_ms=float(extra.get("ttft_slo_ms", _DEFAULT_TTFT_SLO_MS)),
        tpot_slo_ms=float(extra.get("tpot_slo_ms", _DEFAULT_TPOT_SLO_MS)),
        max_new_tokens=request.max_tokens,
    )


class TauScheduler(Scheduler):
    """Wave-aware scheduler: one DispatchSlot per schedule() call.

    plan_wave reserves KV for prompt plus max generate length. This
    class allocates that exact micro-batch and builds SchedulerOutput
    itself; it does not call Scheduler.schedule(). If allocate_slots
    fails for one request, that request is finished with ERROR and the
    next request in the slot is tried.

    on_prefill_complete is recorded in update_from_output, not schedule().
    New arrivals stay in waiting until the active wave has no unfinished
    admitted requests.

    ``__init__`` disables chunked prefill, prefix caching, long-prefill
    caps, speculative decode, and async scheduling so a Prefill slot is
    not split across forwards.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        vllm_config = kwargs.get("vllm_config")
        if vllm_config is None and args:
            vllm_config = args[0]
        if vllm_config is None:
            raise TypeError("TauScheduler requires vllm_config")
        _enforce_wave_runtime_contract(vllm_config)
        super().__init__(*args, **kwargs)
        self.planner = TauBatchPlanner()
        self.dispatcher = WaveDispatcher(WaveDispatchPolicy.OVERLAP)
        self.max_reqs_per_microbatch = (
            self.scheduler_config.tau_batch_max_reqs_per_microbatch
        )
        self.max_microbatches = _resolve_max_microbatches(
            self.scheduler_config.tau_batch_max_microbatches,
            self.max_num_running_reqs,
            self.max_reqs_per_microbatch,
        )
        self.min_waiting_to_plan = self.scheduler_config.tau_batch_min_waiting
        self._wave: WavePlan | None = None
        self._inflight: dict[int, tuple[DispatchSlot, int]] = {}
        trace_path = resolve_trace_path(self.scheduler_config.tau_batch_trace)
        self._tracer: JsonlTracer | None = None
        if trace_path:
            self._tracer = JsonlTracer(trace_path)
            logger.warning(
                "TauScheduler JSONL trace: %s (created on first write)",
                trace_path,
            )
        logger.warning(
            "TauScheduler: take at most %d waiting, pack size <= %d, "
            "at most %d micro-batches (overflow deferred, no padding).",
            self.max_num_running_reqs,
            self.max_reqs_per_microbatch,
            self.max_microbatches,
        )
        if self.min_waiting_to_plan > 0:
            logger.warning(
                "TauScheduler: plan_wave waits for %d waiting requests "
                "(--tau-batch-min-waiting). Set 0 to plan immediately.",
                self.min_waiting_to_plan,
            )

    def _trace(self, event: str, **fields: Any) -> None:
        if self._tracer is None:
            return
        self._tracer.record(event, **fields)

    def trace_queue(
        self, action: str, scheduler_output: SchedulerOutput, depth: int
    ) -> None:
        """Record batch_queue push/pop. Called from EngineCore."""
        fwd_id = None
        inflight = self._inflight.get(id(scheduler_output))
        if inflight is not None:
            fwd_id = inflight[1]
        self._trace(action, fwd_id=fwd_id, queue_depth=depth)

    def schedule(self) -> SchedulerOutput:
        for _ in range(2):
            self._maybe_start_wave()
            if self._wave is None:
                return SchedulerOutput.make_empty()
            slot, allowed = self._next_slot()
            if slot is None:
                if not self._admitted_unfinished():
                    self._clear_wave()
                    continue
                return SchedulerOutput.make_empty()
            return self._schedule_slot(slot, allowed)
        return SchedulerOutput.make_empty()

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        slot = None
        fwd_id = None
        inflight = self._inflight.pop(id(scheduler_output), None)
        if inflight is not None:
            slot, fwd_id = inflight
        result = super().update_from_output(scheduler_output, model_runner_output)
        if slot is not None and slot.phase is DispatchPhase.PREFILL:
            self.dispatcher.on_prefill_complete(slot.microbatch_index)
        if slot is not None:
            self._trace(
                "done",
                fwd_id=fwd_id,
                wave_id=self._wave.wave_id if self._wave is not None else None,
                batch_idx=slot.microbatch_index,
                phase=slot.phase.value,
                req_ids=list(scheduler_output.num_scheduled_tokens),
            )
        if self._wave is not None and not self._admitted_unfinished():
            self._clear_wave()
        return result

    def _clear_wave(self) -> None:
        if self._wave is not None:
            self._trace("wave_end", wave_id=self._wave.wave_id)
        self._wave = None
        self.dispatcher.reset()

    def _maybe_start_wave(self) -> None:
        if self._wave is not None and self._admitted_unfinished():
            return
        self._clear_wave()
        self._drop_unfittable_waiting()
        snapshots = self._waiting_snapshot()
        if not snapshots:
            return
        if (
            self.min_waiting_to_plan > 0
            and len(snapshots) < self.min_waiting_to_plan
        ):
            return
        plan = self.planner.plan_wave(snapshots, self._pack_context())
        if plan is None:
            return
        self._wave = plan
        self.dispatcher.start(plan)
        self._trace(
            "wave_plan",
            wave_id=plan.wave_id,
            batches=[list(mb.req_ids) for mb in plan.microbatches],
            deferred_ids=sorted(plan.deferred_ids),
            waiting=len(snapshots),
            pp_size=self.parallel_config.pipeline_parallel_size,
            policy=self.dispatcher.policy.value,
        )

    def _next_slot(self) -> tuple[DispatchSlot | None, set[str]]:
        assert self._wave is not None
        p = len(self._wave.microbatches)
        skips = 0
        while skips < p:
            slot = self.dispatcher.peek_slot()
            if slot is None:
                return None, set()
            allowed = self._alive_ids(slot)
            if allowed:
                return slot, allowed
            self.dispatcher.commit_slot(slot)
            skips += 1
        return None, set()

    def _schedule_slot(
        self, slot: DispatchSlot, allowed: set[str]
    ) -> SchedulerOutput:
        assert self._wave is not None
        reqs = [
            self.requests[rid]
            for rid in self._wave.microbatches[slot.microbatch_index].req_ids
            if rid in allowed
        ]
        scheduled_new: list[Request] = []
        scheduled_running: list[Request] = []
        scheduled_resumed: list[Request] = []
        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}

        for req in reqs:
            num_new = self._num_new_tokens(req)
            if num_new <= 0:
                continue
            if (
                req.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED)
                and len(self.running) >= self.max_num_running_reqs
            ):
                self._drop_request(req, "max_num_seqs is full")
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
                req_to_new_blocks[req.request_id] = (
                    self.kv_cache_manager.get_blocks(req.request_id)
                )
            num_scheduled_tokens[req.request_id] = num_new
            if kind == "new":
                scheduled_new.append(req)
            elif kind == "resumed":
                scheduled_resumed.append(req)
            else:
                scheduled_running.append(req)

        if not num_scheduled_tokens:
            return SchedulerOutput.make_empty()

        out = self._emit_output(
            scheduled_new,
            scheduled_running,
            scheduled_resumed,
            req_to_new_blocks,
            num_scheduled_tokens,
        )
        self.dispatcher.commit_slot(slot)
        fwd_id = 0
        if self._tracer is not None:
            fwd_id = self._tracer.next_fwd_id()
            out.tau_fwd_id = fwd_id
        self._inflight[id(out)] = (slot, fwd_id)
        self._trace(
            "emit",
            fwd_id=fwd_id,
            wave_id=self._wave.wave_id,
            batch_idx=slot.microbatch_index,
            phase=slot.phase.value,
            req_ids=list(num_scheduled_tokens),
            num_tokens=int(out.total_num_scheduled_tokens),
        )
        return out

    def _num_new_tokens(self, request: Request) -> int:
        num_new = (
            request.num_tokens
            + request.num_output_placeholders
            - request.num_computed_tokens
        )
        return min(
            num_new, self.max_model_len - 1 - request.num_computed_tokens
        )

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

    def _drop_request(self, request: Request, reason: str) -> None:
        logger.error(
            "TauScheduler dropping request %s: %s",
            request.request_id,
            reason,
        )
        self._trace("drop", req_id=request.request_id, reason=reason)
        self.finish_requests(request.request_id, RequestStatus.FINISHED_ERROR)

    def _drop_unfittable_waiting(self) -> None:
        free = self.kv_cache_manager.block_pool.get_num_free_blocks()
        drop_ids: list[str] = []
        for req in list(self.waiting):
            need = estimate_kv_blocks(
                req.num_prompt_tokens, req.max_tokens, self.block_size
            )
            if need > free:
                logger.error(
                    "TauScheduler dropping %s: reserved %d KV blocks "
                    "exceeds free %d",
                    req.request_id,
                    need,
                    free,
                )
                self._trace(
                    "drop",
                    req_id=req.request_id,
                    reason="reserved KV exceeds free blocks",
                    need_blocks=need,
                    free_blocks=free,
                )
                drop_ids.append(req.request_id)
        if drop_ids:
            self.finish_requests(drop_ids, RequestStatus.FINISHED_ERROR)

    def _emit_output(
        self,
        scheduled_new: list[Request],
        scheduled_running: list[Request],
        scheduled_resumed: list[Request],
        req_to_new_blocks: dict[str, KVCacheBlocks],
        num_scheduled_tokens: dict[str, int],
    ) -> SchedulerOutput:
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

    def _waiting_snapshot(self) -> list[TauRequestSnapshot]:
        return [snapshot_from_request(req) for req in self.waiting]

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
        )

    def _alive_ids(self, slot: DispatchSlot) -> set[str]:
        assert self._wave is not None
        ids = self._wave.microbatches[slot.microbatch_index].req_ids
        alive: set[str] = set()
        for rid in ids:
            req = self.requests.get(rid)
            if req is not None and not req.is_finished():
                alive.add(rid)
        return alive

    def _admitted_unfinished(self) -> bool:
        if self._wave is None:
            return False
        for rid in self._wave.admitted_ids:
            req = self.requests.get(rid)
            if req is not None and not req.is_finished():
                return True
        return False
