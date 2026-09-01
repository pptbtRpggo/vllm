# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Any

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.tau_batch.dispatch import (
    DispatchPhase,
    DispatchSlot,
    WaveDispatcher,
    WaveDispatchPolicy,
)
from vllm.v1.core.sched.tau_batch.planner import TauBatchPlanner
from vllm.v1.core.sched.tau_batch.types import (
    PackContext,
    TauRequestSnapshot,
    WavePlan,
)
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

# Prototype defaults when sampling_params.extra_args has no SLO.
_DEFAULT_TTFT_SLO_MS = 10_000.0
_DEFAULT_TPOT_SLO_MS = 100.0


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
    )


class _Holdback:
    """Hide non-allowed requests from waiting / skipped_waiting / running."""

    def __init__(self, scheduler: Scheduler, allowed: set[str]) -> None:
        self._scheduler = scheduler
        self._allowed = allowed
        self._waiting = list(scheduler.waiting)
        self._skipped = list(scheduler.skipped_waiting)
        self._running = list(scheduler.running)

    def apply(self) -> None:
        s = self._scheduler
        s.waiting = self._filter_queue(self._waiting)
        s.skipped_waiting = self._filter_queue(self._skipped)
        s.running = [r for r in self._running if r.request_id in self._allowed]

    def restore(self) -> None:
        s = self._scheduler
        running_after = list(s.running)
        waiting_after = list(s.waiting)
        skipped_after = list(s.skipped_waiting)
        running_ids = {r.request_id for r in running_after}
        s.running = self._restore_running(running_after)
        s.waiting = self._restore_queue(self._waiting, waiting_after, running_ids)
        s.skipped_waiting = self._restore_queue(
            self._skipped, skipped_after, running_ids
        )

    def _filter_queue(self, reqs: list[Request]) -> RequestQueue:
        q = create_request_queue(self._scheduler.policy)
        for req in reqs:
            if req.request_id in self._allowed:
                q.add_request(req)
        return q

    def _restore_running(self, running_after: list[Request]) -> list[Request]:
        after_by_id = {r.request_id: r for r in running_after}
        seen: set[str] = set()
        restored: list[Request] = []
        for req in self._running:
            rid = req.request_id
            if rid in self._allowed:
                if rid in after_by_id:
                    restored.append(after_by_id[rid])
                    seen.add(rid)
            else:
                restored.append(req)
                seen.add(rid)
        for req in running_after:
            if req.request_id not in seen:
                restored.append(req)
        return restored

    def _restore_queue(
        self,
        original: list[Request],
        after: list[Request],
        running_ids: set[str],
    ) -> RequestQueue:
        after_ids = {r.request_id for r in after}
        q = create_request_queue(self._scheduler.policy)
        seen: set[str] = set()
        for req in original:
            rid = req.request_id
            if rid in running_ids:
                continue
            if rid in self._allowed:
                if rid in after_ids:
                    q.add_request(req)
                    seen.add(rid)
            else:
                q.add_request(req)
                seen.add(rid)
        for req in after:
            if req.request_id not in seen and req.request_id not in running_ids:
                q.add_request(req)
        return q


class TauScheduler(Scheduler):
    """Wave-aware scheduler: one DispatchSlot per schedule() call.

    Holdback makes only the current micro-batch visible, then reuses
    Scheduler.schedule() for KV allocation and request lifecycle.
    commit_slot runs only when the parent scheduled the full allowed set.
    on_prefill_complete is recorded in update_from_output, not schedule().

    New arrivals stay in waiting until the active wave has no unfinished
    admitted requests. The next plan_wave uses a fresh waiting snapshot.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.planner = TauBatchPlanner()
        self.dispatcher = WaveDispatcher(WaveDispatchPolicy.OVERLAP)
        self.max_microbatches: int | None = None
        self.max_reqs_per_microbatch: int | None = None
        self._wave: WavePlan | None = None
        self._slot_by_output: dict[int, DispatchSlot] = {}

    def schedule(self) -> SchedulerOutput:
        for _ in range(2):
            self._maybe_start_wave()
            if self._wave is None:
                return SchedulerOutput.make_empty()
            slot, allowed = self._next_slot()
            if slot is None:
                if not self._admitted_unfinished():
                    self._wave = None
                    continue
                return SchedulerOutput.make_empty()
            return self._schedule_slot(slot, allowed)
        return SchedulerOutput.make_empty()

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        slot = self._slot_by_output.pop(id(scheduler_output), None)
        result = super().update_from_output(scheduler_output, model_runner_output)
        if slot is not None and slot.phase is DispatchPhase.PREFILL:
            self.dispatcher.on_prefill_complete(slot.microbatch_index)
        if self._wave is not None and not self._admitted_unfinished():
            self._wave = None
        return result

    def _maybe_start_wave(self) -> None:
        if self._wave is not None and self._admitted_unfinished():
            return
        self._wave = None
        snapshots = self._waiting_snapshot()
        if not snapshots:
            return
        plan = self.planner.plan_wave(snapshots, self._pack_context())
        if plan is None:
            return
        self._wave = plan
        self.dispatcher.start(plan)

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
        hold = _Holdback(self, allowed)
        hold.apply()
        try:
            out = super().schedule()
        finally:
            hold.restore()
        scheduled = set(out.num_scheduled_tokens)
        if scheduled == allowed and out.total_num_scheduled_tokens > 0:
            self.dispatcher.commit_slot(slot)
            self._slot_by_output[id(out)] = slot
        return out

    def _waiting_snapshot(self) -> list[TauRequestSnapshot]:
        reqs = list(self.waiting) + list(self.skipped_waiting)
        return [snapshot_from_request(req) for req in reqs]

    def _pack_context(self) -> PackContext:
        pp = max(1, self.parallel_config.pipeline_parallel_size)
        p = self.max_microbatches if self.max_microbatches is not None else pp
        batch = self.max_reqs_per_microbatch
        if batch is None:
            batch = max(1, self.max_num_running_reqs // p)
        return PackContext(
            now=time.time(),
            max_num_seqs=self.max_num_running_reqs,
            max_microbatches=p,
            max_reqs_per_microbatch=batch,
            pp_size=pp,
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
