# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU probe of EngineCore.step_with_batch_queue on v0.13.

v0.13 stores (future, scheduler_output) in the queue. A 0-token
schedule must not execute_model or enqueue.
"""

from __future__ import annotations

import queue
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass

import pytest

from tests.v1.core.tau_batch.test_scheduler import (
    _add_wave,
    _sampled,
    _tau_scheduler,
)
from vllm.v1.core.sched.tau_batch.dispatch import DispatchPhase, WaveDispatchPolicy
from vllm.v1.core.sched.tau_batch.scheduler import TauScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine.core import EngineCore

pytestmark = pytest.mark.cpu_test

_BATCH_BY_IDS = {
    frozenset({"r0", "r1"}): 0,
    frozenset({"r2", "r3"}): 1,
}


def _slot_name(out: SchedulerOutput) -> str:
    ids = frozenset(out.num_scheduled_tokens)
    batch = _BATCH_BY_IDS.get(ids)
    if batch is None:
        for packed, idx in _BATCH_BY_IDS.items():
            if ids <= packed:
                batch = idx
                break
    if batch is None:
        label = ",".join(sorted(ids))
    else:
        label = f"B{batch}"
    phase = "pre" if out.scheduled_new_reqs else "dec"
    return f"{label}_{phase}"


def _peek_name(sched: TauScheduler) -> str | None:
    slot = sched.dispatcher.peek_slot()
    if slot is None:
        return None
    short = "pre" if slot.phase is DispatchPhase.PREFILL else "dec"
    return f"B{slot.microbatch_index}_{short}"


def _queue_names(core: "_QueueCore") -> tuple[str, ...]:
    return tuple(_slot_name(item[1]) for item in reversed(core.batch_queue))


@dataclass(frozen=True)
class QueueStep:
    step: int
    peek: str | None
    scheduled: str | None
    action: str
    popped: str | None
    queue_after: tuple[str, ...]


class _LazyFuture(Future):
    """Not done until result() so v0.13 can fill the queue without popping."""

    def __init__(self, value) -> None:
        super().__init__()
        self._value = value

    def result(self, timeout=None):
        if not self.done():
            self.set_result(self._value)
        return super().result(timeout)


class _ImmediateExecutor:
    def __init__(self) -> None:
        self._last: _LazyFuture | None = None

    def execute_model(self, scheduler_output, non_block: bool = False):
        self._last = _LazyFuture(_sampled(scheduler_output))
        return self._last

    def sample_tokens(self, grammar_output, non_block: bool = False):
        assert self._last is not None
        return self._last


class _QueueCore:
    def __init__(self, scheduler: TauScheduler, queue_size: int) -> None:
        self.scheduler = scheduler
        self.batch_queue_size = queue_size
        self.batch_queue: deque = deque(maxlen=queue_size)
        self.model_executor = _ImmediateExecutor()
        self.is_ec_producer = False
        self.is_pooling_model = False
        self.aborts_queue: queue.Queue = queue.Queue()

    def log_error_detail(self, scheduler_output):
        return nullcontext()

    def _log_err_callback(self, scheduler_output):
        return lambda _f: None

    def _process_aborts_queue(self) -> None:
        return

    step_with_batch_queue = EngineCore.step_with_batch_queue


def _run_queue(
    *,
    policy: WaveDispatchPolicy,
    queue_size: int,
    n_req: int = 4,
    max_tokens: int = 2,
    max_steps: int = 12,
) -> list[QueueStep]:
    sched = _tau_scheduler(
        max_microbatches=2,
        max_reqs_per_microbatch=2,
        pipeline_parallel_size=2,
    )
    sched.dispatcher = sched.dispatcher.__class__(policy)
    _add_wave(sched, n=n_req, max_tokens=max_tokens)
    core = _QueueCore(sched, queue_size)
    traces: list[QueueStep] = []
    for step in range(max_steps):
        if not sched.has_requests() and not core.batch_queue:
            break
        peek = _peek_name(sched)
        before = [id(item[1]) for item in core.batch_queue]
        before_names = {
            id(item[1]): _slot_name(item[1]) for item in core.batch_queue
        }
        outputs, _executed = core.step_with_batch_queue()
        after_ids = [id(item[1]) for item in core.batch_queue]
        added = [i for i in after_ids if i not in before]
        popped_ids = [i for i in before if i not in after_ids]
        scheduled = None
        if added:
            so = next(item[1] for item in core.batch_queue if id(item[1]) in added)
            scheduled = _slot_name(so)
        popped = before_names[popped_ids[0]] if popped_ids else None
        if added and popped:
            action = "fill+pop"
        elif added:
            action = "fill"
        elif popped and not added:
            action = "wait/pop"
        else:
            action = "idle"
        traces.append(
            QueueStep(
                step=step,
                peek=peek,
                scheduled=scheduled,
                action=action,
                popped=popped,
                queue_after=_queue_names(core),
            )
        )
        if action == "idle":
            break
    return traces


def test_overlap_fills_then_pops_oldest_prefill():
    traces = _run_queue(
        policy=WaveDispatchPolicy.OVERLAP, queue_size=2, max_tokens=2
    )
    assert traces[0].action == "fill"
    assert traces[0].scheduled == "B0_pre"
    assert traces[0].queue_after == ("B0_pre",)
    assert traces[1].action == "fill+pop"
    assert traces[1].scheduled == "B1_pre"
    assert traces[1].popped == "B0_pre"
    assert traces[2].peek == "B0_dec"


def test_drain_zero_token_does_not_enqueue():
    traces = _run_queue(
        policy=WaveDispatchPolicy.DRAIN, queue_size=2, max_tokens=2
    )
    assert [t.scheduled for t in traces[:2]] == ["B0_pre", "B1_pre"]
    wait_steps = [t for t in traces[2:] if t.action == "wait/pop"]
    assert wait_steps
    assert wait_steps[0].popped == "B1_pre"
    first_decode = next(t for t in traces if t.scheduled == "B0_dec")
    assert first_decode.step > wait_steps[0].step
