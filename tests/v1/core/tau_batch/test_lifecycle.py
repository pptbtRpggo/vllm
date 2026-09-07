# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest

from tests.v1.core.tau_batch.test_batch_queue import (
    _ImmediateExecutor,
    _QueueCore,
)
from tests.v1.core.tau_batch.test_scheduler import (
    _add_requests,
    _req,
    _sampled,
    _tau_scheduler,
)
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import FinishReason
from vllm.v1.engine.core import EngineCore
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


class _TrackingExecutor(_ImmediateExecutor):
    """A worker-state probe, not a device or PP communication simulation."""

    def __init__(self, max_num_seqs):
        super().__init__()
        self.max_num_seqs = max_num_seqs
        self.cached_ids = set()
        self.received = []
        self.samples = 0

    def execute_model(self, scheduler_output, non_block=False):
        assert len(scheduler_output.num_scheduled_tokens) <= self.max_num_seqs
        self.received.append(scheduler_output)
        self.cached_ids.difference_update(scheduler_output.finished_req_ids)
        self.cached_ids.update(r.req_id for r in scheduler_output.scheduled_new_reqs)
        assert set(scheduler_output.num_scheduled_tokens) <= self.cached_ids
        return super().execute_model(scheduler_output, non_block)

    def sample_tokens(self, grammar_output, non_block=False):
        assert self.received[-1].total_num_scheduled_tokens > 0
        self.samples += 1
        return super().sample_tokens(grammar_output, non_block)


def _drive(core, *, queued=True):
    outputs = []
    for _ in range(100):
        if not core.scheduler.has_requests() and not core.batch_queue:
            return outputs
        result, _ = core.step_with_batch_queue() if queued else EngineCore.step(core)
        for batch in (result or {}).values():
            outputs.extend(batch.outputs)
    pytest.fail("scheduler did not become idle after all requests completed")


@pytest.mark.parametrize("queue_size", [1, 2, 4])
@pytest.mark.parametrize("queued", [False, True])
def test_multiple_waves_finish_cleanup_and_become_idle(queue_size, queued):
    sched = _tau_scheduler(
        max_num_seqs=2, max_reqs_per_microbatch=4, max_microbatches=0
    )
    core = _QueueCore(sched, queue_size)
    executor = core.model_executor = _TrackingExecutor(max_num_seqs=2)
    initial_free = sched.kv_cache_manager.block_pool.get_num_free_blocks()
    for wave in range(2):
        requests = _add_requests(sched, n=6, max_tokens=3)
        outputs = _drive(core, queued=queued)
        finished = [out for out in outputs if out.finish_reason is not None]
        assert sorted(out.request_id for out in finished) == [f"r{i}" for i in range(6)]
        assert all(out.finish_reason == FinishReason.LENGTH for out in finished)
        assert all(req.num_output_tokens == 3 for req in requests)
        assert not sched.requests and not sched._inflight
        assert not sched._pending_errors and not sched.finished_req_ids
        assert not executor.cached_ids
        assert sched.kv_cache_manager.block_pool.get_num_free_blocks() == initial_free
        assert sched._next_wave_id == wave + 1
    control = [o for o in executor.received if not o.total_num_scheduled_tokens]
    assert control and all(o.finished_req_ids for o in control)
    if queued:
        assert executor.samples == sum(
            bool(o.num_scheduled_tokens) for o in executor.received
        )


@pytest.mark.parametrize("all_fail", [False, True])
@pytest.mark.parametrize("queued", [False, True])
def test_allocation_errors_notify_each_client_once_and_cleanup(all_fail, queued):
    sched = _tau_scheduler()
    reqs = _add_requests(sched, n=2, max_tokens=1)
    reqs[1].client_index = 7
    core = _QueueCore(sched, 2)
    core.model_executor = _TrackingExecutor(16)
    real_allocate = sched.kv_cache_manager.allocate_slots

    def allocate(req, *args, **kwargs):
        if all_fail or req.request_id == "r0":
            return None
        return real_allocate(req, *args, **kwargs)

    with patch.object(sched.kv_cache_manager, "allocate_slots", side_effect=allocate):
        outputs = _drive(core, queued=queued)
    assert len(outputs) == 2
    by_id = {o.request_id: o for o in outputs}
    assert by_id["r0"].finish_reason == FinishReason.ERROR
    assert by_id["r1"].finish_reason == (
        FinishReason.ERROR if all_fail else FinishReason.LENGTH
    )
    assert not sched._pending_errors and not core.model_executor.cached_ids


def test_unfittable_request_error_routes_to_its_client_without_a_token_forward():
    sched = _tau_scheduler()
    request = _req("too-big", tpot_slo_ms=10, max_tokens=1_000_000)
    request.client_index = 7
    sched.add_request(request)
    cleanup = sched.schedule()
    assert cleanup.total_num_scheduled_tokens == 0
    assert cleanup.finished_req_ids == {"too-big"}
    assert not sched.is_idle_output(cleanup)
    result = sched.update_from_output(cleanup, _sampled(cleanup))
    assert len(result[7].outputs) == 1
    assert result[7].outputs[0].finish_reason == FinishReason.ERROR
    assert not sched.has_requests()
    again = sched.update_from_output(SchedulerOutput.make_empty(), _sampled(cleanup))
    assert not any(batch.outputs for batch in again.values())


def test_runtime_request_capacity_validation_precedes_kv_allocation():
    sched = _tau_scheduler()
    requests = _add_requests(sched, n=2)
    sched._ensure_active_list()
    sched.max_num_running_reqs = 1
    with patch.object(sched.kv_cache_manager, "allocate_slots") as allocate:
        with pytest.raises(ValueError, match="max_num_seqs"):
            sched.schedule()
        allocate.assert_not_called()
    assert all(req.status == RequestStatus.WAITING for req in requests)
    assert all(req.num_computed_tokens == 0 for req in requests)


def test_cancelled_wave_drains_before_replacement_and_keeps_trace_identity(tmp_path):
    from vllm.v1.core.sched.tau_batch.trace import load_events

    path = tmp_path / "cancel.jsonl"
    sched = _tau_scheduler(tau_batch_trace=str(path))
    old = _add_requests(sched, n=4)
    pre0, pre1 = sched.schedule(), sched.schedule()
    sched.finish_requests(
        [req.request_id for req in old], RequestStatus.FINISHED_ABORTED
    )
    for i in range(4):
        sched.add_request(_req(f"new{i}", tpot_slo_ms=10))
    sched.update_from_output(pre0, _sampled(pre0))
    cleanup = sched.schedule()
    assert not cleanup.num_scheduled_tokens
    assert sched._wave_id == 0 and len(sched._inflight) == 1
    sched.update_from_output(pre1, _sampled(pre1))
    assert sched._list is None
    new = sched.schedule()
    assert new.num_scheduled_tokens == {"new0": 8, "new1": 8}
    assert sched._wave_id == 1
    sched.update_from_output(cleanup, _sampled(cleanup))
    state = sched.dispatcher.task_state(1)
    assert not state.prefill_dispatched and not state.prefill_completed
    events = load_events(path)
    done = [e for e in events if e["event"] == "done"]
    assert [(e["fwd_id"], e["wave_id"]) for e in done] == [(1, 0), (2, 0)]


def test_resubmitted_id_cannot_receive_cancelled_forward_tokens():
    sched = _tau_scheduler()
    _add_requests(sched, n=1)
    old = sched.schedule()
    sched.finish_requests("r0", RequestStatus.FINISHED_ABORTED)
    replacement = _req("r0", tpot_slo_ms=10, prompt_len=12)
    sched.add_request(replacement)
    cleanup = sched.schedule()
    assert not cleanup.num_scheduled_tokens
    result = sched.update_from_output(old, _sampled(old, token_id=42))
    assert not any(batch.outputs for batch in result.values())
    assert replacement.num_output_tokens == 0
    assert replacement.num_computed_tokens == 0
    sched.update_from_output(cleanup, _sampled(cleanup))
    new = sched.schedule()
    assert new.num_scheduled_tokens == {"r0": 12}
    assert sched._wave_id == 1


def test_default_scheduler_zero_token_control_is_delivered_without_sampling():
    sched = _tau_scheduler()
    control = SchedulerOutput.make_empty()
    control.finished_req_ids = {"finished"}
    control.kv_connector_metadata = object()
    # Exercise the base scheduler contract, without Tau's pure-wait override.
    sched.is_idle_output = SchedulerInterface.is_idle_output.__get__(sched)
    sched.schedule = lambda: control
    sched.has_requests = lambda: True
    core = _QueueCore(sched, 2)
    executor = core.model_executor = _TrackingExecutor(16)
    executor.cached_ids.add("finished")
    _, executed = core.step_with_batch_queue()
    assert executor.received == [control]
    assert not executor.cached_ids
    assert executor.samples == 0
    assert not executed and not core.batch_queue


@pytest.mark.parametrize("queue_size", [2, 4])
def test_abort_during_batch_queue_pop_drains_old_work_before_new_wave(queue_size):
    sched = _tau_scheduler(max_microbatches=0)
    old = _add_requests(sched, n=2 * queue_size)
    core = _QueueCore(sched, queue_size)
    executor = core.model_executor = _TrackingExecutor(16)
    aborted = False

    def abort_on_first_result():
        nonlocal aborted
        if aborted:
            return
        aborted = True
        # The batch queue is full and the oldest result is about to be applied.
        assert len(sched._inflight) == queue_size
        sched.finish_requests(
            [req.request_id for req in old], RequestStatus.FINISHED_ABORTED
        )
        for i in range(4):
            sched.add_request(_req(f"new{i}", tpot_slo_ms=10, max_tokens=2))

    core._process_aborts_queue = abort_on_first_result
    original_execute = executor.execute_model

    def execute(out, non_block=False):
        if any(req.req_id.startswith("new") for req in out.scheduled_new_reqs):
            # schedule() has recorded this new output, but no old output may remain.
            assert all(forward.wave_id == 1 for forward in sched._inflight.values())
        return original_execute(out, non_block)

    executor.execute_model = execute
    outputs = _drive(core)
    assert aborted
    assert all(out.request_id.startswith("new") for out in outputs)
    assert sorted(out.request_id for out in outputs if out.finished) == [
        f"new{i}" for i in range(4)
    ]
    assert not executor.cached_ids and not sched._inflight


def test_tau_pure_wait_does_not_execute_in_single_step_path():
    sched = _tau_scheduler(tau_batch_min_waiting=2)
    _add_requests(sched, n=1)
    core = _QueueCore(sched, 1)
    executor = core.model_executor = _TrackingExecutor(16)
    assert EngineCore.step(core) == ({}, False)
    assert not executor.received
