# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.sched.tau_batch import (
    DispatchPhase,
    DispatchPolicy,
    DispatchSlot,
    ListDispatcher,
    MicroBatchList,
    MicroBatchTask,
)

pytestmark = pytest.mark.cpu_test


def _list(p: int = 3) -> MicroBatchList:
    tasks = tuple(
        MicroBatchTask(req_ids=(f"r{i}",), index=i) for i in range(p)
    )
    admitted = frozenset(rid for task in tasks for rid in task.req_ids)
    return MicroBatchList(
        tasks=tasks,
        admitted_ids=admitted,
        deferred_ids=frozenset(),
    )


def _commit_prefills(disp: ListDispatcher, n: int) -> None:
    for i in range(n):
        slot = disp.peek_slot()
        assert slot == DispatchSlot(i, DispatchPhase.PREFILL)
        disp.commit_slot(slot)


def test_peek_before_start_returns_none():
    disp = ListDispatcher()
    assert disp.peek_slot() is None
    with pytest.raises(ValueError, match="committable"):
        disp.commit_slot(DispatchSlot(0, DispatchPhase.PREFILL))


def test_empty_list_start_raises():
    disp = ListDispatcher()
    empty = MicroBatchList(
        tasks=(),
        admitted_ids=frozenset(),
        deferred_ids=frozenset(),
    )
    with pytest.raises(ValueError, match="non-empty"):
        disp.start(empty)


def test_default_policy_is_overlap():
    assert ListDispatcher().policy is DispatchPolicy.OVERLAP


def test_peek_does_not_advance():
    disp = ListDispatcher()
    disp.start(_list())
    a = disp.peek_slot()
    b = disp.peek_slot()
    assert a == b == DispatchSlot(0, DispatchPhase.PREFILL)


def test_both_policies_issue_prefills_in_order():
    for policy in (DispatchPolicy.OVERLAP, DispatchPolicy.DRAIN):
        disp = ListDispatcher(policy)
        disp.start(_list(3))
        _commit_prefills(disp, 3)
        # Prefill fill done; decode gated on completes.
        assert disp.peek_slot() is None


def test_complete_during_prefill_fill_does_not_insert_decode():
    disp = ListDispatcher(DispatchPolicy.OVERLAP)
    disp.start(_list(3))
    disp.commit_slot(disp.peek_slot())
    disp.on_prefill_complete(0)
    # Still must issue B1, B2 prefills.
    assert disp.peek_slot() == DispatchSlot(1, DispatchPhase.PREFILL)


def test_overlap_decode_after_only_first_prefill_complete():
    disp = ListDispatcher(DispatchPolicy.OVERLAP)
    disp.start(_list(3))
    _commit_prefills(disp, 3)
    disp.on_prefill_complete(0)
    assert disp.peek_slot() == DispatchSlot(0, DispatchPhase.DECODE)


def test_drain_waits_until_all_prefills_complete():
    disp = ListDispatcher(DispatchPolicy.DRAIN)
    disp.start(_list(3))
    _commit_prefills(disp, 3)
    disp.on_prefill_complete(0)
    assert disp.peek_slot() is None
    disp.on_prefill_complete(1)
    assert disp.peek_slot() is None
    disp.on_prefill_complete(2)
    assert disp.peek_slot() == DispatchSlot(0, DispatchPhase.DECODE)


def test_overlap_cyclic_decode_skips_unready_cursor():
    disp = ListDispatcher(DispatchPolicy.OVERLAP)
    disp.start(_list(3))
    _commit_prefills(disp, 3)
    disp.on_prefill_complete(0)
    disp.commit_slot(disp.peek_slot())
    # cursor is now 1; B1 prefill not done.
    assert disp.peek_slot() is None
    disp.on_prefill_complete(1)
    assert disp.peek_slot() == DispatchSlot(1, DispatchPhase.DECODE)


def test_commit_mismatch_raises():
    disp = ListDispatcher()
    disp.start(_list())
    with pytest.raises(ValueError, match="does not match"):
        disp.commit_slot(DispatchSlot(2, DispatchPhase.PREFILL))


def test_on_prefill_complete_before_start_is_ignored():
    disp = ListDispatcher()
    disp.on_prefill_complete(0)
    disp.start(_list())
    _commit_prefills(disp, 3)
    assert disp.peek_slot() is None


def test_reset_clears_list_and_peek():
    disp = ListDispatcher()
    disp.start(_list())
    _commit_prefills(disp, 1)
    disp.reset()
    assert disp.active_list is None
    assert disp.peek_slot() is None
