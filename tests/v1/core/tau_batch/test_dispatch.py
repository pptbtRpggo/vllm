# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.sched.tau_batch import (
    DispatchPhase,
    DispatchSlot,
    MicroBatchPlan,
    WaveDispatcher,
    WaveDispatchPolicy,
    WavePlan,
)

pytestmark = pytest.mark.cpu_test


def _plan(p: int = 3) -> WavePlan:
    batches = tuple(
        MicroBatchPlan(req_ids=(f"r{i}",), index=i) for i in range(p)
    )
    admitted = frozenset(rid for b in batches for rid in b.req_ids)
    return WavePlan(
        wave_id=0,
        microbatches=batches,
        admitted_ids=admitted,
        deferred_ids=frozenset(),
    )


def _commit_prefills(disp: WaveDispatcher, n: int) -> None:
    for i in range(n):
        slot = disp.peek_slot()
        assert slot == DispatchSlot(i, DispatchPhase.PREFILL)
        disp.commit_slot(slot)


def test_peek_before_start_returns_none():
    disp = WaveDispatcher()
    assert disp.peek_slot() is None
    with pytest.raises(ValueError, match="committable"):
        disp.commit_slot(DispatchSlot(0, DispatchPhase.PREFILL))


def test_empty_wave_start_raises():
    disp = WaveDispatcher()
    empty = WavePlan(
        wave_id=0,
        microbatches=(),
        admitted_ids=frozenset(),
        deferred_ids=frozenset(),
    )
    with pytest.raises(ValueError, match="non-empty"):
        disp.start(empty)


def test_default_policy_is_overlap():
    assert WaveDispatcher().policy is WaveDispatchPolicy.OVERLAP


def test_peek_does_not_advance():
    disp = WaveDispatcher()
    disp.start(_plan())
    a = disp.peek_slot()
    b = disp.peek_slot()
    assert a == b == DispatchSlot(0, DispatchPhase.PREFILL)


def test_both_policies_issue_prefills_in_order():
    for policy in (WaveDispatchPolicy.OVERLAP, WaveDispatchPolicy.DRAIN):
        disp = WaveDispatcher(policy)
        disp.start(_plan(3))
        _commit_prefills(disp, 3)
        # Prefill fill done; decode gated on completes.
        assert disp.peek_slot() is None


def test_complete_during_prefill_fill_does_not_insert_decode():
    disp = WaveDispatcher(WaveDispatchPolicy.OVERLAP)
    disp.start(_plan(3))
    disp.commit_slot(disp.peek_slot())
    disp.on_prefill_complete(0)
    # Still must issue B1, B2 prefills.
    assert disp.peek_slot() == DispatchSlot(1, DispatchPhase.PREFILL)


def test_overlap_decode_after_only_first_prefill_complete():
    disp = WaveDispatcher(WaveDispatchPolicy.OVERLAP)
    disp.start(_plan(3))
    _commit_prefills(disp, 3)
    disp.on_prefill_complete(0)
    assert disp.peek_slot() == DispatchSlot(0, DispatchPhase.DECODE)


def test_drain_waits_until_all_prefills_complete():
    disp = WaveDispatcher(WaveDispatchPolicy.DRAIN)
    disp.start(_plan(3))
    _commit_prefills(disp, 3)
    disp.on_prefill_complete(0)
    assert disp.peek_slot() is None
    disp.on_prefill_complete(1)
    assert disp.peek_slot() is None
    disp.on_prefill_complete(2)
    assert disp.peek_slot() == DispatchSlot(0, DispatchPhase.DECODE)


def test_overlap_cyclic_decode_skips_unready_cursor():
    disp = WaveDispatcher(WaveDispatchPolicy.OVERLAP)
    disp.start(_plan(3))
    _commit_prefills(disp, 3)
    disp.on_prefill_complete(0)
    disp.commit_slot(disp.peek_slot())
    # cursor is now 1; B1 prefill not done.
    assert disp.peek_slot() is None
    disp.on_prefill_complete(1)
    assert disp.peek_slot() == DispatchSlot(1, DispatchPhase.DECODE)


def test_commit_mismatch_raises():
    disp = WaveDispatcher()
    disp.start(_plan())
    with pytest.raises(ValueError, match="does not match"):
        disp.commit_slot(DispatchSlot(2, DispatchPhase.PREFILL))


def test_on_prefill_complete_before_start_is_ignored():
    disp = WaveDispatcher()
    disp.on_prefill_complete(0)
    disp.start(_plan())
    _commit_prefills(disp, 3)
    assert disp.peek_slot() is None


def test_reset_clears_plan_and_peek():
    disp = WaveDispatcher()
    disp.start(_plan())
    _commit_prefills(disp, 1)
    disp.reset()
    assert disp.plan is None
    assert disp.peek_slot() is None
