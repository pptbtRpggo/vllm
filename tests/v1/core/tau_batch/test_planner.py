# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.sched.tau_batch import (
    GreedyWaveStrategy,
    PackContext,
    TauBatchPlanner,
    TauRequestSnapshot,
    WavePlan,
)

pytestmark = pytest.mark.cpu_test


def _req(
    req_id: str,
    *,
    tpot_slo_ms: float = 50.0,
    ttft_slo_ms: float = 200.0,
    arrival_time: float = 0.0,
    prompt_len: int = 128,
) -> TauRequestSnapshot:
    return TauRequestSnapshot(
        request_id=req_id,
        arrival_time=arrival_time,
        prompt_len=prompt_len,
        ttft_slo_ms=ttft_slo_ms,
        tpot_slo_ms=tpot_slo_ms,
    )


def _ctx(
    *,
    max_num_seqs: int = 16,
    max_microbatches: int = 4,
    max_reqs_per_microbatch: int = 4,
    now: float = 0.0,
) -> PackContext:
    return PackContext(
        now=now,
        max_num_seqs=max_num_seqs,
        max_microbatches=max_microbatches,
        max_reqs_per_microbatch=max_reqs_per_microbatch,
    )


def _assert_invariants(
    plan: WavePlan, requests: list[TauRequestSnapshot], ctx: PackContext
) -> None:
    input_ids = {r.request_id for r in requests}
    assert plan.admitted_ids <= input_ids
    assert plan.deferred_ids == input_ids - plan.admitted_ids
    assert not (plan.admitted_ids & plan.deferred_ids)
    packed = [rid for b in plan.microbatches for rid in b.req_ids]
    assert frozenset(packed) == plan.admitted_ids
    assert len(packed) == len(set(packed))
    assert plan.microbatches
    for i, batch in enumerate(plan.microbatches):
        assert batch.index == i
        assert batch.req_ids
        assert len(batch.req_ids) <= ctx.max_reqs_per_microbatch
    assert len(plan.admitted_ids) <= ctx.max_num_seqs
    assert len(plan.microbatches) <= ctx.max_microbatches


def test_empty_snapshot_returns_none():
    planner = TauBatchPlanner()
    assert planner.plan_wave([], _ctx()) is None


def test_invalid_slo_raises():
    planner = TauBatchPlanner()
    with pytest.raises(ValueError, match="ttft_slo_ms"):
        planner.plan_wave([_req("a", ttft_slo_ms=0)], _ctx())
    with pytest.raises(ValueError, match="tpot_slo_ms"):
        planner.plan_wave([_req("a", tpot_slo_ms=-1)], _ctx())


def test_duplicate_id_raises():
    planner = TauBatchPlanner()
    with pytest.raises(ValueError, match="duplicate"):
        planner.plan_wave([_req("a"), _req("a")], _ctx())


def test_invalid_prompt_len_raises():
    planner = TauBatchPlanner()
    with pytest.raises(ValueError, match="prompt_len"):
        planner.plan_wave([_req("a", prompt_len=0)], _ctx())


def test_invalid_pack_context_raises():
    planner = TauBatchPlanner()
    with pytest.raises(ValueError, match="max_num_seqs"):
        planner.plan_wave([_req("a")], _ctx(max_num_seqs=0))
    with pytest.raises(ValueError, match="max_microbatches"):
        planner.plan_wave([_req("a")], _ctx(max_microbatches=0))
    with pytest.raises(ValueError, match="max_reqs_per_microbatch"):
        planner.plan_wave([_req("a")], _ctx(max_reqs_per_microbatch=0))


def test_greedy_admits_all_when_capacity_fits():
    planner = TauBatchPlanner()
    requests = [_req("r1"), _req("r2"), _req("r3")]
    ctx = _ctx(max_num_seqs=8, max_microbatches=2, max_reqs_per_microbatch=4)
    plan = planner.plan_wave(requests, ctx)
    assert plan is not None
    _assert_invariants(plan, requests, ctx)
    assert plan.admitted_ids == {"r1", "r2", "r3"}
    assert plan.deferred_ids == frozenset()
    assert plan.extra["strategy"] == "greedy"


def test_greedy_defers_when_over_max_num_seqs():
    planner = TauBatchPlanner()
    requests = [_req(f"r{i}", tpot_slo_ms=float(10 + i)) for i in range(5)]
    ctx = _ctx(max_num_seqs=2, max_microbatches=4, max_reqs_per_microbatch=4)
    plan = planner.plan_wave(requests, ctx)
    assert plan is not None
    _assert_invariants(plan, requests, ctx)
    assert len(plan.admitted_ids) == 2
    # Tight TPOT first: r0 (10ms), r1 (11ms).
    assert plan.admitted_ids == {"r0", "r1"}
    assert plan.deferred_ids == {"r2", "r3", "r4"}


def test_greedy_prefers_tighter_tpot_then_earlier_arrival():
    planner = TauBatchPlanner()
    requests = [
        _req("late_tight", tpot_slo_ms=10.0, arrival_time=5.0),
        _req("early_loose", tpot_slo_ms=80.0, arrival_time=1.0),
        _req("early_tight", tpot_slo_ms=10.0, arrival_time=1.0),
    ]
    ctx = _ctx(max_num_seqs=2, max_microbatches=1, max_reqs_per_microbatch=2)
    plan = planner.plan_wave(requests, ctx)
    assert plan is not None
    _assert_invariants(plan, requests, ctx)
    assert plan.microbatches[0].req_ids == ("early_tight", "late_tight")
    assert plan.deferred_ids == {"early_loose"}


def test_greedy_splits_into_microbatches():
    planner = TauBatchPlanner()
    requests = [_req(f"r{i}", tpot_slo_ms=float(i)) for i in range(1, 7)]
    ctx = _ctx(max_num_seqs=16, max_microbatches=3, max_reqs_per_microbatch=2)
    plan = planner.plan_wave(requests, ctx)
    assert plan is not None
    _assert_invariants(plan, requests, ctx)
    assert len(plan.microbatches) == 3
    assert [b.req_ids for b in plan.microbatches] == [
        ("r1", "r2"),
        ("r3", "r4"),
        ("r5", "r6"),
    ]


def test_greedy_defers_beyond_microbatch_capacity():
    planner = TauBatchPlanner()
    requests = [_req(f"r{i}") for i in range(10)]
    ctx = _ctx(max_num_seqs=16, max_microbatches=2, max_reqs_per_microbatch=3)
    plan = planner.plan_wave(requests, ctx)
    assert plan is not None
    _assert_invariants(plan, requests, ctx)
    assert len(plan.admitted_ids) == 6
    assert len(plan.deferred_ids) == 4
    assert len(plan.microbatches) == 2


def test_plan_wave_is_deterministic_except_wave_id():
    requests = [
        _req("b", tpot_slo_ms=20.0, arrival_time=2.0),
        _req("a", tpot_slo_ms=20.0, arrival_time=1.0),
        _req("c", tpot_slo_ms=10.0, arrival_time=3.0),
    ]
    ctx = _ctx(max_num_seqs=2, max_microbatches=2, max_reqs_per_microbatch=1)
    p1 = TauBatchPlanner(wave_id_start=0)
    p2 = TauBatchPlanner(wave_id_start=7)
    a = p1.plan_wave(requests, ctx)
    b = p2.plan_wave(requests, ctx)
    assert a is not None and b is not None
    assert a.wave_id == 0
    assert b.wave_id == 7
    assert a.microbatches == b.microbatches
    assert a.admitted_ids == b.admitted_ids
    assert a.deferred_ids == b.deferred_ids


def test_wave_id_increments_on_success():
    planner = TauBatchPlanner(wave_id_start=3)
    ctx = _ctx()
    first = planner.plan_wave([_req("a")], ctx)
    second = planner.plan_wave([_req("b")], ctx)
    assert first is not None and second is not None
    assert first.wave_id == 3
    assert second.wave_id == 4


def test_custom_strategy_is_used():
    class AdmitNone(GreedyWaveStrategy):
        def pack(self, requests, ctx):
            ids = frozenset(r.request_id for r in requests)
            return WavePlan(
                wave_id=0,
                microbatches=(),
                admitted_ids=frozenset(),
                deferred_ids=ids,
                extra={"strategy": "none"},
            )

    planner = TauBatchPlanner(strategy=AdmitNone())
    assert planner.plan_wave([_req("a"), _req("b")], _ctx()) is None
