# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.sched.tau_batch import (
    EosEvent,
    GreedyListStrategy,
    MicroBatchList,
    NoOpEosStrategy,
    PackContext,
    TauBatchPlanner,
    TauRequestSnapshot,
    annotate_request_budget,
    estimate_kv_blocks,
    tau_max_ms,
    ttft_slack_ms,
    wait_ms,
)

pytestmark = pytest.mark.cpu_test


def _req(
    req_id: str,
    *,
    tpot_slo_ms: float = 50.0,
    ttft_slo_ms: float = 200.0,
    arrival_time: float = 0.0,
    prompt_len: int = 128,
    max_new_tokens: int = 16,
) -> TauRequestSnapshot:
    return TauRequestSnapshot(
        request_id=req_id,
        arrival_time=arrival_time,
        prompt_len=prompt_len,
        ttft_slo_ms=ttft_slo_ms,
        tpot_slo_ms=tpot_slo_ms,
        max_new_tokens=max_new_tokens,
    )


def _ctx(
    *,
    max_num_seqs: int = 4,
    max_microbatches: int = 4,
    now: float = 0.0,
    kv_free_blocks: int | None = None,
    block_size: int | None = None,
    max_num_batched_tokens: int | None = None,
    pp_size: int | None = None,
) -> PackContext:
    return PackContext(
        now=now,
        max_num_seqs=max_num_seqs,
        max_microbatches=max_microbatches,
        pp_size=pp_size,
        kv_free_blocks=kv_free_blocks,
        block_size=block_size,
        max_num_batched_tokens=max_num_batched_tokens,
    )


def _assert_invariants(
    plan: MicroBatchList, requests: list[TauRequestSnapshot], ctx: PackContext
) -> None:
    input_ids = {r.request_id for r in requests}
    assert plan.admitted_ids <= input_ids
    assert plan.deferred_ids == input_ids - plan.admitted_ids
    assert not (plan.admitted_ids & plan.deferred_ids)
    packed = [rid for task in plan.tasks for rid in task.req_ids]
    assert frozenset(packed) == plan.admitted_ids
    assert len(packed) == len(set(packed))
    assert plan.tasks
    for i, task in enumerate(plan.tasks):
        assert task.index == i
        assert task.req_ids
        assert len(task.req_ids) <= ctx.max_num_seqs
    if ctx.max_microbatches >= 1:
        assert len(plan.tasks) <= ctx.max_microbatches


def test_wait_ms_clamps_before_arrival():
    assert wait_ms(arrival_time=2.0, now=1.0) == 0.0
    assert wait_ms(arrival_time=1.0, now=1.5) == 500.0


def test_ttft_slack_is_slo_minus_wait():
    assert ttft_slack_ms(200.0, 50.0) == 150.0
    assert ttft_slack_ms(200.0, 250.0) == -50.0


def test_tau_max_divides_tpot_by_pp():
    assert tau_max_ms(80.0, None) == 80.0
    assert tau_max_ms(80.0, 4) == 20.0


def test_annotate_request_budget_fills_wait_and_slack():
    req = _req("a", arrival_time=1.0, ttft_slo_ms=200.0, tpot_slo_ms=80.0)
    filled = annotate_request_budget(req, now=1.5, pp_size=4)
    assert filled.wait_ms == 500.0
    assert filled.ttft_slack_ms == -300.0
    assert filled.tau_max_ms == 20.0
    assert req.wait_ms == 0.0


def test_planner_annotates_budgets_before_pack():
    class Capture(GreedyListStrategy):
        def pack(self, requests, ctx):
            self.seen = list(requests)
            return super().pack(requests, ctx)

    capture = Capture()
    planner = TauBatchPlanner(strategy=capture)
    planner.plan(
        [_req("a", arrival_time=1.0, ttft_slo_ms=200.0, tpot_slo_ms=80.0)],
        _ctx(now=1.25, pp_size=2),
    )
    assert capture.seen[0].wait_ms == 250.0
    assert capture.seen[0].ttft_slack_ms == -50.0
    assert capture.seen[0].tau_max_ms == 40.0


def _eos_event(*finished: str, remaining: tuple[str, ...] = ()) -> EosEvent:
    return EosEvent(
        finished_ids=finished,
        wave_id=0,
        batch_idx=0,
        phase="decode",
        remaining_ids=remaining,
        waiting_ids=("late",),
    )


def test_default_eos_strategy_is_noop():
    planner = TauBatchPlanner()
    assert isinstance(planner.eos_strategy, NoOpEosStrategy)
    planner.on_eos(_eos_event("r0"))


def test_planner_forwards_eos_to_strategy():
    seen: list[EosEvent] = []

    class Capture:
        def on_eos(self, event: EosEvent) -> None:
            seen.append(event)

    planner = TauBatchPlanner(eos_strategy=Capture())
    event = _eos_event("r0", remaining=("r1",))
    planner.on_eos(event)
    assert seen == [event]


def test_empty_snapshot_returns_none():
    planner = TauBatchPlanner()
    assert planner.plan([], _ctx()) is None


def test_invalid_slo_raises():
    planner = TauBatchPlanner()
    with pytest.raises(ValueError, match="ttft_slo_ms"):
        planner.plan([_req("a", ttft_slo_ms=0)], _ctx())
    with pytest.raises(ValueError, match="tpot_slo_ms"):
        planner.plan([_req("a", tpot_slo_ms=-1)], _ctx())


def test_duplicate_id_raises():
    planner = TauBatchPlanner()
    with pytest.raises(ValueError, match="duplicate"):
        planner.plan([_req("a"), _req("a")], _ctx())


def test_invalid_prompt_len_raises():
    planner = TauBatchPlanner()
    with pytest.raises(ValueError, match="prompt_len"):
        planner.plan([_req("a", prompt_len=0)], _ctx())


def test_invalid_pack_context_raises():
    planner = TauBatchPlanner()
    with pytest.raises(ValueError, match="max_num_seqs"):
        planner.plan([_req("a")], _ctx(max_num_seqs=0))
    with pytest.raises(ValueError, match="max_microbatches"):
        planner.plan([_req("a")], _ctx(max_microbatches=-1))


def test_greedy_admits_all_when_capacity_fits():
    planner = TauBatchPlanner()
    requests = [_req("r1"), _req("r2"), _req("r3")]
    ctx = _ctx(max_num_seqs=4, max_microbatches=2)
    plan = planner.plan(requests, ctx)
    assert plan is not None
    _assert_invariants(plan, requests, ctx)
    assert plan.admitted_ids == {"r1", "r2", "r3"}
    assert plan.deferred_ids == frozenset()
    assert plan.extra["strategy"] == "greedy"


def test_greedy_does_not_cut_pool_by_max_num_seqs():
    planner = TauBatchPlanner()
    requests = [_req(f"r{i}", tpot_slo_ms=float(10 + i)) for i in range(5)]
    ctx = _ctx(max_num_seqs=2, max_microbatches=4)
    plan = planner.plan(requests, ctx)
    assert plan is not None
    _assert_invariants(plan, requests, ctx)
    assert plan.admitted_ids == {"r0", "r1", "r2", "r3", "r4"}
    assert plan.deferred_ids == frozenset()
    assert [len(task.req_ids) for task in plan.tasks] == [2, 2, 1]


def test_greedy_prefers_tighter_tpot_then_earlier_arrival():
    planner = TauBatchPlanner()
    requests = [
        _req("late_tight", tpot_slo_ms=10.0, arrival_time=5.0),
        _req("early_loose", tpot_slo_ms=80.0, arrival_time=1.0),
        _req("early_tight", tpot_slo_ms=10.0, arrival_time=1.0),
    ]
    ctx = _ctx(max_num_seqs=2, max_microbatches=1)
    plan = planner.plan(requests, ctx)
    assert plan is not None
    _assert_invariants(plan, requests, ctx)
    assert plan.tasks[0].req_ids == ("early_tight", "late_tight")
    assert plan.deferred_ids == {"early_loose"}


def test_greedy_splits_into_microbatches():
    planner = TauBatchPlanner()
    requests = [_req(f"r{i}", tpot_slo_ms=float(i)) for i in range(1, 7)]
    ctx = _ctx(max_num_seqs=2, max_microbatches=3)
    plan = planner.plan(requests, ctx)
    assert plan is not None
    _assert_invariants(plan, requests, ctx)
    assert len(plan.tasks) == 3
    assert [task.req_ids for task in plan.tasks] == [
        ("r1", "r2"),
        ("r3", "r4"),
        ("r5", "r6"),
    ]


def test_greedy_defers_beyond_microbatch_capacity():
    planner = TauBatchPlanner()
    requests = [_req(f"r{i}") for i in range(10)]
    ctx = _ctx(max_num_seqs=3, max_microbatches=2)
    plan = planner.plan(requests, ctx)
    assert plan is not None
    _assert_invariants(plan, requests, ctx)
    assert len(plan.admitted_ids) == 6
    assert len(plan.deferred_ids) == 4
    assert len(plan.tasks) == 2


def test_plan_is_deterministic():
    requests = [
        _req("b", tpot_slo_ms=20.0, arrival_time=2.0),
        _req("a", tpot_slo_ms=20.0, arrival_time=1.0),
        _req("c", tpot_slo_ms=10.0, arrival_time=3.0),
    ]
    ctx = _ctx(max_num_seqs=1, max_microbatches=2)
    a = TauBatchPlanner().plan(requests, ctx)
    b = TauBatchPlanner().plan(requests, ctx)
    assert a is not None and b is not None
    assert a.tasks == b.tasks
    assert a.admitted_ids == b.admitted_ids
    assert a.deferred_ids == b.deferred_ids


def test_custom_strategy_is_used():
    class AdmitNone(GreedyListStrategy):
        def pack(self, requests, ctx):
            ids = frozenset(r.request_id for r in requests)
            return MicroBatchList(
                tasks=(),
                admitted_ids=frozenset(),
                deferred_ids=ids,
                extra={"strategy": "none"},
            )

    planner = TauBatchPlanner(strategy=AdmitNone())
    assert planner.plan([_req("a"), _req("b")], _ctx()) is None


def test_invalid_max_new_tokens_raises():
    planner = TauBatchPlanner()
    with pytest.raises(ValueError, match="max_new_tokens"):
        planner.plan([_req("a", max_new_tokens=-1)], _ctx())


def test_kv_context_requires_block_size():
    planner = TauBatchPlanner()
    with pytest.raises(ValueError, match="block_size"):
        planner.plan([_req("a")], _ctx(kv_free_blocks=8))


def test_greedy_defers_when_kv_blocks_exhausted():
    planner = TauBatchPlanner()
    requests = [
        _req("r0", tpot_slo_ms=10.0, prompt_len=16, max_new_tokens=16),
        _req("r1", tpot_slo_ms=11.0, prompt_len=16, max_new_tokens=16),
        _req("r2", tpot_slo_ms=12.0, prompt_len=16, max_new_tokens=16),
    ]
    ctx = _ctx(
        max_num_seqs=4,
        max_microbatches=4,
        kv_free_blocks=4,
        block_size=16,
    )
    plan = planner.plan(requests, ctx)
    assert plan is not None
    _assert_invariants(plan, requests, ctx)
    assert plan.admitted_ids == {"r0", "r1"}
    assert plan.deferred_ids == {"r2"}
    assert estimate_kv_blocks(16, 16, 16) == 2


def test_pack_pool_respects_list_cap():
    planner = TauBatchPlanner()
    requests = [_req(f"r{i}", tpot_slo_ms=float(i + 1)) for i in range(64)]
    ctx = _ctx(max_num_seqs=4, max_microbatches=8)
    plan = planner.plan(requests, ctx)
    assert plan is not None
    _assert_invariants(plan, requests, ctx)
    assert len(plan.admitted_ids) == 32
    assert len(plan.deferred_ids) == 32
    assert len(plan.tasks) == 8
    assert [len(task.req_ids) for task in plan.tasks] == [4] * 8
    assert [task.index for task in plan.tasks] == list(range(8))


def test_zero_list_cap_packs_whole_pool():
    planner = TauBatchPlanner()
    requests = [_req(f"r{i}", tpot_slo_ms=float(i + 1)) for i in range(64)]
    ctx = _ctx(max_num_seqs=4, max_microbatches=0)
    plan = planner.plan(requests, ctx)
    assert plan is not None
    _assert_invariants(plan, requests, ctx)
    assert len(plan.admitted_ids) == 64
    assert plan.deferred_ids == frozenset()
    assert len(plan.tasks) == 16
    assert [len(task.req_ids) for task in plan.tasks] == [4] * 16


def test_pack_defers_when_list_cap_is_tight():
    planner = TauBatchPlanner()
    requests = [_req(f"r{i}", tpot_slo_ms=float(i + 1)) for i in range(32)]
    ctx = _ctx(max_num_seqs=4, max_microbatches=2)
    plan = planner.plan(requests, ctx)
    assert plan is not None
    _assert_invariants(plan, requests, ctx)
    assert len(plan.admitted_ids) == 8
    assert len(plan.deferred_ids) == 24
    assert len(plan.tasks) == 2


def test_short_pool_is_packed_without_padding():
    planner = TauBatchPlanner()
    requests = [_req(f"r{i}", tpot_slo_ms=float(i + 1)) for i in range(10)]
    ctx = _ctx(max_num_seqs=4, max_microbatches=8)
    plan = planner.plan(requests, ctx)
    assert plan is not None
    _assert_invariants(plan, requests, ctx)
    assert len(plan.admitted_ids) == 10
    assert plan.deferred_ids == frozenset()
    assert [len(task.req_ids) for task in plan.tasks] == [4, 4, 2]


def test_greedy_skips_unfittable_and_takes_next():
    planner = TauBatchPlanner()
    requests = [
        _req("big", tpot_slo_ms=1.0, prompt_len=64, max_new_tokens=64),
        _req("small", tpot_slo_ms=10.0, prompt_len=16, max_new_tokens=16),
    ]
    ctx = _ctx(
        max_num_seqs=2,
        max_microbatches=2,
        kv_free_blocks=3,
        block_size=16,
    )
    plan = planner.plan(requests, ctx)
    assert plan is not None
    _assert_invariants(plan, requests, ctx)
    assert plan.admitted_ids == {"small"}
    assert plan.deferred_ids == {"big"}


@pytest.mark.parametrize(
    "cap,expected",
    [
        (1, [("a", "c")]),
        (2, [("a", "c"), ("b",)]),
        (0, [("a", "c"), ("b",)]),
    ],
)
def test_token_overflow_scans_later_requests(cap, expected):
    requests = [_req(rid, prompt_len=n) for rid, n in [("a", 7), ("b", 6), ("c", 3)]]
    ctx = _ctx(max_microbatches=cap, max_num_batched_tokens=10)
    plan = TauBatchPlanner().plan(requests, ctx)
    assert [task.req_ids for task in plan.tasks] == expected
    assert plan.deferred_ids == ({"b"} if cap == 1 else set())


def test_individually_oversized_prompt_is_deferred():
    requests = [_req("a", prompt_len=11), _req("b", prompt_len=10)]
    plan = TauBatchPlanner().plan(requests, _ctx(max_num_batched_tokens=10))
    assert plan.tasks[0].req_ids == ("b",)
    assert plan.deferred_ids == {"a"}
    assert TauBatchPlanner().plan(requests[:1], _ctx(max_num_batched_tokens=10)) is None


def test_skipped_prompt_does_not_consume_kv_reservation():
    requests = [
        _req(rid, prompt_len=n, max_new_tokens=0)
        for rid, n in [("a", 7), ("b", 6), ("c", 3)]
    ]
    ctx = _ctx(
        max_microbatches=1, max_num_batched_tokens=10, kv_free_blocks=10, block_size=1
    )
    plan = TauBatchPlanner().plan(requests, ctx)
    assert plan.tasks[0].req_ids == ("a", "c")


def test_custom_packer_cannot_exceed_token_limit():
    class Oversized:
        def pack(self, requests, ctx):
            from vllm.v1.core.sched.tau_batch import MicroBatchTask

            return MicroBatchList(
                tasks=(MicroBatchTask(("a", "b"), 0),),
                admitted_ids=frozenset({"a", "b"}),
                deferred_ids=frozenset(),
            )

    with pytest.raises(ValueError, match="prompt tokens"):
        TauBatchPlanner(strategy=Oversized()).plan(
            [_req("a", prompt_len=6), _req("b", prompt_len=6)],
            _ctx(max_num_batched_tokens=10),
        )


def test_invalid_token_limit_rejected():
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        TauBatchPlanner().plan([_req("a")], _ctx(max_num_batched_tokens=0))


@pytest.mark.parametrize("task_cap", [1, 2, 4])
def test_worker_request_capacity_limits_each_task_not_whole_wave(task_cap):
    requests = [_req(str(i)) for i in range(6)]
    ctx = _ctx(max_num_seqs=task_cap, max_microbatches=0)
    plan = TauBatchPlanner().plan(requests, ctx)
    _assert_invariants(plan, requests, ctx)
    assert len(plan.admitted_ids) == 6
    assert all(len(task.req_ids) <= task_cap for task in plan.tasks)


def test_custom_plan_cannot_exceed_worker_request_capacity():
    class Oversized:
        def pack(self, requests, ctx):
            from vllm.v1.core.sched.tau_batch import MicroBatchTask

            return MicroBatchList(
                tasks=(MicroBatchTask(("a", "b", "c"), 0),),
                admitted_ids=frozenset({"a", "b", "c"}),
                deferred_ids=frozenset(),
            )

    with pytest.raises(ValueError, match="max_num_seqs"):
        TauBatchPlanner(strategy=Oversized()).plan(
            [_req(rid) for rid in "abc"],
            _ctx(max_num_seqs=2),
        )


@pytest.mark.parametrize("free,admitted", [(8, ("a", "b")), (4, ("a",))])
def test_custom_plan_kv_reservation_accepts_exact_capacity(free, admitted):
    from vllm.v1.core.sched.tau_batch import MicroBatchTask

    class Custom:
        def pack(self, requests, ctx):
            return MicroBatchList(
                tasks=tuple(
                    MicroBatchTask((rid,), i) for i, rid in enumerate(admitted)
                ),
                admitted_ids=frozenset(admitted),
                deferred_ids=frozenset({"a", "b"} - set(admitted)),
            )

    requests = [_req(rid, prompt_len=32, max_new_tokens=32) for rid in "ab"]
    plan = TauBatchPlanner(strategy=Custom()).plan(
        requests, _ctx(kv_free_blocks=free, block_size=16)
    )
    assert plan.admitted_ids == set(admitted)


def test_custom_plan_kv_reservation_covers_entire_wave():
    from vllm.v1.core.sched.tau_batch import MicroBatchTask

    class Custom:
        def pack(self, requests, ctx):
            return MicroBatchList(
                tasks=(MicroBatchTask(("a",), 0), MicroBatchTask(("b",), 1)),
                admitted_ids=frozenset({"a", "b"}),
                deferred_ids=frozenset(),
            )

    with pytest.raises(ValueError, match="reserves 8 KV blocks, only 4"):
        TauBatchPlanner(strategy=Custom()).plan(
            [_req(rid, prompt_len=32, max_new_tokens=32) for rid in "ab"],
            _ctx(kv_free_blocks=4, block_size=16),
        )


def test_full_tasks_do_not_rescan_untouched_waiting_tail(monkeypatch):
    from vllm.v1.core.sched.tau_batch import strategy

    examined = 0
    original = strategy._kv_blocks_if_fits

    def count(*args):
        nonlocal examined
        examined += 1
        return original(*args)

    monkeypatch.setattr(strategy, "_kv_blocks_if_fits", count)
    requests = [_req(str(i), prompt_len=8) for i in range(1000)]
    requests.insert(0, _req("oversized", prompt_len=1000))
    plan = TauBatchPlanner().plan(
        requests,
        _ctx(max_microbatches=0, max_num_seqs=4, max_num_batched_tokens=32),
    )
    assert len(plan.tasks) == 250
    assert plan.deferred_ids == {"oversized"}
    # A linear work bound, not a machine-dependent timing threshold.
    assert examined <= 2 * len(requests)


def test_token_skips_precede_untouched_tail_in_next_microbatch():
    requests = [
        _req(rid, prompt_len=n)
        for rid, n in [("a", 7), ("b", 6), ("c", 3), ("d", 4), ("e", 5)]
    ]
    plan = TauBatchPlanner().plan(
        requests,
        _ctx(max_microbatches=0, max_num_seqs=2, max_num_batched_tokens=10),
    )
    assert [task.req_ids for task in plan.tasks] == [("a", "c"), ("b", "d"), ("e",)]
