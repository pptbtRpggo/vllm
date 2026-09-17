# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.distributed.pp_partition import (
    RankCost,
    fit_rank_costs,
    partition_layers,
    plan_from_trace_dir,
)


def _write_rank_jsonl(path, recs):
    path.write_text(
        "".join(json.dumps(rec) + "\n" for rec in recs),
        encoding="utf-8",
    )


def _rec(
    *,
    pp_rank: int,
    pp_size: int,
    step: int,
    start: int,
    end: int,
    compute_ms: float,
    send_ms: float | None,
    num_tokens: int = 8,
    num_ctx_tokens: int = 0,
    num_generation_tokens: int = 8,
    send_bytes: int | None = 4096,
):
    return {
        "step": step,
        "ts_unix": 0.0,
        "pp_rank": pp_rank,
        "pp_size": pp_size,
        "tp_rank": 0,
        "start_layer": start,
        "end_layer": end,
        "num_tokens": num_tokens,
        "num_reqs": 1,
        "num_ctx_requests": 1 if num_ctx_tokens else 0,
        "num_ctx_tokens": num_ctx_tokens,
        "num_generation_requests": 1 if num_generation_tokens else 0,
        "num_generation_tokens": num_generation_tokens,
        "recv_ms": None if pp_rank == 0 else 0.5,
        "compute_ms": compute_ms,
        "send_ms": send_ms,
        "send_transfer_ms": send_ms,
        "recv_bytes": None if pp_rank == 0 else send_bytes,
        "send_bytes": send_bytes if send_ms is not None else None,
    }


def test_fit_rank_costs_from_jsonl(tmp_path):
    recs0 = [
        _rec(
            pp_rank=0,
            pp_size=2,
            step=i,
            start=0,
            end=16,
            compute_ms=16.0,
            send_ms=2.0,
        )
        for i in range(8)
    ]
    recs1 = [
        _rec(
            pp_rank=1,
            pp_size=2,
            step=i,
            start=16,
            end=32,
            compute_ms=32.0,
            send_ms=None,
        )
        for i in range(8)
    ]
    _write_rank_jsonl(tmp_path / "pp_stage_pp0_tp0.jsonl", recs0)
    _write_rank_jsonl(tmp_path / "pp_stage_pp1_tp0.jsonl", recs1)

    costs = fit_rank_costs(
        {0: recs0, 1: recs1},
        warmup_steps=5,
    )
    assert len(costs) == 2
    assert costs[0].n_layers == 16
    assert costs[0].t_layer_ms == pytest.approx(1.0)
    assert costs[0].t_comm_out_ms == pytest.approx(2.0)
    assert costs[1].t_layer_ms == pytest.approx(2.0)
    assert costs[1].t_comm_out_ms is None


def test_fit_uses_modeled_transfer_cost_instead_of_blocking_wait():
    rec = _rec(
        pp_rank=0, pp_size=1, step=6, start=0, end=16,
        compute_ms=16, send_ms=104,
    )
    rec["send_transfer_ms"] = 4.0
    costs = fit_rank_costs({0: [rec]})
    assert costs[0].t_comm_out_ms == 4.0


def test_replay_recorded_scales_are_not_applied_twice(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_PP_HETERO", "1,2/4")
    for rank in range(2):
        rec = _rec(
            pp_rank=rank, pp_size=2, step=6, start=rank * 16,
            end=(rank + 1) * 16, compute_ms=16 * (rank + 1),
            send_ms=104 if rank == 0 else None,
        )
        rec["compute_scale"] = rank + 1
        rec["comm_scale"] = 4 if rank == 0 else 1
        rec["send_transfer_ms"] = 4 if rank == 0 else None
        _write_rank_jsonl(tmp_path / f"pp_stage_pp{rank}_tp0.jsonl", [rec])
    live = plan_from_trace_dir(tmp_path, hetero="", min_pp_size=2)
    replay = plan_from_trace_dir(tmp_path, min_pp_size=2)
    assert live == replay
    assert replay.rank_costs[0].t_comm_out_ms == 4.0
    assert replay.rank_costs[1].t_layer_ms == 2.0

    changed = plan_from_trace_dir(tmp_path, hetero="1,2/8", min_pp_size=2)
    assert changed.rank_costs[0].t_comm_out_ms == 8.0


def test_throughput_balances_identical_ranks():
    costs = [
        RankCost(0, 16, t_layer_ms=1.0, t_comm_out_ms=0.1, n_steps=10),
        RankCost(1, 16, t_layer_ms=1.0, t_comm_out_ms=None, n_steps=10),
    ]
    plan = partition_layers(
        costs,
        objective="throughput",
        num_layers=32,
        min_pp_size=2,
        max_pp_size=2,
    )
    assert plan.partitions == [16, 16]
    assert plan.cost_ms == pytest.approx(16.1)


def test_latency_drops_ranks_when_comm_is_expensive():
    costs = [
        RankCost(0, 16, t_layer_ms=1.0, t_comm_out_ms=100.0, n_steps=10),
        RankCost(1, 16, t_layer_ms=1.0, t_comm_out_ms=None, n_steps=10),
    ]
    plan = partition_layers(
        costs,
        objective="latency",
        num_layers=32,
        min_pp_size=1,
        max_pp_size=2,
    )
    assert plan.partitions == [32]
    assert plan.cost_ms == pytest.approx(32.0)


def test_forced_pp2_puts_more_layers_on_faster_rank():
    # Rank 0 is the source and is slower; rank 1 is 10x faster.
    costs = [
        RankCost(0, 8, t_layer_ms=10.0, t_comm_out_ms=1.0, n_steps=10),
        RankCost(1, 24, t_layer_ms=1.0, t_comm_out_ms=None, n_steps=10),
    ]
    plan = partition_layers(
        costs,
        objective="throughput",
        num_layers=32,
        min_pp_size=2,
        max_pp_size=2,
    )
    assert sum(plan.partitions) == 32
    # Bottleneck is max(10*n0, 1, n1) with n0+n1=32.
    # n0=2 -> max(20, 30)=30; n0=3 -> max(30, 29)=30.
    assert plan.partitions[0] in (2, 3)
    assert plan.partitions[1] == 32 - plan.partitions[0]
    assert plan.cost_ms == pytest.approx(31.0)


def test_plan_from_trace_dir_roundtrip(tmp_path):
    recs0 = [
        _rec(
            pp_rank=0,
            pp_size=2,
            step=i,
            start=0,
            end=16,
            compute_ms=16.0,
            send_ms=0.2,
        )
        for i in range(10)
    ]
    recs1 = [
        _rec(
            pp_rank=1,
            pp_size=2,
            step=i,
            start=16,
            end=32,
            compute_ms=16.0,
            send_ms=None,
        )
        for i in range(10)
    ]
    _write_rank_jsonl(tmp_path / "pp_stage_pp0_tp0.jsonl", recs0)
    _write_rank_jsonl(tmp_path / "pp_stage_pp1_tp0.jsonl", recs1)

    plan = plan_from_trace_dir(
        tmp_path,
        objective="throughput",
        warmup_steps=5,
        min_pp_size=2,
        max_pp_size=2,
    )
    assert plan.env_value == "16,16"
    assert plan.pp_size == 2
    payload = plan.to_dict()
    assert payload["VLLM_PP_LAYER_PARTITION"] == "16,16"
    assert payload["partitions"] == [16, 16]
    assert payload["objective"] == "throughput"
    assert len(payload["rank_costs"]) == 2


def test_plan_from_trace_dir_compute_scale_unbalances(tmp_path):
    recs0 = [
        _rec(
            pp_rank=0,
            pp_size=2,
            step=i,
            start=0,
            end=16,
            compute_ms=16.0,
            send_ms=0.2,
        )
        for i in range(10)
    ]
    recs1 = [
        _rec(
            pp_rank=1,
            pp_size=2,
            step=i,
            start=16,
            end=32,
            compute_ms=16.0,
            send_ms=None,
        )
        for i in range(10)
    ]
    _write_rank_jsonl(tmp_path / "pp_stage_pp0_tp0.jsonl", recs0)
    _write_rank_jsonl(tmp_path / "pp_stage_pp1_tp0.jsonl", recs1)

    plan = plan_from_trace_dir(
        tmp_path,
        objective="throughput",
        warmup_steps=5,
        min_pp_size=2,
        max_pp_size=2,
        hetero="1,2",
    )
    assert sum(plan.partitions) == 32
    assert plan.partitions[0] > plan.partitions[1]
    assert plan.rank_costs[1].t_layer_ms == pytest.approx(2.0)


def test_plan_from_trace_dir_comm_scale_multiplies_hop(tmp_path):
    recs0 = [
        _rec(
            pp_rank=0,
            pp_size=2,
            step=i,
            start=0,
            end=16,
            compute_ms=16.0,
            send_ms=0.5,
        )
        for i in range(10)
    ]
    recs1 = [
        _rec(
            pp_rank=1,
            pp_size=2,
            step=i,
            start=16,
            end=32,
            compute_ms=16.0,
            send_ms=None,
        )
        for i in range(10)
    ]
    _write_rank_jsonl(tmp_path / "pp_stage_pp0_tp0.jsonl", recs0)
    _write_rank_jsonl(tmp_path / "pp_stage_pp1_tp0.jsonl", recs1)

    plan = plan_from_trace_dir(
        tmp_path,
        objective="throughput",
        warmup_steps=5,
        min_pp_size=2,
        max_pp_size=2,
        hetero="/4",
    )
    assert plan.rank_costs[0].t_comm_out_ms == pytest.approx(2.0)
    assert plan.env_value == "16,16"


def _simulate_blocking(compute, hops, num_batches=64):
    """Independent rendezvous timeline; no max(stage occupancy) formula.

    A sender can proceed once its forward and the receiver's previous step
    finish. Both devices remain occupied until that transfer completes.
    """
    free = [0.0] * len(compute)
    completions = []
    for _ in range(num_batches):
        input_ready = 0.0
        for rank, duration in enumerate(compute):
            forward_end = max(input_ready, free[rank]) + duration
            if rank < len(hops):
                transfer_end = max(forward_end, free[rank + 1]) + hops[rank]
                free[rank] = transfer_end
                input_ready = transfer_end
            else:
                free[rank] = forward_end
        completions.append(free[-1])
    return completions


def test_blocking_two_stage_regression():
    # The old inbound-only objective chose 20/12 with predicted cost 20.
    costs = [RankCost(0, 16, 1, 8, 10), RankCost(1, 16, 1, None, 10)]
    plan = partition_layers(costs, min_pp_size=2)
    assert plan.partitions == [16, 16]
    assert plan.cost_ms == 24
    assert plan.cost_model == "blocking"
    timeline = _simulate_blocking([20, 12], [8])
    assert timeline[-1] - timeline[-2] == 28
    assert plan.to_dict()["cost_kind"] == "steady_state_cycle_ms"
    assert plan.to_dict()["memory_feasibility_checked"] is False


def test_ideal_overlap_is_explicit_opt_in():
    costs = [RankCost(0, 16, 1, 8, 10), RankCost(1, 16, 1, None, 10)]
    plan = partition_layers(costs, min_pp_size=2, overlap_comm=True)
    assert plan.partitions == [16, 16]
    assert plan.cost_ms == 16
    assert plan.cost_model == "ideal_overlap"


def test_three_stages_charge_both_links_to_middle_rank():
    costs = [
        RankCost(0, 10, 1, 1, 10),
        RankCost(1, 10, 1, 9, 10),
        RankCost(2, 10, 1, None, 10),
    ]
    plan = partition_layers(costs, min_pp_size=3)
    timeline = _simulate_blocking(plan.partitions, [1, 9])
    assert plan.cost_ms == timeline[-1] - timeline[-2] == 17
    assert plan.partitions[1] < plan.partitions[0]


def test_dropping_trailing_rank_removes_its_incoming_hop():
    costs = [
        RankCost(0, 4, 1, 1, 10),
        RankCost(1, 4, 1, 100, 10),
        RankCost(2, 4, 100, None, 10),
    ]
    plan = partition_layers(costs)
    assert plan.partitions == [6, 6]
    assert plan.cost_ms == 7
    single = partition_layers(costs, max_pp_size=1)
    assert single.partitions == [12]
    assert single.cost_ms == 12


def test_blocking_dp_matches_exhaustive_event_simulation():
    import itertools
    import random

    rng = random.Random(203)
    for _ in range(30):
        ranks = rng.randint(2, 4)
        layers = rng.randint(ranks, 8)
        speeds = [rng.randint(1, 5) for _ in range(ranks)]
        hops = [rng.randint(0, 9) for _ in range(ranks - 1)]
        costs = [
            RankCost(r, 1, speeds[r], hops[r] if r < ranks - 1 else None, 10)
            for r in range(ranks)
        ]
        candidates = []
        for pp_size in range(1, ranks + 1):
            for cuts in itertools.combinations(range(1, layers), pp_size - 1):
                edges = (0, *cuts, layers)
                parts = [b - a for a, b in zip(edges, edges[1:])]
                compute = [parts[r] * speeds[r] for r in range(pp_size)]
                timeline = _simulate_blocking(compute, hops[:pp_size - 1])
                candidates.append((timeline[-1] - timeline[-2], parts))
        plan = partition_layers(costs, num_layers=layers)
        assert plan.cost_ms == min(c[0] for c in candidates)
        assert (plan.cost_ms, plan.partitions) in candidates


def test_zero_communication_reduces_to_compute_balance():
    costs = [RankCost(0, 4, 1, 0, 10), RankCost(1, 4, 2, None, 10)]
    blocking = partition_layers(costs, min_pp_size=2)
    ideal = partition_layers(costs, min_pp_size=2, overlap_comm=True)
    assert blocking.partitions == ideal.partitions
    assert blocking.cost_ms == ideal.cost_ms


def test_latency_counts_each_transfer_only_once():
    costs = [RankCost(0, 1, 2, 8, 10), RankCost(1, 1, 3, None, 10)]
    plan = partition_layers(costs, objective="latency", min_pp_size=2)
    assert plan.cost_ms == 13
    assert plan.cost_model == "sequential"


def test_legacy_wall_time_requires_explicit_opt_in():
    rec = _rec(pp_rank=0, pp_size=2, step=6, start=0, end=16,
               compute_ms=16, send_ms=104)
    del rec["send_transfer_ms"]
    with pytest.raises(ValueError, match="includes peer waiting"):
        fit_rank_costs({0: [rec]})
    with pytest.warns(UserWarning, match="including peer waits"):
        costs = fit_rank_costs({0: [rec]}, allow_wall_time_comm=True)
    assert costs[0].t_comm_out_ms == 104
    assert costs[0].comm_source == "wall_time"


@pytest.mark.parametrize("bad_value", [-1, float("nan"), float("inf")])
def test_invalid_link_cost_is_not_silently_discarded(bad_value):
    rec = _rec(pp_rank=0, pp_size=2, step=6, start=0, end=16,
               compute_ms=16, send_ms=1)
    rec["send_transfer_ms"] = bad_value
    with pytest.raises(ValueError, match="invalid transfer cost"):
        fit_rank_costs({0: [rec]})


@pytest.mark.parametrize("workload,warmup", [("decode", 0), ("all", 10)])
def test_empty_filtered_trace_does_not_fall_back_to_wrong_samples(workload, warmup):
    rec = _rec(pp_rank=0, pp_size=1, step=6, start=0, end=16,
               compute_ms=16, send_ms=None, num_ctx_tokens=8,
               num_generation_tokens=0)
    with pytest.raises(ValueError, match="no usable trace steps"):
        fit_rank_costs({0: [rec]}, workload=workload, warmup_steps=warmup)


def test_replay_scale_provenance_uses_the_filtered_samples(tmp_path):
    warmup = _rec(pp_rank=0, pp_size=1, step=0, start=0, end=8,
                  compute_ms=800, send_ms=None)
    measured = _rec(pp_rank=0, pp_size=1, step=6, start=0, end=8,
                    compute_ms=8, send_ms=None)
    warmup["compute_scale"] = 100
    measured["compute_scale"] = 1
    _write_rank_jsonl(tmp_path / "pp_stage_pp0_tp0.jsonl", [warmup, measured])
    plan = plan_from_trace_dir(tmp_path, hetero="2")
    assert plan.cost_ms == 16
    assert plan.rank_costs[0].n_steps == 1


def test_standalone_planner_defaults_to_blocking(tmp_path, capsys):
    from vllm.distributed.pp_partition import main

    for rank in range(2):
        rec = _rec(pp_rank=rank, pp_size=2, step=6, start=16 * rank,
                   end=16 * (rank + 1), compute_ms=16,
                   send_ms=8 if rank == 0 else None)
        _write_rank_jsonl(tmp_path / f"pp_stage_pp{rank}_tp0.jsonl", [rec])
    main([str(tmp_path), "--pp-size", "2"])
    out = capsys.readouterr().out
    assert "cost_model=blocking" in out
    assert "predicted_cost_ms=24.0000" in out
    main([str(tmp_path), "--pp-size", "2", "--overlap-comm"])
    assert "predicted_cost_ms=16.0000" in capsys.readouterr().out
