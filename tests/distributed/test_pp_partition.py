# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest

from tests.distributed.pp_trace_fixtures import write_trace
from vllm.distributed.pp_partition import (
    RankCost,
    partition_layers,
)


def _write_rank_jsonl(path, recs):
    write_trace(path, recs)


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
        "compute_wall_ms": compute_ms,
        "send_ms": send_ms,
        "send_transfer_ms": 99999,
        "send_service_ms": send_ms,
        "send_service_source": "measured_serving_overlap",
        "recv_bytes": None if pp_rank == 0 else send_bytes,
        "send_bytes": send_bytes if send_ms is not None else None,
    }


def test_throughput_balances_identical_ranks():
    costs = [
        RankCost(0, 16, t_layer_ms=1.0, t_comm_out_ms=0.1, n_steps=10),
        RankCost(1, 16, t_layer_ms=1.0, t_comm_out_ms=None, n_steps=10),
    ]
    plan = partition_layers(
        costs,
        allow_unchecked_memory=True,
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
        allow_unchecked_memory=True,
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
        allow_unchecked_memory=True,
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
    plan = partition_layers(costs, allow_unchecked_memory=True, min_pp_size=2)
    assert plan.partitions == [16, 16]
    assert plan.cost_ms == 24
    assert plan.cost_model == "blocking"
    timeline = _simulate_blocking([20, 12], [8])
    assert timeline[-1] - timeline[-2] == 28
    assert plan.to_dict()["cost_kind"] == "steady_state_cycle_ms"
    assert plan.to_dict()["memory_feasibility_checked"] is False


def test_ideal_overlap_is_explicit_opt_in():
    costs = [RankCost(0, 16, 1, 8, 10), RankCost(1, 16, 1, None, 10)]
    plan = partition_layers(
        costs, allow_unchecked_memory=True, min_pp_size=2, overlap_comm=True
    )
    assert plan.partitions == [16, 16]
    assert plan.cost_ms == 16
    assert plan.cost_model == "ideal_overlap"


def test_three_stages_charge_both_links_to_middle_rank():
    costs = [
        RankCost(0, 10, 1, 1, 10),
        RankCost(1, 10, 1, 9, 10),
        RankCost(2, 10, 1, None, 10),
    ]
    plan = partition_layers(costs, allow_unchecked_memory=True, min_pp_size=3)
    timeline = _simulate_blocking(plan.partitions, [1, 9])
    assert plan.cost_ms == timeline[-1] - timeline[-2] == 17
    assert plan.partitions[1] < plan.partitions[0]


def test_dropping_trailing_rank_removes_its_incoming_hop():
    costs = [
        RankCost(0, 4, 1, 1, 10),
        RankCost(1, 4, 1, 100, 10),
        RankCost(2, 4, 100, None, 10),
    ]
    plan = partition_layers(costs, allow_unchecked_memory=True)
    assert plan.partitions == [6, 6]
    assert plan.cost_ms == 7
    single = partition_layers(costs, allow_unchecked_memory=True, max_pp_size=1)
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
                timeline = _simulate_blocking(compute, hops[: pp_size - 1])
                candidates.append(
                    (
                        timeline[-1] - timeline[-2],
                        parts,
                        sum(compute) + sum(hops[: pp_size - 1]),
                    )
                )
        plan = partition_layers(costs, allow_unchecked_memory=True, num_layers=layers)
        assert plan.cost_ms == min(c[0] for c in candidates)
        sequential = plan.to_dict()["predicted_sequential_ms"]
        assert (plan.cost_ms, plan.partitions, sequential) in candidates
        assert (plan.cost_ms, sequential) == min((c[0], c[2]) for c in candidates)


def test_zero_communication_reduces_to_compute_balance():
    costs = [RankCost(0, 4, 1, 0, 10), RankCost(1, 4, 2, None, 10)]
    blocking = partition_layers(costs, allow_unchecked_memory=True, min_pp_size=2)
    ideal = partition_layers(
        costs, allow_unchecked_memory=True, min_pp_size=2, overlap_comm=True
    )
    assert blocking.partitions == ideal.partitions
    assert blocking.cost_ms == ideal.cost_ms


def test_latency_counts_each_transfer_only_once():
    costs = [RankCost(0, 1, 2, 8, 10), RankCost(1, 1, 3, None, 10)]
    plan = partition_layers(
        costs, allow_unchecked_memory=True, objective="latency", min_pp_size=2
    )
    assert plan.cost_ms == 13
    assert plan.cost_model == "sequential"


def test_throughput_tie_break_is_global_not_a_prefix_tie():
    # The best two-stage prefix is 6/2 (max=6, sum=12). 7/1 has a worse
    # prefix max=7 but a better sum=10. The final 1000 ms stage hides both
    # maxima, so a lexicographic prefix DP would incorrectly discard 7/1.
    costs = [
        RankCost(0, 4, 1, 0, 1),
        RankCost(1, 4, 3, 0, 1),
        RankCost(2, 1, 0, None, 1, t_fixed_ms=1000, min_layers=1, max_layers=1),
    ]
    plan = partition_layers(costs, min_pp_size=3, allow_unchecked_memory=True)
    assert plan.partitions == [7, 1, 1]
    assert plan.cost_ms == 1000
    assert plan.to_dict()["predicted_sequential_ms"] == 1010
    assert plan.to_dict()["tie_breaker"] == "sequential_cost"


def test_tie_break_does_not_trade_away_throughput_or_memory_feasibility():
    from tests.distributed.test_pp_memory import _device, _profile

    costs = [RankCost(0, 4, 1, 0, 1), RankCost(1, 4, 3, None, 1)]
    plan = partition_layers(costs, min_pp_size=2, allow_unchecked_memory=True)
    assert plan.partitions == [6, 2]  # Faster sum at 7/1 has worse throughput.
    memory = _profile([_device(0, 5, [1] * 8), _device(1, 8, [1] * 8)])
    bounded = partition_layers(costs, min_pp_size=2, memory_profile=memory)
    assert bounded.partitions == [5, 3]
