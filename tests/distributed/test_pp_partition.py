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
    assert plan.cost_ms == pytest.approx(16.0)


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
    assert plan.cost_ms == pytest.approx(30.0)


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
