# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import json
import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vllm.distributed.pp_memory import DeviceMemory, PPMemoryProfile
from vllm.distributed.pp_partition import (
    RankCost,
    partition_layers,
    plan_from_trace_dir,
)
from vllm.distributed.pp_profile import (
    format_serve_command,
    profile_pp_partition,
    write_profile_result,
)


def _scope():
    return dict(model="test-model", revision=None, dtype="float16",
                quantization=None, kv_cache_dtype="auto", block_size=16,
                max_model_len=128, max_num_seqs=2, max_num_batched_tokens=32,
                gpu_memory_utilization=0.9)


def _device(rank, budget, weights, kv=None, tp=0, **overrides):
    values = dict(
        pp_rank=rank, tp_rank=tp, budget_bytes=budget,
        layer_weights_bytes=tuple(weights),
        layer_kv_bytes=tuple(kv if kv is not None else [0] * len(weights)),
        runtime_bytes=0, activation_bytes=0, workspace_bytes=0,
        communication_bytes=0, graph_bytes=0, safety_margin_bytes=0,
        first_stage_bytes=0, last_stage_bytes=0,
    )
    return DeviceMemory(**(values | overrides))


def _profile(devices, tp=1):
    return PPMemoryProfile(len(devices[0].layer_weights_bytes), len(devices) // tp,
                           tp, _scope(), tuple(devices))


def _costs(n=32, speeds=(1, 1), comm=100):
    return [RankCost(r, n // len(speeds), speed,
                     comm if r < len(speeds) - 1 else None, 10)
            for r, speed in enumerate(speeds)]


@pytest.mark.parametrize("objective,overlap", [("latency", False),
                                               ("throughput", False),
                                               ("throughput", True)])
def test_memory_prevents_dropping_rank_or_overfilling_fast_rank(objective, overlap):
    memory = _profile([_device(r, 16, [1] * 32) for r in range(2)])
    plan = partition_layers(_costs(speeds=(1, 2)), objective=objective,
                            overlap_comm=overlap, memory_profile=memory)
    assert plan.partitions == [16, 16]
    result = plan.to_dict()
    assert result["memory_feasibility_checked"] is True
    assert result["memory_feasibility_basis"] == "supplied_memory_bounds"
    assert [r["headroom_bytes"] for r in result["memory_usage"]] == [0, 0]


def test_kv_and_all_fixed_reserves_are_charged():
    devices = [
        _device(r, 25, [2] * 8, [1] * 8, runtime_bytes=1,
                activation_bytes=2, workspace_bytes=3, communication_bytes=1,
                graph_bytes=2, safety_margin_bytes=4)
        for r in range(2)
    ]
    memory = _profile(devices)
    plan = partition_layers(_costs(8), num_layers=8, memory_profile=memory)
    assert plan.partitions == [4, 4]
    assert all(row["required_bytes"] == 25 for row in memory.plan_usage([4, 4]))
    with pytest.raises(ValueError, match="No memory-feasible partition"):
        partition_layers(_costs(8), num_layers=8,
                         memory_profile=_profile([replace(d, budget_bytes=24)
                                                  for d in devices]))


def test_head_follows_final_selected_rank_and_single_rank_has_both_endpoints():
    memory = _profile([
        _device(0, 10, [1] * 8, first_stage_bytes=2, last_stage_bytes=3),
        _device(1, 20, [1] * 8, last_stage_bytes=3),
        _device(2, 20, [1] * 8, last_stage_bytes=3),
    ])
    plan = partition_layers(_costs(8, (1, 1, 1)), num_layers=8,
                            objective="latency", max_pp_size=2,
                            memory_profile=memory)
    assert plan.pp_size == 2  # 8 + embedding 2 + head 3 cannot fit on rank 0.
    usage = plan.to_dict()["memory_usage"]
    assert usage[0]["endpoint_bytes"] == 2
    assert usage[1]["endpoint_bytes"] == 3
    assert all(row["pp_rank"] < 2 for row in usage)
    with pytest.raises(ValueError, match="No memory-feasible partition"):
        partition_layers(_costs(8, (1, 1, 1)), num_layers=8, max_pp_size=1,
                         memory_profile=memory)


def test_nonuniform_layer_memory_uses_actual_interval():
    memory = _profile([_device(0, 10, [8, 1, 1, 1]),
                       _device(1, 10, [8, 1, 1, 1])])
    plan = partition_layers(_costs(4, (1, 10), comm=0), memory_profile=memory)
    assert plan.partitions == [3, 1]
    assert memory.plan_usage([3, 1])[0]["weights_bytes"] == 10


def test_all_tp_devices_must_fit_not_just_tp_zero():
    memory = _profile([
        _device(0, 100, [1] * 8, tp=0), _device(0, 3, [1] * 8, tp=1),
        _device(1, 100, [1] * 8, tp=0), _device(1, 5, [1] * 8, tp=1),
    ], tp=2)
    plan = partition_layers(_costs(8, (1, 10)), memory_profile=memory)
    assert plan.partitions == [3, 5]
    assert len(plan.to_dict()["memory_usage"]) == 4


def test_missing_memory_is_rejected_unless_explicitly_unchecked():
    with pytest.raises(ValueError, match="memory feasibility requires"):
        partition_layers(_costs())
    plan = partition_layers(_costs(), allow_unchecked_memory=True)
    assert plan.to_dict()["memory_feasibility_checked"] is False
    assert "vllm serve" not in format_serve_command(plan)


@pytest.mark.parametrize("field,value", [("budget_bytes", 0),
    ("budget_bytes", -1), ("budget_bytes", 1.5), ("runtime_bytes", True),
    ("graph_bytes", float("nan")), ("layer_weights_bytes", (1, -1)),
    ("layer_kv_bytes", (1,))])
def test_invalid_memory_data_is_rejected(field, value):
    with pytest.raises(ValueError):
        _device(0, 10, [1, 1], **{field: value})


def test_profile_requires_complete_tp_and_layer_coverage():
    d = _device(0, 10, [1, 1])
    with pytest.raises(ValueError, match="every PP/TP"):
        PPMemoryProfile(2, 1, 2, _scope(), (d,))
    with pytest.raises(ValueError, match="every PP/TP"):
        PPMemoryProfile(2, 1, 2, _scope(), (d, d))
    with pytest.raises(ValueError, match="all model layers"):
        PPMemoryProfile(3, 1, 1, _scope(), (d,))


def _write_traces(path, memory, tp_size=None):
    for r in range(memory.pp_size):
        start = r * memory.num_layers // memory.pp_size
        end = (r + 1) * memory.num_layers // memory.pp_size
        record = dict(pp_rank=r, tp_rank=0, step=6, start_layer=start,
                      end_layer=end, num_tokens=8, compute_ms=end - start,
                      send_transfer_ms=1 if r < memory.pp_size - 1 else None,
                      tp_size=memory.tp_size if tp_size is None else tp_size)
        (path / f"pp_stage_pp{r}_tp0.jsonl").write_text(json.dumps(record) + "\n")


def test_json_sidecar_roundtrip_and_serve_scope(tmp_path, monkeypatch):
    memory = _profile([_device(r, 16, [1] * 32) for r in range(2)])
    path = tmp_path / "pp_memory_profile.json"
    path.write_text(json.dumps(memory.to_dict()))
    assert PPMemoryProfile.from_file(path) == memory
    _write_traces(tmp_path, memory)
    plan = plan_from_trace_dir(tmp_path)
    assert plan.partitions == [16, 16]
    write_profile_result(plan, tmp_path)
    payload = json.loads((tmp_path / "pp_partition_plan.json").read_text())
    assert payload["memory_profile"]["serving_config"] == _scope()
    monkeypatch.setenv("VLLM_PP_COMM_BANDWIDTH_GBPS", "8")
    command = format_serve_command(plan)
    assert "--max-model-len 128" in command
    assert "--max-num-seqs 2" in command
    assert "--tensor-parallel-size 1" in command
    assert "VLLM_PP_COMM_BANDWIDTH_GBPS=8" in command
    _write_traces(tmp_path, memory, tp_size=2)
    with pytest.raises(ValueError, match="TP size"):
        plan_from_trace_dir(tmp_path)


def _engine_config(memory):
    scope = memory.serving_config
    return SimpleNamespace(
        model_config=SimpleNamespace(
            **{k: scope[k] for k in ("model", "revision", "dtype", "quantization",
                                    "max_model_len")},
            get_total_num_hidden_layers=lambda: memory.num_layers),
        cache_config=SimpleNamespace(cache_dtype=scope["kv_cache_dtype"],
                                     block_size=scope["block_size"],
                                     gpu_memory_utilization=0.9),
        scheduler_config=SimpleNamespace(
            max_num_seqs=scope["max_num_seqs"],
            max_num_batched_tokens=scope["max_num_batched_tokens"]),
        parallel_config=SimpleNamespace(tensor_parallel_size=memory.tp_size,
                                        pipeline_parallel_size=memory.pp_size),
    )


def test_live_engine_scope_checked_before_generate_and_shutdown_on_failure(tmp_path):
    memory = _profile([_device(r, 16, [1] * 32) for r in range(2)])
    config = _engine_config(memory)
    memory.validate_engine_config(config)
    config.scheduler_config.max_num_seqs = 10
    events = []
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(vllm_config=config,
            engine_core=SimpleNamespace(shutdown=lambda: events.append("shutdown"))),
        generate=lambda *a, **kw: events.append("generate"),
    )
    with pytest.raises(ValueError, match="max_num_seqs"):
        profile_pp_partition(dump_dir=tmp_path, llm_factory=lambda: llm,
                             memory_profile=memory)
    assert events == ["shutdown"]


def test_missing_memory_stops_live_run_before_factory(tmp_path):
    called = []
    with pytest.raises(ValueError, match="memory feasibility requires"):
        profile_pp_partition(dump_dir=tmp_path,
                             llm_factory=lambda: called.append(True))
    assert not called


def test_memory_constrained_dp_matches_exhaustive_search():
    rng = random.Random(911)
    for _ in range(25):
        n, p = 7, 3
        speeds = [rng.randint(1, 4) for _ in range(p)]
        devices = [_device(r, rng.randint(8, 25),
                           [rng.randint(1, 4) for _ in range(n)], [1] * n,
                           first_stage_bytes=2, last_stage_bytes=3)
                   for r in range(p)]
        memory = _profile(devices)
        feasible = []
        for size in range(1, p + 1):
            for cuts in itertools.combinations(range(1, n), size - 1):
                edges = (0, *cuts, n)
                parts = [b - a for a, b in zip(edges, edges[1:])]
                fits = True
                for r, (a, b) in enumerate(zip(edges, edges[1:])):
                    # Independent sum, not the production fits/estimate helper.
                    demand = sum(devices[r].layer_weights_bytes[a:b]) + b - a
                    demand += 2 if r == 0 else 0
                    demand += 3 if r == size - 1 else 0
                    fits &= demand <= devices[r].budget_bytes
                if fits:
                    cost = max(count * speeds[r] + (1 if r else 0)
                               + (1 if r < size - 1 else 0)
                               for r, count in enumerate(parts))
                    feasible.append((cost, parts))
        costs = _costs(n, speeds, comm=1)
        if not feasible:
            with pytest.raises(ValueError, match="No memory-feasible"):
                partition_layers(costs, num_layers=n, memory_profile=memory)
        else:
            plan = partition_layers(costs, num_layers=n, memory_profile=memory)
            assert plan.cost_ms == min(item[0] for item in feasible)
            assert (plan.cost_ms, plan.partitions) in feasible


def test_cli_requires_memory_and_honors_explicit_profile(tmp_path, capsys):
    from vllm.entrypoints.cli.pp_profile import PPProfileSubcommand
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    memory = _profile([_device(r, 16, [1] * 32) for r in range(2)])
    _write_traces(tmp_path, memory)
    path = tmp_path / "memory.json"
    path.write_text(json.dumps(memory.to_dict()))
    parser = FlexibleArgumentParser()
    PPProfileSubcommand().subparser_init(parser.add_subparsers(dest="subparser"))
    argv = ["pp-profile", "--skip-run", "--trace-dir", str(tmp_path)]
    with pytest.raises(ValueError, match="memory feasibility requires"):
        PPProfileSubcommand.cmd(parser.parse_args(argv))
    PPProfileSubcommand.cmd(parser.parse_args(argv + ["--memory-profile", str(path)]))
    assert "memory_feasibility_checked=True" in capsys.readouterr().out
    payload = json.loads((tmp_path / "pp_partition_plan.json").read_text())
    assert payload["partitions"] == [16, 16]
    assert len(payload["memory_usage"]) == 2


def test_live_memory_profile_is_saved_and_replay_is_equivalent(tmp_path):
    memory = _profile([_device(r, 16, [1] * 32) for r in range(2)])
    events = []
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(vllm_config=_engine_config(memory),
            engine_core=SimpleNamespace(shutdown=lambda: events.append("shutdown"))),
        generate=lambda *a, **kw: _write_traces(tmp_path, memory),
    )
    live = profile_pp_partition(dump_dir=tmp_path, llm_factory=lambda: llm,
                                memory_profile=memory)
    replay = profile_pp_partition(dump_dir=tmp_path, skip_run=True)
    assert events == ["shutdown"]
    assert live.to_dict() == replay.to_dict()
    assert live.partitions == [16, 16]


def test_bad_supplied_profile_cannot_be_ignored_by_opt_out(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": 1, "devices": []}))
    with pytest.raises(ValueError, match="invalid memory profile"):
        profile_pp_partition(dump_dir=tmp_path, memory_profile=path,
                             allow_unchecked_memory=True, skip_run=True)


@pytest.mark.parametrize("override", ["kv_cache_memory_bytes", "num_gpu_blocks_override"])
def test_unsupported_fixed_kv_allocation_cannot_bypass_bounds(override):
    memory = _profile([_device(r, 16, [1] * 32) for r in range(2)])
    config = _engine_config(memory)
    setattr(config.cache_config, override, 1000)
    with pytest.raises(ValueError, match="fixed KV allocation overrides"):
        memory.validate_engine_config(config)


def test_standalone_planner_loads_memory_profile(tmp_path, capsys):
    from vllm.distributed.pp_partition import main

    memory = _profile([_device(r, 16, [1] * 32) for r in range(2)])
    _write_traces(tmp_path, memory)
    path = tmp_path / "memory.json"
    path.write_text(json.dumps(memory.to_dict()))
    main([str(tmp_path), "--memory-profile", str(path)])
    output = capsys.readouterr().out
    assert "VLLM_PP_LAYER_PARTITION=16,16" in output
    assert "memory_feasibility_checked=True" in output
