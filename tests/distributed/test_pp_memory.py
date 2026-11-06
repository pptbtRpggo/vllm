# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import json
import random
from dataclasses import replace

import pytest

from tests.distributed.pp_planner_fixtures import (
    _costs,
    _device,
    _engine_config,
    _profile,
    _scope,
)
from tests.distributed.pp_trace_fixtures import write_fit_profile, write_trace
from vllm.distributed.pp_memory import PPMemoryProfile
from vllm.distributed.pp_partition import (
    partition_layers,
    plan_from_trace_dir,
)


@pytest.mark.parametrize(
    "objective,overlap",
    [("latency", False), ("throughput", False), ("throughput", True)],
)
def test_memory_prevents_dropping_rank_or_overfilling_fast_rank(objective, overlap):
    memory = _profile([_device(r, 16, [1] * 32) for r in range(2)])
    plan = partition_layers(
        _costs(speeds=(1, 2)),
        objective=objective,
        overlap_comm=overlap,
        memory_profile=memory,
    )
    assert plan.partitions == [16, 16]
    result = plan.to_dict()
    assert result["memory_feasibility_checked"] is True
    assert result["memory_feasibility_basis"] == "supplied_memory_bounds"
    assert [r["headroom_bytes"] for r in result["memory_usage"]] == [0, 0]


def test_kv_and_all_fixed_reserves_are_charged():
    devices = [
        _device(
            r,
            25,
            [2] * 8,
            [1] * 8,
            runtime_bytes=1,
            activation_bytes=2,
            workspace_bytes=3,
            communication_bytes=1,
            graph_bytes=2,
            safety_margin_bytes=4,
        )
        for r in range(2)
    ]
    memory = _profile(devices)
    plan = partition_layers(_costs(8), num_layers=8, memory_profile=memory)
    assert plan.partitions == [4, 4]
    assert all(row["required_bytes"] == 25 for row in memory.plan_usage([4, 4]))
    with pytest.raises(ValueError, match="No memory-feasible partition"):
        partition_layers(
            _costs(8),
            num_layers=8,
            memory_profile=_profile([replace(d, budget_bytes=24) for d in devices]),
        )


def test_head_follows_final_selected_rank_and_single_rank_has_both_endpoints():
    memory = _profile(
        [
            _device(0, 10, [1] * 8, first_stage_bytes=2, last_stage_bytes=3),
            _device(1, 20, [1] * 8, last_stage_bytes=3),
            _device(2, 20, [1] * 8, last_stage_bytes=3),
        ]
    )
    plan = partition_layers(
        _costs(8, (1, 1, 1)),
        num_layers=8,
        objective="latency",
        max_pp_size=2,
        memory_profile=memory,
    )
    assert plan.pp_size == 2  # 8 + embedding 2 + head 3 cannot fit on rank 0.
    usage = plan.to_dict()["memory_usage"]
    assert usage[0]["endpoint_bytes"] == 2
    assert usage[1]["endpoint_bytes"] == 3
    assert all(row["pp_rank"] < 2 for row in usage)
    with pytest.raises(ValueError, match="No memory-feasible partition"):
        partition_layers(
            _costs(8, (1, 1, 1)), num_layers=8, max_pp_size=1, memory_profile=memory
        )


def test_nonuniform_layer_memory_uses_actual_interval():
    memory = _profile([_device(0, 10, [8, 1, 1, 1]), _device(1, 10, [8, 1, 1, 1])])
    plan = partition_layers(_costs(4, (1, 10), comm=0), memory_profile=memory)
    assert plan.partitions == [3, 1]
    assert memory.plan_usage([3, 1])[0]["weights_bytes"] == 10


def test_all_tp_devices_must_fit_not_just_tp_zero():
    memory = _profile(
        [
            _device(0, 100, [1] * 8, tp=0),
            _device(0, 3, [1] * 8, tp=1),
            _device(1, 100, [1] * 8, tp=0),
            _device(1, 5, [1] * 8, tp=1),
        ],
        tp=2,
    )
    plan = partition_layers(_costs(8, (1, 10)), memory_profile=memory)
    assert plan.partitions == [3, 5]
    assert len(plan.to_dict()["memory_usage"]) == 4


def test_missing_memory_is_rejected_unless_explicitly_unchecked():
    with pytest.raises(ValueError, match="memory feasibility requires"):
        partition_layers(_costs())
    plan = partition_layers(_costs(), allow_unchecked_memory=True)
    assert plan.to_dict()["memory_feasibility_checked"] is False
    assert plan.to_dict()["memory_feasibility_basis"] == "unchecked"


@pytest.mark.parametrize(
    "field,value",
    [
        ("budget_bytes", 0),
        ("budget_bytes", -1),
        ("budget_bytes", 1.5),
        ("runtime_bytes", True),
        ("graph_bytes", float("nan")),
        ("layer_weights_bytes", (1, -1)),
        ("layer_kv_bytes", (1,)),
    ],
)
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
    records_by_rank = {}
    for r in range(memory.pp_size):
        start = r * memory.num_layers // memory.pp_size
        end = (r + 1) * memory.num_layers // memory.pp_size
        record = dict(
            pp_rank=r,
            pp_size=memory.pp_size,
            num_reqs=1,
            batch_shape=[dict(query_tokens=8, context_tokens=16, prompt_tokens=0)],
            tp_rank=0,
            step=6,
            start_layer=start,
            end_layer=end,
            num_tokens=8,
            compute_ms=end - start,
            compute_wall_ms=end - start,
            send_service_ms=1 if r < memory.pp_size - 1 else None,
            send_ms=1 if r < memory.pp_size - 1 else None,
            send_transfer_ms=1 if r < memory.pp_size - 1 else None,
            tp_size=memory.tp_size if tp_size is None else tp_size,
        )
        records_by_rank[r] = [record]
        write_trace(path / f"pp_stage_pp{r}_tp0.jsonl", [record])
    write_fit_profile(path / "fit", records_by_rank)


def test_json_sidecar_roundtrip_and_serve_scope(tmp_path, monkeypatch):
    memory = _profile([_device(r, 16, [1] * 32) for r in range(2)])
    path = tmp_path / "pp_memory_profile.json"
    path.write_text(json.dumps(memory.to_dict()))
    assert PPMemoryProfile.from_file(path) == memory
    _write_traces(tmp_path, memory)
    plan = plan_from_trace_dir(tmp_path, fit_trace_dirs=[tmp_path / "fit"])
    assert plan.partitions == [16, 16]
    plan_path = tmp_path / "pp_partition_plan.json"
    plan_path.write_text(json.dumps(plan.to_dict()))
    payload = json.loads(plan_path.read_text())
    assert payload["memory_profile"]["serving_config"] == _scope()
    config = _engine_config(memory)
    memory.validate_engine_config(config)
    config.scheduler_config.max_num_seqs += 1
    with pytest.raises(ValueError, match="max_num_seqs"):
        memory.validate_engine_config(config)
    _write_traces(tmp_path, memory, tp_size=2)
    with pytest.raises(ValueError, match="TP size"):
        plan_from_trace_dir(tmp_path, fit_trace_dirs=[tmp_path / "fit"])


def test_memory_constrained_dp_matches_exhaustive_search():
    rng = random.Random(911)
    for _ in range(25):
        n, p = 7, 3
        speeds = [rng.randint(1, 4) for _ in range(p)]
        devices = [
            _device(
                r,
                rng.randint(8, 25),
                [rng.randint(1, 4) for _ in range(n)],
                [1] * n,
                first_stage_bytes=2,
                last_stage_bytes=3,
            )
            for r in range(p)
        ]
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
                    cost = max(
                        count * speeds[r] + (1 if r else 0) + (1 if r < size - 1 else 0)
                        for r, count in enumerate(parts)
                    )
                    feasible.append((cost, parts))
        costs = _costs(n, speeds, comm=1)
        if not feasible:
            with pytest.raises(ValueError, match="No memory-feasible"):
                partition_layers(costs, num_layers=n, memory_profile=memory)
        else:
            plan = partition_layers(costs, num_layers=n, memory_profile=memory)
            assert plan.cost_ms == min(item[0] for item in feasible)
            assert (plan.cost_ms, plan.partitions) in feasible


def test_bad_supplied_profile_cannot_be_ignored_by_opt_out(tmp_path):
    # The offline core reads traces before memory; reach the supplied-memory
    # guard with a complete trace fixture instead of failing on absent traces.
    memory = _profile([_device(r, 16, [1] * 32) for r in range(2)])
    _write_traces(tmp_path, memory)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": 1, "devices": []}))
    with pytest.raises(ValueError, match="invalid memory profile"):
        plan_from_trace_dir(
            tmp_path,
            fit_trace_dirs=[tmp_path / "fit"],
            memory_profile=path,
            allow_unchecked_memory=True,
        )


@pytest.mark.parametrize(
    "override", ["kv_cache_memory_bytes", "num_gpu_blocks_override"]
)
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
    main(
        [
            str(tmp_path),
            "--fit-trace-dir",
            str(tmp_path / "fit"),
            "--memory-profile",
            str(path),
        ]
    )
    output = capsys.readouterr().out
    assert "VLLM_PP_LAYER_PARTITION=16,16" in output
    assert "memory_feasibility_checked=True" in output


def _fixed_blocks_profile(*, blocks=16, **scope_changes):
    memory = _profile([_device(r, 100, [1] * 8, kv=[2] * 8) for r in range(2)])
    scope = memory.serving_config | scope_changes
    policy = dict(
        mode="fixed_blocks",
        num_gpu_blocks=blocks,
        block_size=scope["block_size"],
        kv_cache_dtype=scope["kv_cache_dtype"],
    )
    return replace(memory, serving_config=scope, cache_policy=policy)


def test_v1_serialization_key_set_and_override_guard_are_unchanged(tmp_path):
    memory = _profile([_device(r, 100, [1] * 8) for r in range(2)])
    payload = memory.to_dict()
    assert set(payload) == {
        "version",
        "num_layers",
        "pp_size",
        "tp_size",
        "serving_config",
        "devices",
    }
    assert payload["version"] == 1
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(payload))
    restored = PPMemoryProfile.from_file(path)
    assert restored == memory
    config = _engine_config(restored)
    restored.validate_engine_config(config)
    config.cache_config.num_gpu_blocks_override = 16
    with pytest.raises(ValueError, match="fixed KV allocation overrides"):
        restored.validate_engine_config(config)


def test_v2_fixed_blocks_roundtrip_preserves_policy_and_memory_costs(tmp_path):
    memory = _fixed_blocks_profile()
    payload = memory.to_dict()
    assert payload["version"] == 2
    assert payload["cache_policy"] == dict(
        mode="fixed_blocks", num_gpu_blocks=16, block_size=16, kv_cache_dtype="auto"
    )
    path = tmp_path / "v2.json"
    path.write_text(json.dumps(payload))
    restored = PPMemoryProfile.from_file(path)
    assert restored == memory
    assert restored.plan_usage([4, 4]) == memory.plan_usage([4, 4])
    assert restored.plan_usage([4, 4])[0]["kv_cache_bytes"] == 8
    # Device remapping retains the allocation contract and role-dependent costs.
    selected = restored.select_devices((1, 0))
    assert selected.cache_policy == memory.cache_policy
    assert selected.to_dict()["version"] == 2


@pytest.mark.parametrize("policy", [None, {}, {"mode": "fixed_blocks"}])
def test_v1_rejects_any_serialized_cache_policy_even_null(tmp_path, policy):
    memory = _profile([_device(r, 100, [1] * 8) for r in range(2)])
    payload = memory.to_dict() | {"cache_policy": policy}
    path = tmp_path / "invalid-v1.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="cache_policy"):
        PPMemoryProfile.from_file(path)


@pytest.mark.parametrize("policy", [None, {}, {"mode": "fixed_bytes", "bytes": 4096}])
def test_v2_requires_an_explicit_supported_policy(tmp_path, policy):
    memory = _profile([_device(r, 100, [1] * 8) for r in range(2)])
    payload = memory.to_dict() | {"version": 2}
    if policy is not None:
        payload["cache_policy"] = policy
    path = tmp_path / "invalid-v2.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="cache_policy"):
        PPMemoryProfile.from_file(path)


def test_v2_rejects_null_policy_instead_of_reinterpreting_as_v1(tmp_path):
    memory = _profile([_device(r, 100, [1] * 8) for r in range(2)])
    path = tmp_path / "null-v2.json"
    path.write_text(json.dumps(memory.to_dict() | {"version": 2, "cache_policy": None}))
    with pytest.raises(ValueError, match="cache_policy"):
        PPMemoryProfile.from_file(path)


@pytest.mark.parametrize(
    "change",
    [
        {"mode": "fixed_bytes"},
        {"num_gpu_blocks": 0},
        {"num_gpu_blocks": -1},
        {"num_gpu_blocks": True},
        {"num_gpu_blocks": 16.0},
        {"block_size": 0},
        {"block_size": True},
        {"block_size": 32},
        {"kv_cache_dtype": ""},
        {"kv_cache_dtype": "float16"},
        {"bytes": 4096},
    ],
)
def test_v2_policy_is_exact_positive_and_matches_serving_scope(change):
    memory = _fixed_blocks_profile()
    with pytest.raises(ValueError, match="cache_policy"):
        replace(memory, cache_policy=memory.cache_policy | change)


@pytest.mark.parametrize(
    "missing", ["mode", "num_gpu_blocks", "block_size", "kv_cache_dtype"]
)
def test_v2_rejects_missing_policy_fields(missing):
    memory = _fixed_blocks_profile()
    policy = dict(memory.cache_policy)
    policy.pop(missing)
    with pytest.raises(ValueError, match="cache_policy"):
        replace(memory, cache_policy=policy)


@pytest.mark.parametrize("version", [0, 3, True, "2"])
def test_unknown_memory_protocol_version_is_rejected(tmp_path, version):
    path = tmp_path / "unknown-version.json"
    path.write_text(
        json.dumps(_fixed_blocks_profile().to_dict() | {"version": version})
    )
    with pytest.raises(ValueError, match="version"):
        PPMemoryProfile.from_file(path)


def test_fixed_blocks_capacity_rounds_each_request_before_multiplying():
    # ceil(17 / 16) * 2 = 4 blocks; ceil((17 * 2) / 16) = 3 is insufficient.
    with pytest.raises(ValueError, match="capacity"):
        _fixed_blocks_profile(blocks=3, max_model_len=17, max_num_seqs=2)
    memory = _fixed_blocks_profile(blocks=4, max_model_len=17, max_num_seqs=2)
    assert memory.cache_policy["num_gpu_blocks"] == 4
    with pytest.raises(ValueError, match="capacity"):
        _fixed_blocks_profile(blocks=15)
    assert _fixed_blocks_profile(blocks=16).cache_policy["num_gpu_blocks"] == 16


def test_v2_requires_exact_live_fixed_blocks_and_rejects_fixed_bytes():
    memory = _fixed_blocks_profile()
    config = _engine_config(memory)
    for count in (None, 15, 17, True):
        config.cache_config.num_gpu_blocks_override = count
        with pytest.raises(ValueError, match="num_gpu_blocks_override"):
            memory.validate_engine_config(config)
    config.cache_config.num_gpu_blocks_override = 16
    memory.validate_engine_config(config)
    config.cache_config.kv_cache_memory_bytes = 4096
    with pytest.raises(ValueError, match="kv_cache_memory_bytes"):
        memory.validate_engine_config(config)


@pytest.mark.parametrize("key,value", [("block_size", 32), ("cache_dtype", "float16")])
def test_v2_live_block_geometry_and_dtype_still_match_scope(key, value):
    memory = _fixed_blocks_profile()
    config = _engine_config(memory)
    config.cache_config.num_gpu_blocks_override = 16
    setattr(config.cache_config, key, value)
    with pytest.raises(ValueError, match="serving config"):
        memory.validate_engine_config(config)


def test_v2_keeps_complete_tp_coverage_requirement():
    memory = _fixed_blocks_profile()
    with pytest.raises(ValueError, match="every PP/TP"):
        replace(memory, devices=memory.devices[:1])
