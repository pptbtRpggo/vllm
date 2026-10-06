# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def experiment(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "pp_sla_experiment", directory / "pp_sla_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def observations():
    return [
        dict(
            pp_rank=rank,
            tp_rank=0,
            layer_storage_bytes={
                str(i): 1000 for i in range(rank * 12, (rank + 1) * 12)
            },
            non_layer_storage_bytes=2000 if rank == 0 else 3000 if rank == 3 else 0,
            total_bytes=64 * 1024**3,
            peak_reserved_bytes=12 * 512 * 1024**2 + 12000,
        )
        for rank in range(4)
    ]


def build(module, rows):
    return module.memory_bounds(
        rows,
        dict(
            num_hidden_layers=48,
            hidden_size=8192,
            num_attention_heads=64,
            num_key_value_heads=8,
        ),
        dict(
            dtype="float16",
            kv_cache_dtype="auto",
            block_size=128,
            gpu_memory_utilization=0.85,
        ),
        blocks=1024,
    )


def test_fixed_kv_pool_not_counted_twice(experiment):
    profile = build(experiment, observations())
    for device in profile["devices"]:
        assert device["layer_kv_bytes"] == [512 * 1024**2] * 48
        assert device["runtime_bytes"] == 6 * 1024**3
        assert device["first_stage_bytes"] == 2000
        assert device["last_stage_bytes"] == 3000
        assert device["layer_weights_bytes"] == [1000] * 48


def test_measured_large_residual_increases_reserve(experiment):
    rows = observations()
    rows[2]["peak_reserved_bytes"] += 10 * 1024**3
    profile = build(experiment, rows)
    assert all(d["runtime_bytes"] == 10 * 1024**3 for d in profile["devices"])


def test_memory_observations_must_cover_all_layers(experiment):
    rows = observations()
    del rows[0]["layer_storage_bytes"]["0"]
    with pytest.raises(ValueError, match="incomplete"):
        build(experiment, rows)


def test_two_group_network_sets_only_cross_group_extra_delay(experiment):
    matrix = experiment.two_group_network(25, 1)
    assert matrix["bandwidth_gbps"] == [[None, 25, 25], [25, 25], [None], []]
    assert matrix["latency_ms"] == [[0, 1, 1], [1, 1], [0], []]
    with pytest.raises(ValueError, match="positive"):
        experiment.two_group_network(0, 1)
    assert experiment.two_group_network(25, 0)["latency_ms"] == [
        [0, 0, 0],
        [0, 0],
        [0],
        [],
    ]
    with pytest.raises(ValueError, match="nonnegative"):
        experiment.two_group_network(25, -1)
