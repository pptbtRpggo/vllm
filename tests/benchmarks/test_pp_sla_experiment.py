# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

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


def test_pp2_network_has_one_cross_group_link(experiment):
    native = experiment.two_group_network(None, 0, mode="native", pp_size=2)
    assert native == {"bandwidth_gbps": [[None], []]}
    network = experiment.two_group_network(
        100,
        0,
        mode="target_total",
        native_bandwidth_gbps=156,
        native_latency_ms=0.02,
        pp_size=2,
    )
    assert network["bandwidth_gbps"] == [[100], []]
    assert network["native_bandwidth_gbps"] == [[156], []]
    assert network["native_latency_ms"] == [[0.02], []]


def test_qwen_pp2_memory_uses_its_kv_geometry(experiment):
    rows = [
        dict(
            pp_rank=rank,
            tp_rank=0,
            layer_storage_bytes={
                str(i): 1000 for i in range(rank * 14, (rank + 1) * 14)
            },
            non_layer_storage_bytes=2000 if rank == 0 else 3000,
            total_bytes=64 * 1024**3,
            peak_reserved_bytes=14 * 256 * 1024**2 + 17000,
        )
        for rank in range(2)
    ]
    profile = experiment.memory_bounds(
        rows,
        dict(
            num_hidden_layers=28,
            hidden_size=3584,
            num_attention_heads=28,
            num_key_value_heads=4,
        ),
        dict(
            dtype="float16",
            kv_cache_dtype="auto",
            block_size=128,
            gpu_memory_utilization=0.85,
        ),
        blocks=1024,
    )
    assert profile["pp_size"] == 2
    assert len(profile["devices"]) == 2
    for device in profile["devices"]:
        assert device["layer_kv_bytes"] == [256 * 1024**2] * 28
        assert device["first_stage_bytes"] == 2000
        assert device["last_stage_bytes"] == 3000


def qwen_experiment(module, tmp_path, model_layers=28, **overrides):
    model, data = tmp_path / "model", tmp_path / "data"
    model.mkdir()
    data.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            dict(
                num_hidden_layers=model_layers,
                hidden_size=3584,
                num_attention_heads=28,
                num_key_value_heads=4,
            )
        )
    )
    for i, name in enumerate(("warmup", "profile", "evaluation")):
        (data / f"{name}.jsonl").write_text(
            json.dumps(
                dict(
                    id=i,
                    prompt=[1],
                    input_tokens=1,
                    output_tokens=1,
                    prompt_sha256=str(i),
                )
            )
            + "\n"
        )
    args = dict(
        model=str(model),
        data=str(data),
        output=str(tmp_path / "results"),
        port=0,
        kv_blocks=1024,
        pp_size=2,
        compute_scales="2,4",
        served_model_name="qwen2-7b",
        network_mode="native",
        cross_bandwidth_gbps=None,
        cross_extra_latency_ms=0,
    )
    args.update(overrides)
    return module.Experiment(SimpleNamespace(**args))


@pytest.mark.parametrize(
    "pp_size,layers,scales,parts,visible,budget",
    [
        (2, 28, "2,4", [14, 14], "0,1", 512),
        (2, 28, None, [14, 14], "0,1", 4096),
        (4, 48, None, [12, 12, 12, 12], "0,1,2,3", 2048),
    ],
)
def test_launch_uses_model_layers_and_pp_devices(
    experiment, tmp_path, monkeypatch, pp_size, layers, scales, parts, visible, budget
):
    exp = qwen_experiment(
        experiment,
        tmp_path,
        model_layers=layers,
        pp_size=pp_size,
        compute_scales=scales,
        max_num_batched_tokens=budget,
    )
    assert exp.uniform_partition == parts
    captured = {}

    class Process:
        pid = 1
        returncode = 0

        def poll(self):
            return None

        def wait(self, timeout):
            return 0

    def launch(command, **kwargs):
        captured.update(command=command, env=kwargs["env"])
        return Process()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Socket(Response):
        def setsockopt(self, *args):
            pass

        def bind(self, address):
            captured["address"] = address

    monkeypatch.setattr(experiment.subprocess, "Popen", launch)
    monkeypatch.setattr(
        experiment.urllib.request, "urlopen", lambda *a, **k: Response()
    )
    monkeypatch.setattr(experiment.os, "killpg", lambda *args: None)
    monkeypatch.setattr(experiment.time, "sleep", lambda *args: None)
    monkeypatch.setattr(experiment.socket, "socket", Socket)
    with exp.server("test", exp.uniform_partition):
        pass
    command = captured["command"]
    assert command[command.index("--pipeline-parallel-size") + 1] == str(pp_size)
    assert command[command.index("--max-num-batched-tokens") + 1] == str(budget)
    assert exp.serving["max_num_batched_tokens"] == budget
    assert captured["env"]["ASCEND_RT_VISIBLE_DEVICES"] == visible
    assert captured["env"]["VLLM_PP_HETERO"] == (
        "2.0,4.0" if pp_size == 2 else "1.0,1.0,2.0,4.0"
    )


@pytest.mark.parametrize("scales", ["2,4,4", "0.5,4", "nan,4"])
def test_invalid_pp2_scales_are_rejected(experiment, tmp_path, scales):
    with pytest.raises(ValueError, match="compute scale"):
        qwen_experiment(experiment, tmp_path, compute_scales=scales)


def test_profile_runs_pp2_trace_memory_and_dp(experiment, tmp_path, monkeypatch):
    from tests.distributed.pp_trace_fixtures import write_trace
    from tests.distributed.test_pp_layer_cost import layer_trace

    exp = qwen_experiment(experiment, tmp_path)
    exp.args.profile_concurrency = 8
    exp.args.profile_requests = 8
    exp.args.warmup_output_tokens = 16
    rows = layer_trace(repeats=(1, 3))
    for rank, records in rows.items():
        for record in records:
            start, end = rank * 14, (rank + 1) * 14
            record.update(
                start_layer=start,
                end_layer=end,
                layer_compute_ms={str(i): rank + 1 for i in range(start, end)},
            )
            record["compute_wall_ms"] = (
                sum(record["layer_compute_ms"].values())
                + record["non_layer_compute_ms"]
            )
    observations = [
        dict(
            pp_rank=rank,
            tp_rank=0,
            layer_storage_bytes={
                str(i): 1000 for i in range(rank * 14, (rank + 1) * 14)
            },
            non_layer_storage_bytes=2000 if rank == 0 else 3000,
            total_bytes=64 * 1024**3,
            peak_reserved_bytes=14 * 256 * 1024**2 + 17000,
        )
        for rank in range(2)
    ]

    @contextmanager
    def server(key, partition, trace):
        assert partition == [14, 14] and trace
        folder = exp.root / key / "traces"
        folder.mkdir(parents=True)
        for rank, records in rows.items():
            write_trace(folder / f"pp_stage_pp{rank}_tp0.jsonl", records)
        yield

    monkeypatch.setattr(exp, "server", server)
    monkeypatch.setattr(exp, "load", lambda *args: None)
    monkeypatch.setattr(
        exp,
        "rpc",
        lambda method, *args: (
            observations if method == "get_pp_memory_observation" else None
        ),
    )
    exp.profile()
    plans = json.loads((exp.root / "plans.json").read_text())
    assert plans["uniform"] == [14, 14]
    for objective in ("latency", "throughput"):
        assert len(plans[objective]) == 2 and sum(plans[objective]) == 28
        plan = json.loads((exp.root / "profile" / (objective + ".json")).read_text())
        assert plan["memory_feasibility_checked"]
