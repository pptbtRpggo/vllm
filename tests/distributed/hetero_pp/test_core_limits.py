# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise core-limit validation and the real worker hook without Ascend."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3] / "vllm/distributed/hetero_pp"


def load(name, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        f"core_limit_{name}", ROOT / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_native_worker_binds_stage_quota_before_native_distributed_init(monkeypatch):
    import json

    from vllm.distributed.hetero_pp import native_worker

    stages = [
        {"node_ip": ip, "soc": "910B", "layers": 2}
        for ip in ["10.0.0.1", "10.0.0.1", "10.0.0.2", "10.0.0.2"]
    ]
    stages[3]["npu_core_fraction"] = "1/4"
    monkeypatch.setenv(
        "VLLM_HETERO_PP_PLAN",
        json.dumps({"stages": stages, "validation_mode": "910b_only"}),
    )
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "2,2,2,2")
    monkeypatch.setenv("VLLM_HOST_IP", "10.0.0.2")
    calls: list[str | tuple[str, str]] = []

    def apply_limit(soc, fraction):
        calls.append((soc, fraction))
        return {"fraction": fraction}

    monkeypatch.setattr(
        native_worker,
        "apply_npu_core_limit",
        apply_limit,
    )
    import torch

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(get_device_name=lambda device: "Ascend910B4-1"),
        raising=False,
    )

    class AscendBase:
        def _init_worker_distributed_environment(self):
            calls.append("native HCCL")

    class Worker(native_worker.NativeWorkerMixin, AscendBase):
        def _check_native_startup(self):
            calls.append("cross-rank configuration check")

    worker = Worker()
    worker.rank, worker.local_rank = 3, 1
    worker.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            nnodes=2,
            world_size=4,
            pipeline_parallel_size=4,
            tensor_parallel_size=1,
            data_parallel_size=1,
        ),
        model_config=SimpleNamespace(get_total_num_hidden_layers=lambda: 8),
    )
    worker._init_worker_distributed_environment()
    assert calls == [("910B", "1/4"), "native HCCL", "cross-rank configuration check"]
    assert worker._hetero_npu_core_limit == {"fraction": "1/4"}


def test_native_worker_preserves_serving_with_transformer_partition():
    from vllm.distributed.hetero_pp.native_worker import NativeWorkerMixin

    class AscendBase:
        def load_model(self):
            transformer = SimpleNamespace(start_layer=2, end_layer=4)
            self.model_runner = SimpleNamespace(
                model=SimpleNamespace(transformer=transformer)
            )

    class Worker(NativeWorkerMixin, AscendBase):
        pass

    worker = Worker()
    worker.rank = 1
    worker._native_plan = SimpleNamespace(layer_ranges=[(0, 2), (2, 4)])
    worker.load_model()
    assert worker._verified_native_layers() == (2, 4)
    worker.model_runner.model.transformer.start_layer = 0
    with pytest.raises(ValueError, match="rank 1 loaded layers"):
        worker._verified_native_layers()


def test_native_worker_observes_transformers_module_list_partition():
    from torch import nn

    from vllm.distributed.hetero_pp.native_worker import NativeWorkerMixin
    from vllm.model_executor.models.utils import PPMissingLayer

    model = nn.Module()
    model.layers = nn.ModuleList(
        [
            PPMissingLayer(),
            nn.Linear(1, 1),
            nn.Linear(1, 1),
            PPMissingLayer(),
        ]
    )
    worker = NativeWorkerMixin()
    worker.rank = 1
    worker.model_runner = SimpleNamespace(model=model)
    worker._native_plan = SimpleNamespace(layer_ranges=[(0, 1), (1, 3), (3, 4)])
    assert worker._verified_native_layers() == (1, 3)


def test_native_ack_finishes_native_receive_dependencies_before_done(monkeypatch):
    import torch

    from vllm.distributed.hetero_pp import native_worker, profiling

    calls = []
    payload = {"hidden_states": torch.zeros(1)}
    group = SimpleNamespace(cpu_group=object(), rank_in_group=2)
    link = native_worker._NativeLink(group, lambda: calls.append("synchronize"))
    monkeypatch.setattr(link, "validate_payload", lambda values: None)
    step = SimpleNamespace(
        metadata={"step": 0, "trace_session": "profile", "batch_id": "batch"}
    )
    timed = profiling._AckTraceLink(step, link, 1)
    work = SimpleNamespace(wait=lambda: calls.append("wait"))
    monkeypatch.setattr(
        timed.control, "send", lambda message, peer, identity: calls.append(message)
    )

    def receive():
        calls.append("native receive")
        return payload, [work], [lambda: calls.append("postprocess")]

    values, handles, postprocess = link.receive(timed, receive)
    assert values is payload
    assert handles == postprocess == []
    assert calls == [
        "READY",
        "native receive",
        "wait",
        "postprocess",
        "synchronize",
        "DONE",
    ]
    assert step.recv_ack["identity"]["payload_bytes"] == 4


def test_topology_round_trips_limits_and_rejects_bad_direct_input(monkeypatch):
    config = load("config", monkeypatch)
    stages = [
        {"node_ip": ip, "soc": soc, "layers": 2}
        for ip, soc in [("10.0.0.1", "910B")] * 2 + [("10.0.0.2", "310P")] * 2
    ]
    stages[2]["npu_core_fraction"] = "0.5"
    stages[3]["npu_core_fraction"] = "1/4"
    plan = config.PipelinePlan.from_dict({"stages": stages}, 8)
    assert [stage.npu_core_fraction for stage in plan.stages] == [
        "1",
        "1",
        "1/2",
        "1/4",
    ]
    stages[3]["npu_core_fraction"] = True
    with pytest.raises(ValueError, match="npu_core_fraction"):
        config.PipelinePlan.from_dict({"stages": stages}, 8)


@pytest.mark.parametrize(
    "soc, fraction, capacities, expected",
    [
        ("910B", "1/2", {0: 20, 1: 40}, {0: 10, 1: 20}),
        ("310P", "1/4", {0: 8}, {0: 2}),
        ("910B", "1", {}, {}),
    ],
)
def test_worker_sets_limits_before_gloo_and_reads_back(
    monkeypatch, soc, fraction, capacities, expected
):
    module = load("worker", monkeypatch)
    applied = dict(capacities)
    calls: list[str | tuple[int, int]] = []

    def set_limit(device, resource_type, count):
        assert device == 3
        calls.append((resource_type, count))
        applied[resource_type] = count
        return 0

    monkeypatch.setitem(
        sys.modules,
        "acl",
        SimpleNamespace(
            rt=SimpleNamespace(
                get_device=lambda: (3, 0),
                get_device_res_limit=lambda device, resource: (applied[resource], 0),
                set_device_res_limit=set_limit,
            )
        ),
    )
    ps = SimpleNamespace(
        init_distributed_environment=lambda *a, **kw: calls.append("gloo"),
        ensure_model_parallel_initialized=lambda *a, **kw: None,
    )
    deps = {
        "vllm.distributed": SimpleNamespace(parallel_state=ps),
        "vllm.distributed.hetero_pp.groups": SimpleNamespace(
            install_group_factory=lambda: None
        ),
        "vllm_ascend.distributed.parallel_state": SimpleNamespace(
            init_ascend_model_parallel=lambda config: None
        ),
    }
    for name, value in deps.items():
        monkeypatch.setitem(sys.modules, name, value)

    worker = module.StagingWorkerMixin()
    worker.rank = worker.local_rank = 0
    worker.distributed_init_method = "tcp://localhost:12345"
    worker.parallel_config = SimpleNamespace(world_size=4, pipeline_parallel_size=4)
    worker._hetero_timeout_s = 10
    worker._hetero_stage_soc = soc
    worker._hetero_npu_core_fraction = fraction
    worker._init_worker_distributed_environment()
    assert calls == [*expected.items(), "gloo"]
    if expected:
        assert worker._hetero_npu_core_limit["fraction"] == fraction
        assert list(worker._hetero_npu_core_limit["effective"].values()) == list(
            expected.values()
        )


@pytest.mark.parametrize("error", ["set", "readback"])
def test_worker_requires_success_and_matching_core_limit(monkeypatch, error):
    module = load("worker", monkeypatch)
    monkeypatch.setitem(
        sys.modules,
        "acl",
        SimpleNamespace(
            rt=SimpleNamespace(
                get_device=lambda: (0, 0),
                get_device_res_limit=lambda device, resource: (8, 0),
                set_device_res_limit=lambda *args: 507000 if error == "set" else 0,
            )
        ),
    )
    with pytest.raises(RuntimeError, match="core limit"):
        module.apply_npu_core_limit("310P", "1/4")
