# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the real PP wrappers with CPU tensors and fake transport/timing.

The Ascend parent is stubbed; these tests do not validate HCCL or NPU kernels.
"""

import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm.distributed.pp_hetero import PPHeteroConfig
from vllm.distributed.pp_stage_trace import PPStageTracer
from vllm.sequence import IntermediateTensors


def _worker_module(backend, monkeypatch):
    if backend == "cuda":
        from vllm.v1.worker import gpu_worker

        return gpu_worker, gpu_worker.Worker

    utils = ModuleType("vllm_ascend.utils")
    utils.enable_sp = lambda: backend == "ascend_sp"
    parent = ModuleType("vllm_ascend.worker.worker")
    parent.NPUWorker = type("NPUWorker", (), {})
    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", utils)
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.worker", parent)
    path = Path(__file__).parents[2] / "vllm/v1/worker/pp_ascend_worker.py"
    spec = importlib.util.spec_from_file_location("test_ascend_pp_wrapper", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.PPAscendWorker


@pytest.mark.parametrize("backend", ["cuda", "ascend", "ascend_sp"])
@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("tracing", [False, True])
def test_worker_comm_delay_excludes_peer_wait(
    backend, rank, tracing, tmp_path, monkeypatch
):
    module, worker_cls = _worker_module(backend, monkeypatch)
    now = [0.0]
    sleeps = []
    calls = []
    payload = {"hidden_states": torch.empty(1_000_000, dtype=torch.uint8)}
    tp = SimpleNamespace(world_size=2)
    wire_bytes = 1_000_000 if backend == "ascend_sp" else 500_000
    baseline_ms = wire_bytes / 1_000_000  # 8 Gbit/s.

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    def recv(**kwargs):
        calls.append("recv")
        now[0] += (100 + baseline_ms) / 1000
        return payload

    def send(tensors, **kwargs):
        calls.append("send")
        assert tensors is payload
        assert kwargs["all_gather_group"] is (None if backend == "ascend_sp" else tp)
        now[0] += (300 + baseline_ms) / 1000

    group = SimpleNamespace(
        rank_in_group=rank, world_size=3, is_first_rank=rank == 0,
        is_last_rank=rank == 2, recv_tensor_dict=recv, send_tensor_dict=send,
    )
    monkeypatch.setattr(module, "get_pp_group", lambda: group)
    monkeypatch.setattr(module, "get_tp_group", lambda: tp)
    monkeypatch.setattr("vllm.distributed.pp_hetero.time.perf_counter", lambda: now[0])
    monkeypatch.setattr("vllm.distributed.pp_hetero.time.sleep", sleep)

    def forward(scheduler_output, intermediate):
        calls.append("compute")
        if rank:
            assert intermediate.tensors is payload
        else:
            assert intermediate is None
        now[0] += 0.010
        return None if rank == 2 else IntermediateTensors(payload)

    tracer = (
        PPStageTracer(str(tmp_path), rank, 3, torch.device("cpu"), tp_size=2)
        if tracing else None
    )
    worker = object.__new__(worker_cls)
    worker.device = torch.device("cpu")
    worker._pp_stage_tracer = tracer
    worker._pp_hetero = PPHeteroConfig(
        comm_scales=(4, 7), comm_bandwidth_gbps=(8, 8)
    )
    worker.model_runner = SimpleNamespace(
        execute_model=forward,
        model=SimpleNamespace(start_layer=rank * 16, end_layer=(rank + 1) * 16),
    )
    worker.annotate_profile = lambda _: nullcontext()
    worker.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=3),
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=False)
        ),
    )
    scheduler = SimpleNamespace(
        total_num_scheduled_tokens=1, num_scheduled_tokens={"r": 1},
        scheduled_new_reqs=[],
    )
    try:
        assert worker.execute_model(scheduler) is None
    finally:
        if tracer:
            tracer.close()

    expected_calls = (["recv"] if rank else []) + ["compute"]
    expected_calls += ["send"] if rank < 2 else []
    assert calls == expected_calls
    expected_delays = []
    if rank:
        expected_delays.append(baseline_ms * ((4, 7)[rank - 1] - 1) / 1000)
    if rank < 2:
        expected_delays.append(baseline_ms * ((4, 7)[rank] - 1) / 1000)
    assert sleeps == pytest.approx(expected_delays)
    if tracer:
        record = json.loads(Path(tracer.path).read_text())
        if rank:
            assert record["recv_bytes"] == wire_bytes
            assert record["recv_ms"] == pytest.approx(
                100 + baseline_ms * (4, 7)[rank - 1]
            )
        if rank < 2:
            assert record["send_bytes"] == wire_bytes
            assert record["send_ms"] == pytest.approx(300 + baseline_ms * (4, 7)[rank])
            assert record["send_transfer_ms"] == pytest.approx(
                baseline_ms * (4, 7)[rank]
            )
            assert record["comm_scale"] == (4, 7)[rank]
        else:
            assert record["send_transfer_ms"] is None
