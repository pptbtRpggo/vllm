# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tests.distributed.test_pp_stream import DeferredStream
from vllm.distributed.tp_hetero import TPCollectiveDelay, TPHeteroConfig


@pytest.mark.parametrize("tracing", [False, True])
def test_tp_worker_has_no_execute_sync_and_keeps_pending_batches_separate(
    tracing, monkeypatch
):
    stream = DeferredStream(monkeypatch)
    rows = []
    deferred = []
    writer = SimpleNamespace(
        check=lambda: None,
        submit_ready=lambda event, resolve: deferred.append((event, resolve)),
    )

    def all_reduce(x):
        stream.work(0.003)
        return x

    comm = SimpleNamespace(
        all_reduce=all_reduce,
        all_gather=lambda x, dim: x,
        reduce_scatter=lambda x, dim: x,
    )

    class Layer(torch.nn.Module):
        def forward(self, duration):
            stream.work(duration)
            comm.all_reduce(torch.empty(1_000_000, dtype=torch.uint8))
            stream.work(duration)
            comm.all_reduce(torch.empty(1_000_000, dtype=torch.uint8))
            return 42

    model = SimpleNamespace(
        model=SimpleNamespace(
            start_layer=0, end_layer=1, layers=torch.nn.ModuleList([Layer()])
        )
    )

    class Parent:
        def execute_model(self, scheduler):
            return self.model_runner.model.model.layers[0](
                scheduler.total_num_scheduled_tokens / 1000
            )

    parent = ModuleType("vllm_ascend.worker.worker")
    parent.NPUWorker = Parent
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.worker", parent)
    path = Path(__file__).parents[2] / "vllm/v1/worker/tp_ascend_worker.py"
    spec = importlib.util.spec_from_file_location("test_tp_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "sync_torch_device",
        lambda _: pytest.fail("execute must not synchronize"),
    )
    monkeypatch.setattr(
        module, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0)
    )
    config = TPHeteroConfig((2, 1), 1, 25)
    worker = object.__new__(module.TPAscendWorker)
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True)
    )
    worker._tp_stream_delay = stream
    worker._tp_trace_writer = writer if tracing else None
    worker._tp_trace_remaining = 1000
    worker._tp_hetero = config
    worker._tp_compute_model = None
    worker._tp_compute_delay = None
    worker._tp_logged = False
    worker._tp_collectives = TPCollectiveDelay(comm, config, 2, stream)
    worker._tp_collectives.install()
    worker.model_runner = SimpleNamespace(model=model)
    try:
        for tokens in (2, 5):
            assert (
                worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=tokens))
                == 42
            )
        assert not rows
        stream.drain()
        if tracing:
            for event, resolve in deferred:
                assert event.query()
                rows.append(resolve())
            assert [r["scheduled_tokens"] for r in rows] == [2, 5]
            assert [r["compute_base_ms"] for r in rows] == pytest.approx([4, 10])
            assert [r["compute_extra_requested_ms"] for r in rows] == pytest.approx(
                [4, 10]
            )
            assert all(
                r["collective_extra_requested_ms"] == pytest.approx(0.64) for r in rows
            )
            assert all(r["collective_counts"]["all_reduce"] == 2 for r in rows)
    finally:
        worker._tp_compute_delay.uninstall()
        worker._tp_collectives.uninstall()
