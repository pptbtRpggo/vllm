# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.tp_hetero import (
    TPCollectiveDelay,
    TPComputeDelay,
    TPHeteroConfig,
)
from vllm.pp_hetero_env import TP_ASCEND_WORKER, maybe_override_pp_worker


@pytest.fixture(autouse=True)
def isolate_hetero_environment(monkeypatch):
    import os

    for key in os.environ:
        if key.startswith(("VLLM_PP_", "VLLM_TP_")):
            monkeypatch.delenv(key)


def test_tp_worker_selection_and_no_silent_cuda(monkeypatch):
    monkeypatch.setenv("VLLM_TP_COMPUTE_SCALES", "1,2,4,1")
    cfg = SimpleNamespace(worker_cls="vllm_ascend.worker.worker.NPUWorker")
    maybe_override_pp_worker(cfg)
    assert cfg.worker_cls == TP_ASCEND_WORKER
    maybe_override_pp_worker(cfg)  # EngineCore may validate the config again.
    assert cfg.worker_cls == TP_ASCEND_WORKER
    with pytest.raises(ValueError, match="Ascend"):
        maybe_override_pp_worker(SimpleNamespace(worker_cls="auto"))
    monkeypatch.setenv("VLLM_PP_HETERO", "1,2")
    with pytest.raises(ValueError, match="cannot be combined"):
        maybe_override_pp_worker(cfg)


def test_tp_config_and_collective_cut_bytes(monkeypatch):
    monkeypatch.setenv("VLLM_TP_COMPUTE_SCALES", "1,2,4,1")
    monkeypatch.setenv("VLLM_TP_CROSS_GROUP_SIZE", "2")
    monkeypatch.setenv("VLLM_TP_CROSS_EXTRA_BANDWIDTH_GBPS", "25")
    config = TPHeteroConfig.from_env(4)
    assert config.scale(2) == 4
    assert config.cross_bytes("all_reduce", 1_000_000, 4) == 1_000_000
    assert config.cross_bytes("all_gather", 1_000_000, 4) == 2_000_000
    assert config.cross_bytes("reduce_scatter", 1_000_000, 4) == 500_000
    assert config.extra_ms("all_reduce", 1_000_000, 4) == pytest.approx(0.32)
    monkeypatch.setenv("VLLM_TP_CROSS_EXTRA_LATENCY_MS", "1")
    config = TPHeteroConfig.from_env(4)
    assert config.extra_ms("all_reduce", 1_000_000, 4) == pytest.approx(1.32)
    monkeypatch.setenv("VLLM_TP_CROSS_EXTRA_LATENCY_MS", "-1")
    with pytest.raises(ValueError, match="extra latency"):
        TPHeteroConfig.from_env(4)
    monkeypatch.setenv("VLLM_TP_CROSS_EXTRA_LATENCY_MS", "0")
    monkeypatch.setenv("VLLM_TP_COMPUTE_SCALES", "1,2")
    with pytest.raises(ValueError, match="one value"):
        TPHeteroConfig.from_env(4)
    monkeypatch.setenv("VLLM_TP_COMPUTE_SCALES", "0.5,1,1,1")
    with pytest.raises(ValueError, match="cannot speed up"):
        TPHeteroConfig.from_env(4)


def test_tp_latency_only_selects_worker_and_requires_cross_group(monkeypatch):
    monkeypatch.setenv("VLLM_TP_CROSS_EXTRA_LATENCY_MS", "1")
    monkeypatch.setenv("VLLM_TP_CROSS_GROUP_SIZE", "2")
    cfg = SimpleNamespace(worker_cls="vllm_ascend.worker.worker.NPUWorker")
    maybe_override_pp_worker(cfg)
    assert cfg.worker_cls == TP_ASCEND_WORKER
    config = TPHeteroConfig.from_env(4)
    assert config.extra_ms("all_reduce", 1_000_000, 4) == pytest.approx(1)
    monkeypatch.setenv("VLLM_TP_CROSS_GROUP_SIZE", "4")
    with pytest.raises(ValueError, match="must split"):
        TPHeteroConfig.from_env(4)


def test_compute_wait_precedes_collectives_and_does_not_scale_communication():
    from tests.distributed.test_pp_stream import DeferredStream

    stream = DeferredStream()
    starts = []

    def all_reduce(x):
        stream.enqueue(lambda: starts.append(stream.now))
        stream.work(0.003)
        return x

    comm = SimpleNamespace(
        all_reduce=all_reduce,
        all_gather=lambda x, dim: x,
        reduce_scatter=lambda x, dim: x,
    )
    collective = TPCollectiveDelay(comm, TPHeteroConfig((2, 1), 2, None), 2, stream)
    collective.install()

    class Layer(torch.nn.Module):
        def forward(self, x):
            stream.work(0.002)
            x = comm.all_reduce(x)
            stream.work(0.001)
            return comm.all_reduce(x)

    layer = Layer()
    model = SimpleNamespace(
        model=SimpleNamespace(
            start_layer=0, end_layer=1, layers=torch.nn.ModuleList([layer])
        )
    )
    compute = TPComputeDelay(model, 2, stream)
    compute.install()
    collective.compute_delay = compute
    compute.reset()
    layer(torch.empty(1))
    compute.validate()
    stream.drain()
    assert stream.waits == pytest.approx([2, 1])
    assert starts == pytest.approx([1004, 1009])
    assert stream.now == pytest.approx(1012)
    compute.uninstall()
    collective.uninstall()


def test_native_collective_path_submits_no_device_work():
    from tests.distributed.test_pp_stream import DeferredStream

    stream = DeferredStream()
    comm = SimpleNamespace(
        all_reduce=lambda x: x,
        all_gather=lambda x, dim: x,
        reduce_scatter=lambda x, dim: x,
    )
    delay = TPCollectiveDelay(comm, TPHeteroConfig((1, 1), 2, None), 2, stream)
    delay.install()
    comm.all_reduce(torch.empty(1))
    assert not stream.pending
    delay.uninstall()


def test_collective_delay_and_trace_are_ordered_after_native_communication():
    from tests.distributed.test_pp_stream import DeferredStream

    stream = DeferredStream()
    comm = SimpleNamespace(
        all_reduce=lambda x: (stream.work(0.001), x)[1],
        all_gather=lambda x, dim: x,
        reduce_scatter=lambda x, dim: x,
    )
    delay = TPCollectiveDelay(comm, TPHeteroConfig((1, 1), 1, 25), 2, stream)
    delay.install()
    delay.trace_enabled = True
    snapshots = []
    for size in (1_000_000, 2_000_000):
        delay.reset()
        comm.all_reduce(torch.empty(size, dtype=torch.uint8))
        snapshots.append(delay.stats)
    stream.drain()
    assert [s.extra_total_ms for s in snapshots] == pytest.approx([0.32, 0.64])
    assert [
        s.events[0][0].elapsed_time(s.events[0][1]) for s in snapshots
    ] == pytest.approx([1.32, 1.64])
    delay.uninstall()
