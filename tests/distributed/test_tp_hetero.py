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


def test_collective_delay_only_during_forward(monkeypatch):
    now = [0.0]
    sleeps = []
    monkeypatch.setattr("vllm.distributed.tp_hetero.time.perf_counter", lambda: now[0])

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr("vllm.distributed.tp_hetero.time.sleep", sleep)

    class StreamDelay:
        def enqueue(self, callback):
            callback()

    class Comm:
        def all_reduce(self, x):
            now[0] += 0.001
            return x

        def all_gather(self, x, dim):
            now[0] += 0.001
            return x

        def reduce_scatter(self, x, dim):
            now[0] += 0.001
            return x

    comm = Comm()
    original = comm.all_reduce
    delay = TPCollectiveDelay(
        comm, TPHeteroConfig((1, 2, 4, 1), 2, 25), 4, StreamDelay()
    )
    delay.install()
    x = torch.empty(1_000_000, dtype=torch.uint8)
    comm.all_reduce(x)
    assert not sleeps
    delay.active = True
    comm.all_reduce(x)
    assert sleeps == pytest.approx([0.00032])
    assert delay.total_ms == pytest.approx(1.32)
    assert delay.actual_extra_total_ms == pytest.approx(0.32)
    assert delay.counts["all_reduce"] == 1
    delay.uninstall()
    assert comm.all_reduce == original


def test_short_sleep_compensates_measured_wakeup_overhead(monkeypatch):
    now = [0.0]
    sleeps = []
    monkeypatch.setattr("vllm.distributed.tp_hetero.time.perf_counter", lambda: now[0])

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds + 0.0001

    monkeypatch.setattr("vllm.distributed.tp_hetero.time.sleep", sleep)

    class StreamDelay:
        def enqueue(self, callback):
            callback()

    comm = SimpleNamespace(
        all_reduce=lambda x: x,
        all_gather=lambda x, dim: x,
        reduce_scatter=lambda x, dim: x,
    )
    delay = TPCollectiveDelay(
        comm, TPHeteroConfig((1, 1), 1, 25), 2, StreamDelay()
    )
    delay.install()
    delay.active = True
    x = torch.empty(1_000_000, dtype=torch.uint8)
    comm.all_reduce(x)
    comm.all_reduce(x)
    assert sleeps[1] < sleeps[0]
    assert delay.extra_total_ms == pytest.approx(0.64)
    assert delay.actual_extra_total_ms < 0.84
    delay.uninstall()


def test_native_baseline_does_not_enqueue_callbacks():
    class StreamDelay:
        def enqueue(self, callback):
            raise AssertionError("native TP must not enqueue stream callbacks")

    comm = SimpleNamespace(
        all_reduce=lambda x: x,
        all_gather=lambda x, dim: x,
        reduce_scatter=lambda x, dim: x,
    )
    delay = TPCollectiveDelay(
        comm, TPHeteroConfig((1, 1, 1, 1), 4, None), 4, StreamDelay()
    )
    delay.install()
    delay.active = True
    comm.all_reduce(torch.empty(1))
    assert delay.counts["all_reduce"] == 1
    delay.uninstall()


def test_compute_delay_precedes_collectives_and_excludes_comm(monkeypatch):
    now = [0.0]
    order = []
    monkeypatch.setattr("vllm.distributed.tp_hetero.time.perf_counter", lambda: now[0])

    def sleep(seconds):
        order.append("delay")
        now[0] += seconds

    monkeypatch.setattr("vllm.distributed.tp_hetero.time.sleep", sleep)

    class StreamDelay:
        def enqueue(self, callback):
            callback()

    stream = StreamDelay()

    class Comm:
        def all_reduce(self, x):
            order.append("all_reduce")
            now[0] += 0.003
            return x

        def all_gather(self, x, dim):
            return x

        def reduce_scatter(self, x, dim):
            return x

    comm = Comm()
    collective = TPCollectiveDelay(
        comm, TPHeteroConfig((2, 1), 2, None), 2, stream
    )
    collective.install()

    class Layer(torch.nn.Module):
        def forward(self, x):
            now[0] += 0.002
            x = comm.all_reduce(x)
            now[0] += 0.001
            x = comm.all_reduce(x)
            now[0] += 0.0005
            return x

    layer = Layer()
    model = SimpleNamespace(model=SimpleNamespace(
        start_layer=0, end_layer=1, layers=torch.nn.ModuleList([layer])
    ))
    compute = TPComputeDelay(model, 2, stream)
    compute.install()
    compute.reset()
    collective.compute_delay = compute
    collective.active = True
    layer(torch.empty(1))
    compute.validate()
    assert order == ["delay", "all_reduce", "delay", "all_reduce", "delay"]
    assert now[0] == pytest.approx(0.013)
    assert compute.input_compute_ms == pytest.approx(3.5)
    assert compute.requested_delay_ms == pytest.approx(3.5)
    assert collective.counts["all_reduce"] == 2
    compute.uninstall()
    collective.uninstall()
