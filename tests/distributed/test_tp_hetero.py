# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.pp_layer_trace import PPLayerTimer
from vllm.distributed.tp_hetero import TPCollectiveDelay, TPHeteroConfig
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
    monkeypatch.setenv("VLLM_TP_COMPUTE_SCALES", "1,2")
    with pytest.raises(ValueError, match="one value"):
        TPHeteroConfig.from_env(4)
    monkeypatch.setenv("VLLM_TP_COMPUTE_SCALES", "0.5,1,1,1")
    with pytest.raises(ValueError, match="cannot speed up"):
        TPHeteroConfig.from_env(4)


def test_collective_delay_only_during_forward(monkeypatch):
    now = [0.0]
    sleeps = []
    monkeypatch.setattr("vllm.distributed.tp_hetero.time.perf_counter", lambda: now[0])

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr("vllm.distributed.tp_hetero.time.sleep", sleep)
    monkeypatch.setattr("vllm.distributed.tp_hetero.sync_torch_device", lambda _: None)

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
        comm, TPHeteroConfig((1, 2, 4, 1), 2, 25), 4, torch.device("cpu")
    )
    delay.install()
    x = torch.empty(1_000_000, dtype=torch.uint8)
    comm.all_reduce(x)
    assert not sleeps
    delay.active = True
    comm.all_reduce(x)
    assert sleeps == pytest.approx([0.00032])
    assert delay.total_ms == pytest.approx(1.32)
    assert delay.counts["all_reduce"] == 1
    delay.uninstall()
    assert comm.all_reduce == original


def test_native_baseline_uses_same_collective_wrapper(monkeypatch):
    monkeypatch.setattr("vllm.distributed.tp_hetero.sync_torch_device", lambda _: None)
    comm = SimpleNamespace(
        all_reduce=lambda x: x,
        all_gather=lambda x, dim: x,
        reduce_scatter=lambda x, dim: x,
    )
    delay = TPCollectiveDelay(
        comm, TPHeteroConfig((1, 1, 1, 1), 4, None), 4, torch.device("cpu")
    )
    delay.install()
    delay.active = True
    comm.all_reduce(torch.empty(1))
    assert delay.counts["all_reduce"] == 1
    delay.uninstall()


def test_layer_compute_scale_excludes_collective_time(monkeypatch):
    now = [0.0]
    excluded = [0.0]
    monkeypatch.setattr("vllm.distributed.pp_layer_trace.time.perf_counter", lambda: now[0])
    monkeypatch.setattr("vllm.distributed.pp_layer_trace.sync_torch_device", lambda _: None)

    def sleep(seconds):
        now[0] += seconds

    monkeypatch.setattr("vllm.distributed.pp_hetero.time.sleep", sleep)

    class Layer(torch.nn.Module):
        def forward(self, x):
            now[0] += 0.002  # compute before collective
            now[0] += 0.003  # collective, including simulated link delay
            excluded[0] += 3
            now[0] += 0.001  # compute after collective
            return x

    layer = Layer()
    model = SimpleNamespace(model=SimpleNamespace(
        start_layer=0, end_layer=1, layers=torch.nn.ModuleList([layer])
    ))
    timer = PPLayerTimer(model, torch.device("cpu"))
    with timer.capture(2, excluded_ms=lambda: excluded[0]):
        layer(1)
    assert now[0] == pytest.approx(0.009)  # 3 ms compute + 3 ms comm + 3 ms delay
    assert timer.mock_requested_delay_ms == pytest.approx(3)
