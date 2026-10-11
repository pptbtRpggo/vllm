# SPDX-License-Identifier: Apache-2.0
"""Run the Ascend PP stream path with deferred device work and fake transport."""

from types import SimpleNamespace

import pytest
import torch

from tests.distributed.test_pp_hetero_worker import _worker_module
from tests.distributed.test_pp_stream import DeferredStream
from vllm.distributed.pp_hetero import PPHeteroConfig
from vllm.distributed.pp_layer_trace import PPLayerTimer
from vllm.sequence import IntermediateTensors


@pytest.mark.parametrize("backend", ["ascend", "ascend_sp"])
@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("tracing", [False, True])
def test_pp_stream_orders_transport_and_reuses_loaded_model_timer(
    backend, rank, tracing, monkeypatch
):
    module, worker_cls = _worker_module(backend, monkeypatch)
    stream = DeferredStream()
    calls, timer_models, pending, rows = [], [], [], []
    tp = SimpleNamespace(world_size=1)
    payload = {"hidden_states": torch.empty(8)}

    def recv(**kwargs):
        calls.append("recv")
        stream.work(0.003)
        return payload

    def send(tensors, **kwargs):
        calls.append("send")
        assert tensors is payload
        assert kwargs["all_gather_group"] is (None if backend == "ascend_sp" else tp)
        stream.work(0.003)

    group = SimpleNamespace(
        rank_in_group=rank,
        world_size=3,
        is_first_rank=rank == 0,
        recv_tensor_dict=recv,
        send_tensor_dict=send,
    )
    monkeypatch.setattr(module, "get_pp_group", lambda: group)
    monkeypatch.setattr(module, "get_tp_group", lambda: tp)
    monkeypatch.setattr(
        module, "sync_torch_device", lambda _: pytest.fail("no per-step sync")
    )

    def device_delay(*, queued):
        assert queued is False  # PP retains native ProcessGroup submission order.
        return stream

    def layer_timer(model, _device):
        timer_models.append(model)
        # Discover the real decoder modules; only device events are emulated.
        return PPLayerTimer(model, torch.device("cpu"))

    monkeypatch.setattr(
        "vllm.distributed.ascend_device_delay.AscendDeviceDelay", device_delay
    )
    monkeypatch.setattr("vllm.distributed.pp_layer_trace.PPLayerTimer", layer_timer)

    class Layer(torch.nn.Module):
        def forward(self, duration):
            stream.work(duration)
            return duration

    layer = Layer()
    model = SimpleNamespace(
        start_layer=0, end_layer=1, layers=torch.nn.ModuleList([layer])
    )

    def forward(scheduler, intermediate):
        calls.append("compute")
        assert intermediate is None if rank == 0 else intermediate.tensors is payload
        layer(scheduler.total_num_scheduled_tokens / 1000)
        return None if rank == 2 else IntermediateTensors(payload)

    def reserve(scheduler, **metadata):
        return SimpleNamespace(tokens=scheduler.total_num_scheduled_tokens, **metadata)

    tracer = SimpleNamespace(
        compute_model="layer-measured",
        reserve_async_step=reserve,
        submit_ready_record=lambda event, resolve: pending.append((event, resolve)),
    )
    worker = object.__new__(worker_cls)
    worker.device = SimpleNamespace(type="npu")
    worker._pp_stage_tracer = tracer if tracing else None
    worker._pp_hetero = PPHeteroConfig(compute_scales=(2, 2, 2))
    worker.model_runner = SimpleNamespace(execute_model=forward, model=model)
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True)
    )
    for tokens in (2, 5):
        scheduler = SimpleNamespace(
            total_num_scheduled_tokens=tokens,
            num_scheduled_tokens={},
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[], num_computed_tokens=[]),
        )
        assert worker.execute_model(scheduler) is None
    assert (
        calls
        == ((["recv"] if rank else []) + ["compute"] + (["send"] if rank < 2 else []))
        * 2
    )
    assert timer_models == ([model] if tracing else [])
    assert not layer._forward_hooks and not layer._forward_pre_hooks
    assert all(not event.query() for event, _ in pending)
    stream.drain()
    for event, resolve in pending:
        assert event.query()
        rows.append(resolve())
    if tracing:
        assert [r.tokens for r in rows] == [2, 5]
        assert [r.compute_base_ms for r in rows] == pytest.approx([2, 5])
        assert [r.compute_delay_ms for r in rows] == pytest.approx([2, 5])
        assert [r.layer_compute_ms["0"] for r in rows] == pytest.approx([4, 10])
        if rank:
            assert all(r.recv_ms == pytest.approx(3) for r in rows)
        if rank < 2:
            assert all(r.send_ms == pytest.approx(3) for r in rows)
