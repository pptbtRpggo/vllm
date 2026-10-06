# SPDX-License-Identifier: Apache-2.0
"""Deferred stream tasks exercise the ordering used by the Ascend PP worker."""

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.pp_hetero import PPHeteroConfig, PPNetwork
from vllm.distributed.pp_stream import PPStreamExecution, PPStreamStep


class DeferredStream:
    def __init__(self, monkeypatch):
        self.now = 1.0
        self.pending = []
        monkeypatch.setattr("time.perf_counter", lambda: self.now)
        monkeypatch.setattr("time.perf_counter_ns", lambda: round(self.now * 1e9))
        monkeypatch.setattr("time.sleep", self.advance)

    def advance(self, seconds):
        self.now += seconds

    def enqueue(self, callback):
        self.pending.append(callback)

    def work(self, seconds):
        self.enqueue(lambda: self.advance(seconds))

    def drain(self):
        while self.pending:
            self.pending.pop(0)()

    def check(self):
        pass


def test_recv_delay_precedes_compute_and_send_delay_precedes_next_batch(monkeypatch):
    stream = DeferredStream(monkeypatch)
    execution = PPStreamExecution(stream)
    hetero = PPHeteroConfig(network=PPNetwork.from_json('{"bandwidth_gbps":[[8],[]]}'))
    steps = [PPStreamStep(), PPStreamStep()]
    for step in steps:
        execution.begin_comm(step, "recv")
        stream.work(0.003)
        execution.end_comm(step, "recv", hetero.extra_transfer_ms(0, 1_000_000))
        execution.compute(lambda: stream.work(0.002), step, 2, "shape-affine")
        execution.begin_comm(step, "send")
        stream.work(0.003)
        execution.end_comm(step, "send", hetero.extra_transfer_ms(0, 1_000_000))
    assert not steps[0].compute_timing
    stream.drain()
    for step in steps:
        assert step.compute_timing["compute_base_ms"] == pytest.approx(2)
        assert step.compute_timing["compute_delay_ms"] == pytest.approx(2)
        assert step.windows["recv_end_ns"] <= step.compute_start_ns
        assert step.compute_end_ns <= step.windows["send_start_ns"]
        assert step.recv_ms == pytest.approx(4)
        assert step.send_ms == pytest.approx(4)
    assert steps[0].windows["send_end_ns"] <= steps[1].windows["recv_start_ns"]


def test_layer_statistics_survive_next_microbatch_and_hooks_are_removed(monkeypatch):
    stream = DeferredStream(monkeypatch)
    execution = PPStreamExecution(stream)

    class Layer(torch.nn.Module):
        def forward(self, x):
            stream.work(x)
            return x

    layer = Layer()
    timer = SimpleNamespace(layers={0: layer}, endpoints={})
    steps = [PPStreamStep(), PPStreamStep()]
    for step, duration in zip(steps, (0.002, 0.005)):
        execution.compute(lambda d=duration: layer(d), step, 2, "layer-measured", timer)
    assert not layer._forward_hooks and not layer._forward_pre_hooks
    stream.drain()
    assert steps[0].layers["0"] == pytest.approx(4)
    assert steps[1].layers["0"] == pytest.approx(10)
    assert steps[0].compute_timing["compute_delay_ms"] == pytest.approx(2)
    assert steps[1].compute_timing["compute_delay_ms"] == pytest.approx(5)


def test_untraced_fast_stage_queues_no_callbacks(monkeypatch):
    stream = DeferredStream(monkeypatch)
    execution = PPStreamExecution(stream)
    assert execution.compute(lambda: 42, None, 1, "layer-measured") == 42
    assert not stream.pending


@pytest.mark.parametrize("tracing", [False, True])
def test_ascend_worker_submits_two_batches_without_device_sync(
    tracing, tmp_path, monkeypatch
):
    import json

    from tests.distributed.test_pp_hetero_worker import _worker_module
    from vllm.distributed.pp_stage_trace import PPStageTracer
    from vllm.sequence import IntermediateTensors

    stream = DeferredStream(monkeypatch)
    module, worker_cls = _worker_module("ascend", monkeypatch)
    monkeypatch.setattr(
        module,
        "sync_torch_device",
        lambda _: pytest.fail("execute must not synchronize"),
    )
    monkeypatch.setenv("VLLM_PP_COMPUTE_MODEL", "shape-affine")
    payload = {"hidden_states": torch.empty(1_000_000, dtype=torch.uint8)}

    def recv(**kwargs):
        stream.work(0.003)
        return payload

    def send(tensors, **kwargs):
        assert tensors is payload
        stream.work(0.003)

    group = SimpleNamespace(
        rank_in_group=1,
        world_size=3,
        is_first_rank=False,
        recv_tensor_dict=recv,
        send_tensor_dict=send,
    )
    monkeypatch.setattr(module, "get_pp_group", lambda: group)
    monkeypatch.setattr(module, "get_tp_group", lambda: SimpleNamespace(world_size=1))
    worker = object.__new__(worker_cls)
    worker.device = SimpleNamespace(type="npu")
    worker._pp_stream_delay = stream
    worker._pp_stream_execution = PPStreamExecution(stream)
    worker._pp_hetero = PPHeteroConfig(
        compute_scales=(1, 2, 1), comm_scales=(2, 2), comm_bandwidth_gbps=(8, 8)
    )
    tracer = (
        PPStageTracer(str(tmp_path), 1, 3, torch.device("cpu")) if tracing else None
    )
    worker._pp_stage_tracer = tracer

    def forward(scheduler, intermediate):
        assert intermediate.tensors is payload
        stream.work(scheduler.total_num_scheduled_tokens / 1000)
        return IntermediateTensors(payload)

    worker.model_runner = SimpleNamespace(
        execute_model=forward,
        model=SimpleNamespace(start_layer=1, end_layer=2),
        requests={"r": SimpleNamespace(prompt_token_ids=[0])},
    )
    for tokens in (2, 5):
        if tracer:
            tracer.is_warmup = tokens == 2
        scheduler = SimpleNamespace(
            total_num_scheduled_tokens=tokens,
            num_scheduled_tokens={"r": tokens},
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=["r"], num_computed_tokens=[32]
            ),
        )
        assert worker.execute_model(scheduler) is None
    stream.drain()
    if tracer:
        tracer.close()
        rows = [
            json.loads(line)
            for line in (tmp_path / "pp_stage_pp1_tp0.jsonl").read_text().splitlines()
        ]
        assert [r["step"] for r in rows] == [0, 1]
        assert [r["is_warmup"] for r in rows] == [True, False]
        assert [r["num_tokens"] for r in rows] == [2, 5]
        assert [r["compute_base_ms"] for r in rows] == pytest.approx([2, 5])
        assert [r["compute_wall_ms"] for r in rows] == pytest.approx([4, 10])
        assert all(r["recv_ms"] == pytest.approx(4) for r in rows)
        assert all(r["send_ms"] == pytest.approx(4) for r in rows)


def test_zero_delay_network_worker_queues_no_callback_or_sync(monkeypatch):
    from tests.distributed.test_pp_hetero_worker import _worker_module
    from vllm.sequence import IntermediateTensors

    stream = DeferredStream(monkeypatch)
    module, worker_cls = _worker_module("ascend", monkeypatch)
    monkeypatch.setattr(
        module, "sync_torch_device", lambda _: pytest.fail("zero delay must not sync")
    )
    monkeypatch.setenv("VLLM_PP_COMPUTE_MODEL", "layer-measured")
    payload = {"hidden_states": torch.empty(1)}
    sent = []
    group = SimpleNamespace(
        rank_in_group=1,
        world_size=3,
        is_first_rank=False,
        recv_tensor_dict=lambda **kwargs: payload,
        send_tensor_dict=lambda tensors, **kwargs: sent.append(tensors),
    )
    monkeypatch.setattr(module, "get_pp_group", lambda: group)
    monkeypatch.setattr(module, "get_tp_group", lambda: SimpleNamespace(world_size=1))
    worker = object.__new__(worker_cls)
    worker.device = SimpleNamespace(type="npu")
    worker._pp_stream_delay = stream
    worker._pp_stream_execution = PPStreamExecution(stream)
    worker._pp_stage_tracer = None
    worker._pp_hetero = PPHeteroConfig(
        network=PPNetwork.from_json('{"bandwidth_gbps":[[null,null],[null],[]]}')
    )
    worker.model_runner = SimpleNamespace(
        execute_model=lambda scheduler, intermediate: IntermediateTensors(
            intermediate.tensors
        )
    )
    assert worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=1)) is None
    assert sent == [payload]
    assert not stream.pending
