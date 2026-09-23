# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.pp_layer_trace import PPLayerTimer
from vllm.distributed.pp_stage_trace import PPStageTracer


class Decoder(torch.nn.Module):
    def __init__(self, now, duration):
        super().__init__()
        self.now, self.duration = now, duration

    def forward(self, x):
        self.now[0] += self.duration
        return x + 1


def model_and_clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("time.perf_counter", lambda: now[0])
    layers = torch.nn.ModuleList([Decoder(now, n / 1000) for n in (1, 2, 3)])
    model = SimpleNamespace(
        model=SimpleNamespace(start_layer=1, end_layer=3, layers=layers)
    )
    return model, layers, now


def test_cpu_layer_timing_and_cleanup_on_failure(monkeypatch):
    model, layers, now = model_and_clock(monkeypatch)
    timer = PPLayerTimer(model, torch.device("cpu"))
    with timer.capture():
        assert layers[2](layers[1](0)) == 2
    assert timer.elapsed_ms() == pytest.approx({"1": 2, "2": 3})
    with pytest.raises(ValueError, match="every decoder layer"), timer.capture():
        layers[1](0)
    with pytest.raises(ValueError, match="one call"), timer.capture():
        layers[1](0)
        layers[1](0)
    for layer in layers:
        assert not layer._forward_hooks and not layer._forward_pre_hooks
    with timer.capture():
        layers[2](layers[1](0))
    assert timer.elapsed_ms() == pytest.approx({"1": 2, "2": 3})


@pytest.mark.parametrize("backend", ["cuda", "npu"])
def test_device_events_use_current_stream_without_per_layer_sync(backend, monkeypatch):
    model, layers, now = model_and_clock(monkeypatch)
    events = []

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing
            events.append(self)

        def record(self):
            self.at = now[0]

        def elapsed_time(self, end):
            return (end.at - self.at) * 1000

        def synchronize(self):
            pytest.fail("per-layer synchronization changes execution")

    monkeypatch.setattr(torch, backend, SimpleNamespace(Event=Event), raising=False)
    timer = PPLayerTimer(model, SimpleNamespace(type=backend))
    for _ in range(2):
        with timer.capture():
            layers[2](layers[1](0))
        assert timer.elapsed_ms() == pytest.approx({"1": 2, "2": 3})
    assert len(events) == 4  # Reused between microbatches.


def test_stage_trace_serializes_layers_and_residual(tmp_path, monkeypatch):
    model, layers, now = model_and_clock(monkeypatch)
    tracer = PPStageTracer(
        str(tmp_path), 1, 2, torch.device("cpu"), compute_model="layer-measured"
    )
    config = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=True))

    def forward():
        now[0] += 0.004  # Endpoint/runner work before local layers.
        return layers[2](layers[1](0))

    result, timing = tracer.measure_stretched_compute(
        forward,
        lambda ms: ms,
        model_runner=SimpleNamespace(model=model),
        vllm_config=config,
    )
    assert result == 2
    kwargs = dict(
        num_tokens=1,
        num_reqs=1,
        num_ctx_requests=0,
        num_ctx_tokens=0,
        num_generation_requests=1,
        num_generation_tokens=1,
        recv_ms=None,
        send_ms=None,
        recv_bytes=None,
        send_bytes=None,
        start_layer=1,
        end_layer=3,
        **timing,
    )
    tracer.record(**kwargs)
    with pytest.raises(ValueError, match="fresh layer measurement"):
        tracer.record(**kwargs)
    tracer.close()
    row = json.loads((tmp_path / "pp_stage_pp1_tp0.jsonl").read_text())
    assert row["compute_model"] == "layer-measured"
    assert row["layer_compute_ms"] == pytest.approx({"1": 2, "2": 3})
    assert row["non_layer_compute_ms"] == pytest.approx(4)
    assert row["compute_wall_ms"] == pytest.approx(9)


@pytest.mark.parametrize(
    "eager,tp,scale,compiled,error",
    [
        (False, 1, 1, 0, "enforce-eager"),
        (True, 2, 1, 0, "TP=1"),
        (True, 1, 2, 0, "mock compute delay"),
        (True, 1, 1, 3, "compilation mode NONE"),
    ],
)
def test_unsupported_execution_fails_before_forward(
    eager, tp, scale, compiled, error, tmp_path
):
    tracer = PPStageTracer(
        str(tmp_path),
        0,
        2,
        torch.device("cpu"),
        tp_size=tp,
        compute_model="layer-measured",
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=eager),
        compilation_config=SimpleNamespace(mode=compiled),
    )
    try:
        with pytest.raises(ValueError, match=error):
            tracer.measure_stretched_compute(
                lambda: pytest.fail("unexpected forward"),
                lambda ms: ms,
                vllm_config=config,
                compute_scale=scale,
            )
    finally:
        tracer.close()


def test_endpoints_are_separate_and_not_double_counted(tmp_path, monkeypatch):
    model, layers, now = model_and_clock(monkeypatch)
    model.model.embed_tokens = Decoder(now, 0.004)
    model.model.norm = Decoder(now, 0.002)
    model.logits_processor = Decoder(now, 0.006)
    # LM head.forward is intentionally never invoked, matching vLLM's path.
    model.lm_head = Decoder(now, 100)
    tracer = PPStageTracer(
        str(tmp_path), 0, 1, torch.device("cpu"), compute_model="layer-measured"
    )
    config = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=True))

    def forward():
        now[0] += 0.003
        x = model.model.embed_tokens(0)
        x = layers[2](layers[1](x))
        return model.logits_processor(model.model.norm(x))

    try:
        _, timing = tracer.measure_stretched_compute(
            forward,
            lambda ms: ms,
            model_runner=SimpleNamespace(model=model),
            vllm_config=config,
        )
        measured = tracer._layer_measurement
        assert measured["embedding_ms"] == pytest.approx(4)
        assert measured["lm_head_ms"] == pytest.approx(6)
        assert measured["final_norm_ms"] == pytest.approx(2)
        assert measured["runner_overhead_ms"] == pytest.approx(3)
        assert measured["non_layer_compute_ms"] == pytest.approx(15)
        assert timing["compute_wall_ms"] == pytest.approx(20)
        for m in (model.model.embed_tokens, model.model.norm, model.logits_processor):
            assert not m._forward_hooks and not m._forward_pre_hooks
    finally:
        tracer.close()
