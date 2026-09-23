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


@pytest.mark.parametrize("backend", ["cpu", "cuda", "npu"])
def test_layer_mock_measures_actual_wait_and_endpoints(backend, tmp_path, monkeypatch):
    model, layers, now = model_and_clock(monkeypatch)
    model.model.embed_tokens = Decoder(now, 0.004)
    model.model.norm = Decoder(now, 0.002)
    model.logits_processor = Decoder(now, 0.006)
    syncs = []
    if backend != "cpu":

        class Event:
            def __init__(self, **kwargs):
                pass

            def record(self):
                pytest.fail("mock costs must measure synchronized wall time")

        monkeypatch.setattr(
            torch,
            backend,
            SimpleNamespace(
                Event=Event,
                is_available=lambda: True,
                synchronize=lambda *args: syncs.append(now[0]),
            ),
            raising=False,
        )
    device = SimpleNamespace(type=backend)
    timer = PPLayerTimer(model, device)
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds + 0.001  # Include actual oversleep in measured costs.

    monkeypatch.setattr("time.sleep", sleep)

    def forward():
        now[0] += 0.003  # Runner overhead must not be multiplied.
        x = model.model.embed_tokens(0)
        x = layers[2](layers[1](x))
        return model.logits_processor(model.model.norm(x))

    with timer.capture(2):
        assert forward() == 5
    assert sleeps == pytest.approx([0.004, 0.002, 0.003, 0.002, 0.006])
    assert timer.elapsed_ms() == pytest.approx({"1": 5, "2": 7})
    assert timer.endpoint_elapsed_ms() == pytest.approx(
        dict(embedding_ms=9, final_norm_ms=5, lm_head_ms=13)
    )
    assert timer.mock_delay_ms == pytest.approx(22)
    assert timer.mock_requested_delay_ms == pytest.approx(17)
    assert now[0] == pytest.approx(0.042)
    assert len(syncs) == (0 if backend == "cpu" else 10)
    for module in [*layers, *timer.endpoints.values()]:
        assert not module._forward_hooks and not module._forward_pre_hooks


def test_tracer_layer_mock_does_not_apply_stage_delay(tmp_path, monkeypatch):
    model, layers, now = model_and_clock(monkeypatch)

    def sleep(seconds):
        now[0] += seconds + 0.001

    monkeypatch.setattr("time.sleep", sleep)
    tracer = PPStageTracer(
        str(tmp_path), 1, 2, torch.device("cpu"), compute_model="layer-measured"
    )

    def forward():
        now[0] += 0.004
        return layers[2](layers[1](0))

    try:
        result, timing = tracer.measure_stretched_compute(
            forward,
            lambda ms: pytest.fail("duplicate stage delay"),
            model_runner=SimpleNamespace(model=model),
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(enforce_eager=True)
            ),
            compute_scale=2,
        )
        assert result == 2
        assert timing == pytest.approx(
            dict(
                compute_ms=14, compute_base_ms=9, compute_wall_ms=16, compute_delay_ms=7
            )
        )
        assert tracer._layer_measurement["layer_compute_ms"] == pytest.approx(
            {"1": 5, "2": 7}
        )
        assert tracer._layer_measurement["runner_overhead_ms"] == pytest.approx(4)
    finally:
        tracer.close()


def test_mock_hooks_removed_after_forward_failure(monkeypatch):
    model, layers, now = model_and_clock(monkeypatch)
    monkeypatch.setattr("time.sleep", lambda s: now.__setitem__(0, now[0] + s))
    timer = PPLayerTimer(model, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="failed"), timer.capture(2):
        layers[1](0)
        raise RuntimeError("failed")
    for layer in layers:
        assert not layer._forward_hooks and not layer._forward_pre_hooks
    with timer.capture(2):
        layers[2](layers[1](0))
    assert timer.elapsed_ms() == pytest.approx({"1": 4, "2": 6})


@pytest.mark.parametrize(
    "eager,tp,compiled,error",
    [
        (False, 1, 0, "enforce-eager"),
        (True, 2, 0, "TP=1"),
        (True, 1, 3, "compilation mode NONE"),
    ],
)
def test_untraced_layer_mock_rejects_unsupported_execution(
    eager, tp, compiled, error, monkeypatch
):
    from vllm.distributed.pp_hetero import PPHeteroConfig, execute_pp_compute

    monkeypatch.setenv("VLLM_PP_COMPUTE_MODEL", "layer-measured")
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=eager),
            compilation_config=SimpleNamespace(mode=compiled),
            parallel_config=SimpleNamespace(tensor_parallel_size=tp),
        )
    )
    with pytest.raises(ValueError, match=error):
        execute_pp_compute(
            worker,
            lambda: pytest.fail("unexpected forward"),
            None,
            PPHeteroConfig(compute_scales=(2,)),
            0,
        )


def test_untraced_fast_rank_keeps_mock_hooks_when_other_rank_is_slow(monkeypatch):
    from vllm.distributed.pp_hetero import PPHeteroConfig, execute_pp_compute

    model, layers, now = model_and_clock(monkeypatch)
    monkeypatch.setenv("VLLM_PP_COMPUTE_MODEL", "layer-measured")
    monkeypatch.setattr("time.sleep", lambda s: pytest.fail("fast rank must not sleep"))
    worker = SimpleNamespace(
        device=torch.device("cpu"),
        model_runner=SimpleNamespace(model=model),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=True),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
        ),
    )
    for _ in range(2):
        value, _ = execute_pp_compute(
            worker,
            lambda: layers[2](layers[1](0)),
            None,
            PPHeteroConfig(compute_scales=(1, 2)),
            0,
        )
        assert value == 2
        assert worker._pp_layer_mock_timer.wall_timing
        assert worker._pp_layer_mock_timer.mock_delay_ms == 0
    assert now[0] == pytest.approx(0.010)
