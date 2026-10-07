# SPDX-License-Identifier: Apache-2.0
"""Deferred device operations check dependencies without an NPU."""

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.pp_stream import PPStreamExecution, PPStreamStep


class DeferredStream:
    def __init__(self, monkeypatch=None):
        self.now = 1000.0
        self.pending = []
        self.clock_offset_ns = 0
        self.clock_error_ns = 0
        self.waits = []

    def enqueue(self, fn):
        self.pending.append(fn)

    def work(self, seconds):
        self.enqueue(lambda: setattr(self, "now", self.now + seconds * 1000))

    def drain(self):
        while self.pending:
            self.pending.pop(0)()

    def buffer(self):
        return [0, 0, 0]

    def begin_interval(self, buffer=None):
        if buffer is None:
            buffer = self.buffer()
        self.enqueue(lambda: buffer.__setitem__(0, self.now))
        return buffer

    def end_interval(self, buffer, factor=0, extra_ms=0):
        def finish():
            buffer[1] = self.now
            delay = (self.now - buffer[0]) * factor + extra_ms
            self.waits.append(delay)
            self.now += delay
            buffer[2] = self.now

        self.enqueue(finish)

    def wait_ms(self, milliseconds):
        if milliseconds:
            self.enqueue(lambda: setattr(self, "now", self.now + milliseconds))

    def event(self):
        event = SimpleNamespace(timestamp=None)
        self.enqueue(lambda: setattr(event, "timestamp", self.now))
        event.query = lambda: event.timestamp is not None
        event.elapsed_time = lambda other: other.timestamp - event.timestamp
        return event

    def snapshot(self, buffer):
        host = [0, 0, 0]
        self.enqueue(lambda: host.__setitem__(slice(None), buffer))
        return host

    def elapsed_time(self, start, end):
        return end.timestamp - start.timestamp

    def durations(self, host):
        return host[1] - host[0], host[2] - host[1]

    def window(self, host):
        return round(host[0] * 1e6), round(host[2] * 1e6)

    def check(self):
        pass


def test_pp_one_stage_wait_and_trace_attribution_preserve_batch_snapshots():
    stream = DeferredStream()
    execution = PPStreamExecution(stream)
    rows, deferred = [], []
    tracer = SimpleNamespace(
        submit_ready_record=lambda event, resolve: deferred.append((event, resolve))
    )

    class Layer(torch.nn.Module):
        def forward(self, duration):
            stream.work(duration)
            return duration

    first, second = Layer(), Layer()
    timer = SimpleNamespace(layers={0: first, 1: second}, endpoints={})
    for duration in (0.002, 0.005):
        step = PPStreamStep()
        execution.begin_comm(step, "recv")
        stream.work(0.003)
        execution.end_comm(step, "recv", 1)
        execution.compute(
            lambda d=duration: second(first(d)), step, 2, "layer-measured", timer
        )
        execution.begin_comm(step, "send")
        stream.work(0.003)
        execution.end_comm(step, "send", 1)
        execution.finish_trace(step, SimpleNamespace(), tracer, True)
    assert not first._forward_hooks and not second._forward_hooks
    assert not deferred[0][0].query()
    stream.drain()
    for event, resolve in deferred:
        assert event.query()
        rows.append(resolve())
    assert [r.compute_base_ms for r in rows] == pytest.approx([4, 10])
    assert [r.compute_delay_ms for r in rows] == pytest.approx([4, 10])
    assert [r.layer_compute_ms["0"] for r in rows] == pytest.approx([4, 10])
    assert all(r.compute_delay_placement == "stage-attributed" for r in rows)
    assert all(
        r.recv_ms == pytest.approx(4) and r.send_ms == pytest.approx(4) for r in rows
    )
    assert stream.waits == pytest.approx([1, 4, 1, 1, 10, 1])
    assert rows[0].send_end_ns <= rows[1].recv_start_ns


def test_untraced_fast_stage_queues_no_device_operations():
    stream = DeferredStream()
    execution = PPStreamExecution(stream)
    assert execution.compute(lambda: 42, None, 1, "layer-measured") == 42
    execution.end_comm(None, "send", 0)
    assert not stream.pending


def test_stage_buffer_reuse_orders_snapshot_before_next_execution():
    stream = DeferredStream()
    execution = PPStreamExecution(stream)
    snapshots = []
    for duration in (0.002, 0.005):
        execution.compute(lambda d=duration: stream.work(d), None, 3, "layer-measured")
        snapshots.append(stream.snapshot(execution._compute_buffer))
    stream.drain()
    assert [stream.durations(h) for h in snapshots] == pytest.approx([(2, 4), (5, 10)])


def test_background_layer_trace_uses_completed_event_reader():
    stream = DeferredStream()
    native_event = stream.event

    def event():
        result = native_event()

        def forbidden(_end):
            pytest.fail("background trace must not drain TorchNPU queues")

        result.elapsed_time = forbidden
        return result

    stream.event = event

    class Layer(torch.nn.Module):
        def forward(self):
            stream.work(0.002)

    timer = SimpleNamespace(layers={0: Layer()}, endpoints={})
    step = PPStreamStep()
    execution = PPStreamExecution(stream)
    execution.compute(timer.layers[0], step, 1, "layer-measured", timer)
    pending = []
    tracer = SimpleNamespace(
        submit_ready_record=lambda event, resolve: pending.append((event, resolve))
    )
    execution.finish_trace(step, SimpleNamespace(), tracer, True)
    stream.drain()
    assert pending[0][0].query()
    assert pending[0][1]().layer_compute_ms == pytest.approx({"0": 2})
