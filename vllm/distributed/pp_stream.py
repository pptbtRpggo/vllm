# SPDX-License-Identifier: Apache-2.0
"""Device-side PP stage delay and deferred event traces on Ascend.

A stage has one proportional wait, before native send establishes its producer
stream dependency. Layer events measure unmodified module computation. The
stage delay is attributed proportionally for DP and explicitly tagged.
"""

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import Any, TypeVar

T = TypeVar("T")


def _attributed_layer_timing(stream, events, base_ms, delay_ms):
    """Allocate one observed stage wait across measured modules and residual.

    The layer events measure unmodified execution. The returned layer costs
    include a proportional share of the stage wait, not per-layer sleeps.
    Read events only in the background after the completion event is ready.
    """
    slowdown = (base_ms + delay_ms) / base_ms if base_ms else 1.0
    durations = {
        index: stream.elapsed_time(start, end) * slowdown
        for index, (start, end) in events.items()
    }
    layers = {str(i): ms for i, ms in durations.items() if isinstance(i, int)}
    endpoints = {i + "_ms": ms for i, ms in durations.items() if isinstance(i, str)}
    residual = base_ms + delay_ms - sum(layers.values())
    overhead = residual - sum(endpoints.values())
    if min(residual, overhead) < -0.01:
        raise ValueError("layer events exceed stage timing")
    return {
        "layer_compute_ms": layers,
        **endpoints,
        "non_layer_compute_ms": max(0.0, residual),
        "runner_overhead_ms": max(0.0, overhead),
    }


@dataclass
class PPStreamStep:
    intervals: dict[str, Any] = field(default_factory=dict)
    layer_events: dict[int | str, tuple[Any, Any]] = field(default_factory=dict)
    recv_bytes: int | None = None
    send_bytes: int | None = None


class PPStreamExecution:
    def __init__(self, stream) -> None:
        self.stream = stream
        self._compute_buffer = None

    def begin_comm(self, step: PPStreamStep, kind: str) -> None:
        step.intervals[kind] = self.stream.begin_interval()

    def end_comm(self, step: PPStreamStep | None, kind: str, extra_ms: float) -> None:
        if step is None:
            self.stream.wait_ms(extra_ms)
        else:
            self.stream.end_interval(step.intervals[kind], extra_ms=extra_ms)

    def compute(
        self,
        fn: Callable[[], T],
        step: PPStreamStep | None,
        scale: float,
        mode: str,
        timer=None,
    ) -> T:
        if step is None and scale == 1:
            return fn()
        if mode not in ("shape-affine", "layer-measured"):
            raise ValueError(f"unknown compute profiling mode: {mode}")
        if step is not None and mode == "layer-measured" and timer is None:
            raise ValueError("layer-measured tracing requires local layers")
        if step is None:
            if self._compute_buffer is None:
                self._compute_buffer = self.stream.buffer()
            interval = self.stream.begin_interval(self._compute_buffer)
        else:
            interval = self.stream.begin_interval()
            step.intervals["compute"] = interval
        with self._record_layer_events(step, timer):
            result = fn()
        # Exactly one proportional wait, irrespective of the profiling mode.
        self.stream.end_interval(interval, factor=max(0.0, scale - 1))
        return result

    @contextmanager
    def _record_layer_events(self, step, timer):
        """Limit module hooks to this forward and always remove them on exit."""
        if step is None or timer is None:
            yield
            return
        handles, seen, starts = [], [], {}

        def before(index, _module, _args):
            if index in starts:
                raise ValueError("layer-measured expects one call per decoder layer")
            starts[index] = self.stream.event()

        def after(index, _module, _args, _output):
            step.layer_events[index] = (starts[index], self.stream.event())
            if isinstance(index, int):
                seen.append(index)

        try:
            for index, module in {**timer.layers, **timer.endpoints}.items():
                handles.append(module.register_forward_pre_hook(partial(before, index)))
                handles.append(module.register_forward_hook(partial(after, index)))
            yield
            if seen != list(timer.layers):
                raise ValueError(
                    "layer-measured did not observe every decoder layer in order"
                )
        finally:
            for handle in handles:
                handle.remove()

    def finish_trace(self, step, record, tracer, layer_measured):
        snapshots = {
            kind: self.stream.snapshot(buf) for kind, buf in step.intervals.items()
        }
        completion = self.stream.event()

        def resolve():
            base, delay = self.stream.durations(snapshots["compute"])
            record.compute_base_ms = base
            record.compute_ms = record.compute_wall_ms = base + delay
            record.compute_delay_ms = delay
            record.compute_delay_placement = "stage-attributed"
            record.timing_source = "device_clock"
            record.clock_error_ns = self.stream.clock_error_ns
            record.recv_bytes, record.send_bytes = step.recv_bytes, step.send_bytes
            for kind in ("recv", "send"):
                if kind in snapshots:
                    native, extra = self.stream.durations(snapshots[kind])
                    setattr(record, kind + "_ms", native + extra)
                    start, end = self.stream.window(snapshots[kind])
                    setattr(record, kind + "_start_ns", start)
                    setattr(record, kind + "_end_ns", end)
            record.comm_delay_in_window = True
            if layer_measured:
                timing = _attributed_layer_timing(
                    self.stream, step.layer_events, base, delay
                )
                for name, duration in timing.items():
                    setattr(record, name, duration)
            return record

        tracer.submit_ready_record(completion, resolve)
