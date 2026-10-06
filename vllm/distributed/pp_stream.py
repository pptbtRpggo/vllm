# SPDX-License-Identifier: Apache-2.0
"""Ascend PP timing and mock delays ordered on the native current stream.

Native send/recv must establish their usual dependency on this stream before
end_comm is enqueued. Callbacks never launch device work or write files.
Each step owns its statistics until its final callback submits the trace.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class PPStreamStep:
    windows: dict[str, Any] = field(default_factory=dict)
    layers: dict[str, float] = field(default_factory=dict)
    endpoints: dict[str, float] = field(default_factory=dict)
    compute_timing: dict[str, float] = field(default_factory=dict)
    compute_start_ns: int = 0
    compute_end_ns: int = 0
    requested_compute_ms: float = 0.0
    actual_compute_sleep_ms: float = 0.0
    recv_ms: float | None = None
    send_ms: float | None = None
    recv_bytes: int | None = None
    send_bytes: int | None = None


class PPStreamExecution:
    def __init__(self, stream) -> None:
        self.stream = stream
        self._sleep_overhead: dict[str, float] = {}

    def _sleep(self, requested_ms: float, key: str) -> float:
        sleep_ms = max(0.0, requested_ms - self._sleep_overhead.get(key, 0.0))
        started = time.perf_counter()
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000)
        actual_ms = (time.perf_counter() - started) * 1000
        if sleep_ms > 0:
            self._sleep_overhead[key] = 0.75 * self._sleep_overhead.get(
                key, 0.0
            ) + 0.25 * max(0.0, actual_ms - sleep_ms)
        return actual_ms

    def begin_comm(self, step: PPStreamStep, kind: str) -> None:
        def begin() -> None:
            step.windows[kind + "_start_ns"] = time.perf_counter_ns()

        self.stream.enqueue(begin)

    def end_comm(self, step: PPStreamStep | None, kind: str, extra_ms: float) -> None:
        if step is None and extra_ms == 0:
            return

        def finish() -> None:
            if extra_ms:
                self._sleep(extra_ms, kind)
            if step is not None:
                end = time.perf_counter_ns()
                step.windows[kind + "_end_ns"] = end
                step.windows["comm_delay_in_window"] = True
                setattr(
                    step, kind + "_ms", (end - step.windows[kind + "_start_ns"]) / 1e6
                )

        self.stream.enqueue(finish)

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
        if mode == "layer-measured" and timer is None:
            raise ValueError("layer-measured callback execution requires local layers")
        stats = step if step is not None else PPStreamStep()
        handles = []
        seen = []
        starts: dict[int | str, list[float]] = {}

        def stage_start() -> None:
            stats.compute_start_ns = time.perf_counter_ns()

        self.stream.enqueue(stage_start)

        def before(index, _module, _args):
            if index in starts:
                raise ValueError("layer-measured expects one call per decoder layer")
            start = [0.0]
            starts[index] = start
            self.stream.enqueue(lambda: start.__setitem__(0, time.perf_counter()))

        def after(index, _module, _args, _output):
            start = starts[index]
            if isinstance(index, int):
                seen.append(index)

            def finish() -> None:
                base_ms = (time.perf_counter() - start[0]) * 1000
                requested = base_ms * max(0.0, scale - 1)
                actual = self._sleep(requested, "compute") if requested else 0.0
                stats.requested_compute_ms += requested
                stats.actual_compute_sleep_ms += actual
                duration = (time.perf_counter() - start[0]) * 1000
                if isinstance(index, int):
                    stats.layers[str(index)] = duration
                else:
                    stats.endpoints[index + "_ms"] = duration

            self.stream.enqueue(finish)

        try:
            if timer is not None:
                from functools import partial

                for index, module in {**timer.layers, **timer.endpoints}.items():
                    handles.append(
                        module.register_forward_pre_hook(partial(before, index))
                    )
                    handles.append(module.register_forward_hook(partial(after, index)))
            result = fn()
            if timer is not None and seen != list(timer.layers):
                raise ValueError(
                    "layer-measured did not observe every decoder layer in order"
                )
        finally:
            for handle in handles:
                handle.remove()

        def stage_end() -> None:
            if mode == "shape-affine":
                base = (time.perf_counter_ns() - stats.compute_start_ns) / 1e6
                requested = base * max(0.0, scale - 1)
                stats.requested_compute_ms = requested
                stats.actual_compute_sleep_ms = (
                    self._sleep(requested, "compute") if requested else 0.0
                )
            stats.compute_end_ns = time.perf_counter_ns()
            wall_ms = (stats.compute_end_ns - stats.compute_start_ns) / 1e6
            base_ms = wall_ms - stats.actual_compute_sleep_ms
            stats.compute_timing.update(
                compute_ms=base_ms + stats.requested_compute_ms,
                compute_base_ms=base_ms,
                compute_wall_ms=wall_ms,
                compute_delay_ms=stats.actual_compute_sleep_ms,
            )

        self.stream.enqueue(stage_end)
        return result

    def finish_trace(
        self, step: PPStreamStep, record, tracer, layer_measured: bool
    ) -> None:
        def finish() -> None:
            for name, value in step.compute_timing.items():
                setattr(record, name, value)
            record.recv_ms, record.send_ms = step.recv_ms, step.send_ms
            record.recv_bytes, record.send_bytes = step.recv_bytes, step.send_bytes
            for name, value in step.windows.items():
                setattr(record, name, value)
            if layer_measured:
                record.layer_compute_ms = step.layers
                residual = record.compute_wall_ms - sum(step.layers.values())
                overhead = residual - sum(step.endpoints.values())
                if min(residual, overhead) < -0.01:
                    raise ValueError("layer timing exceeds stream-ordered stage timing")
                record.non_layer_compute_ms = max(0.0, residual)
                record.runner_overhead_ms = max(0.0, overhead)
                for name, value in step.endpoints.items():
                    setattr(record, name, value)
                record.compute_delay_placement = (
                    "layer" if record.compute_scale > 1 else "none"
                )
            record.ts_unix = time.time()
            tracer.submit_async_record(record)

        self.stream.enqueue(finish)
