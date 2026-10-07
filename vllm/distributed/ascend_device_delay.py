# SPDX-License-Identifier: Apache-2.0
"""910B device-side timing and delays; no host callback in the compute stream.

The optional library is built with benchmarks/build_ascend_delay.sh. It is
loaded only when a simulated delay or device-clock trace is requested. Fixed
kernel arguments and persistent timing buffers are compatible with NPU Graph.
"""

from __future__ import annotations

import ctypes
import math
import os
import time

import torch

CYCLES_PER_MS = 50_000


def prepare_ascend_full_graph(worker) -> None:
    """Fill Ascend's FULL Graph workspace registry before capture.

    Some Ascend versions initialize this registry only with compilation mode
    VLLM_COMPILE, although FULL runtime capture also supports mode NONE. Keep
    model/attention execution unchanged and apply the same setup to baselines.
    """
    config = worker.vllm_config
    if config.model_config.enforce_eager:
        return
    if not config.compilation_config.cudagraph_mode.has_full_cudagraphs():
        return
    from vllm_ascend.compilation.acl_graph import get_graph_params, set_graph_params

    if get_graph_params() is None:
        set_graph_params(worker.model_runner.cudagraph_batch_sizes)


class AscendDeviceDelay:
    def __init__(self) -> None:
        self.library = None
        self.wait_buffer = None
        self._elapsed_api = None
        self.clock_offset_ns: int | None = None
        self.clock_error_ns: int | None = None

    def _load(self):
        if self.library is not None:
            return
        path = os.environ.get("VLLM_ASCEND_DELAY_LIBRARY")
        if not path:
            raise RuntimeError(
                "Build benchmarks/build_ascend_delay.sh and set "
                "VLLM_ASCEND_DELAY_LIBRARY to libhetero_delay.so"
            )
        if "910B" not in torch.npu.get_device_name().upper():
            raise ValueError("device delay currently supports the 910B counter only")
        torch.ops.load_library(path)
        ops = torch.ops.vllm_ascend_delay
        try:
            for name in ("mark", "stretch", "wait"):
                getattr(ops, name)
        except AttributeError as error:
            raise RuntimeError(
                "Rebuild VLLM_ASCEND_DELAY_LIBRARY with "
                "benchmarks/build_ascend_delay.sh; queued TorchNPU ops are missing"
            ) from error
        self.library = ops

    @staticmethod
    def buffer():
        return torch.empty(3, device="npu", dtype=torch.int64)

    def begin_interval(self, buffer=None):
        self._load()
        if buffer is None:
            buffer = self.buffer()
        self.library.mark(buffer)
        buffer.record_stream(torch.npu.current_stream())
        return buffer

    def end_interval(self, buffer, factor: float = 0.0, extra_ms: float = 0.0):
        self._load()
        if not math.isfinite(factor) or factor < 0:
            raise ValueError("delay factor must be finite and nonnegative")
        if not math.isfinite(extra_ms) or extra_ms < 0:
            raise ValueError("delay must be finite and nonnegative")
        self.library.stretch(
            buffer,
            round(factor * 1_000_000),
            math.ceil(extra_ms * CYCLES_PER_MS),
        )
        buffer.record_stream(torch.npu.current_stream())

    def wait_ms(self, milliseconds: float) -> None:
        if not math.isfinite(milliseconds) or milliseconds < 0:
            raise ValueError("delay must be finite and nonnegative")
        if milliseconds:
            self._load()
            if self.wait_buffer is None:
                self.wait_buffer = self.buffer()
            self.library.wait(self.wait_buffer, math.ceil(milliseconds * CYCLES_PER_MS))
            self.wait_buffer.record_stream(torch.npu.current_stream())

    @staticmethod
    def event():
        event = torch.npu.Event(enable_timing=True)
        event.record()
        return event

    def elapsed_time(self, start, end) -> float:
        """Read completed trace events without draining TorchNPU's CPU queue.

        TorchNPU Event.elapsed_time empties all submission queues even when
        these events are complete. In a background reader that can wait for
        later collectives while holding the GIL needed by their producer.
        The ACL call only reads this pair; CDLL releases the GIL for the call.
        """
        if not start.query() or not end.query():
            raise RuntimeError("trace events must be complete before reading")
        if self._elapsed_api is None:
            library = ctypes.CDLL("libascendcl.so")
            self._elapsed_api = library.aclrtEventElapsedTime
            self._elapsed_api.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            self._elapsed_api.restype = ctypes.c_int
        result = ctypes.c_float()
        error = self._elapsed_api(
            ctypes.byref(result),
            ctypes.c_void_p(start.npu_event),
            ctypes.c_void_p(end.npu_event),
        )
        if error:
            raise RuntimeError(f"ACL event elapsed-time read failed: {error}")
        if not math.isfinite(result.value) or result.value < 0:
            raise RuntimeError("invalid ACL event elapsed time")
        return result.value

    @staticmethod
    def snapshot(buffer):
        # This copy is ordered before the next reuse of a captured buffer.
        host = torch.empty(
            buffer.shape, dtype=buffer.dtype, device="cpu", pin_memory=True
        )
        host.copy_(buffer, non_blocking=True)
        return host

    def calibrate_clock(self) -> None:
        """Calibrate once at trace startup, never in a request's compute path.

        Cross-rank windows are approximate within clock_error_ns. Choose the
        narrowest CPU bracket instead of treating independently recorded CPU
        submission times as device execution timestamps.
        """
        trials = []
        buffer = self.buffer()
        for _ in range(5):
            torch.npu.synchronize()
            before = time.perf_counter_ns()
            self.begin_interval(buffer)
            host = self.snapshot(buffer)
            event = self.event()
            event.synchronize()
            after = time.perf_counter_ns()
            trials.append((after - before, (before + after) // 2 - int(host[0]) * 20))
        width, self.clock_offset_ns = min(trials)
        self.clock_error_ns = width // 2

    @staticmethod
    def durations(host):
        start, end, waited = map(int, host.tolist())
        if not start <= end <= waited:
            raise RuntimeError("invalid device timing interval")
        return (end - start) / CYCLES_PER_MS, (waited - end) / CYCLES_PER_MS

    def window(self, host):
        if self.clock_offset_ns is None:
            raise RuntimeError("device clock was not calibrated")
        start, _, end = map(int, host.tolist())
        return start * 20 + self.clock_offset_ns, end * 20 + self.clock_offset_ns

    def check(self):
        # CANN asynchronous errors are surfaced by native torch-npu operations.
        pass

    def close(self):
        pass
