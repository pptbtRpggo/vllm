# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit CPU/Gloo P2P for CPU or NPU tensors, without an HCCL group.

This is an inference transport, not a global patch of torch.distributed. Peers
must agree on shape, dtype and tag, as with torch.distributed.isend/irecv.
Only contiguous *CPU bytes* go over Gloo; an NPU's physical storage format and
device index are never transmitted. Each rank chooses its own local tensor.
"""

import threading
import time
from collections.abc import Callable
from contextlib import nullcontext, suppress
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


class StagedWork:
    """Completion includes network IO and, for receives, the destination copy.

    wait() blocks the host and returns True. A timedelta timeout raises
    TimeoutError without cancelling the operation; wait() may then be retried.
    is_completed() reports terminal completion, including failure. wait() must
    be called to observe an error. No Future, cancellation or CUDA graph support
    is provided. This deliberately does not claim the full c10d Work API.
    """

    def __init__(
        self,
        work: Any,
        finish: Callable[[], None],
        buffers: tuple[Any, ...],
        retire: Callable[["StagedWork", Exception | None], None],
    ) -> None:
        self._work = work
        self._finish = finish
        self._buffers = buffers
        self._retire = retire
        self._done = threading.Event()
        self._error: Exception | None = None

    def _run(self) -> None:
        try:
            self._work.wait()
            self._finish()
        except Exception as error:
            self._error = error
        finally:
            # Release device/host storage only after all IO and copies finish.
            self._buffers = ()
            self._finish = lambda: None
            self._work = None
            self._retire(self, self._error)
            self._done.set()

    def wait(self, timeout: timedelta | None = None) -> bool:
        if timeout is not None and not isinstance(timeout, timedelta):
            raise TypeError("timeout must be a datetime.timedelta or None")
        seconds = None if timeout is None else timeout.total_seconds()
        if seconds is not None and seconds < 0:
            raise ValueError("timeout must be nonnegative")
        if not self._done.wait(seconds):
            raise TimeoutError("CPU-staged transfer is still pending; not cancelled")
        if self._error is not None:
            raise RuntimeError("CPU-staged transfer failed") from self._error
        return True

    def is_completed(self) -> bool:
        return self._done.is_set()


class CpuStagingTransport:
    """Point-to-point CPU staging on an explicitly supplied Gloo group.

    dst/src are GLOBAL ranks, even for subgroups. Posting is thread-safe; peers
    must post equal-tag messages in the same order. isend takes a synchronous
    D2H snapshot before starting asynchronous network IO, so the source may be
    reused after isend returns. irecv returns immediately after posting a CPU
    receive. A progress thread waits for Gloo, copies to the local destination,
    and synchronizes that copy before marking the Work complete. The caller
    must wait before consuming or reusing the destination.

    In-flight buffers survive a dropped Work handle. close() drains all pending
    work and reports failures, including failures of dropped handles. Call it
    before destroying the process group. Caller timeouts do not abort Gloo;
    create the process group with an appropriate network timeout as well.

    No cross-SoC HCCL communicator is created, and no model/Worker/Runner or
    vLLM process group is changed by constructing this transport.
    """

    def __init__(self, group: dist.ProcessGroup) -> None:
        if not dist.is_initialized() or group is None:
            raise ValueError("an initialized, explicit Gloo group is required")
        if dist.get_backend(group) != "gloo":
            raise ValueError("CPU staging requires a Gloo group, never HCCL")
        if dist.get_rank(group) < 0:
            raise ValueError("this rank is not a member of the Gloo group")
        self.group = group
        self._rank = dist.get_rank()
        self._ranks = set(dist.get_process_group_ranks(group))
        self._lock = threading.Lock()
        self._pending: set[StagedWork] = set()
        self._errors: list[Exception] = []
        self._closed = False

    def _check(self, tensor: torch.Tensor, peer: int, tag: int) -> None:
        if peer not in self._ranks or peer == self._rank:
            raise ValueError("peer must be a different global rank in the group")
        if not isinstance(tag, int) or not 0 <= tag < 2**31:
            raise ValueError("tag must be an integer in [0, 2**31)")
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError("only dense, unquantized strided tensors are supported")
        if tensor.device.type not in ("cpu", "npu"):
            raise ValueError("CPU staging currently supports CPU and NPU tensors")

    @staticmethod
    def _device(tensor: torch.Tensor):
        if tensor.device.type == "cpu":
            return None, nullcontext()
        module = torch.get_device_module(tensor.device.type)
        return module, module.device(tensor.device)

    @staticmethod
    def _wire(buffer: torch.Tensor) -> torch.Tensor:
        # A one-byte empty-message token makes empty sends match real receives.
        if buffer.numel() == 0:
            return torch.zeros(1, dtype=torch.uint8)
        return buffer.reshape(-1).view(torch.uint8)

    def _retire(self, work: StagedWork, error: Exception | None) -> None:
        with self._lock:
            self._pending.remove(work)
            if error is not None:
                self._errors.append(error)

    def _post(self, post, finish, buffers) -> StagedWork:
        with self._lock:
            if self._closed:
                raise RuntimeError("CPU staging transport is closed")
            work = StagedWork(post(), finish, buffers, self._retire)
            self._pending.add(work)
            threading.Thread(
                target=work._run, name="cpu-staging-progress", daemon=True
            ).start()
            return work

    def isend(self, tensor: torch.Tensor, dst: int, tag: int = 0) -> StagedWork:
        self._check(tensor, dst, tag)
        module, context = self._device(tensor)
        with context, torch.inference_mode():
            buffer = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
            buffer.copy_(tensor.detach(), non_blocking=False)
            if module is not None:
                module.current_stream(tensor.device).synchronize()
        wire = self._wire(buffer)
        return self._post(
            lambda: dist.isend(wire, dst=dst, group=self.group, tag=tag),
            lambda: None,
            (buffer, wire),
        )

    def irecv(self, tensor: torch.Tensor, src: int, tag: int = 0) -> StagedWork:
        self._check(tensor, src, tag)
        buffer = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
        wire = self._wire(buffer)
        # The copy thread has its own device/stream context. Do not overwrite
        # destination storage before earlier work on the posting stream ends.
        module, context = self._device(tensor)
        ready = None
        with context:
            if module is not None:
                ready = module.Event()
                ready.record(module.current_stream(tensor.device))

        def finish() -> None:
            if ready is not None:
                ready.synchronize()
            _, copy_context = self._device(tensor)
            with copy_context, torch.inference_mode():
                tensor.copy_(buffer, non_blocking=False)
                if module is not None:
                    module.current_stream(tensor.device).synchronize()

        return self._post(
            lambda: dist.irecv(wire, src=src, group=self.group, tag=tag),
            finish,
            (tensor, buffer, wire, ready),
        )

    def send(self, tensor: torch.Tensor, dst: int, tag: int = 0) -> None:
        self.isend(tensor, dst, tag).wait()

    def recv(self, tensor: torch.Tensor, src: int, tag: int = 0) -> None:
        self.irecv(tensor, src, tag).wait()

    def close(self, timeout: timedelta | None = None) -> None:
        if timeout is not None and timeout.total_seconds() < 0:
            raise ValueError("timeout must be nonnegative")
        deadline = (
            None if timeout is None else time.monotonic() + timeout.total_seconds()
        )
        with self._lock:
            self._closed = True
            pending = tuple(self._pending)
        for work in pending:
            remaining = (
                None
                if deadline is None
                else timedelta(seconds=max(0, deadline - time.monotonic()))
            )
            # Drain the other transfers before reporting a failed transfer.
            with suppress(RuntimeError):
                work.wait(remaining)
        with self._lock:
            if self._errors:
                raise RuntimeError("CPU staging transport had failed transfers") from (
                    self._errors[0]
                )
