# SPDX-License-Identifier: Apache-2.0
"""Stream-ordered host callbacks for eager Ascend TP and PP experiments.

CANN executes each callback after preceding work on the stream and blocks
subsequent work until the callback returns. Callbacks may read CPU clocks,
sleep and queue completed CPU records. They must never call CANN or torch
device APIs or perform file I/O.
"""

from __future__ import annotations

import ctypes
import itertools
from collections.abc import Callable
from typing import Any

import torch

_Callback = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class TPStreamDelay:
    def __init__(self) -> None:
        if not hasattr(torch, "npu"):
            raise RuntimeError("TP stream delay requires torch-npu")
        try:
            self._acl = ctypes.CDLL("libascendcl.so")
            self._launch = self._acl.aclrtLaunchHostFunc
        except (OSError, AttributeError) as exc:
            raise RuntimeError(
                "TP stream delay requires CANN aclrtLaunchHostFunc"
            ) from exc
        self._launch.argtypes = [ctypes.c_void_p, _Callback, ctypes.c_void_p]
        self._launch.restype = ctypes.c_int
        self._next_id = itertools.count(1)
        self._pending: dict[int, Callable[[], None]] = {}
        self._errors: list[BaseException] = []

        @_Callback
        def run(user_data: Any) -> None:
            token = int(user_data)
            callback = self._pending.pop(token, None)
            if callback is None:
                self._errors.append(RuntimeError("missing TP stream callback"))
                return
            try:
                callback()
            except BaseException as exc:
                # ctypes cannot propagate an exception from a C callback.
                self._errors.append(exc)

        self._callback = run

    def enqueue(self, callback: Callable[[], None]) -> None:
        self.check()
        token = next(self._next_id)
        self._pending[token] = callback
        stream = torch.npu.current_stream().npu_stream
        ret = self._launch(
            ctypes.c_void_p(stream), self._callback, ctypes.c_void_p(token)
        )
        if ret != 0:
            self._pending.pop(token)
            raise RuntimeError(f"aclrtLaunchHostFunc failed with code {ret}")

    def check(self) -> None:
        if self._errors:
            raise RuntimeError("TP stream callback failed") from self._errors.pop(0)

    def close(self) -> None:
        # Call only after the worker's final device synchronization.
        self.check()
        if self._pending:
            raise RuntimeError("TP stream callbacks remain pending")
