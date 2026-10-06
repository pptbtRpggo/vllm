# SPDX-License-Identifier: Apache-2.0
"""Write completed CPU trace records without blocking a device callback."""

from collections.abc import Callable
from queue import SimpleQueue
from threading import Thread
from typing import Any


class AsyncTraceWriter:
    """Finish mutating records before submit; drain device callbacks before close."""

    def __init__(self, write: Callable[[Any], None]) -> None:
        self._write = write
        self._queue: SimpleQueue = SimpleQueue()
        self._stop = object()
        self._error: Exception | None = None
        self._closed = False
        self._thread = Thread(target=self._run, name="vllm-trace-writer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            record = self._queue.get()
            if record is self._stop:
                return
            if self._error is None:
                try:
                    self._write(record)
                except Exception as exc:
                    self._error = exc

    def submit(self, record: Any) -> None:
        if self._closed:
            raise RuntimeError("trace writer is closed")
        self._queue.put(record)

    def check(self) -> None:
        if self._error is not None:
            raise RuntimeError("asynchronous trace write failed") from self._error

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put(self._stop)
            self._thread.join()
        self.check()
