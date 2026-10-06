# SPDX-License-Identifier: Apache-2.0
"""Resolve completed device snapshots and write traces in a background thread."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from queue import SimpleQueue
from threading import Thread
from typing import Any


@dataclass
class _PendingTrace:
    completion: Any
    resolve: Callable[[], Any]


class AsyncTraceWriter:
    """Own submitted records; resolve deferred records after their event completes."""

    def __init__(self, write: Callable[[Any], None], initialize=None) -> None:
        self._write = write
        self._initialize = initialize
        self._queue: SimpleQueue = SimpleQueue()
        self._stop = object()
        self._error: Exception | None = None
        self._closed = False
        self._thread = Thread(target=self._run, name="vllm-trace-writer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        if self._initialize is not None:
            try:
                self._initialize()
            except Exception as exc:
                self._error = exc
        while True:
            record = self._queue.get()
            if record is self._stop:
                return
            if self._error is None:
                try:
                    if isinstance(record, _PendingTrace):
                        while not record.completion.query():
                            time.sleep(0.001)
                        record = record.resolve()
                    self._write(record)
                except Exception as exc:
                    self._error = exc

    def submit(self, record: Any) -> None:
        if self._closed:
            raise RuntimeError("trace writer is closed")
        self._queue.put(record)

    def submit_ready(self, completion: Any, resolve: Callable[[], Any]) -> None:
        """Read pinned snapshots only after their producer event completes.

        Polling and record resolution run in this writer's background thread,
        not in a stream callback or the worker's submission thread.
        """
        self.submit(_PendingTrace(completion, resolve))

    def check(self) -> None:
        if self._error is not None:
            raise RuntimeError("asynchronous trace write failed") from self._error

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put(self._stop)
            self._thread.join()
        self.check()
