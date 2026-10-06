# SPDX-License-Identifier: Apache-2.0

from threading import Event, get_ident

import pytest

from vllm.distributed.async_trace import AsyncTraceWriter


def test_writer_does_not_block_producer_and_close_drains_records():
    entered, release = Event(), Event()
    rows, threads = [], []

    def write(row):
        entered.set()
        assert release.wait(5)
        rows.append(row)
        threads.append(get_ident())

    writer = AsyncTraceWriter(write)
    try:
        writer.submit({"batch": 1})
        assert entered.wait(5)
        writer.submit({"batch": 2})
        assert not rows  # File I/O is blocked, but the producer continued.
    finally:
        release.set()
        writer.close()
    assert rows == [{"batch": 1}, {"batch": 2}]
    assert all(thread != get_ident() for thread in threads)
    with pytest.raises(RuntimeError, match="closed"):
        writer.submit({})


def test_writer_reports_io_errors_when_drained():
    def write(_row):
        raise OSError("disk full")

    writer = AsyncTraceWriter(write)
    writer.submit({})
    with pytest.raises(RuntimeError, match="trace write failed") as error:
        writer.close()
    assert isinstance(error.value.__cause__, OSError)
