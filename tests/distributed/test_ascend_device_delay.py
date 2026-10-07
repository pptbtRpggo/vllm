# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm.distributed.ascend_device_delay import AscendDeviceDelay


def test_delay_submission_never_requests_raw_stream_or_host_sync(monkeypatch):
    import torch

    class Stream:
        @property
        def npu_stream(self):
            pytest.fail("raw stream access would drain TorchNPU's submission queue")

    stream = Stream()
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_stream=lambda: stream), raising=False
    )
    buffers = []

    class Buffer:
        def record_stream(self, actual):
            assert actual is stream

    runtime = AscendDeviceDelay()
    calls = []
    runtime.library = SimpleNamespace(
        mark=lambda buffer: calls.append(("mark", buffer)),
        stretch=lambda *args: calls.append(("stretch", *args)),
        wait=lambda *args: calls.append(("wait", *args)),
    )

    def allocate():
        buffer = Buffer()
        buffers.append(buffer)
        return buffer

    monkeypatch.setattr(runtime, "buffer", allocate)
    buffer = runtime.begin_interval()
    runtime.end_interval(buffer, factor=3, extra_ms=0.5)
    runtime.wait_ms(20)
    runtime.wait_ms(10)
    assert calls == [
        ("mark", buffer),
        ("stretch", buffer, 3_000_000, 25_000),
        ("wait", buffers[1], 1_000_000),
        ("wait", buffers[1], 500_000),
    ]
    assert len(buffers) == 2  # The wait's dispatch anchor is reused.


def test_zero_wait_does_not_load_library_or_allocate(monkeypatch):
    runtime = AscendDeviceDelay()

    def forbidden():
        pytest.fail("zero wait must remain on the native path")

    monkeypatch.setattr(runtime, "_load", forbidden)
    monkeypatch.setattr(runtime, "buffer", forbidden)
    runtime.wait_ms(0)


def test_completed_event_read_avoids_torchnpu_queue_drain():
    runtime = AscendDeviceDelay()

    def forbidden(_other):
        pytest.fail("TorchNPU elapsed_time drains the shared submission queue")

    start = SimpleNamespace(npu_event=123, query=lambda: True, elapsed_time=forbidden)
    end = SimpleNamespace(npu_event=456, query=lambda: True)

    def read(result, first, last):
        assert first.value == 123 and last.value == 456
        result._obj.value = 20.125
        return 0

    runtime._elapsed_api = read
    assert runtime.elapsed_time(start, end) == pytest.approx(20.125)
    start.query = lambda: False
    with pytest.raises(RuntimeError, match="complete"):
        runtime.elapsed_time(start, end)
    start.query = lambda: True
    runtime._elapsed_api = lambda *_args: 1
    with pytest.raises(RuntimeError, match="ACL"):
        runtime.elapsed_time(start, end)


def test_old_library_fails_with_rebuild_instruction(monkeypatch):
    import torch

    monkeypatch.setenv("VLLM_ASCEND_DELAY_LIBRARY", "/old/libhetero_delay.so")
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(get_device_name=lambda: "Ascend910B3"),
        raising=False,
    )
    monkeypatch.setattr(
        torch,
        "ops",
        SimpleNamespace(load_library=lambda path: None, vllm_ascend_delay=object()),
    )
    with pytest.raises(RuntimeError, match="Rebuild VLLM_ASCEND_DELAY_LIBRARY"):
        AscendDeviceDelay()._load()


def test_registered_delay_ops_on_npu():
    import os

    import torch

    pytest.importorskip("torch_npu")
    if not torch.npu.is_available() or not os.getenv("VLLM_ASCEND_DELAY_LIBRARY"):
        pytest.skip("requires an NPU and the built delay library")
    runtime = AscendDeviceDelay()
    runtime._load()
    buffer = torch.zeros(3, device="npu", dtype=torch.int64)
    # These side-effecting operators support eager and CANN Graph capture,
    # with compilation mode NONE. FakeTensor/AOT compilation is unsupported.
    checks = ("test_schema", "test_autograd_registration")
    for operation, arguments in (
        (runtime.library.mark, (buffer,)),
        (runtime.library.stretch, (buffer, 0, 0)),
        (runtime.library.wait, (buffer, 50_000)),
    ):
        torch.library.opcheck(operation, arguments, test_utils=checks)
    torch.npu.synchronize()  # Test boundary only.


def test_pp_direct_submission_preserves_stream_and_delay_arguments(monkeypatch):
    import torch

    from vllm.distributed import ascend_device_delay as module

    stream = SimpleNamespace(npu_stream=987)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_stream=lambda: stream, get_device_name=lambda: "Ascend910B3"
        ),
        raising=False,
    )
    monkeypatch.setenv("VLLM_ASCEND_DELAY_LIBRARY", "/built/libhetero_delay.so")
    calls = []

    def mark(actual_stream, address):
        calls.append(("mark", actual_stream.value, address.value))

    def stretch(actual_stream, address, factor, cycles):
        calls.append(("stretch", actual_stream.value, address.value, factor, cycles))

    def wait(actual_stream, cycles):
        calls.append(("wait", actual_stream.value, cycles))

    library = SimpleNamespace(
        launch_mark=mark, launch_stretch=stretch, launch_wait=wait
    )
    monkeypatch.setattr(module.ctypes, "CDLL", lambda path: library)

    class Buffer:
        def data_ptr(self):
            return 123

        def record_stream(self, actual):
            assert actual is stream

    runtime = AscendDeviceDelay(queued=False)
    monkeypatch.setattr(runtime, "buffer", Buffer)
    buffer = runtime.begin_interval()
    runtime.end_interval(buffer, factor=1, extra_ms=2)
    runtime.wait_ms(5)
    runtime.wait_ms(0)
    assert calls == [
        ("mark", 987, 123),
        ("stretch", 987, 123, 1_000_000, 100_000),
        ("wait", 987, 250_000),
    ]
