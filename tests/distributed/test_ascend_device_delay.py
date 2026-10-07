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
