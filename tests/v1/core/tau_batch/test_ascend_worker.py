# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test the adapter against a fake plugin, with real JSONL instrumentation.

These tests check hook boundaries and output/error preservation on CPU. They
cannot validate the installed Ascend plugin or NPU synchronization on hardware.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm.v1.core.sched.tau_batch import trace as tau_trace

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    calls = []
    output = object()

    class Runner:
        def execute_model(self, scheduler_output, *args, **kwargs):
            calls.append(("runner", scheduler_output, args, kwargs))
            return output

    class NPUWorker:
        def __init__(self):
            self.vllm_config = SimpleNamespace(
                scheduler_config=SimpleNamespace(
                    tau_batch_trace=str(tmp_path / "t.jsonl")
                )
            )
            self.model_runner = None

        def init_device(self):
            calls.append("init_device")
            if self.model_runner is None:
                self.model_runner = Runner()
            return "initialized"

        def execute_model(self, scheduler_output):
            calls.append("recv")
            result = self.model_runner.execute_model(
                scheduler_output, "intermediate", optional="forwarded"
            )
            calls.append("send")
            return result

    for name in ("vllm_ascend", "vllm_ascend.worker", "vllm_ascend.worker.worker"):
        module = ModuleType(name)
        if name.endswith("worker.worker"):
            module.NPUWorker = NPUWorker
        else:
            module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(synchronize=lambda: calls.append("sync")),
        raising=False,
    )
    monkeypatch.setattr(tau_trace, "_worker_tracer", None)
    monkeypatch.setattr(tau_trace, "_worker_tracer_ready", False)
    monkeypatch.setattr(tau_trace, "_pp_rank", lambda: 1)
    path = Path(__file__).resolve().parents[4] / "vllm/v1/worker/tau_ascend_worker.py"
    spec = importlib.util.spec_from_file_location("tau_ascend_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    worker = module.TauAscendWorker()
    assert worker.init_device() == "initialized"
    calls.clear()
    sched = SimpleNamespace(
        tau_fwd_id=7,
        num_scheduled_tokens={"r": 1},
        tau_task=dict(
            phase="decode", n=1, s_sum=9, s_max=9, seq_lens=[9], tokens=1, pp_size=2
        ),
    )
    yield SimpleNamespace(
        worker=worker,
        base=NPUWorker,
        runner=Runner,
        calls=calls,
        output=output,
        sched=sched,
        trace=tmp_path / "t.jsonl",
    )
    if tau_trace._worker_tracer is not None:
        tau_trace._worker_tracer.close()


def test_adapter_records_inside_recv_send_and_preserves_output(adapter, monkeypatch):
    a = adapter
    clock = iter((100, 200))
    monkeypatch.setattr(tau_trace.time, "time_ns", lambda: next(clock, 300))
    assert a.worker.execute_model(a.sched) is a.output
    assert [c[0] if isinstance(c, tuple) else c for c in a.calls] == [
        "recv",
        "runner",
        "sync",
        "send",
    ]
    assert a.calls[1] == (
        "runner",
        a.sched,
        ("intermediate",),
        {"optional": "forwarded"},
    )
    (event,) = [json.loads(line) for line in a.trace.read_text().splitlines()]
    assert event["event"] == "compute"
    assert event["fwd_id"] == 7 and event["pp_rank"] == 1
    assert event["req_ids"] == ["r"]
    assert event["seq_lens"] == [9]
    assert event["start_ts_ns"] == 100 and event["end_ts_ns"] == 200
    assert a.worker.execute_model.__func__ is a.base.execute_model
    # A different plugin instance is not patched globally.
    other = a.runner()
    assert other.execute_model.__func__ is a.runner.execute_model


def test_adapter_skips_untraced_profile_or_cleanup_calls(adapter):
    a = adapter
    a.sched.tau_fwd_id = None
    assert a.worker.execute_model(a.sched) is a.output
    assert "sync" not in a.calls
    assert not a.trace.exists()


def test_adapter_does_not_wrap_twice(adapter):
    a = adapter
    a.worker.init_device()
    a.worker.execute_model(a.sched)
    assert a.calls.count("sync") == 1
    assert len(a.trace.read_text().splitlines()) == 1


@pytest.mark.parametrize("failure", ["runner", "sync", "missing_sync_api"])
def test_adapter_propagates_errors_without_compute_record(
    adapter, monkeypatch, failure
):
    a = adapter
    error = RuntimeError("failed")

    def fail(*args, **kwargs):
        raise error

    if failure == "runner":
        # Install a fresh runner before init_device captures the original method.
        a.worker.model_runner = SimpleNamespace(execute_model=fail)
        a.worker.init_device()
    elif failure == "sync":
        monkeypatch.setattr(torch.npu, "synchronize", fail)
    else:
        monkeypatch.delattr(torch.npu, "synchronize")
    with pytest.raises(
        AttributeError if failure == "missing_sync_api" else RuntimeError
    ):
        a.worker.execute_model(a.sched)
    assert "send" not in a.calls
    assert not a.trace.exists()


def test_adapter_preserves_async_return_without_materializing(adapter):
    class AsyncOutput:
        def get_output(self):
            raise AssertionError("The adapter must not consume the async result")

    a = adapter
    output = AsyncOutput()
    a.worker.model_runner = SimpleNamespace(execute_model=lambda *args, **kw: output)
    a.worker.init_device()
    assert a.worker.execute_model(a.sched) is output
    assert a.calls.count("sync") == 1


def test_adapter_jsonl_passes_bench_checker_for_both_ranks(adapter, monkeypatch):
    a = adapter
    for fid, phase in enumerate(("prefill", "decode"), 1):
        a.sched.tau_fwd_id = fid
        a.sched.tau_task["phase"] = phase
        a.sched.tau_task["tokens"] = 9 if phase == "prefill" else 1
        with a.trace.open("a") as stream:
            stream.write(
                json.dumps(dict(event="emit", fwd_id=fid, **a.sched.tau_task)) + "\n"
            )
        for rank in range(2):
            monkeypatch.setattr(tau_trace, "_pp_rank", lambda r=rank: r)
            a.worker.execute_model(a.sched)
    root = Path(__file__).resolve().parents[4]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tools/tau_batch_run.py"),
            "check-trace",
            str(a.trace),
            "--pp",
            "2",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["passed"]
    assert report["events"]["compute"] == 4
    assert report["incomplete_forwards"] == 0
