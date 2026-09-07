# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.v1.core.tau_batch.test_batch_queue import _QueueCore
from tests.v1.core.tau_batch.test_scheduler import (
    _add_requests,
    _sampled,
    _tau_scheduler,
)
from vllm.v1.core.sched.tau_batch import trace as tau_trace
from vllm.v1.core.sched.tau_batch.plot_trace import pipeline_to_html, spans_to_html
from vllm.v1.core.sched.tau_batch.trace import (
    JsonlTracer,
    load_events,
    pair_forwards,
    pipeline_cells,
    record_worker_phase,
    trace_worker_phase,
)

pytestmark = pytest.mark.cpu_test


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_trace_writes_wave_emit_done(tmp_path: Path) -> None:
    path = tmp_path / "tau.jsonl"
    sched = _tau_scheduler(tau_batch_trace=str(path))
    _add_requests(sched)
    pre0 = sched.schedule()
    pre1 = sched.schedule()
    assert pre0.tau_fwd_id == 1
    assert pre1.tau_fwd_id == 2
    sched.update_from_output(pre0, _sampled(pre0))
    sched.update_from_output(pre1, _sampled(pre1))
    events = _events(path)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "meta"
    assert "wave_plan" in kinds
    assert kinds.count("emit") == 2
    assert kinds.count("done") == 2
    plan = next(e for e in events if e["event"] == "wave_plan")
    assert plan["batches"] == [["r0", "r1"], ["r2", "r3"]]
    assert len(plan["requests"]) == 4
    assert {r["req_id"] for r in plan["requests"]} == {"r0", "r1", "r2", "r3"}
    assert all("wait_ms" in r and "ttft_slack_ms" in r for r in plan["requests"])
    emit0 = next(e for e in events if e["event"] == "emit" and e["fwd_id"] == 1)
    assert emit0["n"] == 2
    assert emit0["s_max"] == 8
    assert emit0["s_sum"] == 16
    assert emit0["tokens"] == 16
    assert emit0["pp_size"] == 2
    assert emit0["phase"] == "prefill"
    assert "task" not in kinds
    task0 = next(e for e in events if e["event"] == "done" and e["fwd_id"] == 1)
    assert task0["n"] == 2
    assert task0["duration_ms"] >= 0
    spans = pair_forwards(events)
    assert len(spans) == 2
    assert spans[0].job.endswith("B0_pre")
    assert spans[0].req_ids == ("r0", "r1")
    assert spans[0].done_mono_ns >= spans[0].emit_mono_ns


def test_trace_not_created_until_first_write(tmp_path: Path) -> None:
    path = tmp_path / "lazy.jsonl"
    sched = _tau_scheduler(tau_batch_trace=str(path))
    assert sched._tracer is not None
    assert not path.exists()
    _add_requests(sched)
    assert not path.exists()
    sched.schedule()
    assert path.exists()
    assert _events(path)[0]["event"] == "meta"


def test_trace_recreates_after_delete(tmp_path: Path) -> None:
    path = tmp_path / "rotate.jsonl"
    sched = _tau_scheduler(tau_batch_trace=str(path))
    _add_requests(sched)
    first = sched.schedule()
    assert path.exists()
    path.unlink()
    sched.update_from_output(first, _sampled(first))
    assert path.exists()
    kinds = [e["event"] for e in _events(path)]
    assert kinds[0] == "meta"
    assert "done" in kinds


def _reset_worker_tracer() -> None:
    tau_trace._worker_tracer = None
    tau_trace._worker_tracer_ready = False


def test_worker_phases_record_recv_compute_send(tmp_path: Path) -> None:
    path = tmp_path / "phases.jsonl"
    cfg = SimpleNamespace(scheduler_config=SimpleNamespace(tau_batch_trace=str(path)))
    _reset_worker_tracer()
    try:
        start = time.time_ns()
        record_worker_phase(
            cfg,
            kind="recv",
            fwd_id=1,
            start_ts_ns=start,
            req_ids=["r0"],
            sync_end=False,
        )
        record_worker_phase(
            cfg,
            kind="compute",
            fwd_id=1,
            start_ts_ns=start,
            req_ids=["r0"],
            sync_end=False,
        )
        record_worker_phase(
            cfg,
            kind="send",
            fwd_id=1,
            start_ts_ns=start,
            req_ids=["r0"],
            sync_end=False,
        )
        kinds = [e["event"] for e in _events(path)]
        assert kinds == ["recv", "compute", "send"]
        recv = _events(path)[0]
        assert recv["fwd_id"] == 1
        assert recv["end_ts_ns"] >= recv["start_ts_ns"]
    finally:
        _reset_worker_tracer()


def test_trace_worker_phase_skips_without_fwd_id(tmp_path: Path) -> None:
    path = tmp_path / "skip.jsonl"
    cfg = SimpleNamespace(scheduler_config=SimpleNamespace(tau_batch_trace=str(path)))
    _reset_worker_tracer()
    try:
        out = SimpleNamespace(tau_fwd_id=None, num_scheduled_tokens={"r0": 1})
        with trace_worker_phase(cfg, out, "recv"):
            pass
        assert not path.exists()
        out.tau_fwd_id = 3
        with trace_worker_phase(cfg, out, "compute", sync_end=False):
            pass
        events = _events(path)
        assert [e["event"] for e in events] == ["compute"]
        assert events[0]["fwd_id"] == 3
        assert events[0]["req_ids"] == ["r0"]
    finally:
        _reset_worker_tracer()


def test_worker_reopens_after_driver_recreates(tmp_path: Path) -> None:
    """rm the JSONL, then the driver recreates it. Worker must follow."""
    path = tmp_path / "shared.jsonl"
    worker = JsonlTracer(str(path), write_meta=False)
    driver = JsonlTracer(str(path), write_meta=True)
    worker.record("stage", fwd_id=1, pp_rank=0)
    driver.record("emit", fwd_id=1)
    path.unlink()
    driver.record("emit", fwd_id=2)
    worker.record("stage", fwd_id=2, pp_rank=1)
    kinds = [e["event"] for e in _events(path)]
    assert kinds[0] == "meta"
    assert kinds.count("emit") == 1
    assert kinds.count("stage") == 1
    assert next(e for e in _events(path) if e["event"] == "stage")["fwd_id"] == 2


def test_trace_off_writes_nothing(tmp_path: Path) -> None:
    sched = _tau_scheduler()
    _add_requests(sched)
    sched.schedule()
    assert list(tmp_path.iterdir()) == []
    assert sched._tracer is None


def test_trace_queue_events_from_batch_queue(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    sched = _tau_scheduler(tau_batch_trace=str(path))
    _add_requests(sched, n=4, max_tokens=2)
    core = _QueueCore(sched, queue_size=2)
    core.step_with_batch_queue()
    core.step_with_batch_queue()
    events = _events(path)
    kinds = [e["event"] for e in events]
    assert "enqueue" in kinds
    assert "dequeue" in kinds
    spans = pair_forwards(load_events(path))
    assert spans
    html = spans_to_html(spans)
    assert "B0_pre" in html


def test_pipeline_cells_preserve_measured_stage_windows() -> None:
    events = [
        {
            "event": "emit",
            "fwd_id": 1,
            "wave_id": 0,
            "batch_idx": 0,
            "phase": "prefill",
            "req_ids": ["r0"],
            "mono_ns": 0,
            "ts_ns": 1000,
        },
        {
            "event": "done",
            "fwd_id": 1,
            "wave_id": 0,
            "batch_idx": 0,
            "phase": "prefill",
            "req_ids": ["r0"],
            "mono_ns": 50,
            "ts_ns": 1050,
        },
        {
            "event": "stage",
            "fwd_id": 1,
            "pp_rank": 0,
            "start_ts_ns": 10,
            "end_ts_ns": 30,
            "ts_ns": 30,
        },
        {
            "event": "stage",
            "fwd_id": 1,
            "pp_rank": 1,
            "start_ts_ns": 11,
            "end_ts_ns": 45,
            "ts_ns": 45,
        },
    ]
    cells = pipeline_cells(events)
    assert [(c.pp_rank, c.start_ts_ns, c.end_ts_ns) for c in cells] == [
        (0, 10, 30),
        (1, 11, 45),
    ]
    assert cells[0].job.endswith("B0_pre")
    html = pipeline_to_html(cells)
    assert "PP0" in html
    assert "PP1" in html


@pytest.mark.parametrize("compute_first", [True, False])
def test_plot_prefers_compute_without_moving_timestamps(compute_first):
    stage = dict(event="stage", fwd_id=1, pp_rank=1, start_ts_ns=1, end_ts_ns=100)
    compute = dict(event="compute", fwd_id=1, pp_rank=1, start_ts_ns=25, end_ts_ns=75)
    events = [compute, stage] if compute_first else [stage, compute]
    # Iterators are supported too; no event stream is consumed twice.
    cells = pipeline_cells(iter(events))
    assert len(cells) == 1
    assert (cells[0].start_ts_ns, cells[0].end_ts_ns) == (25, 75)
    assert cells[0].kind == "compute"


def test_stage_envelope_does_not_add_a_device_barrier(tmp_path, monkeypatch):
    cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(tau_batch_trace=str(tmp_path / "stage.jsonl"))
    )
    _reset_worker_tracer()

    def unexpected_sync():
        raise AssertionError("stage envelope must not synchronize the device")

    monkeypatch.setattr(tau_trace, "_sync_compute_device", unexpected_sync)
    try:
        tau_trace.record_worker_stage(
            cfg, fwd_id=1, start_ts_ns=time.time_ns(), req_ids=["a"]
        )
        assert _events(tmp_path / "stage.jsonl")[0]["event"] == "stage"
    finally:
        _reset_worker_tracer()


def test_metadata_after_worker_open_and_ids_survive_rotation(tmp_path):
    path = tmp_path / "trace.jsonl"
    worker = JsonlTracer(str(path), write_meta=False)
    worker.record("stage", fwd_id=1)
    driver = JsonlTracer(str(path), metadata={"hidden_size": 768})
    assert driver.next_fwd_id() == 1
    meta = next(e for e in _events(path) if e["event"] == "meta")
    assert meta["config"]["hidden_size"] == 768
    path.unlink()
    assert driver.next_fwd_id() == 2
    driver.close()
    worker.close()


def test_trace_off_does_not_build_latency_features(monkeypatch):
    sched = _tau_scheduler()
    _add_requests(sched, n=2)

    def unexpected_features(*args):
        raise AssertionError("trace features constructed while tracing is disabled")

    monkeypatch.setattr(sched, "_task_features", unexpected_features)
    out = sched.schedule()
    assert out.tau_task is None
    assert out.tau_fwd_id is None
    sched.update_from_output(out, _sampled(out))
    assert not sched._inflight


def test_decode_features_describe_context_including_input_token(tmp_path):
    path = tmp_path / "decode.jsonl"
    sched = _tau_scheduler(tau_batch_trace=str(path))
    _add_requests(sched, n=2)
    pre = sched.schedule()
    sched.update_from_output(pre, _sampled(pre))
    dec = sched.schedule()
    assert dec.tau_task["seq_lens"] == [9, 9]
    assert dec.tau_task["tokens"] == 2
    assert dec.tau_task["s_sum"] == 18
    events = _events(path)
    assert "hidden_size" in events[0]["config"]
    assert all("hidden_size" not in e for e in events if e["event"] == "emit")


def test_npu_sync_error_is_not_silently_ignored(monkeypatch):
    import torch

    def fail():
        raise RuntimeError("NPU sync failed")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(is_available=lambda: True, synchronize=fail),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="NPU sync failed"):
        tau_trace._sync_compute_device()


@pytest.mark.parametrize("failure", ["execution", "synchronization"])
def test_phase_failure_does_not_record_success(tmp_path, monkeypatch, failure):
    path = tmp_path / "failed.jsonl"
    cfg = SimpleNamespace(scheduler_config=SimpleNamespace(tau_batch_trace=str(path)))
    out = SimpleNamespace(tau_fwd_id=1, num_scheduled_tokens={"r": 1})
    _reset_worker_tracer()

    def fail():
        raise RuntimeError("failed")

    try:
        monkeypatch.setattr(tau_trace, "_sync_compute_device", fail)
        with (
            pytest.raises(RuntimeError, match="failed"),
            trace_worker_phase(cfg, out, "compute"),
        ):
            if failure == "execution":
                fail()
        assert not path.exists() or not _events(path)
    finally:
        if tau_trace._worker_tracer is not None:
            tau_trace._worker_tracer.close()
        _reset_worker_tracer()
