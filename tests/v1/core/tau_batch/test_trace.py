# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest

from tests.v1.core.tau_batch.test_batch_queue import _QueueCore
from tests.v1.core.tau_batch.test_scheduler import (
    _add_requests,
    _sampled,
    _tau_scheduler,
)
from vllm.v1.core.sched.tau_batch.plot_trace import pipeline_to_html, spans_to_html
from vllm.v1.core.sched.tau_batch.trace import (
    JsonlTracer,
    load_events,
    pair_forwards,
    pipeline_cells,
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
    assert kinds.count("task") == 2
    task0 = next(e for e in events if e["event"] == "task" and e["fwd_id"] == 1)
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


def test_pipeline_cells_place_rank1_after_rank0() -> None:
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
        (1, 30, 45),
    ]
    assert cells[0].job.endswith("B0_pre")
    html = pipeline_to_html(cells)
    assert "PP0" in html
    assert "PP1" in html
