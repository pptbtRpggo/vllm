# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stdlib runner tests: no vLLM import or accelerator required."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "tau_run", ROOT / "tools/tau_batch_run.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def records(start=1):
    result = []
    for fid, phase in enumerate(("prefill", "decode"), start):
        features = dict(
            phase=phase,
            n=2,
            s_sum=16,
            s_max=8,
            seq_lens=[8, 8],
            tokens=16 if phase == "prefill" else 2,
        )
        result.append(dict(event="emit", fwd_id=fid, **features))
        for rank in range(2):
            result.append(
                dict(
                    event="compute",
                    fwd_id=fid,
                    pp_rank=rank,
                    start_ts_ns=10,
                    end_ts_ns=20,
                    **features,
                )
            )
    return result


def append(path, events):
    with path.open("a") as stream:
        for event in events:
            stream.write(json.dumps(event) + "\n")


def test_trace_complete_and_byte_range(tmp_path):
    path = tmp_path / "trace.jsonl"
    append(path, records())
    offset = path.stat().st_size
    append(path, records(3))
    end = path.stat().st_size
    append(path, [{"event": "emit", "fwd_id": 5}])
    report = runner.check_trace(path, 2, offset, end)
    assert report["passed"]
    assert report["events"] == {"emit": 2, "compute": 4}
    assert report["trace_end_offset"] == end
    assert not runner.check_trace(path, 2)["passed"]


@pytest.mark.parametrize(
    "damage",
    [
        "stage_only",
        "rank",
        "one_missing",
        "duplicate",
        "bad_features",
        "mismatch",
        "orphan",
        "malformed",
    ],
)
def test_trace_rejects_incomplete_or_invalid_labels(tmp_path, damage):
    path = tmp_path / "trace.jsonl"
    events = records()
    if damage == "stage_only":
        for e in events:
            if e["event"] == "compute":
                e["event"] = "stage"
    elif damage == "rank":
        events[1]["pp_rank"] = -1
    elif damage == "one_missing":
        events.pop(1)
    elif damage == "duplicate":
        events.append(events[1])
    elif damage == "bad_features":
        events[1]["seq_lens"] = ["bad", 8]
    elif damage == "mismatch":
        events[1]["tokens"] = 1
    elif damage == "orphan":
        events[1]["fwd_id"] = 99
    else:
        events.append([])
    append(path, events)
    assert not runner.check_trace(path, 2)["passed"]


def test_empty_and_partial_trace_fail(tmp_path):
    path = tmp_path / "trace.jsonl"
    assert not runner.check_trace(path, 2)["passed"]
    path.write_text('{"event":')
    assert runner.check_trace(path, 2)["malformed"] == 1
    with pytest.raises(ValueError):
        runner.check_trace(path, 0)


def test_shell_wrappers_help_from_other_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHON", sys.executable)
    for script in ("serve_tau.sh", "bench_tau.sh"):
        result = subprocess.run(
            ["bash", str(ROOT / script), "--help"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
