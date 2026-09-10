# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stdlib runner tests: no vLLM import or accelerator required."""

import importlib.util
import json
import os
import shutil
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


def test_summary_preserves_slo_statistics_and_separates_warmup(tmp_path, monkeypatch):
    run = tmp_path / "run"
    target = run / "bench"
    scratch = tmp_path / "scratch"
    target.mkdir(parents=True)
    scratch.mkdir()
    monkeypatch.setenv("BENCH_WORK_DIR", str(scratch))
    (run / "server_meta.json").write_text("{}")
    (target / "bench_meta.json").write_text(
        json.dumps(
            {
                "phases": {"result": {"expected": 4, "trace": None}},
            }
        )
    )
    warmup = {"request_goodput": 99, "completed": 10}
    (target / "summary.json").write_text(json.dumps({"warmup": warmup}))
    evaluation = {
        "attainment_rates": {"ttft": 0.75, "tpot": 0.5, "all": 0.25},
        "request_goodput": 0.5,
        "by_profile": {
            "tight": {
                "attainment_rates": {"ttft": 0.5, "tpot": 0.5, "all": 0},
                "request_goodput": 0,
            }
        },
    }
    (scratch / "result.json").write_text(
        json.dumps(
            {
                "completed": 4,
                "request_goodput": 0.5,
                "slo_evaluation": evaluation,
            }
        )
    )
    assert (
        runner.end_bench(
            runner.argparse.Namespace(
                run_dir=str(run),
                result_dir=str(target),
                name="result",
            )
        )
        == 0
    )
    summary = json.loads((target / "summary.json").read_text())
    assert summary["slo_evaluation"] == evaluation
    assert summary["request_goodput"] == 0.5
    assert summary["warmup"] == warmup


def test_latest_run_discovery_and_overrides(tmp_path, monkeypatch):
    root = tmp_path / "repo with spaces"
    (root / "tools").mkdir(parents=True)
    shutil.copy2(ROOT / "bench_tau.sh", root / "bench_tau.sh")
    shutil.copy2(ROOT / "tools/tau_batch_run.py", root / "tools/tau_batch_run.py")
    shutil.copy2(ROOT / "tools/tau_config.py", root / "tools/tau_config.py")
    shutil.copytree(ROOT / "configs", root / "configs")
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "capture", lambda command: "")
    settings = dict.fromkeys(runner.SERVE_SETTING_NAMES, "1")
    settings.update(
        PP="2", MIN_WAITING="0", HOST="127.0.0.1", PORT="8000", SCHEDULER="tau"
    )
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    latest = root / "output/latest"
    env = {key: os.environ[key] for key in ("PATH", "HOME") if key in os.environ}
    env.update(PYTHON=sys.executable, TMPDIR=str(tmp_path))

    def bench(*args, **overrides):
        return subprocess.run(
            ["bash", str(root / "bench_tau.sh"), *args, "--dry-run"],
            env={**env, **overrides},
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=10,
        )

    missing = bench()
    assert missing.returncode == 2
    assert "请先运行 serve_tau.sh" in missing.stderr
    first = tmp_path / "first run"
    for index, run in enumerate((first, tmp_path / "second run")):
        args = runner.argparse.Namespace(
            run_dir=str(run),
            trace=str(run / "trace.jsonl"),
            model=f"/models/model{index}",
            command=["vllm", "serve"],
            dry_run=True,
        )
        runner.prepare_serve(args)
        assert not run.exists()
        if index == 0:
            assert not latest.is_symlink()
        else:
            assert latest.resolve() == first
        args.dry_run = False
        runner.prepare_serve(args)
        assert latest.resolve() == run
        for mode in ("smoke", "collect"):
            result = bench("--mode", mode)
            assert result.returncode == 0, result.stderr
            assert f"--model /models/model{index}" in result.stdout
            assert not (run / "bench").exists()
    assert "--model /models/model0" in bench(RUN_DIR=str(first)).stdout
    explicit = bench(str(first), RUN_DIR=str(latest))
    assert explicit.returncode == 0, explicit.stderr
    assert "--model /models/model0" in explicit.stdout
    latest.unlink()
    latest.symlink_to(tmp_path / "missing")
    assert bench().returncode == 2


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
