# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stdlib runner tests: no vLLM import or accelerator required."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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


def setup_run(tmp_path, monkeypatch, mode="smoke"):
    for key in runner.serve_settings():
        monkeypatch.delenv(key, raising=False)
    run = tmp_path / "run"
    run.mkdir()
    manifest = dict(
        model="/models/model with spaces",
        trace=str(run / "trace.jsonl"),
        settings=runner.serve_settings(),
    )
    runner.write_json(run / "run.json", manifest)
    dataset = tmp_path / "sharegpt.json"
    dataset.write_text("[]")
    args = SimpleNamespace(
        run_dir=str(run),
        mode=mode,
        num_prompts=None,
        output_len=None,
        concurrency=32,
        request_rate=float("inf"),
        base_url=None,
        dataset=str(dataset),
        download=False,
        ready_timeout=1,
        dry_run=False,
        seed=0,
    )
    monkeypatch.setattr(runner, "wait_ready", lambda *args: None)
    return run, manifest, args


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


def fake_benchmark(monkeypatch, manifest, *, stage_only=False, completed_delta=0):
    calls = []

    def launch(command, log):
        def option(name):
            return command[command.index(name) + 1]

        count = int(option("--num-prompts"))
        calls.append(command)
        runner.write_json(
            Path(option("--result-dir")) / option("--result-filename"),
            {"completed": count + completed_delta},
        )
        events = records(1 + 2 * (len(calls) - 1))
        if stage_only:
            for event in events:
                if event["event"] == "compute":
                    event["event"] = "stage"
        append(Path(manifest["trace"]), events)
        return 0

    monkeypatch.setattr(runner, "run_logged", launch)
    return calls


def test_collect_warmup_and_measured_ranges_are_separate(tmp_path, monkeypatch):
    run, manifest, args = setup_run(tmp_path, monkeypatch, "collect")
    calls = fake_benchmark(monkeypatch, manifest)
    assert runner.bench(args) == 0
    assert len(calls) == 2
    assert calls[0][calls[0].index("--num-prompts") + 1] == "32"
    assert calls[1][calls[1].index("--num-prompts") + 1] == "1000"
    assert manifest["model"] in calls[1]
    assert "http://127.0.0.1:8000" in calls[1]
    target = next((run / "bench").iterdir())
    warmup = json.loads((target / "warmup_trace_check.json").read_text())
    result = json.loads((target / "result_trace_check.json").read_text())
    assert result["trace_start_offset"] == warmup["trace_end_offset"]
    assert result["events"]["compute"] == 4
    assert len(json.loads((target / "dataset.json").read_text())["sha256"]) == 64


@pytest.mark.parametrize("stage_only,delta", [(True, 0), (False, -1)])
def test_collect_stops_before_measured_run_on_failed_warmup(
    tmp_path, monkeypatch, stage_only, delta
):
    run, manifest, args = setup_run(tmp_path, monkeypatch, "collect")
    calls = fake_benchmark(
        monkeypatch, manifest, stage_only=stage_only, completed_delta=delta
    )
    with pytest.raises(ValueError, match="Trace is not ready"):
        runner.bench(args)
    assert len(calls) == 1
    report = next((run / "bench").glob("*/warmup_trace_check.json"))
    assert not json.loads(report.read_text())["passed"]


@pytest.mark.parametrize("key,value", [("TP", "2"), ("MIN_WAITING", "4")])
def test_unsupported_collection_settings_fail(tmp_path, monkeypatch, key, value):
    run, manifest, args = setup_run(tmp_path, monkeypatch)
    manifest["settings"][key] = value
    runner.write_json(run / "run.json", manifest)
    with pytest.raises(ValueError):
        runner.bench(args)
    assert not (run / "bench").exists()


def test_server_dry_run_and_existing_run_protection(tmp_path, monkeypatch, capsys):
    run, _, _ = setup_run(tmp_path, monkeypatch)
    args = SimpleNamespace(model="/models/a b", run_dir=str(run), dry_run=True)
    assert runner.serve(args) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["settings"]["MIN_WAITING"] == "0"
    assert "/models/a b" in preview["command"]
    args.dry_run = False
    with pytest.raises(ValueError, match="NEW RUN_DIR"):
        runner.serve(args)


def test_logging_preserves_nonzero_exit(tmp_path, capsys):
    log = tmp_path / "child.log"
    assert (
        runner.run_logged([sys.executable, "-c", "print('output'); exit(7)"], log) == 7
    )
    assert log.read_text() == "output\n"
    assert "output" in capsys.readouterr().out


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
