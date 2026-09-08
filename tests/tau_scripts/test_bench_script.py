# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real bench shell + fake CLI + local models endpoint; no accelerator needed."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def bench_setup(tmp_path):
    run = tmp_path / "service run"
    run.mkdir()
    model = '/models/model $(touch SHOULD_NOT_EXIST) "quoted"'
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            data = json.dumps({"data": [{"id": model}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    trace = tmp_path / "trace.jsonl"
    trace.write_text("old trace outside measured range\n")
    manifest = dict(
        model=model,
        trace=str(trace),
        settings=dict(
            HOST="127.0.0.1",
            PORT=str(server.server_port),
            TP="1",
            PP="2",
            MIN_WAITING="0",
        ),
    )
    (run / "run.json").write_text(json.dumps(manifest))
    dataset = tmp_path / "data $(touch SHOULD_NOT_EXIST).json"
    dataset.write_text("[]")
    calls = tmp_path / "calls.jsonl"
    binary = tmp_path / "bin"
    binary.mkdir()
    fake = binary / "vllm"
    fake.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, signal, sys
from pathlib import Path
args = sys.argv[1:]
def option(name):
    return args[args.index(name) + 1]
calls = Path(os.environ["TAU_TEST_CALLS"])
with calls.open("a") as stream:
    stream.write(json.dumps(args) + "\\n")
step = len(calls.read_text().splitlines())
print("fake benchmark", flush=True)
damage = os.environ.get("TAU_TEST_DAMAGE", "")
if damage == "exit":
    sys.exit(7)
if damage == "block":
    def stop(signum, frame):
        calls.with_suffix(".stopped").write_text(str(signum))
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    calls.with_suffix(".ready").write_text(str(os.getpid()))
    while True:
        signal.pause()
trace = Path(os.environ["TAU_TEST_TRACE"])
with trace.open("a") as stream:
    for i, phase in enumerate(("prefill", "decode")):
        fid = step * 2 + i
        features = dict(phase=phase, n=2, s_sum=16, s_max=8, seq_lens=[8,8],
                        tokens=16 if phase == "prefill" else 2)
        stream.write(json.dumps(dict(event="emit", fwd_id=fid, **features)) + "\\n")
        for rank in (0, 1):
            event = "stage" if damage == "stage_only" else "compute"
            stream.write(json.dumps(dict(event=event, fwd_id=fid, pp_rank=rank,
                         start_ts_ns=100, end_ts_ns=200, **features)) + "\\n")
count = int(option("--num-prompts")) - (1 if damage == "short_count" else 0)
target = Path(option("--result-dir")) / option("--result-filename")
target.write_text(json.dumps(dict(completed=count)))
"""
    )
    fake.chmod(0o755)
    env = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "LANG", "LC_ALL")
        if key in os.environ
    }
    env.update(
        PYTHON=sys.executable,
        PATH=str(binary) + os.pathsep + env.get("PATH", ""),
        TMPDIR=str(tmp_path),
        DATASET=str(dataset),
        TAU_TEST_CALLS=str(calls),
        TAU_TEST_TRACE=str(trace),
    )
    try:
        yield dict(
            run=run,
            model=model,
            trace=trace,
            dataset=dataset,
            calls=calls,
            env=env,
            requests=requests,
            tmp=tmp_path,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def launch(setup, *args, legacy=False):
    prefix = (
        [sys.executable, str(ROOT / "tools/tau_batch_run.py"), "bench"]
        if legacy
        else ["bash", str(ROOT / "bench_tau.sh")]
    )
    return subprocess.run(
        prefix + [str(setup["run"]), *map(str, args)],
        env=setup["env"],
        cwd=setup["tmp"],
        capture_output=True,
        text=True,
        timeout=30,
    )


def calls(setup):
    return [json.loads(line) for line in setup["calls"].read_text().splitlines()]


def option(command, name):
    return command[command.index(name) + 1]


@pytest.mark.parametrize("legacy", [False, True])
def test_traffic_parameters_and_range_isolation(bench_setup, legacy):
    s = bench_setup
    s["env"].update(REQUEST_RATE="5", BURSTINESS="inf", CONCURRENCY="8", SEED="7")
    target = s["tmp"] / "chosen result"
    result = launch(s, "--mode", "collect", "--result-dir", target, legacy=legacy)
    assert result.returncode == 0, result.stdout + result.stderr
    warmup, measured = calls(s)
    assert option(warmup, "--num-prompts") == "32"
    assert option(measured, "--num-prompts") == "1000"
    for command in (warmup, measured):
        assert command[:2] == ["bench", "serve"]
        assert option(command, "--model") == s["model"]
        assert option(command, "--dataset-path") == str(s["dataset"])
        assert option(command, "--endpoint") == "/v1/completions"
        assert option(command, "--request-rate") == "5"
        assert option(command, "--burstiness") == "inf"
        assert option(command, "--max-concurrency") == "8"
        assert option(command, "--seed") == "7"
        assert option(command, "--sharegpt-output-len") == "256"
        assert "--ignore-eos" in command
    w = json.loads((target / "warmup_trace_check.json").read_text())
    r = json.loads((target / "result_trace_check.json").read_text())
    assert w["passed"] and r["passed"]
    assert w["trace_end_offset"] == r["trace_start_offset"]
    assert r["events"] == {"emit": 2, "compute": 4}
    assert json.loads((target / "result_command.json").read_text())["command"] == [
        "vllm",
        *measured,
    ]
    assert "fake benchmark" in (target / "result.log").read_text()
    assert len(json.loads((target / "dataset.json").read_text())["sha256"]) == 64
    assert not (s["tmp"] / "SHOULD_NOT_EXIST").exists()


def test_smoke_defaults_and_explicit_overrides(bench_setup):
    s = bench_setup
    s["env"].update(IGNORE_EOS="0", NUM_PROMPTS="40", CONCURRENCY="9")
    result = launch(s, "--num-prompts", "16", "--concurrency", "3")
    assert result.returncode == 0, result.stdout + result.stderr
    (command,) = calls(s)
    assert option(command, "--num-prompts") == "16"
    assert option(command, "--max-concurrency") == "3"
    assert option(command, "--sharegpt-output-len") == "16"
    assert "--ignore-eos" not in command


def test_dry_run_does_not_send_or_download(bench_setup):
    s = bench_setup
    missing = s["tmp"] / "missing.json"
    result = launch(
        s, "--mode", "collect", "--dry-run", "--download", "--dataset", missing
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("vllm bench serve") == 2
    assert not s["calls"].exists()
    assert not missing.exists()
    assert not (s["run"] / "bench").exists()
    assert s["requests"] == []
    assert not list(s["tmp"].glob("tau_bench.*"))


@pytest.mark.parametrize("damage", ["stage_only", "short_count", "exit"])
def test_failed_warmup_never_starts_measured_requests(bench_setup, damage):
    s = bench_setup
    s["env"]["TAU_TEST_DAMAGE"] = damage
    result = launch(s, "--mode", "collect")
    assert result.returncode == (7 if damage == "exit" else 2)
    assert len(calls(s)) == 1
    (target,) = (s["run"] / "bench").iterdir()
    assert not (target / "result_command.json").exists()
    if damage != "exit":
        assert not json.loads((target / "warmup_trace_check.json").read_text())[
            "passed"
        ]


def test_existing_result_is_not_overwritten(bench_setup):
    target = bench_setup["tmp"] / "existing"
    target.mkdir()
    marker = target / "marker"
    marker.write_text("keep")
    result = launch(bench_setup, "--result-dir", target)
    assert result.returncode == 2
    assert marker.read_text() == "keep"
    assert not bench_setup["calls"].exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("REQUEST_RATE", "0"),
        ("BURSTINESS", "nan"),
        ("CONCURRENCY", "0"),
        ("IGNORE_EOS", "maybe"),
    ],
)
def test_invalid_traffic_config_fails_before_sending(bench_setup, key, value):
    bench_setup["env"][key] = value
    result = launch(bench_setup)
    assert result.returncode == 2
    assert not bench_setup["calls"].exists()


@pytest.mark.parametrize("key,value", [("TP", "2"), ("MIN_WAITING", "4")])
def test_unsupported_service_config_fails_before_sending(bench_setup, key, value):
    manifest = bench_setup["run"] / "run.json"
    config = json.loads(manifest.read_text())
    config["settings"][key] = value
    manifest.write_text(json.dumps(config))
    result = launch(bench_setup)
    assert result.returncode == 2
    assert not bench_setup["calls"].exists()


def test_stop_signal_stops_client_and_prevents_next_phase(bench_setup):
    s = bench_setup
    s["env"]["TAU_TEST_DAMAGE"] = "block"
    process = subprocess.Popen(
        ["bash", str(ROOT / "bench_tau.sh"), str(s["run"]), "--mode", "collect"],
        env=s["env"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    child_pid = None
    try:
        deadline = time.monotonic() + 15
        ready = s["calls"].with_suffix(".ready")
        while not ready.exists():
            if process.poll() is not None or time.monotonic() >= deadline:
                pytest.fail("Fake benchmark did not become ready")
            time.sleep(0.05)
        child_pid = int(ready.read_text())
        process.terminate()
        stdout, _ = process.communicate(timeout=10)
        assert process.returncode == 130, stdout
        assert s["calls"].with_suffix(".stopped").read_text() == str(signal.SIGTERM)
        assert len(calls(s)) == 1
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        child_pid = None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if child_pid is not None:
            with suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)


@pytest.mark.parametrize("legacy", [False, True])
def test_slo_config_snapshot_and_seed_reach_both_phases(bench_setup, legacy):
    s = bench_setup
    config = s["tmp"] / "slo groups.json"
    config.write_text(
        json.dumps(
            {
                "profiles": {"tight": {"ttft_slo_ms": 1000, "tpot_slo_ms": 50}},
                "ratios": {"tight": 1},
            }
        )
    )
    target = s["tmp"] / "slo result"
    result = launch(
        s,
        "--mode",
        "collect",
        "--result-dir",
        target,
        "--slo-config",
        config,
        "--seed",
        "7",
        legacy=legacy,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    snapshot = target / "slo_config.json"
    assert snapshot.read_text() == config.read_text()
    for command in calls(s):
        assert option(command, "--slo-config") == str(snapshot)
        assert option(command, "--slo-seed") == "7"
    metadata = json.loads((target / "slo_config_source.json").read_text())
    assert metadata["path"] == str(config)
    assert metadata["slo_seed"] == 7
