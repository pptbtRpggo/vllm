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


@pytest.mark.parametrize("damage", ["stage_only", "short_count"])
def test_bench_without_trace_skips_coverage_but_checks_completed(bench_setup, damage):
    setup = bench_setup
    manifest_path = setup["run"] / "server_meta.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["trace"] = None
    manifest_path.write_text(json.dumps(manifest))
    setup["env"]["TAU_TEST_DAMAGE"] = damage
    target = setup["tmp"] / "without_trace"
    result = launch(setup, "--result-dir", target)
    assert (result.returncode == 0) == (damage != "short_count"), result.stderr
    assert not json.loads((target / "summary.json").read_text())["trace"]["enabled"]
    assert (
        json.loads((target / "bench_meta.json").read_text())["phases"]["result"][
            "trace"
        ]
        is None
    )
    assert '"trace_check": "disabled"' in result.stdout


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
    (run / "server_meta.json").write_text(json.dumps(manifest))
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
if "--slo-config" in args:
    calls.with_suffix(".slo.json").write_text(Path(option("--slo-config")).read_text())
with calls.open("a") as stream:
    stream.write(json.dumps(args) + "\\n")
step = len(calls.read_text().splitlines())
Path(option("--request-output")).write_text('{}\\n')
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
if damage == "replace":
    replacement = trace.with_suffix(".replacement")
    replacement.write_text(trace.read_text())
    replacement.replace(trace)
if damage == "truncate":
    trace.write_text("")
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
    summary = json.loads((target / "summary.json").read_text())
    w = summary["warmup"]["trace"]
    r = summary["trace"]
    assert w["passed"] and r["passed"]
    assert w["trace_end_offset"] == r["trace_start_offset"]
    assert r["events"] == {"emit": 2, "compute": 4}
    assert json.loads((target / "bench_meta.json").read_text())["phases"]["result"][
        "command"
    ] == [
        "vllm",
        *measured,
    ]
    assert {p.name for p in target.iterdir()} == {
        "bench_meta.json",
        "summary.json",
        "requests.jsonl",
    }
    assert (
        len(json.loads((target / "bench_meta.json").read_text())["dataset"]["sha256"])
        == 64
    )
    assert not Path(option(measured, "--result-dir")).exists()
    assert not (s["tmp"] / "SHOULD_NOT_EXIST").exists()


def test_smoke_defaults_and_explicit_overrides(bench_setup):
    s = bench_setup
    s["env"].update(IGNORE_EOS="0", NUM_PROMPTS="40", CONCURRENCY="9")
    result = launch(s, "--num-prompts", "16", "--concurrency", "3")
    assert result.returncode == 0, result.stdout + result.stderr
    (command,) = calls(s)
    assert option(command, "--num-prompts") == "16"
    assert option(command, "--max-concurrency") == "3"
    assert "CONCURRENCY（最多未完成请求数） = 3" in result.stdout
    assert "NUM_PROMPTS（正式请求数） = 16" in result.stdout
    assert "0（smoke 跳过预热；配置值 32）" in result.stdout
    assert "SLO_ENABLED = False" in result.stdout
    assert "[bench result 正式压测：16 个请求]" in result.stdout
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
    assert (
        "result" not in json.loads((target / "bench_meta.json").read_text())["phases"]
    )
    assert (target / "error.log").is_file()
    if damage != "exit":
        assert not json.loads((target / "summary.json").read_text())["warmup"]["trace"][
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
    manifest = bench_setup["run"] / "server_meta.json"
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
    metadata = json.loads((target / "bench_meta.json").read_text())["slo"]
    assert metadata["config"] == json.loads(config.read_text())
    for command in calls(s):
        snapshot = Path(option(command, "--slo-config"))
        assert snapshot.name == "slo_config.json"
        assert snapshot.parent != target
        assert not snapshot.exists()  # Only the metadata snapshot persists.
        assert option(command, "--slo-seed") == "7"
    assert metadata["path"] == str(config)
    assert metadata["slo_seed"] == 7


@pytest.mark.parametrize("damage", ["replace", "truncate"])
def test_trace_damage_preserves_measurements_and_failure_log(bench_setup, damage):
    bench_setup["env"]["TAU_TEST_DAMAGE"] = damage
    target = bench_setup["tmp"] / "damaged"
    result = launch(bench_setup, "--result-dir", target)
    assert result.returncode == 2
    summary = json.loads((target / "summary.json").read_text())
    assert summary["completed"] == 8
    assert not summary["trace"]["passed"]
    assert damage in summary["trace"]["error"].lower()
    assert (target / "requests.jsonl").exists()
    assert (target / "error.log").is_file()


def test_readiness_failure_keeps_metadata_and_error_log(bench_setup):
    target = bench_setup["tmp"] / "not_ready"
    # The local fixture returns /v1/models only; change expected model.
    path = bench_setup["run"] / "server_meta.json"
    meta = json.loads(path.read_text())
    meta["model"] = "wrong-model"
    path.write_text(json.dumps(meta))
    result = launch(bench_setup, "--result-dir", target)
    assert result.returncode == 2
    assert (target / "bench_meta.json").is_file()
    assert "does not match" in (target / "error.log").read_text()
    assert not bench_setup["calls"].exists()


@pytest.mark.parametrize("sampled", [False, True])
def test_single_bench_config_includes_slo_and_workload(bench_setup, sampled):
    import yaml

    s = bench_setup
    config = yaml.safe_load((ROOT / "configs/bench.yaml").read_text())
    config.update(RUN_DIR=str(s["run"]), DATASET=str(s["dataset"]), MODE="collect")
    config["MODES"]["collect"] = dict(NUM_PROMPTS=6, OUTPUT_LEN=24, CONCURRENCY=3)
    config["WARMUP_REQUESTS"] = 2
    config["SLO"]["enabled"] = True
    config["SLO"]["profiles"]["tight"]["ttft_slo_ms"] = 750
    if sampled:
        config["SLO"]["profiles"]["sampled"] = {
            "ttft_slo_ms": {"distribution": "uniform", "min": 1000, "max": 3000},
            "tpot_slo_ms": {
                "distribution": "normal",
                "mean": 100,
                "std": 10,
                "min": 80,
                "max": 120,
            },
        }
        config["SLO"]["ratios"] = dict(tight=0.2, loose=0.3, sampled=0.5)
    path = s["tmp"] / "bench.yaml"
    path.write_text(yaml.safe_dump(config))
    result = subprocess.run(
        ["bash", str(ROOT / "bench_tau.sh"), "--config", str(path)],
        env=s["env"],
        cwd=s["tmp"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    warmup, formal = calls(s)
    assert "Tau bench 生效参数" in result.stdout
    assert f"CONFIG = {path}" in result.stdout
    assert '"ttft_slo_ms": 750' in result.stdout
    assert '"ratios"' in result.stdout
    assert "SLO_ENABLED = True" in result.stdout
    assert "[bench warmup 预热：2 个请求，不计入正式结果]" in result.stdout
    assert "[bench result 正式压测：6 个请求]" in result.stdout
    assert result.stdout.count("[完整启动命令]") == 2
    assert result.stdout.index("Tau bench 生效参数") < result.stdout.index(
        "fake benchmark"
    )
    assert option(warmup, "--num-prompts") == "2"
    assert option(formal, "--num-prompts") == "6"
    assert option(formal, "--sharegpt-output-len") == "24"
    assert option(formal, "--max-concurrency") == "3"
    payload = json.loads(s["calls"].with_suffix(".slo.json").read_text())
    assert payload == {k: v for k, v in config["SLO"].items() if k != "enabled"}
    (target,) = (s["run"] / "bench").iterdir()
    meta = json.loads((target / "bench_meta.json").read_text())
    assert meta["launch_config"]["path"] == str(path)
    assert meta["slo"]["section"] == "SLO"
    assert meta["slo"]["config"] == payload
    assert {p.name for p in target.iterdir()} == {
        "bench_meta.json",
        "requests.jsonl",
        "summary.json",
    }


def test_native_scheduler_bench_does_not_require_tau_settings(bench_setup):
    s = bench_setup
    path = s["run"] / "server_meta.json"
    metadata = json.loads(path.read_text())
    metadata["trace"] = None
    metadata["settings"].update(SCHEDULER="default", TP="2")
    del metadata["settings"]["MIN_WAITING"]
    path.write_text(json.dumps(metadata))
    target = s["tmp"] / "native_bench"
    result = launch(s, "--result-dir", target)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SCHEDULER = default" in result.stdout
    assert (
        json.loads((target / "summary.json").read_text())["trace"]["enabled"] is False
    )
