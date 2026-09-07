# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the actual shell launcher using a fake vllm, without an NPU."""

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def launch_env(tmp_path):
    env = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL")
        if key in os.environ
    }
    for key in (
        "MODEL",
        "RUN_DIR",
        "RUN",
        "TRACE",
        "HOST",
        "PORT",
        "TP",
        "PP",
        "MAX_MODEL_LEN",
        "MAX_NUM_SEQS",
        "MAX_NUM_BATCHED_TOKENS",
        "MAX_REQS_PER_MB",
        "MAX_MICROBATCHES",
        "MIN_WAITING",
        "GPU_MEM",
        "ASCEND_RT_VISIBLE_DEVICES",
    ):
        env.pop(key, None)
    binary = tmp_path / "bin"
    binary.mkdir()
    fake = binary / "vllm"
    fake.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, signal, subprocess, sys
from pathlib import Path
record = Path(os.environ["TAU_TEST_RECORD"])
record.write_text(json.dumps(dict(pid=os.getpid(), argv=sys.argv[1:],
                                 pythonpath=os.environ.get("PYTHONPATH"))))
print("fake vllm output", flush=True)
if os.environ.get("TAU_TEST_MODE") != "signal":
    sys.exit(7)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
def stop(signum, frame):
    child.terminate()
    child.wait(timeout=5)
    record.with_suffix(".stopped").write_text(str(signum))
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
ready = json.dumps(dict(pid=os.getpid(), child=child.pid))
record.with_suffix(".ready").write_text(ready)
while True:
    signal.pause()
"""
    )
    fake.chmod(0o755)
    local = tmp_path / "local"
    local.mkdir()
    env.update(
        PYTHON=sys.executable,
        TMPDIR=str(local),
        PATH=str(binary) + os.pathsep + env.get("PATH", ""),
        TAU_TEST_RECORD=str(tmp_path / "record.json"),
    )
    return env


def launch(env, *args, cwd=None):
    return subprocess.run(
        ["bash", str(ROOT / "serve_tau.sh"), *map(str, args)],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_defaults_generate_independent_paths_without_side_effects(launch_env):
    previews = []
    for _ in range(2):
        result = launch(launch_env, "/models/example", "--dry-run")
        assert result.returncode == 0, result.stderr
        preview = json.loads(result.stdout)
        previews.append(preview)
        assert not Path(preview["run_dir"]).exists()
        assert not Path(preview["trace"]).exists()
        assert Path(preview["trace"]).parent == Path(launch_env["TMPDIR"]).resolve()
        assert preview["command"][:3] == ["vllm", "serve", "/models/example"]
        assert preview["settings"]["PP"] == "2"
        assert preview["settings"]["TP"] == "1"
        assert preview["settings"]["MAX_NUM_SEQS"] == "4"
        assert preview["settings"]["MAX_REQS_PER_MB"] == "4"
        assert preview["settings"]["MIN_WAITING"] == "0"
    assert previews[0]["run_dir"] != previews[1]["run_dir"]
    assert previews[0]["trace"] != previews[1]["trace"]
    assert not Path(launch_env["TAU_TEST_RECORD"]).exists()


def test_exact_arguments_manifest_logging_and_exit_status(launch_env, tmp_path):
    model = tmp_path / 'model $(touch SHOULD_NOT_EXIST) "quoted"'
    model.mkdir()
    run = tmp_path / "run with spaces"
    trace = tmp_path / "custom trace.jsonl"
    launch_env.update(
        MODEL="/ignored/model",
        RUN_DIR=str(tmp_path / "ignored_run"),
        TRACE=str(trace),
        MAX_NUM_SEQS="8",
        PORT="8123",
    )
    result = launch(launch_env, model.name, "--run-dir", run, cwd=tmp_path)
    assert result.returncode == 7, result.stderr
    manifest = json.loads((run / "run.json").read_text())
    record = json.loads(Path(launch_env["TAU_TEST_RECORD"]).read_text())
    assert manifest["model"] == str(model.resolve())
    assert manifest["trace"] == str(trace.resolve())
    assert manifest["command"][1:] == record["argv"]
    assert manifest["vllm_executable"] == str(tmp_path / "bin" / "vllm")
    assert manifest["settings"]["MAX_REQS_PER_MB"] == "8"
    assert manifest["settings"]["PORT"] == "8123"
    assert record["argv"][record["argv"].index("--port") + 1] == "8123"
    assert (
        record["argv"][record["argv"].index("--tau-batch-max-reqs-per-microbatch") + 1]
        == "8"
    )
    assert record["argv"][record["argv"].index("--worker-cls") + 1].endswith(
        "TauAscendWorker"
    )
    assert record["pythonpath"].split(os.pathsep)[0] == str(ROOT)
    assert "fake vllm output" in (run / "server.log").read_text()
    assert (run / "packages.txt").exists()
    assert (run / "npu.txt").exists()
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()
    assert not (tmp_path / "ignored_run").exists()


@pytest.mark.parametrize("existing", ["run", "trace"])
def test_refuse_existing_outputs_without_launch(launch_env, tmp_path, existing):
    run, trace = tmp_path / "run", tmp_path / "trace.jsonl"
    launch_env["TRACE"] = str(trace)
    if existing == "run":
        run.mkdir()
        (run / "marker").write_text("keep")
    else:
        trace.write_text("keep")
    result = launch(launch_env, "/models/example", "--run-dir", run)
    assert result.returncode == 2
    assert "NEW RUN_DIR" in result.stderr
    assert not Path(launch_env["TAU_TEST_RECORD"]).exists()
    assert (run / "marker" if existing == "run" else trace).read_text() == "keep"


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGINT])
def test_exec_preserves_pid_and_delivers_stop_signal(launch_env, tmp_path, stop_signal):
    launch_env["TAU_TEST_MODE"] = "signal"
    record = Path(launch_env["TAU_TEST_RECORD"])
    process = subprocess.Popen(
        [
            "bash",
            str(ROOT / "serve_tau.sh"),
            "/models/example",
            "--run-dir",
            str(tmp_path / "run"),
        ],
        env=launch_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    child_pid = None
    try:
        deadline = time.monotonic() + 20
        while not record.with_suffix(".ready").exists():
            if process.poll() is not None or time.monotonic() >= deadline:
                pytest.fail("Fake server did not become ready")
            time.sleep(0.05)
        ready = json.loads(record.with_suffix(".ready").read_text())
        child_pid = ready["child"]
        assert ready["pid"] == process.pid
        process.send_signal(stop_signal)
        stdout, _ = process.communicate(timeout=10)
        assert process.returncode == 0, stdout
        assert record.with_suffix(".stopped").read_text() == str(stop_signal)
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if child_pid is not None:
            with suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)


def test_legacy_python_entry_delegates_to_shell_defaults(launch_env, tmp_path):
    launch_env["MAX_NUM_SEQS"] = "6"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/tau_batch_run.py"),
            "serve",
            "/models/example",
            "--run-dir",
            str(tmp_path / "legacy"),
            "--dry-run",
        ],
        env=launch_env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    preview = json.loads(result.stdout)
    assert preview["command"][:2] == ["vllm", "serve"]
    assert preview["settings"]["MAX_REQS_PER_MB"] == "6"
