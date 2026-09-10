# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the actual shell launcher using a fake vllm, without an NPU."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def launch_env(tmp_path, monkeypatch):
    # Keep the launcher's latest link inside the test's disposable repository.
    root = tmp_path / "repo"
    (root / "tools").mkdir(parents=True)
    shutil.copy2(ROOT / "serve_tau.sh", root / "serve_tau.sh")
    shutil.copy2(ROOT / "tools/tau_batch_run.py", root / "tools/tau_batch_run.py")
    shutil.copy2(ROOT / "tools/tau_config.py", root / "tools/tau_config.py")
    shutil.copytree(ROOT / "configs", root / "configs")
    monkeypatch.setattr(sys.modules[__name__], "ROOT", root)
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
                                 pythonpath=os.environ.get("PYTHONPATH"),
                                 trace_env=os.environ.get("TAU_BATCH_TRACE"))))
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
        assert Path(preview["run_dir"]).parent == ROOT / "output"
        assert not Path(preview["run_dir"]).exists()
        assert not Path(preview["trace"]).exists()
        assert Path(preview["trace"]).parent == Path(launch_env["TMPDIR"]).resolve()
        assert preview["command"][:3] == ["vllm", "serve", "/models/example"]
        assert preview["settings"]["PP"] == "2"
        assert preview["settings"]["TP"] == "1"
        assert preview["settings"]["MAX_NUM_SEQS"] == "4"
        assert "MAX_REQS_PER_MB" not in preview["settings"]
        assert preview["settings"]["MIN_WAITING"] == "0"
    assert previews[0]["run_dir"] != previews[1]["run_dir"]
    assert previews[0]["trace"] != previews[1]["trace"]
    assert not Path(launch_env["TAU_TEST_RECORD"]).exists()


def test_trace_off_omits_worker_and_clears_inherited_path(launch_env, tmp_path):
    path = tmp_path / "must_not_be_touched.jsonl"
    path.write_text("keep")
    launch_env.update(TRACE=str(path), TAU_BATCH_TRACE=str(path))
    run = tmp_path / "disabled"
    result = launch(launch_env, "/models/example", "--run-dir", run, "--no-trace")
    assert result.returncode == 7, result.stderr
    manifest = json.loads((run / "server_meta.json").read_text())
    record = json.loads(Path(launch_env["TAU_TEST_RECORD"]).read_text())
    assert manifest["trace"] is None
    assert manifest["settings"]["TRACE_ENABLED"] == "0"
    assert "--tau-batch-trace" not in record["argv"]
    assert "--worker-cls" not in record["argv"]
    assert record["trace_env"] is None
    assert path.read_text() == "keep"


def test_trace_flag_overrides_environment(launch_env):
    launch_env["TRACE_ENABLED"] = "0"
    result = launch(launch_env, "/models/example", "--trace", "--dry-run")
    preview = json.loads(result.stdout)
    assert preview["trace"]
    assert "--tau-batch-trace" in preview["command"]


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
    manifest = json.loads((run / "server_meta.json").read_text())
    record = json.loads(Path(launch_env["TAU_TEST_RECORD"]).read_text())
    assert manifest["model"] == str(model.resolve())
    assert manifest["trace"] == str(trace.resolve())
    assert manifest["command"][1:] == record["argv"]
    assert manifest["vllm_executable"] == str(tmp_path / "bin" / "vllm")
    assert manifest["settings"]["MAX_NUM_SEQS"] == "8"
    assert "MAX_REQS_PER_MB" not in manifest["settings"]
    assert manifest["settings"]["PORT"] == "8123"
    assert "Tau serve 生效参数" in result.stdout
    assert f"MODEL = {model.resolve()}" in result.stdout
    assert "MAX_NUM_SEQS = 8" in result.stdout
    assert "PORT = 8123" in result.stdout
    assert str(ROOT / "configs/serve.yaml") in result.stdout
    assert "[serve 完整启动命令]" in result.stdout
    assert result.stdout.index("Tau serve 生效参数") < result.stdout.index(
        "fake vllm output"
    )
    assert record["argv"][record["argv"].index("--port") + 1] == "8123"
    assert record["argv"][record["argv"].index("--max-num-seqs") + 1] == "8"
    assert record["argv"][record["argv"].index("--worker-cls") + 1].endswith(
        "TauAscendWorker"
    )
    assert record["pythonpath"].split(os.pathsep)[0] == str(ROOT)
    assert "fake vllm output" in (run / "server.log").read_text()
    assert {p.name for p in run.iterdir()} == {"server_meta.json", "server.log"}
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
    assert preview["settings"]["MAX_NUM_SEQS"] == "6"
    assert "MAX_REQS_PER_MB" not in preview["settings"]


@pytest.mark.parametrize("request_cap", [1, 4, 64])
def test_native_request_capacity_is_the_only_cli_limit(launch_env, request_cap):
    launch_env["MAX_NUM_SEQS"] = str(request_cap)
    result = launch(launch_env, "/models/example", "--dry-run")
    assert result.returncode == 0, result.stderr
    preview = json.loads(result.stdout)
    command = preview["command"]
    assert preview["settings"]["MAX_NUM_SEQS"] == str(request_cap)
    assert "MAX_REQS_PER_MB" not in preview["settings"]
    assert command[command.index("--max-num-seqs") + 1] == str(request_cap)
    assert "--tau-batch-max-reqs-per-microbatch" not in command


def test_default_restart_preserves_previous_run_and_updates_latest(launch_env):
    runs = []
    for _ in range(2):
        result = launch(launch_env, "/models/example", "--no-trace")
        assert result.returncode == 7
        latest = ROOT / "output" / "latest"
        assert latest.is_symlink()
        runs.append(latest.resolve())
    assert runs[0] != runs[1]
    for run in runs:
        assert run.parent == ROOT / "output"
        assert (run / "server_meta.json").is_file()
        assert "fake vllm output" in (run / "server.log").read_text()


def test_edit_service_config_then_launch_without_arguments(launch_env):
    config = ROOT / "configs/serve.yaml"
    config.write_text(
        config.read_text()
        .replace('MODEL: ""', 'MODEL: "/models/configured"')
        .replace("DTYPE: null", "DTYPE: float16")
        .replace("MAX_NUM_SEQS: 4", "MAX_NUM_SEQS: 7")
        .replace("TRACE_ENABLED: true", "TRACE_ENABLED: false")
    )
    result = launch(launch_env)
    assert result.returncode == 7, result.stderr
    run = (ROOT / "output/latest").resolve()
    meta = json.loads((run / "server_meta.json").read_text())
    assert meta["model"] == "/models/configured"
    assert meta["settings"]["DTYPE"] == "float16"
    assert meta["command"][meta["command"].index("--dtype") + 1] == "float16"
    assert "DTYPE = float16" in result.stdout
    assert meta["settings"]["MAX_NUM_SEQS"] == "7"
    assert meta["trace"] is None
    assert meta["launch_config"]["path"] == str(config)
    assert meta["launch_config"]["cli_overrides"] == []


def test_bad_service_config_fails_before_creating_output(launch_env):
    config = ROOT / "configs/serve.yaml"
    config.write_text(config.read_text() + "UNKNOWN_OPTION: 7\n")
    result = launch(launch_env, "/models/test")
    assert result.returncode == 2
    assert "UNKNOWN_OPTION" in result.stderr
    assert not (ROOT / "output").exists()


def test_default_scheduler_uses_native_defaults_and_clears_tau_state(launch_env):
    config = ROOT / "configs/serve.yaml"
    config.write_text(
        config.read_text().replace("SCHEDULER: tau", "SCHEDULER: default")
    )
    launch_env.update(
        TAU_BATCH_TRACE="/tmp/inherited.jsonl", MIN_WAITING="99", MAX_MICROBATCHES="9"
    )
    result = launch(launch_env, "/models/test")
    assert result.returncode == 7, result.stderr
    run = (ROOT / "output/latest").resolve()
    meta = json.loads((run / "server_meta.json").read_text())
    record = json.loads(Path(launch_env["TAU_TEST_RECORD"]).read_text())
    assert meta["settings"]["SCHEDULER"] == "default"
    assert meta["settings"]["MAX_NUM_SEQS"] is None
    assert meta["trace"] is None and record["trace_env"] is None
    assert "MIN_WAITING" not in meta["settings"]
    assert "MAX_MICROBATCHES" not in meta["settings"]
    for flag in (
        "--dtype",
        "--scheduler-cls",
        "--worker-cls",
        "--max-num-seqs",
        "--max-num-batched-tokens",
    ):
        assert flag not in record["argv"]
    assert not any("tau-batch" in arg for arg in record["argv"])
    assert "由 vLLM/平台决定" in result.stdout


@pytest.mark.parametrize("enabled", [True, False])
def test_default_scheduler_specific_options_reach_cli(launch_env, enabled):
    import yaml

    path = ROOT / "configs/serve.yaml"
    data = yaml.safe_load(path.read_text())
    data["SCHEDULER"] = "default"
    data["SCHEDULERS"]["default"].update(
        MAX_NUM_SEQS=32,
        MAX_NUM_BATCHED_TOKENS=4096,
        ENABLE_CHUNKED_PREFILL=enabled,
        ENABLE_PREFIX_CACHING=enabled,
        ASYNC_SCHEDULING=enabled,
        SCHEDULING_POLICY="fcfs",
    )
    path.write_text(yaml.safe_dump(data))
    result = launch(launch_env, "/models/test", "--dry-run")
    assert result.returncode == 0, result.stderr
    cmd = json.loads(result.stdout)["command"]
    assert cmd[cmd.index("--max-num-seqs") + 1] == "32"
    assert cmd[cmd.index("--max-num-batched-tokens") + 1] == "4096"
    assert cmd[cmd.index("--scheduling-policy") + 1] == "fcfs"
    for flag in ("enable-chunked-prefill", "enable-prefix-caching", "async-scheduling"):
        assert ("--" if enabled else "--no-") + flag in cmd
    assert "--scheduler-cls" not in cmd


def test_switch_back_to_tau_uses_only_tau_profile(launch_env):
    config = ROOT / "configs/serve.yaml"
    config.write_text(
        config.read_text().replace("SCHEDULER: tau", "SCHEDULER: default")
    )
    result = launch(launch_env, "/models/test", "--scheduler", "tau", "--dry-run")
    assert result.returncode == 0, result.stderr
    preview = json.loads(result.stdout)
    assert preview["settings"]["SCHEDULER"] == "tau"
    assert preview["settings"]["MAX_NUM_SEQS"] == "4"
    assert preview["settings"]["MAX_NUM_BATCHED_TOKENS"] == "8192"
    assert preview["trace"]
    for flag in (
        "--no-enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--no-async-scheduling",
    ):
        assert flag in preview["command"]


@pytest.mark.parametrize("scheduler,extra", [("default", ["--trace"]), ("unknown", [])])
def test_invalid_scheduler_combination_fails_before_launch(
    launch_env, scheduler, extra
):
    result = launch(launch_env, "/models/test", "--scheduler", scheduler, *extra)
    assert result.returncode == 2
    assert not Path(launch_env["TAU_TEST_RECORD"]).exists()
    assert not (ROOT / "output").exists()
