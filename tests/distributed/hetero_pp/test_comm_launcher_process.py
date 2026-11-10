# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only checks of the communication launcher's real subprocess handling."""

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / "examples/online_serving/hetero_pp/check_cpu_gloo.sh"
HELPER = ENTRY.with_suffix(".py")


def load_launcher():
    assert HELPER.is_file(), "communication launcher is not implemented"
    spec = importlib.util.spec_from_file_location("comm_process_launcher", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bash_help_needs_no_torch_or_npu():
    assert ENTRY.is_file(), "common bash entrypoint is not implemented"
    env = dict(os.environ, COMM_CHECK_BOOTSTRAP_PYTHON=sys.executable)
    result = subprocess.run(
        ["bash", str(ENTRY), "--help"],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "--head" in result.stdout and "--devices" in result.stdout


def test_prepare_environment_keeps_user_visibility_and_disables_assert_optimization():
    app = load_launcher()
    original = {"PYTHONOPTIMIZE": "2", "ASCEND_RT_VISIBLE_DEVICES": "3,7"}
    actual = app.prepare_environment(ROOT, original)
    assert "PYTHONOPTIMIZE" not in actual
    assert actual["ASCEND_RT_VISIBLE_DEVICES"] == "3,7"
    assert actual["PYTHONPATH"].split(os.pathsep)[0] == str(ROOT)
    assert original["PYTHONOPTIMIZE"] == "2"


def test_actual_child_exit_code_and_log_are_preserved(tmp_path):
    app = load_launcher()
    log = tmp_path / "child.log"
    result = app.run_process(
        [
            sys.executable,
            "-c",
            "print('fixture failure', flush=True); raise SystemExit(7)",
        ],
        os.environ.copy(),
        log,
        timeout=5,
    )
    assert result == 7
    assert "fixture failure" in log.read_text()


def test_timeout_terminates_only_its_own_child_group(tmp_path):
    app = load_launcher()
    log = tmp_path / "timeout.log"
    pid_file = tmp_path / "child.pid"
    heartbeat = tmp_path / "heartbeat"
    child_program = (
        "import pathlib, sys, time\n"
        "p = pathlib.Path(sys.argv[1])\n"
        "for i in range(1500):\n"
        "    p.write_text(str(i))\n"
        "    time.sleep(0.02)\n"
    )
    program = (
        "import os, pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', sys.argv[3], sys.argv[2]]); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
        "print('started', flush=True); time.sleep(30)"
    )
    with pytest.raises(TimeoutError):
        app.run_process(
            [
                sys.executable,
                "-c",
                program,
                str(pid_file),
                str(heartbeat),
                child_program,
            ],
            os.environ.copy(),
            log,
            timeout=0.5,
        )
    assert "started" in log.read_text()
    assert pid_file.read_text().strip().isdigit()
    last = heartbeat.read_text()
    time.sleep(0.15)
    assert heartbeat.read_text() == last


def test_sigterm_runs_cleanup_for_the_owned_child(tmp_path):
    load_launcher()
    heartbeat = tmp_path / "heartbeat"
    child = (
        "import pathlib,sys,time\n"
        "p=pathlib.Path(sys.argv[1])\n"
        "for i in range(1500):\n"
        "    p.write_text(str(i)); time.sleep(0.02)\n"
    )
    parent = (
        "import importlib.util,os,sys,pathlib; "
        "spec=importlib.util.spec_from_file_location('launcher',sys.argv[1]); "
        "app=importlib.util.module_from_spec(spec); spec.loader.exec_module(app); "
        "app.install_signal_handlers(); "
        "app.run_process([sys.executable,'-c',sys.argv[2],sys.argv[3]], "
        "os.environ.copy(),pathlib.Path(sys.argv[4]),timeout=30)"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            parent,
            str(HELPER),
            child,
            str(heartbeat),
            str(tmp_path / "signal_child.log"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not heartbeat.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert heartbeat.exists()
        process.terminate()
        process.communicate(timeout=5)
        assert process.returncode != 0
        last = heartbeat.read_text()
        time.sleep(0.15)
        assert heartbeat.read_text() == last
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=5)


def test_selected_device_list_is_explicit_and_validated():
    app = load_launcher()
    assert (
        app.prepare_environment(ROOT, {}, devices="1,3")["ASCEND_RT_VISIBLE_DEVICES"]
        == "1,3"
    )
    for invalid in ("0", "0,0", "0,x", "0,1,2", "0,-1"):
        with pytest.raises(ValueError):
            app.prepare_environment(ROOT, {}, devices=invalid)


def test_ownership_snapshot_includes_worker_sessions_but_excludes_other_jobs(
    monkeypatch,
):
    app = load_launcher()
    table = {
        100: (1, 100, 300, "S"),
        101: (100, 101, 301, "S"),
        102: (101, 101, 302, "S"),
        200: (1, 200, 299, "S"),
    }
    monkeypatch.setattr(app, "process_table", lambda: table)
    family = app.owned_family(100)
    assert set(family) == {100, 101, 102}
    assert family[101][2] == 301


def test_cleanup_does_not_signal_reused_pids(monkeypatch):
    app = load_launcher()
    original = {100: (1, 100, 300, "S"), 101: (100, 101, 301, "S")}
    current = {100: (1, 100, 300, "S"), 101: (1, 101, 999, "S")}
    signals = []
    monkeypatch.setattr(app, "process_table", lambda: current)
    monkeypatch.setattr(app.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    app.signal_family(original, 15)
    assert signals == [(100, 15)]


def test_linux_proc_parser_handles_spaces_parentheses_and_non_utf8_names(
    tmp_path, monkeypatch
):
    app = load_launcher()
    directory = tmp_path / "123"
    directory.mkdir()
    fields = ["S", "1", "123"] + ["0"] * 16 + ["501"]
    (directory / "stat").write_bytes(
        b"123 (a strange ) name\xff) " + " ".join(fields).encode()
    )
    monkeypatch.setattr(app, "PROC_ROOT", tmp_path)
    assert app.process_table() == {123: (1, 123, 501, "S")}


def test_stop_process_never_adopts_a_reused_root_or_its_children(monkeypatch):
    app = load_launcher()
    known = {100: (1, 100, 300, "S"), 101: (100, 101, 301, "S")}
    current = {100: (1, 100, 999, "S"), 102: (100, 102, 1000, "S")}
    signals = []
    monkeypatch.setattr(app, "process_table", lambda: current)
    monkeypatch.setattr(app.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(app.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    class OldProcess:
        pid = 100
        returncode = 0

        def wait(self, timeout=None):
            return self.returncode

    app.stop_process(OldProcess(), known)
    assert signals == []
