# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monitoring records and host-owned CSV output without actual NPUs or Docker."""

import csv
import importlib.util
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / "examples/online_serving/hetero_pp"


@pytest.fixture
def monitor(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    spec = importlib.util.spec_from_file_location(
        "npu_monitor_test", DIRECTORY / "monitor_npu.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NPU:
    def device_count(self):
        return 2

    def set_device(self, device):
        self.device = device

    def get_device_name(self, device):
        return "Ascend910B4-1"

    def mem_get_info(self, device):
        return 4 * 1024**3, 16 * 1024**3

    def utilization(self, device):
        return 0 if device == 0 else 75


def test_csv_records_both_devices_and_preserves_zero_utilization(monitor):
    stream = io.StringIO()
    assert (
        monitor.collect(
            NPU(), stream, interval=1, samples=1, node="host-a", role="head"
        )
        == 0
    )
    rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
    assert [row["logical_device"] for row in rows] == ["0", "1"]
    assert [row["utilization_pct"] for row in rows] == ["0.0", "75.0"]
    assert all(row["memory_used_bytes"] == str(12 * 1024**3) for row in rows)
    assert all(row["node"] == "host-a" and row["status"] == "ok" for row in rows)


def test_unsupported_utilization_is_blank_instead_of_false_zero(monitor):
    class Unsupported(NPU):
        def utilization(self, device):
            raise RuntimeError("not supported")

    stream = io.StringIO()
    assert monitor.collect(Unsupported(), stream, samples=1) == 0
    rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
    assert all(
        row["utilization_pct"] == "" and row["status"] == "partial" for row in rows
    )
    assert all("not supported" in row["error"] for row in rows)


def test_failed_device_initialization_is_reported_and_not_retried(monitor):
    class Failed(NPU):
        calls = 0

        def set_device(self, device):
            self.calls += 1
            raise RuntimeError("device unavailable")

    npu = Failed()
    stream = io.StringIO()
    assert monitor.collect(npu, stream, samples=3, interval=0.001) == 1
    rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
    assert len(rows) == 2
    assert npu.calls == 2
    assert all(row["status"] == "error" for row in rows)


def test_stop_marker_ends_continuous_collection(monitor, tmp_path):
    marker = tmp_path / "stop"

    class StopAfterSample(NPU):
        def utilization(self, device):
            marker.touch()
            return 0

    stream = io.StringIO()
    assert monitor.collect(StopAfterSample(), stream, stop_file=marker) == 0
    assert len(list(csv.DictReader(io.StringIO(stream.getvalue())))) == 2


def test_host_wrapper_saves_csv_owned_by_current_user_and_keeps_failure(tmp_path):
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/bash\n"
        'if [[ "$1" == ps ]]; then echo hetero-pp-worker; exit 0; fi\n'
        'if [[ "$1" == exec ]]; then\n'
        '  printf "timestamp_utc,logical_device\\n2026-11-12T00:00:00Z,0\\n"\n'
        "  exit 7\n"
        "fi\nexit 99\n"
    )
    docker.chmod(0o700)
    env = dict(PATH=f"{tmp_path}:{os.environ['PATH']}", HOME=str(tmp_path))
    result = subprocess.run(
        ["bash", str(DIRECTORY / "monitor_npu.sh"), "--duration", "1"],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 7
    files = list((tmp_path / "vllm_hetero_pp_logs").glob("npu_monitor_*.csv"))
    assert len(files) == 1
    assert files[0].stat().st_uid == os.getuid()
    assert "2026-11-12" in files[0].read_text()


def test_native_stdout_does_not_pollute_csv(tmp_path):
    (tmp_path / "torch.py").write_text(
        'print("native import message")\n'
        "class Device:\n"
        " def device_count(self): return 1\n"
        ' def set_device(self, d): print("device initialized")\n'
        ' def get_device_name(self, d): return "Ascend310P3"\n'
        " def mem_get_info(self, d): return 100, 200\n"
        " def utilization(self, d): return 10\n"
        "npu=Device()\n"
    )
    (tmp_path / "torch_npu.py").write_text("")
    env = dict(PATH=os.environ["PATH"], PYTHONPATH=str(tmp_path), HOME=str(tmp_path))
    result = subprocess.run(
        [
            sys.executable,
            str(DIRECTORY / "monitor_npu.py"),
            "--collect",
            "--samples",
            "1",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert len(rows) == 1 and rows[0]["device_name"] == "Ascend310P3"
    assert "native import message" not in result.stdout
    assert "native import message" in result.stderr
