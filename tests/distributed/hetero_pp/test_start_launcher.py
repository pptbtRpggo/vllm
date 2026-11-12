# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch planning and failure handling without Ray, TorchNPU or model loading."""

import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / "examples/online_serving/hetero_pp"
SCRIPT = DIRECTORY / "start_hetero_pp.sh"


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "model with spaces"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 48})
    )
    return path


def invoke(model, *extra, environment=None):
    return subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--python",
            sys.executable,
            "--head",
            "192.0.2.10",
            "--peer",
            "192.0.2.20",
            "--model",
            str(model),
            "--node-ip",
            "192.0.2.10",
            "--dry-run",
            *extra,
        ],
        text=True,
        capture_output=True,
        timeout=10,
        env=environment,
    )


@pytest.mark.parametrize("ip,role", [("192.0.2.10", "head"), ("192.0.2.20", "worker")])
def test_same_script_plans_both_roles_without_starting_ray_or_loading_npu(
    model, ip, role
):
    result = invoke(model, "--node-ip", ip)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["role"] == role
    assert not (model.parent / "output").exists()
    stages = plan["plan"]["stages"]
    assert [stage["layers"] for stage in stages] == [18, 18, 6, 6]
    assert [stage["soc"] for stage in stages] == ["910B", "910B", "310P", "310P"]
    ray = plan["ray_command"]
    assert ray[1:5] == ["-m", "ray.scripts.scripts", "start", "--num-cpus=4"]
    assert '--resources={"NPU":2}' in ray
    assert "--min-worker-port=0" in ray
    assert "--max-worker-port=0" in ray
    if role == "head":
        assert "--head" in ray
        serve = plan["serve_command"]
        assert serve[1:3] == ["-m", "vllm.distributed.hetero_pp.serve"]
        assert serve[serve.index("--model") + 1] == str(model)
        assert "--no-async-scheduling" in serve
        assert "--distributed-executor-backend" not in serve
        assert "--nnodes" not in serve
    else:
        assert "--address=192.0.2.10:16379" in ray
        assert plan["serve_command"] is None


@pytest.mark.parametrize(
    "extra",
    [
        ("--layers", "18,18,6"),
        ("--layers", "18,18,6,0"),
        ("--layers", "18,18,6,5"),
        ("--head", "192.0.2.20"),
        ("--node-ip", "192.0.2.99"),
        ("--ray-port", "8000"),
        ("--startup-timeout", "nan"),
        ("--max-num-seqs", "0"),
    ],
)
def test_invalid_configuration_is_rejected_before_launch(model, extra):
    result = invoke(model, *extra)
    assert result.returncode != 0
    assert not result.stdout.strip().startswith("{")


def test_dry_run_keeps_visibility_and_does_not_expose_other_environment(model):
    env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES="0,1", PRIVATE_TEST_SECRET="hide")
    result = invoke(model, environment=env)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["environment"]["ASCEND_RT_VISIBLE_DEVICES"] == "0,1"
    assert "PRIVATE_TEST_SECRET" not in result.stdout
    assert plan["environment"]["VLLM_USE_V2_MODEL_RUNNER"] == "0"


@pytest.fixture
def launcher(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    source = DIRECTORY / "start_hetero_pp.py"
    spec = importlib.util.spec_from_file_location("start_launcher_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("head_port", [16379, 16389])
def test_ray_head_port_does_not_overlap_reserved_worker_ports(
    launcher, model, head_port
):
    # Exercise Ray's real configuration checker without starting any Ray process.
    parameter = pytest.importorskip("ray._private.parameter")
    args = launcher.argument_parser().parse_args(
        [
            "--head",
            "192.0.2.10",
            "--peer",
            "192.0.2.20",
            "--model",
            str(model),
            "--ray-port",
            str(head_port),
        ]
    )
    command = launcher.make_launch(args, args.head, sys.executable, {})["ray_command"]
    ports = dict(
        item[2:].split("=", 1)
        for item in command
        if item.startswith("--") and "=" in item
    )
    params = parameter.RayParams(
        gcs_server_port=int(ports["port"]),
        min_worker_port=int(ports.get("min-worker-port", "10002")),
        max_worker_port=int(ports.get("max-worker-port", "19999")),
    )
    params.update_pre_selected_port()


def test_automatic_role_rejects_unrelated_machine_and_wrong_interface(launcher):
    interfaces = {"eth0": ["192.0.2.20"], "eth1": ["10.0.0.2"]}
    actual = launcher.select_network(
        "192.0.2.10", "192.0.2.20", interfaces, "192.0.2.20"
    )
    assert actual["node_ip"] == "192.0.2.20"
    assert actual["interface"] == "eth0"
    with pytest.raises(ValueError):
        launcher.select_network(
            "192.0.2.10", "192.0.2.20", interfaces, "10.0.0.2", "eth1"
        )


def test_failed_ray_start_never_starts_model_or_reports_joined(
    launcher, model, tmp_path, monkeypatch
):
    args = launcher.argument_parser().parse_args(
        [
            "--head",
            "192.0.2.10",
            "--peer",
            "192.0.2.20",
            "--model",
            str(model),
        ]
    )
    launch = launcher.make_launch(args, "192.0.2.10", sys.executable, {})
    monkeypatch.setattr(
        launcher, "capture", lambda *a, **kw: (7, "existing Ray instance")
    )
    with pytest.raises(RuntimeError, match="Ray"):
        launcher.start_services(args, launch, {}, tmp_path, sys.executable)
    assert not (tmp_path / "ray_started.json").exists()
    assert not (tmp_path / "server_exit.json").exists()
    assert "existing Ray instance" in (tmp_path / "ray_start.log").read_text()


def test_worker_join_is_not_reported_as_model_validation(
    launcher, model, tmp_path, monkeypatch
):
    args = launcher.argument_parser().parse_args(
        [
            "--head",
            "192.0.2.10",
            "--peer",
            "192.0.2.20",
            "--model",
            str(model),
        ]
    )
    launch = launcher.make_launch(args, "192.0.2.20", sys.executable, {})
    monkeypatch.setattr(launcher, "capture", lambda *a, **kw: (0, "Ray started"))
    assert launcher.start_services(args, launch, {}, tmp_path, sys.executable) == 0
    receipt = json.loads((tmp_path / "ray_started.json").read_text())
    assert receipt["role"] == "worker"
    assert receipt["model_inference_validated"] is False
    assert not (tmp_path / "complete.json").exists()


def test_cluster_failure_prevents_head_model_launch(
    launcher, model, tmp_path, monkeypatch
):
    args = launcher.argument_parser().parse_args(
        [
            "--head",
            "192.0.2.10",
            "--peer",
            "192.0.2.20",
            "--model",
            str(model),
        ]
    )
    launch = launcher.make_launch(args, "192.0.2.10", sys.executable, {})
    responses = iter([(0, "Ray started"), (1, "peer missing")])
    monkeypatch.setattr(launcher, "capture", lambda *a, **kw: next(responses))
    with pytest.raises(RuntimeError, match="Ray"):
        launcher.start_services(args, launch, {}, tmp_path, sys.executable)
    assert not (tmp_path / "server_exit.json").exists()
    assert "peer missing" in (tmp_path / "cluster_wait.log").read_text()


def test_api_failure_keeps_real_exit_status_and_log(launcher, tmp_path):
    log = tmp_path / "api.log"
    command = [sys.executable, "-c", "print('model load failed'); raise SystemExit(7)"]
    assert launcher.serve_foreground(command, dict(os.environ), log, 5, 1) == 7
    assert "model load failed" in log.read_text()
    assert not (tmp_path / "api_ready.json").exists()


def test_api_exiting_zero_before_health_is_not_ready(launcher, tmp_path):
    with pytest.raises(RuntimeError, match="API"):
        launcher.serve_foreground(
            [sys.executable, "-c", "print('no server')"],
            dict(os.environ),
            tmp_path / "api.log",
            5,
            1,
        )
    assert not (tmp_path / "api_ready.json").exists()


def test_api_startup_timeout_stops_only_the_child(launcher, tmp_path):
    with pytest.raises(TimeoutError):
        launcher.serve_foreground(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            dict(os.environ),
            tmp_path / "api.log",
            0,
            1,
        )
    assert not (tmp_path / "api_ready.json").exists()


def test_real_health_response_is_not_reported_as_inference_validation(
    launcher, tmp_path
):
    with socket.socket() as selection:
        selection.bind(("127.0.0.1", 0))
        port = selection.getsockname()[1]
    code = """
import sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == '/health' else 404)
        self.end_headers()
server = HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler)
server.timeout = 5
server.handle_request()
server.server_close()
time.sleep(0.1)
"""
    rc = launcher.serve_foreground(
        [sys.executable, "-c", code, str(port)],
        dict(os.environ),
        tmp_path / "api.log",
        5,
        port,
    )
    assert rc == 0
    receipt = json.loads((tmp_path / "api_ready.json").read_text())
    assert receipt["health_http_status"] == 200
    assert receipt["model_inference_validated"] is False
    assert not (tmp_path / "complete.json").exists()
