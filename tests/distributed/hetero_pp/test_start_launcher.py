# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch planning and failure handling without Ray, TorchNPU or model loading."""

import importlib.util
import json
import os
import socket
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

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


def test_profile_host_sync_reaches_the_real_profile_subprocess(model):
    result = invoke(model, "--profile", "--profile-compute-timing", "host-sync")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["profile_environment"]["PROFILE_COMPUTE_TIMING"] == "host-sync"
    assert plan["serve_command"] is None


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


@pytest.mark.parametrize(
    "head,peer", [("910b", "910b"), ("310p", "910b"), ("310p", "310p")]
)
@pytest.mark.parametrize("ip", ["192.0.2.10", "192.0.2.20"])
@pytest.mark.parametrize("profile", [False, True])
def test_configured_socs_reach_service_and_profile_plans(
    model, head, peer, ip, profile
):
    extra = ("--profile",) if profile else ()
    result = invoke(
        model, "--head-soc", head, "--peer-soc", peer, "--node-ip", ip, *extra
    )
    assert result.returncode == 0, result.stderr
    launch = json.loads(result.stdout)
    names = {"910b": "910B", "310p": "310P"}
    assert [s["soc"] for s in launch["plan"]["stages"]] == [names[head]] * 2 + [
        names[peer]
    ] * 2
    assert [s["layers"] for s in launch["plan"]["stages"]] == [12] * 4
    assert launch["plan"]["validation_mode"] == (
        "910b_only" if head == peer == "910b" else "explicit"
    )
    assert launch["environment"]["VLLM_HETERO_PP_PLAN"] == json.dumps(launch["plan"])
    if ip == "192.0.2.10":
        assert (launch["serve_command"] is not None) is not profile
        assert (launch["profile_command"] is not None) is profile
    else:
        assert launch["serve_command"] is None
        assert launch["profile_command"] is None


@pytest.mark.parametrize("total,counts", [(32, [8] * 4), (50, [13, 13, 12, 12])])
def test_homogeneous_default_partition_uses_model_layer_count(model, total, counts):
    (model / "config.json").write_text(
        json.dumps({"architectures": ["LlamaForCausalLM"], "num_hidden_layers": total})
    )
    result = invoke(model, "--peer-soc", "910b")
    assert result.returncode == 0, result.stderr
    assert [s["layers"] for s in json.loads(result.stdout)["plan"]["stages"]] == counts


def test_reverse_soc_order_preserves_explicit_uneven_layers(model):
    result = invoke(
        model, "--head-soc", "310p", "--peer-soc", "910b", "--layers", "6,6,18,18"
    )
    assert result.returncode == 0, result.stderr
    assert [s["layers"] for s in json.loads(result.stdout)["plan"]["stages"]] == [
        6,
        6,
        18,
        18,
    ]


@pytest.mark.parametrize(
    "declared,actual,accepted",
    [
        ("910b", "910B", True),
        ("910b", "310P", False),
        ("310p", "310P", True),
        ("310p", "910B", False),
    ],
)
@pytest.mark.parametrize("role", ["head", "worker"])
def test_main_checks_configured_local_soc_before_starting_ray(
    launcher, model, tmp_path, monkeypatch, declared, actual, accepted, role
):
    ip = "192.0.2.10" if role == "head" else "192.0.2.20"
    network = {"node_ip": ip, "interface": "eth0"}
    monkeypatch.setattr(launcher, "ipv4_interfaces", lambda: [])
    monkeypatch.setattr(launcher, "select_network", lambda *a, **kw: network)
    route = SimpleNamespace(connect=lambda *a: None, bind=lambda *a: None)
    monkeypatch.setattr(launcher.socket, "socket", lambda *a: nullcontext(route))
    monkeypatch.setattr(
        launcher, "discover_runtime", lambda *a: (sys.executable, {}, {"soc": actual})
    )
    calls = []

    def start_services(args, launch, *rest):
        calls.append(launch)
        return 0

    monkeypatch.setattr(launcher, "start_services", start_services)
    output = tmp_path / "launch"
    rc = launcher.main(
        [
            "--head",
            "192.0.2.10",
            "--peer",
            "192.0.2.20",
            "--node-ip",
            ip,
            "--model",
            str(model),
            "--head-soc" if role == "head" else "--peer-soc",
            declared,
            "--layers",
            "12,12,12,12",
            "--output",
            str(output),
        ]
    )
    assert rc == (0 if accepted else 1)
    assert bool(calls) is accepted
    if accepted:
        assert calls[0]["plan"]["stages"][0 if role == "head" else 2]["soc"] == actual
    else:
        assert (output / "failure.json").exists()
        assert not (output / "ray_started.json").exists()


def test_service_uses_vllm_capacity_and_batch_defaults(model):
    result = invoke(model)
    assert result.returncode == 0, result.stderr
    command = json.loads(result.stdout)["serve_command"]
    for flag in (
        "--max-model-len",
        "--max-num-batched-tokens",
        "--max-num-seqs",
        "--gpu-memory-utilization",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
    ):
        assert flag not in command
    assert "--enforce-eager" in command
    assert "--no-async-scheduling" in command


def test_service_forwards_explicit_capacity_and_batch_overrides(model):
    arguments = (
        "--max-model-len",
        "2048",
        "--max-num-batched-tokens",
        "1024",
        "--max-num-seqs",
        "64",
        "--gpu-memory-utilization",
        "0.95",
    )
    result = invoke(model, *arguments)
    assert result.returncode == 0, result.stderr
    command = json.loads(result.stdout)["serve_command"]
    for flag, value in zip(arguments[::2], arguments[1::2]):
        assert command.count(flag) == 1
        assert command[command.index(flag) + 1] == value


def test_profile_keeps_its_bounded_capacity_and_batch_defaults(model):
    result = invoke(model, "--profile")
    assert result.returncode == 0, result.stderr
    settings = json.loads(result.stdout)["profile_environment"]
    assert settings["PROFILE_MAX_MODEL_LEN"] == "512"
    assert settings["PROFILE_MAX_NUM_BATCHED_TOKENS"] == "256"
    assert settings["PROFILE_MAX_NUM_SEQS"] == "4"
    assert settings["PROFILE_GPU_MEMORY_UTILIZATION"] == "0.8"


def test_profile_forwards_explicit_capacity_overrides(model):
    result = invoke(
        model,
        "--profile",
        "--max-model-len",
        "1024",
        "--max-num-batched-tokens",
        "512",
        "--max-num-seqs",
        "8",
        "--gpu-memory-utilization",
        "0.95",
    )
    assert result.returncode == 0, result.stderr
    settings = json.loads(result.stdout)["profile_environment"]
    assert settings["PROFILE_MAX_MODEL_LEN"] == "1024"
    assert settings["PROFILE_MAX_NUM_BATCHED_TOKENS"] == "512"
    assert settings["PROFILE_MAX_NUM_SEQS"] == "8"
    assert settings["PROFILE_GPU_MEMORY_UTILIZATION"] == "0.95"


@pytest.mark.parametrize(
    "extra,context", [((), None), (("--max-model-len", "2048"), "2048")]
)
def test_direct_serve_script_uses_vllm_defaults_and_forwards_overrides(
    model, tmp_path, extra, context
):
    python = tmp_path / "python"
    python.write_text(
        f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n"
    )
    python.chmod(0o700)
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    result = subprocess.run(
        ["bash", str(DIRECTORY / "serve.sh"), *extra],
        env=dict(
            os.environ,
            PATH=f"{tmp_path}:{os.environ['PATH']}",
            MODEL=str(model),
            PLAN=str(plan),
        ),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    command = json.loads(result.stdout)
    if context is None:
        assert "--max-model-len" not in command
    else:
        assert command.count("--max-model-len") == 1
        assert command[command.index("--max-model-len") + 1] == context
    for flag in (
        "--max-num-batched-tokens",
        "--max-num-seqs",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
    ):
        assert flag not in command


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
        ("--head-soc", "guess"),
        ("--peer-soc", "guess"),
        ("--head", "192.0.2.20"),
        ("--node-ip", "192.0.2.99"),
        ("--ray-port", "8000"),
        ("--startup-timeout", "nan"),
        ("--max-num-seqs", "0"),
        ("--max-num-batched-tokens", "0"),
        ("--max-model-len", "0"),
        ("--gpu-memory-utilization", "1"),
        ("--gpu-memory-utilization", "nan"),
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


def test_builtin_chat_template_is_forwarded_only_to_head_api(model):
    result = invoke(model, "--chat-template", "codellama-instruct")
    assert result.returncode == 0, result.stderr
    command = json.loads(result.stdout)["serve_command"]
    template = Path(command[command.index("--chat-template") + 1])
    assert template == DIRECTORY / "codellama_instruct.jinja"
    assert template.is_file()
    worker = invoke(
        model, "--node-ip", "192.0.2.20", "--chat-template", "/only/on/head.jinja"
    )
    assert worker.returncode == 0, worker.stderr
    assert json.loads(worker.stdout)["serve_command"] is None


def test_explicit_chat_template_with_spaces_and_existing_default(model, tmp_path):
    template = tmp_path / "custom chat.jinja"
    template.write_text("{{ messages[0]['content'] }}")
    result = invoke(model, "--chat-template", str(template))
    assert result.returncode == 0, result.stderr
    command = json.loads(result.stdout)["serve_command"]
    assert command[command.index("--chat-template") + 1] == str(template)
    default = invoke(model)
    assert "--chat-template" not in json.loads(default.stdout)["serve_command"]


@pytest.mark.parametrize("contents", [None, "  \n"])
def test_missing_or_empty_template_is_rejected_before_start(model, tmp_path, contents):
    template = tmp_path / "invalid.jinja"
    if contents is not None:
        template.write_text(contents)
    result = invoke(model, "--chat-template", str(template))
    assert result.returncode != 0
    assert "模板" in result.stderr
    assert not result.stdout.strip().startswith("{")


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


@pytest.mark.parametrize("ip", ["192.0.2.10", "192.0.2.20"])
def test_profile_mode_plans_ray_and_head_profile_instead_of_api(model, tmp_path, ip):
    shared = tmp_path / "shared"
    shared.mkdir()
    result = invoke(
        model,
        "--node-ip",
        ip,
        "--profile",
        "--trace-root",
        str(shared),
        "--gpu-memory-utilization",
        "0.7",
        "--max-num-seqs",
        "8",
    )
    assert result.returncode == 0, result.stderr
    launch = json.loads(result.stdout)
    assert launch["serve_command"] is None
    if ip == "192.0.2.10":
        assert launch["profile_command"] == [
            "bash",
            str(DIRECTORY / "profile_edgeshard.sh"),
        ]
        assert launch["profile_environment"]["PROFILE_GPU_MEMORY_UTILIZATION"] == "0.7"
        assert launch["profile_environment"]["PROFILE_MAX_NUM_SEQS"] == "8"
        assert launch["profile_environment"]["TRACE_ROOT"] == str(shared)
    else:
        assert launch["profile_command"] is None


def test_profile_uses_default_local_directory(model):
    result = invoke(model, "--profile")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["profile_environment"]["TRACE_ROOT"].endswith(
        "/vllm_hetero_pp_logs/profiles"
    )


def test_fixed_kv_configuration_is_replayed_in_normal_service(model):
    result = invoke(model, "--block-size", "128", "--num-gpu-blocks-override", "64")
    assert result.returncode == 0, result.stderr
    command = json.loads(result.stdout)["serve_command"]
    assert command[command.index("--block-size") + 1] == "128"
    assert command[command.index("--num-gpu-blocks-override") + 1] == "64"


@pytest.mark.parametrize("profile_exit", [0, 7])
def test_profile_head_runs_after_cluster_ready_without_serving(
    launcher,
    model,
    tmp_path,
    monkeypatch,
    profile_exit,
):
    args = launcher.argument_parser().parse_args(
        [
            "--head",
            "192.0.2.10",
            "--peer",
            "192.0.2.20",
            "--model",
            str(model),
            "--profile",
            "--trace-root",
            str(tmp_path),
        ]
    )
    launch = launcher.make_launch(args, args.head, sys.executable, {})
    calls = []

    def capture(command, env, timeout):
        calls.append("ray" if "start" in command else "cluster/storage")
        return 0, "ready"

    def profile(command, env, log, timeout):
        assert len(calls) == 2
        calls.append("profile")
        assert env["PLAN"] == str(tmp_path / "plan.json")
        assert env["PY"] == sys.executable
        return profile_exit

    monkeypatch.setattr(launcher, "capture", capture)
    monkeypatch.setattr(launcher, "profile_foreground", profile)
    monkeypatch.setattr(
        launcher, "serve_foreground", lambda *a: pytest.fail("API started")
    )
    assert (
        launcher.start_services(args, launch, {}, tmp_path, sys.executable)
        == profile_exit
    )
    assert calls == ["ray", "cluster/storage", "profile"]
    receipt = json.loads((tmp_path / "profile_exit.json").read_text())
    assert receipt["profile_and_planning_completed"] is (profile_exit == 0)
    assert receipt["measured_performance_improvement"] is False


def test_profile_foreground_preserves_failure_and_output(launcher, tmp_path):
    log = tmp_path / "profile.log"
    rc = launcher.profile_foreground(
        [sys.executable, "-c", "print('profile table'); raise SystemExit(7)"],
        os.environ.copy(),
        log,
        5,
    )
    assert rc == 7
    assert "profile table" in log.read_text()


def test_profile_foreground_timeout_stops_owned_child(launcher, tmp_path):
    pidfile = tmp_path / "pid"
    program = (
        "import os, time; from pathlib import Path; "
        f"Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    with pytest.raises(TimeoutError, match="profiling"):
        launcher.profile_foreground(
            [sys.executable, "-c", program], os.environ.copy(), tmp_path / "log", 0.5
        )
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


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
