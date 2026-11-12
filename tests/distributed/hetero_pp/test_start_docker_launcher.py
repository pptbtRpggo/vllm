# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the host launcher against network/Docker boundaries, without NPUs."""

import os
import shlex
import signal
import stat
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "examples/online_serving/hetero_pp/start_hetero_pp_docker.sh"


@pytest.fixture
def environment(tmp_path):
    ip = tmp_path / "ip"
    ip.write_text(
        "#!/bin/bash\n"
        'case "$*" in\n'
        '  "-o -4 addr show") '
        'echo "2: enp3s0 inet ${TEST_LOCAL_IP}/24 scope global enp3s0" ;;\n'
        '  "-4 route get "*) echo "141.147.135.54 dev enp3s0 src ${TEST_LOCAL_IP}" ;;\n'
        "  *) exit 9 ;;\n"
        "esac\n"
    )
    ip.chmod(0o700)
    model = tmp_path / "model with spaces"
    model.mkdir()
    return dict(
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        HOME=str(tmp_path),
        TEST_LOCAL_IP="141.147.135.53",
        TEST_MODEL=str(model),
    )


def invoke(environment, *arguments):
    return subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--head",
            "53",
            "--peer",
            "54",
            "--model",
            environment["TEST_MODEL"],
            *arguments,
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize(
    "local,soc", [("141.147.135.53", "910b"), ("141.147.135.54", "310p")]
)
def test_local_ip_selects_image_and_nic_without_npu_smi(environment, local, soc):
    environment["TEST_LOCAL_IP"] = local
    result = invoke(environment, "--dry-run")
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert command[:2] == ["docker", "run"]
    assert ("seccomp=unconfined" in command) == (soc == "310p")
    assert ("--security-opt" in command) == (soc == "310p")
    assert "--entrypoint" not in command
    assert "--init" not in command
    image = (
        "quay.io/ascend/vllm-ascend:v0.23.0-openeuler"
        if soc == "910b"
        else "quay.io/ascend/vllm-ascend:v0.23.0-310p-openeuler"
    )
    assert image in command
    assert "GLOO_SOCKET_IFNAME=enp3s0" in command
    assert f"VLLM_HOST_IP={local}" in command
    assert "--head" in command and "141.147.135.53" in command
    assert "--peer" in command and "141.147.135.54" in command
    assert command[command.index("--ray-port") + 1] == "16379"
    assert command[command.index("--api-port") + 1] == "8000"
    assert not any("ASCEND_RT_VISIBLE_DEVICES" in arg for arg in command)
    assert command.count("--devices") == 0
    mounts = [command[i + 1] for i, arg in enumerate(command) if arg == "--mount"]
    assert (
        f"type=bind,src={environment['TEST_MODEL']},dst=/workspace/model,readonly"
        in mounts
    )


def test_image_ports_and_noncontiguous_host_devices_can_be_overridden(environment):
    result = invoke(
        environment,
        "--image",
        "admin/910b:tested",
        "--devices",
        "0,2",
        "--interface",
        "enp3s0",
        "--ray-port",
        "18001",
        "--api-port",
        "18002",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert "admin/910b:tested" in command
    assert "/dev/davinci0" in command and "/dev/davinci2" in command
    assert "/dev/davinci1" not in command
    assert "GLOO_SOCKET_IFNAME=enp3s0" in command
    assert command[command.index("--ray-port") + 1] == "18001"
    assert command[command.index("--api-port") + 1] == "18002"


@pytest.mark.parametrize(
    "arguments",
    [
        ("--head", "256"),
        ("--peer", "53"),
        ("--devices", "0,0"),
        ("--ray-port", "8000"),
        ("--api-port", "65536"),
        ("--layers", "18,18,6"),
    ],
)
def test_invalid_configuration_never_generates_launch(environment, arguments):
    result = invoke(environment, *arguments, "--dry-run")
    assert result.returncode != 0
    assert not result.stdout.startswith("docker run")


def test_unrelated_host_is_not_treated_as_310p(environment):
    environment["TEST_LOCAL_IP"] = "141.147.135.55"
    result = invoke(environment, "--dry-run")
    assert result.returncode != 0


def test_empty_port_environment_uses_defaults_and_container_options_are_forwarded(
    environment,
):
    environment.update(RAY_PORT="", SERVICE_PORT="")
    result = invoke(environment, "--startup-timeout", "900", "--dry-run")
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert command[command.index("--ray-port") + 1] == "16379"
    assert command[command.index("--api-port") + 1] == "8000"
    assert command[-2:] == ["--startup-timeout", "900"]


def test_interface_must_own_the_selected_local_address(environment):
    result = invoke(environment, "--interface", "missing0", "--dry-run")
    assert result.returncode != 0
    assert not result.stdout.startswith("docker run")


def test_logs_default_to_host_home_and_dry_run_does_not_create_them(environment):
    result = invoke(environment, "--dry-run")
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    mounts = [command[i + 1] for i, arg in enumerate(command) if arg == "--mount"]
    base = Path(environment["HOME"]) / "vllm_hetero_pp_logs"
    assert f"type=bind,src={base}/preview-head,dst=/work/hetero_logs" in mounts
    assert command[command.index("--output") + 1] == "/work/hetero_logs/start"
    assert not base.exists()


def test_custom_log_directory_is_forwarded_as_a_host_mount(environment, tmp_path):
    base = tmp_path / "readable logs"
    result = invoke(environment, "--log-dir", str(base), "--dry-run")
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert f"type=bind,src={base}/preview-head,dst=/work/hetero_logs" in command
    assert not base.exists()


def test_container_logs_remain_readable_with_restrictive_initial_umask(
    environment, tmp_path
):
    # The Docker boundary maps the generated host log directory for this fixture.
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/bash\n"
        "while (($#)); do\n"
        '  if [[ "$1" == --mount && "$2" == *dst=/work/hetero_logs ]]; then\n'
        '    value="${2#type=bind,src=}"\n'
        '    export TEST_LOG_ROOT="${value%,dst=/work/hetero_logs}"\n'
        "    shift 2; continue\n"
        "  fi\n"
        '  if [[ "$1" == -e ]]; then export "$2"; shift 2; continue; fi\n'
        '  if [[ "$1" == *v0.23.0-openeuler ]]; then shift; exec "$@"; fi\n'
        "  shift\n"
        "done\nexit 99\n"
    )
    docker.chmod(0o700)
    fake_bash = tmp_path / "bash"
    fake_bash.write_text(
        '#!/bin/sh\n: "${TEST_LOG_ROOT:?missing log mount}"\n'
        'mkdir "$TEST_LOG_ROOT/start"\n'
        'echo readable > "$TEST_LOG_ROOT/start/result.log"\n'
    )
    fake_bash.chmod(0o700)
    result = subprocess.run(
        [
            "/bin/bash",
            str(SCRIPT),
            "--head",
            "53",
            "--peer",
            "54",
            "--model",
            environment["TEST_MODEL"],
        ],
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
        umask=0o077,
    )
    assert result.returncode == 0, result.stderr
    base = Path(environment["HOME"]) / "vllm_hetero_pp_logs"
    runs = list(base.glob("hetero_pp_head_*"))
    assert len(runs) == 1
    assert stat.S_IMODE(base.stat().st_mode) == 0o700
    assert stat.S_IMODE(runs[0].stat().st_mode) == 0o700
    assert stat.S_IMODE((runs[0] / "start").stat().st_mode) == 0o755
    log = runs[0] / "start/result.log"
    assert stat.S_IMODE(log.stat().st_mode) == 0o644
    assert log.read_text() == "readable\n"


def test_container_failure_is_preserved_without_removing_existing_container(
    environment, tmp_path
):
    docker = tmp_path / "docker"
    docker.write_text('#!/bin/bash\n[[ "$1" == run ]] || exit 99\nexit 7\n')
    docker.chmod(0o700)
    result = invoke(environment)
    assert result.returncode == 7


def test_worker_container_stays_alive_after_ray_join(environment, tmp_path):
    # Exercise the real container shell: joining Ray must not exit the container.
    environment["TEST_LOCAL_IP"] = "141.147.135.54"
    result = invoke(environment, "--dry-run")
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    image = command.index("quay.io/ascend/vllm-ascend:v0.23.0-310p-openeuler")
    assert command[image + 1 : image + 3] == ["/bin/bash", "-c"]
    assert "HETERO_PP_ROLE=worker" in command
    fake_bash = tmp_path / "bash"
    fake_bash.write_text('#!/bin/sh\necho "Ray joined fixture"\nexit 0\n')
    fake_bash.chmod(0o700)
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/bash\n"
        "while (($#)); do\n"
        '  if [[ "$1" == -e ]]; then export "$2"; shift 2; continue; fi\n'
        '  if [[ "$1" == *v0.23.0-310p-openeuler ]]; then\n'
        '    shift; exec "$@"\n'
        "  fi\n"
        "  shift\n"
        "done\nexit 99\n"
    )
    docker.chmod(0o700)
    log = tmp_path / "worker.log"
    with log.open("w") as stream:
        process = subprocess.Popen(
            [
                "/bin/bash",
                str(SCRIPT),
                "--head",
                "53",
                "--peer",
                "54",
                "--model",
                environment["TEST_MODEL"],
            ],
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 3
            while (
                "Ray joined fixture" not in log.read_text()
                and time.monotonic() < deadline
            ):
                time.sleep(0.02)
            assert "Ray joined fixture" in log.read_text()
            assert process.poll() is None
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3)
