# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise Docker command generation without starting Docker or using NPUs."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "examples/online_serving/hetero_pp/check_cpu_gloo_docker.sh"


def run_script(*arguments, env=None):
    return subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize("soc", ["910b", "310p"])
def test_dry_run_mounts_source_and_devices_without_overwriting_cann(soc):
    result = run_script("--soc", soc, "--head", "192.168.1.10", "--dry-run")
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.strip())
    assert command[:2] == ["docker", "run"]
    assert command[command.index("--network") + 1] == "host"
    assert "--privileged" not in command
    assert "/dev/davinci0" in command
    assert "/dev/davinci1" in command
    mounts = [command[i + 1] for i, arg in enumerate(command) if arg == "--mount"]
    assert f"type=bind,src={ROOT},dst=/work/vllm" in mounts
    assert not any("dst=/usr/local/Ascend," in mount for mount in mounts)
    image = command[command.index("--entrypoint") + 2]
    assert "v0.23.0" in image
    assert ".post" not in image
    assert image.endswith("-310p") == (soc == "310p")
    assert command[-4:] == ["--head", "192.168.1.10", "--devices", "0,1"]


def test_custom_image_and_test_options_are_forwarded_as_separate_arguments():
    result = run_script(
        "--image",
        "local/admin:approved",
        "--head",
        "192.168.1.10",
        "--devices",
        "4,5",
        "--port",
        "19190",
        "--timeout",
        "300",
        "--interface",
        "eth1",
        "--cann-env",
        "/opt/sdk/setenv.bash",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.strip())
    assert "local/admin:approved" in command
    assert "/dev/davinci4" in command and "/dev/davinci5" in command
    assert command[-8:] == [
        "--port",
        "19190",
        "--timeout",
        "300",
        "--interface",
        "eth1",
        "--cann-env",
        "/opt/sdk/setenv.bash",
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--soc", "910b"],
        ["--head", "192.168.1.10"],
        ["--soc", "910b", "--head", "192.168.1.10", "--devices", "0,0"],
        ["--soc", "910b", "--head", "192.168.1.10", "--port", "65535"],
        ["--soc", "910b", "--head", "192.168.1.10", "--unexpected"],
    ],
)
def test_invalid_arguments_do_not_generate_a_docker_command(arguments):
    result = run_script(*arguments, "--dry-run")
    assert result.returncode != 0
    assert not result.stdout.startswith("docker run")


def test_container_failure_is_not_reported_as_success(tmp_path):
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\nexit 7\n")
    docker.chmod(0o700)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}")
    result = run_script("--soc", "910b", "--head", "192.168.1.10", env=env)
    assert result.returncode == 7


def test_manual_template_passes_host_settings_without_expanding_them(tmp_path):
    docker = tmp_path / "docker"
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\nexit 7\n')
    docker.chmod(0o700)
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        IMAGE="admin/310p:approved",
        MASTER_ADDR="192.168.1.10",
        MASTER_PORT="29501",
        HOST_PROJECT_DIR=str(tmp_path / "source with spaces"),
    )
    manual = ROOT / "examples/online_serving/hetero_pp/check_cpu_gloo_docker_manual.sh"
    result = subprocess.run(
        ["bash", str(manual)], env=env, text=True, capture_output=True, timeout=10
    )
    assert result.returncode == 7, result.stderr
    arguments = result.stdout.splitlines()
    assert "admin/310p:approved" in arguments
    assert f"{tmp_path}/source with spaces:/work/vllm" in arguments
    assert "MASTER_ADDR=192.168.1.10" in arguments
    assert "MASTER_PORT=29501" in arguments
    assert "--rm" in arguments and "--network=host" in arguments
    assert "rm" not in arguments
    assert "SERVICE_PORT" not in arguments


def test_manual_passes_values_even_when_docker_environment_is_cleared(tmp_path):
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/bash\n"
        'if [[ "${FAKE_STAGE:-}" != clean ]]; then\n'
        "  exec env -u MASTER_ADDR -u MASTER_PORT -u NPU_DEVICES "
        'FAKE_STAGE=clean "$0" "$@"\n'
        "fi\n"
        "while (($#)); do\n"
        '  if [[ "$1" == -e ]]; then export "$2"; shift; fi\n'
        "  shift\n"
        "done\n"
        'printf "%s\\n" "${MASTER_ADDR:-MISSING}" '
        '"${MASTER_PORT:-MISSING}" "${NPU_DEVICES:-MISSING}"\n'
    )
    docker.chmod(0o700)
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        MASTER_ADDR="144.147.135.53",
        MASTER_PORT="29501",
        NPU_DEVICES="0,1",
    )
    env.pop("FAKE_STAGE", None)
    manual = ROOT / "examples/online_serving/hetero_pp/check_cpu_gloo_docker_manual.sh"
    result = subprocess.run(
        ["bash", str(manual)], env=env, text=True, capture_output=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["144.147.135.53", "29501", "0,1"]
