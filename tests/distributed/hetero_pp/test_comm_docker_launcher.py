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
    assert "v0.23.0.post1" in image
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
