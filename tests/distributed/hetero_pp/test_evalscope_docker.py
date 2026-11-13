# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test client orchestration and setup with Docker/download boundaries replaced."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).resolve().parents[3] / "examples/online_serving/hetero_pp"
LAUNCHER = DIRECTORY / "start_evalscope_docker.sh"
BOOTSTRAP = DIRECTORY / "evalscope_container_setup.sh"


def launch(tmp_path, *extra, fail=""):
    capture = tmp_path / "docker.calls"
    docker = tmp_path / "docker"
    docker.write_text(
        '#!/bin/bash\nprintf "%q " "$@" >> "$CAPTURE"\n'
        'echo >> "$CAPTURE"\n'
        'if [[ "$1 $2" == "image inspect" ]]; then\n'
        '  echo "PATH=/image/python/bin:/usr/bin:/bin"\nfi\n'
        'if [[ "$1" == "$FAIL_STAGE" ]]; then exit 9; fi\n'
    )
    docker.chmod(0o700)
    result = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--name",
            "evalscope-test",
            "--output-dir",
            str(tmp_path / "results with spaces"),
            *extra,
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "CAPTURE": str(capture),
            "FAIL_STAGE": fail,
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    calls = (
        [shlex.split(line) for line in capture.read_text().splitlines()]
        if capture.exists()
        else []
    )
    return result, calls


def test_client_copies_code_and_data_and_exposes_venv_without_npus(tmp_path):
    result, calls = launch(tmp_path)
    assert result.returncode == 0, result.stderr
    create = next(call for call in calls if call[0] == "create")
    assert "quay.io/ascend/vllm-ascend:v0.23.0-openeuler" in create
    assert "--net=host" in create
    assert "--device" not in create and "--privileged" not in create
    assert "--security-opt" not in create and "--rm" not in create
    assert not any("Ascend/driver" in value for value in create)
    assert (
        "PATH=/tmp/evalscope-client/.venv/bin:/tmp/evalscope-client/tools:/image/python/bin:/usr/bin:/bin"
        in create
    )
    assert "TORCH_DEVICE_BACKEND_AUTOLOAD=0" in create
    assert create[create.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert create[create.index("--entrypoint") + 1] == "/bin/bash"
    copies = [call for call in calls if call[0] == "cp"]
    assert {call[-1] for call in copies} == {
        "evalscope-test:/tmp/evalscope-setup.sh",
        "evalscope-test:/tmp/benchmark_gsm8k.sh",
        "evalscope-test:/tmp/evalscope-gsm8k",
    }
    assert calls[-1] == ["start", "-ai", "evalscope-test"]


def test_copy_failure_never_starts_or_deletes_container(tmp_path):
    result, calls = launch(tmp_path, fail="cp")
    assert result.returncode == 9
    assert not any(call[0] in ("start", "rm", "stop") for call in calls)


def test_client_forwards_explicit_http_mirror_and_trust_host(tmp_path):
    result, calls = launch(
        tmp_path,
        "--index-url",
        "http://mirrors.tools.huawei.com/pypi/simple",
        "--trusted-host",
        "mirrors.tools.huawei.com",
    )
    assert result.returncode == 0, result.stderr
    create = next(call for call in calls if call[0] == "create")
    assert "EVALSCOPE_INDEX_URL=http://mirrors.tools.huawei.com/pypi/simple" in create
    assert "EVALSCOPE_TRUSTED_HOST=mirrors.tools.huawei.com" in create


@pytest.fixture
def setup_environment(tmp_path):
    home = tmp_path / "client"
    bindir = home / ".venv/bin"
    bindir.mkdir(parents=True)
    results = tmp_path / "results"
    results.mkdir()
    capture = tmp_path / "pip.calls"
    # Replace only package downloads. The setup script, copies, PATH, help
    # checks, ready marker, symlink and final interactive Bash are real.
    python = bindir / "python"
    python.write_text(
        '#!/bin/bash\nprintf "%q " "$@" >> "$CAPTURE"\necho >> "$CAPTURE"\n'
        'if [[ "$1 $2 $3 $4" == "-m pip --isolated install" ]]; then\n'
        "  if [[ ${FAIL_INSTALL:-0} == 1 ]]; then exit 23; fi\n"
        "  cat > \"$HOME/.venv/bin/evalscope\" <<'CLI'\n"
        "#!/bin/bash\n"
        'if [[ "$1" == perf && "$2" == --help ]]; then\n'
        '  echo "--parallel --number --warmup-num --outputs-dir --stream"\n'
        'else echo "evalscope fixture-version"; fi\n'
        'CLI\nchmod +x "$HOME/.venv/bin/evalscope"\n'
        'elif [[ "$3" == freeze ]]; then echo "evalscope==fixture-version"; fi\n'
    )
    python.chmod(0o700)
    env = {
        **os.environ,
        "HOME": str(home),
        "CAPTURE": str(capture),
        "EVALSCOPE_BOOTSTRAP_PYTHON": sys.executable,
    }

    def invoke(**extra):
        return subprocess.run(
            [
                "bash",
                str(BOOTSTRAP),
                str(DIRECTORY / "benchmark_gsm8k.sh"),
                str(DIRECTORY / "data/gsm8k"),
                str(results),
            ],
            env={**env, **extra},
            input="command -v evalscope\nbenchmark_gsm8k.sh --dry-run\nexit\n",
            capture_output=True,
            text=True,
            timeout=10,
        )

    return invoke, home, results, capture


def test_setup_prepares_manual_commands_and_resume_does_not_reinstall(
    setup_environment,
):
    invoke, home, results, capture = setup_environment
    first = invoke()
    assert first.returncode == 0, first.stderr
    assert (home / ".evalscope-ready").exists()
    assert (home / "tools/benchmark_gsm8k.sh").stat().st_mode & 0o100
    assert (home / "tools/data/gsm8k/test.jsonl").read_bytes() == (
        DIRECTORY / "data/gsm8k/test.jsonl"
    ).read_bytes()
    assert (home / "vllm_hetero_pp_logs").resolve() == results
    assert ".venv/bin" in (home / ".bashrc").read_text()
    assert "准备完成" in first.stdout
    assert str(home / ".venv/bin/evalscope") in first.stdout
    assert str(home / "tools/data/gsm8k/test.jsonl") in first.stdout
    previous = capture.read_text()
    assert "--trusted-host" not in previous
    second = invoke()
    assert second.returncode == 0, second.stderr
    assert capture.read_text() == previous


def test_install_failure_never_marks_ready_or_claims_success(setup_environment):
    invoke, home, results, _ = setup_environment
    result = invoke(FAIL_INSTALL="1")
    assert result.returncode == 23
    assert not (home / ".evalscope-ready").exists()
    assert "准备完成" not in result.stdout
    assert (results / "evalscope_install.log").exists()


def test_setup_passes_http_mirror_trust_to_the_venv_installer(setup_environment):
    invoke, _, _, capture = setup_environment
    result = invoke(
        EVALSCOPE_INDEX_URL="http://mirrors.tools.huawei.com/pypi/simple",
        EVALSCOPE_TRUSTED_HOST="mirrors.tools.huawei.com",
    )
    assert result.returncode == 0, result.stderr
    install = shlex.split(capture.read_text().splitlines()[0])
    assert (
        install[install.index("--index-url") + 1]
        == "http://mirrors.tools.huawei.com/pypi/simple"
    )
    assert install[install.index("--trusted-host") + 1] == "mirrors.tools.huawei.com"
