# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the benchmark launcher without EvalScope, HTTP, or NPUs."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples/online_serving/hetero_pp/benchmark_gsm8k.sh"
)


@pytest.fixture
def benchmark(tmp_path):
    dataset = tmp_path / "gsm8k questions.jsonl"
    dataset.write_text('{"question":"What is 1 + 1?","answer":"2"}\n')
    executable = tmp_path / "evalscope fake"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ ${2:-} == --help ]]; then\n"
        "  echo '--parallel --number --warmup-num --outputs-dir --stream'\n"
        "  exit 0\n"
        "fi\n"
        "if [[ ${1:-} == --version ]]; then echo 'fake-version'; exit 0; fi\n"
        'printf "%s\\0" "$@" > "$CAPTURE"\n'
        "echo 'fake benchmark output (no requests sent)'\n"
        'exit "${FAKE_FAIL:-0}"\n'
    )
    executable.chmod(0o755)
    output = tmp_path / "new results"
    capture = tmp_path / "captured.args"

    def invoke(*extra, env=None, shell="bash"):
        return subprocess.run(
            [
                shell,
                str(SCRIPT),
                "--dataset-path",
                str(dataset),
                "--evalscope",
                str(executable),
                "--output-dir",
                str(output),
                *extra,
            ],
            env={**os.environ, "CAPTURE": str(capture), **(env or {})},
            capture_output=True,
            text=True,
            timeout=10,
        )

    return invoke, dataset, output, capture


def read_args(capture):
    return capture.read_bytes().decode().rstrip("\0").split("\0")


def values(args, name):
    following = args[args.index(name) + 1 :]
    result = []
    for value in following:
        if value.startswith("--"):
            break
        result.append(value)
    return result


def test_default_sweep_keeps_each_request_count_and_streams(benchmark):
    invoke, dataset, output, capture = benchmark
    result = invoke()
    assert result.returncode == 0, result.stderr
    args = read_args(capture)
    assert args[0] == "perf"
    assert values(args, "--parallel") == ["8", "16", "32", "64"]
    assert values(args, "--number") == ["512"] * 4
    assert values(args, "--dataset-path") == [str(dataset)]
    assert values(args, "--dataset") == ["openqa"]
    assert values(args, "--warmup-num") == ["64"]
    assert values(args, "--max-tokens") == ["128"]
    assert "--stream" in args and "--open-loop" not in args
    assert (output / "run.log").read_text().startswith("fake benchmark output")
    assert (output / "command.sh").is_file()


def test_overrides_and_single_concurrency(benchmark, tmp_path):
    invoke, _, _, capture = benchmark
    tokenizer = tmp_path / "a tokenizer"
    tokenizer.mkdir()
    result = invoke(
        "--parallel",
        "16",
        "--number",
        "128",
        "--max-tokens",
        "64",
        "--url",
        "http://example.test:8100/v1/chat/completions",
        "--model",
        "my model",
        "--tokenizer-path",
        str(tokenizer),
    )
    assert result.returncode == 0, result.stderr
    args = read_args(capture)
    assert values(args, "--parallel") == ["16"]
    assert values(args, "--number") == ["128"]
    assert values(args, "--tokenizer-path") == [str(tokenizer)]
    assert values(args, "--model") == ["my model"]


def test_evalscope_error_propagates_and_preserves_log(benchmark):
    invoke, _, output, _ = benchmark
    result = invoke(env={"FAKE_FAIL": "23"})
    assert result.returncode == 23
    assert "fake benchmark output" in (output / "run.log").read_text()


@pytest.mark.parametrize(
    "extra",
    [
        ("--parallel", "0"),
        ("--parallel", "8", "oops"),
        ("--number", "0"),
        ("--number", "32"),
        ("--max-tokens", "-1"),
        ("--warmup-num", "1"),
        ("--unknown-option",),
    ],
)
def test_invalid_arguments_never_start_benchmark(benchmark, extra):
    invoke, _, output, capture = benchmark
    result = invoke(*extra)
    assert result.returncode == 2
    assert not capture.exists()
    assert not output.exists()


def test_existing_output_is_never_overwritten(benchmark):
    invoke, _, output, capture = benchmark
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    result = invoke()
    assert result.returncode == 2
    assert (output / "keep.txt").read_text() == "keep"
    assert not capture.exists()


def test_missing_dataset_never_starts_benchmark(benchmark):
    invoke, dataset, output, capture = benchmark
    dataset.unlink()
    result = invoke()
    assert result.returncode == 2
    assert not output.exists() and not capture.exists()


def test_dry_run_does_not_require_evalscope_or_write_results(benchmark):
    invoke, _, output, capture = benchmark
    result = invoke("--evalscope", "/missing/evalscope", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "--parallel 8 16 32 64" in result.stdout
    assert not output.exists() and not capture.exists()


def test_sh_invocation_reexecutes_bash(benchmark):
    invoke, _, _, capture = benchmark
    result = invoke(shell="sh")
    assert result.returncode == 0, result.stderr
    assert capture.exists()
