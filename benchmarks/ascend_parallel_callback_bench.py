# SPDX-License-Identifier: Apache-2.0
"""Compare native, zero-delay, active-callback and tracing TP/PP in one server."""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from pp_sla import read_samples, run_load, save

CASES = (
    "native_a",
    "zero_a",
    "callback_zero_a",
    "trace_zero_a",
    "native_b",
    "trace_zero_b",
    "callback_zero_b",
    "zero_b",
    "native_c",
)


def set_control(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state))
    temporary.replace(path)


def run_architecture(args, architecture, warm, samples):
    root = Path(args.output) / architecture
    root.mkdir(parents=True, exist_ok=True)
    control_file = root / "control.json"
    set_control(
        control_file,
        dict(mode="native", label="startup", phase="warmup", concurrency=1),
    )
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("VLLM_PP_", "VLLM_TP_")):
            env.pop(key)
    env.update(
        VLLM_CALLBACK_BENCH_CONTROL=str(control_file.resolve()),
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_NO_USAGE_STATS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTORCH_NPU_ALLOC_CONF="expandable_segments:True",
        PYTHONPATH=str(Path("benchmarks").resolve())
        + ":"
        + str(Path.cwd())
        + ":"
        + env.get("PYTHONPATH", ""),
    )
    if architecture == "tp":
        env.update(
            VLLM_TP_COMPUTE_SCALES="1,1,1,1",
            VLLM_TP_MOCK_TRACE=str((root / "trace").resolve()),
        )
        worker = "TPCallbackBenchWorker"
    else:
        env.update(
            VLLM_PP_STAGE_TRACE=str((root / "trace").resolve()),
            VLLM_PP_COMPUTE_MODEL="layer-measured",
        )
        worker = "PPCallbackBenchWorker"
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--served-model-name",
        "callback-34b",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dtype",
        "float16",
        "--tensor-parallel-size",
        "4" if architecture == "tp" else "1",
        "--pipeline-parallel-size",
        "1" if architecture == "tp" else "4",
        "--distributed-executor-backend",
        "mp",
        "--worker-cls",
        "ascend_parallel_callback_worker." + worker,
        "--enforce-eager",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "256",
        "--max-num-batched-tokens",
        "2048",
        "--block-size",
        "128",
        "--gpu-memory-utilization",
        "0.85",
        "--num-gpu-blocks-override",
        "1024",
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--disable-log-requests",
    ]
    save(
        root / "settings.json",
        dict(
            command=command,
            cases=CASES,
            formal_output_tokens=16,
            concurrency=[1, 8, 32],
            counts={"1": 16, "8": 64, "32": 128},
            git_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        ),
    )
    url = f"http://127.0.0.1:{args.port}"
    with (root / "server.log").open("w") as log:
        proc = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"{architecture} server exited {proc.returncode}"
                    )
                try:
                    with urllib.request.urlopen(url + "/health", timeout=2):
                        break
                except (OSError, TimeoutError):
                    time.sleep(2)
            else:
                raise TimeoutError(f"{architecture} server startup timeout")
            measured_single = set()
            for label in CASES:
                mode = label.rsplit("_", 1)[0]
                points = [8, 32]
                if mode not in measured_single:
                    points.insert(0, 1)
                    measured_single.add(mode)
                for concurrency in points:
                    state = dict(mode=mode, label=label, concurrency=concurrency)
                    set_control(control_file, dict(**state, phase="warmup"))
                    warmed = asyncio.run(
                        run_load(
                            warm[: max(8, concurrency)],
                            url,
                            "callback-34b",
                            concurrency,
                            output_override=16,
                        )
                    )
                    if warmed["summary"]["failed"]:
                        raise RuntimeError(f"{architecture}/{label} warmup failed")
                    set_control(control_file, dict(**state, phase="formal"))
                    chosen = (
                        samples[:16]
                        if concurrency == 1
                        else (samples[:64] if concurrency == 8 else samples[128:256])
                    )
                    result = asyncio.run(
                        run_load(
                            chosen,
                            url,
                            "callback-34b",
                            concurrency,
                            output_override=16,
                        )
                    )
                    save(
                        root / f"{label}_c{concurrency}.json",
                        dict(warmup=warmed, formal=result),
                    )
                    print(
                        architecture,
                        label,
                        f"c{concurrency}",
                        json.dumps(result["summary"]),
                        flush=True,
                    )
                    if result["summary"]["failed"]:
                        raise RuntimeError(f"{architecture}/{label} requests failed")
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=18790)
    parser.add_argument("--architecture", choices=["tp", "pp", "both"], default="both")
    parser.add_argument("--data-dir", default="output/pp_sla_34b_20260929/data")
    args = parser.parse_args()
    warm = read_samples(Path(args.data_dir) / "warmup.jsonl")
    samples = read_samples(Path(args.data_dir) / "evaluation.jsonl")
    assert len(warm) >= 32 and len(samples) >= 256
    assert not (
        {r["prompt_sha256"] for r in warm[:32]}
        & {r["prompt_sha256"] for r in samples[:256]}
    )
    for architecture in (
        ("tp", "pp") if args.architecture == "both" else (args.architecture,)
    ):
        run_architecture(args, architecture, warm, samples)


if __name__ == "__main__":
    main()
