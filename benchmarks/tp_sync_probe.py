# SPDX-License-Identifier: Apache-2.0
"""Small, same-workload comparison of native TP and mock synchronization points."""

from __future__ import annotations

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


def run_server(kind: str, output: Path, model: str, warm, formal, port: int) -> None:
    folder = output / kind
    folder.mkdir(parents=True)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("VLLM_PP_", "VLLM_TP_")):
            env.pop(key)
    env.update(
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_NO_USAGE_STATS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTORCH_NPU_ALLOC_CONF="expandable_segments:True",
        PYTHONPATH=str(Path.cwd()) + ":" + env.get("PYTHONPATH", ""),
    )
    mode_file = folder / "mode.txt"
    mode_file.write_text("none\n")
    if kind == "probe":
        env["VLLM_TP_SYNC_PROBE_MODE_FILE"] = str(mode_file.resolve())
        env["VLLM_TP_SYNC_PROBE_TRACE_DIR"] = str((folder / "worker_trace").resolve())
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model,
        "--served-model-name",
        "tp-34b",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "float16",
        "--tensor-parallel-size",
        "4",
        "--pipeline-parallel-size",
        "1",
        "--distributed-executor-backend",
        "mp",
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
    if kind == "probe":
        command += ["--worker-cls", "benchmarks.tp_sync_probe_worker.TPSyncProbeWorker"]
    save(
        folder / "settings.json",
        {"command": command, "formal_ids": [r["id"] for r in formal]},
    )
    with (folder / "server.log").open("w") as log:
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
                    raise RuntimeError(f"{kind} server exited {proc.returncode}")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=2
                    ):
                        break
                except (OSError, TimeoutError):
                    time.sleep(2)
            else:
                raise TimeoutError(f"{kind} server startup timed out")
            modes = (
                ["native"]
                if kind == "native"
                else [
                    "none",
                    "event_layer",
                    "layer",
                    "event_collective",
                    "collective",
                    "event_both",
                    "both",
                    "none_repeat",
                ]
            )
            for mode in modes:
                mode_file.write_text(("none" if mode == "none_repeat" else mode) + "\n")
                url = f"http://127.0.0.1:{port}"
                warmed = asyncio.run(
                    run_load(warm, url, "tp-34b", 8, output_override=8)
                )
                save(folder / f"warmup_{mode}.json", warmed)
                if warmed["summary"]["failed"]:
                    raise RuntimeError(f"warmup {mode} failed")
                result = asyncio.run(
                    run_load(formal, url, "tp-34b", 8, output_override=16)
                )
                save(folder / f"formal_{mode}.json", result)
                print(kind, mode, json.dumps(result["summary"]), flush=True)
                if result["summary"]["failed"]:
                    raise RuntimeError(f"formal {mode} failed")
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: tp_sync_probe.py MODEL OUTPUT")
    model, output = sys.argv[1], Path(sys.argv[2]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = Path("output/pp_sla_34b_20260929/data")
    warm = read_samples(data / "warmup.jsonl")[:16]
    formal = read_samples(data / "evaluation.jsonl")[:32]
    for kind in ("native", "probe"):
        run_server(kind, output, model, warm, formal, 18775)


if __name__ == "__main__":
    main()
