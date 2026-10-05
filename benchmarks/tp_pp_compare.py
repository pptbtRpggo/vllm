# SPDX-License-Identifier: Apache-2.0
"""Run TP=4 against the exact requests used by the P99 PP comparison.

This records a matched workload; it does not make software network emulation
equivalent to a physical two-host link. Repeat runs before drawing conclusions.
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from pp_sla import read_samples, run_load, save

SCENARIOS = {
    "p99_a": dict(base=8, raised=12, formal=(64, 160), bandwidth=100, latency=1e-6),
    "p99_b": dict(base=32, raised=48, formal=(160, 416), bandwidth=100, latency=1e-6),
    "p99_c": dict(base=16, raised=24, formal=(480, 640), bandwidth=25, latency=1),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-pp", type=Path, required=True,
                        help="Existing PP uniform formal_c{base}.json for request checks")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--validate-only", action="store_true",
                        help="Check paired requests and settings without starting a server")
    args = parser.parse_args()
    spec = SCENARIOS[args.scenario]
    if args.output.exists() and not args.validate_only:
        raise FileExistsError(f"output already exists: {args.output}")
    warm = read_samples(args.data / "warmup.jsonl")
    evaluation = read_samples(args.data / "evaluation.jsonl")
    formal = evaluation[slice(*spec["formal"])]
    assert len(formal) == spec["formal"][1] - spec["formal"][0]
    assert len(warm) >= spec["raised"]
    expected = json.loads(args.reference_pp.read_text())["records"]
    assert [r["prompt_sha256"] for r in expected] == [
        r["prompt_sha256"] for r in formal
    ], "TP and PP formal prompts or order differ"
    assert [r["requested_output_tokens"] for r in expected] == [
        r["output_tokens"] for r in formal
    ], "TP and PP requested output lengths differ"
    assert not ({r["prompt_sha256"] for r in warm[:spec["raised"]]} &
                {r["prompt_sha256"] for r in formal})
    if args.validate_only:
        print(json.dumps({
            "scenario": args.scenario,
            "requests": len(formal),
            "prompts_and_output_lengths_match_pp": True,
            "concurrency": [spec["base"], spec["raised"]],
            "network": [spec["bandwidth"], spec["latency"]],
        }))
        return

    args.output.mkdir(parents=True)
    env = os.environ.copy()
    for key in (*[k for k in env if k.startswith("VLLM_PP_")],
                *[k for k in env if k.startswith("VLLM_TP_")]):
        env.pop(key)
    env.update(
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_TP_COMPUTE_SCALES="1,1,2,4",
        VLLM_TP_CROSS_GROUP_SIZE="2",
        VLLM_TP_CROSS_EXTRA_BANDWIDTH_GBPS=str(spec["bandwidth"]),
        VLLM_TP_CROSS_EXTRA_LATENCY_MS=str(spec["latency"]),
    )
    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", args.model,
        "--served-model-name", "tp-34b", "--host", "127.0.0.1",
        "--port", str(args.port), "--dtype", "float16",
        "--tensor-parallel-size", "4", "--pipeline-parallel-size", "1",
        "--distributed-executor-backend", "mp", "--enforce-eager",
        "--max-model-len", "4096", "--max-num-seqs", "256",
        "--max-num-batched-tokens", "2048", "--block-size", "128",
        "--gpu-memory-utilization", "0.85", "--num-gpu-blocks-override", "1024",
        "--no-enable-prefix-caching", "--enable-chunked-prefill",
        "--disable-log-requests",
    ]
    save(args.output / "experiment.json", {
        "scenario": args.scenario,
        "model": args.model,
        "formal_indices": spec["formal"],
        "concurrency": [spec["base"], spec["raised"]],
        "compute_scales": [1, 1, 2, 4],
        "network_mode": "extra",
        "cross_extra_bandwidth_gbps": spec["bandwidth"],
        "cross_extra_latency_ms": spec["latency"],
        "command": command,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "reference_pp": str(args.reference_pp),
    })
    with (args.output / "server.log").open("w") as log:
        proc = subprocess.Popen(command, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited {proc.returncode}; see server.log")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{args.port}/health", timeout=2
                    ) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(2)
            else:
                raise TimeoutError("server startup timed out")
            for concurrency in (spec["base"], spec["raised"]):
                url = f"http://127.0.0.1:{args.port}"
                warmed = asyncio.run(run_load(
                    warm[:max(16, concurrency)], url, "tp-34b", concurrency,
                    output_override=16,
                ))
                save(args.output / f"warmup_c{concurrency}.json", warmed)
                if warmed["summary"]["failed"]:
                    raise RuntimeError(f"warmup failed at concurrency {concurrency}")
                result = asyncio.run(run_load(
                    formal, url, "tp-34b", concurrency,
                ))
                save(args.output / f"formal_c{concurrency}.json", result)
                print(args.scenario, concurrency, json.dumps(result["summary"]), flush=True)
                if result["summary"]["failed"]:
                    raise RuntimeError(f"formal requests failed at concurrency {concurrency}")
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


if __name__ == "__main__":
    main()
