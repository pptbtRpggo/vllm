# SPDX-License-Identifier: Apache-2.0
"""Ascend PP=4 experiment: measured costs, memory bounds, and serving SLA.

Run after pp_sla.py prepare. Source the Ascend environment first. This is a
single-host compute-heterogeneity experiment; it does not emulate two hosts.
Memory bounds are for this driver's fixed KV pool, not engine auto-partitioning.
"""

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from contextlib import contextmanager, suppress
from dataclasses import asdict
from pathlib import Path

from pp_sla import capacity_summary, read_samples, run_load, save

GIB = 1024**3
REPO = Path(__file__).resolve().parents[1]


def memory_bounds(observations, model_config, serving, blocks):
    """Account for the same KV token capacity on every tested partition.

    Weights and peaks are observed; KV bytes follow model geometry and the fixed
    allocation. Runtime reserve is at least 6 GiB, plus a 2 GiB safety margin.
    These reserves are conservative assumptions, not measurements of all shards.
    """
    layers = model_config["num_hidden_layers"]
    if len(observations) != 4:
        raise ValueError("expected four TP=1 worker observations")
    by_rank = {r["pp_rank"]: r for r in observations}
    if set(by_rank) != set(range(4)) or any(r["tp_rank"] for r in observations):
        raise ValueError("expected distinct PP ranks 0..3 and TP=1")
    weights = {}
    for row in observations:
        for index, size in row["layer_storage_bytes"].items():
            if int(index) in weights or size <= 0:
                raise ValueError("duplicate or empty layer storage")
            weights[int(index)] = size
    if set(weights) != set(range(layers)):
        raise ValueError("incomplete layer storage observations")
    if serving["dtype"] != "float16" or serving["kv_cache_dtype"] != "auto":
        raise ValueError("this experiment expects FP16 weights and KV")
    head_dim = model_config.get(
        "head_dim", model_config["hidden_size"] // model_config["num_attention_heads"]
    )
    kv_per_layer = (
        blocks
        * serving["block_size"]
        * 2
        * model_config["num_key_value_heads"]
        * head_dim
        * 2
    )
    residuals = []
    for row in observations:
        known = sum(row["layer_storage_bytes"].values())
        known += row["non_layer_storage_bytes"]
        known += len(row["layer_storage_bytes"]) * kv_per_layer
        residuals.append(max(0, row["peak_reserved_bytes"] - known))
    reserve = max(6 * GIB, max(residuals))
    devices = []
    for rank in range(4):
        row = by_rank[rank]
        devices.append(
            dict(
                pp_rank=rank,
                tp_rank=0,
                budget_bytes=int(
                    row["total_bytes"] * serving["gpu_memory_utilization"]
                ),
                layer_weights_bytes=[weights[i] for i in range(layers)],
                layer_kv_bytes=[kv_per_layer] * layers,
                runtime_bytes=reserve,
                activation_bytes=0,
                workspace_bytes=0,
                communication_bytes=0,
                graph_bytes=0,
                safety_margin_bytes=2 * GIB,
                first_stage_bytes=by_rank[0]["non_layer_storage_bytes"],
                last_stage_bytes=by_rank[3]["non_layer_storage_bytes"],
            )
        )
    return dict(
        version=1,
        num_layers=layers,
        pp_size=4,
        tp_size=1,
        serving_config=serving,
        devices=devices,
    )


class Experiment:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.output).resolve()
        self.url = f"http://127.0.0.1:{args.port}"
        self.model_name = "pp-sla-34b"
        self.serving = dict(
            model=str(Path(args.model).resolve()),
            revision=None,
            dtype="float16",
            quantization=None,
            kv_cache_dtype="auto",
            block_size=128,
            max_model_len=4096,
            max_num_seqs=256,
            max_num_batched_tokens=2048,
            gpu_memory_utilization=0.85,
        )
        self.data = {
            name: read_samples(Path(args.data) / f"{name}.jsonl")
            for name in ("warmup", "profile", "evaluation")
        }
        hashes = [r["prompt_sha256"] for rs in self.data.values() for r in rs]
        if len(hashes) != len(set(hashes)):
            raise ValueError("sample splits must contain unique disjoint prompts")
        if any(
            r["input_tokens"] + r["output_tokens"] > 4096
            for rs in self.data.values()
            for r in rs
        ):
            raise ValueError("samples exceed the configured model context")
        self.config = json.loads((Path(args.model) / "config.json").read_text())
        if self.config["num_hidden_layers"] != 48:
            raise ValueError("this protocol requires the 48-layer CodeLlama model")

    def rpc(self, method, args=()):
        body = json.dumps(dict(method=method, args=list(args))).encode()
        request = urllib.request.Request(
            self.url + "/collective_rpc",
            body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
        if "error" in result or "results" not in result:
            raise RuntimeError(result)
        return result["results"]

    @contextmanager
    def server(self, key, parts, trace=False):
        folder = self.root / key
        folder.mkdir(parents=True, exist_ok=False)
        env = os.environ.copy()
        for name in list(env):
            if name.startswith(("VLLM_PP_", "VLLM_PROFILER")):
                del env[name]
        env.update(
            VLLM_PP_LAYER_PARTITION=",".join(map(str, parts)),
            VLLM_PP_HETERO="1,1,2,4",
            VLLM_PP_COMPUTE_MODEL="layer-measured",
            VLLM_PP_DEVICE_ORDER="0,1,2,3",
            VLLM_SERVER_DEV_MODE="1",
            ASCEND_RT_VISIBLE_DEVICES="0,1,2,3",
            VLLM_WORKER_MULTIPROC_METHOD="spawn",
            VLLM_NO_USAGE_STATS="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            PYTORCH_NPU_ALLOC_CONF="expandable_segments:True",
            PYTHONPATH=str(REPO) + ":" + env.get("PYTHONPATH", ""),
        )
        if trace:
            env.update(
                VLLM_PP_STAGE_TRACE=str(folder / "traces"),
                VLLM_PP_TRACE_SESSION=uuid.uuid4().hex,
            )
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.serving["model"],
            "--served-model-name",
            self.model_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.args.port),
            "--dtype",
            "float16",
            "--pipeline-parallel-size",
            "4",
            "--tensor-parallel-size",
            "1",
            "--distributed-executor-backend",
            "mp",
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
            str(self.args.kv_blocks),
            "--enforce-eager",
            "--no-enable-prefix-caching",
            "--enable-chunked-prefill",
            "--disable-log-requests",
        ]
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", self.args.port))
        info = dict(
            parts=parts,
            trace=trace,
            command=command,
            environment={k: v for k, v in env.items() if k.startswith("VLLM_PP_")},
            started=time.time(),
        )
        with (folder / "server.log").open("w") as log:
            proc = subprocess.Popen(
                command,
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            info["pid"] = proc.pid
            save(folder / "lifecycle.json", info)
            print("START", key, parts, flush=True)
            try:
                deadline = time.monotonic() + 900
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        raise RuntimeError(
                            f"{key} server exited; see {folder}/server.log"
                        )
                    try:
                        with urllib.request.urlopen(self.url + "/health", timeout=2):
                            break
                    except OSError:
                        time.sleep(2)
                else:
                    raise TimeoutError("server startup exceeded 900 seconds")
                info["startup_s"] = time.time() - info["started"]
                print("READY", key, round(info["startup_s"], 1), flush=True)
                yield
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=30)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
                with suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=15)
                info.update(finished=time.time(), returncode=proc.returncode)
                save(folder / "lifecycle.json", info)
                time.sleep(3)

    def load(self, key, split, count, concurrency, output_override=None):
        if not concurrency <= count <= len(self.data[split]):
            raise ValueError("insufficient samples or fewer requests than users")
        result = asyncio.run(
            run_load(
                self.data[split][:count],
                self.url,
                self.model_name,
                concurrency,
                output_override,
            )
        )
        result.update(split=split, sample_scope="pilot" if count < 1000 else "formal")
        save(self.root / key, result)
        print("RESULT", key, json.dumps(result["summary"]), flush=True)
        if result["summary"]["failed"]:
            raise RuntimeError(f"{key}: request failures; results saved")
        return result

    def profile(self):
        from vllm.distributed.pp_layer_cost import measured_layer_rank_costs
        from vllm.distributed.pp_memory import PPMemoryProfile
        from vllm.distributed.pp_partition import load_trace_records, partition_layers

        with self.server("profile", [12, 12, 12, 12], trace=True):
            concurrency = self.args.profile_concurrency
            self.rpc("set_pp_profile_warmup", [True])
            self.load(
                "profile/warmup.json",
                "warmup",
                max(8, concurrency),
                concurrency,
                self.args.warmup_output_tokens,
            )
            self.rpc("set_pp_profile_warmup", [False])
            self.load(
                "profile/requests.json",
                "profile",
                self.args.profile_requests,
                concurrency,
            )
            observations = self.rpc("get_pp_memory_observation")
            save(self.root / "profile/memory_observations.json", observations)
        bounds = memory_bounds(
            observations, self.config, self.serving, self.args.kv_blocks
        )
        save(self.root / "profile/memory.json", bounds)
        memory = PPMemoryProfile.from_file(self.root / "profile/memory.json")
        if any(r["headroom_bytes"] < 0 for r in memory.plan_usage([12] * 4)):
            raise ValueError("uniform allocation exceeds conservative memory bound")
        costs = measured_layer_rank_costs(
            [load_trace_records(self.root / "profile/traces")],
            workload="all",
            warmup_steps=0,
            num_layers=48,
        )
        save(self.root / "profile/costs.json", [asdict(c) for c in costs])
        plans = {"uniform": [12, 12, 12, 12]}
        for objective in ("latency", "throughput"):
            plan = partition_layers(
                costs,
                objective=objective,
                num_layers=48,
                min_pp_size=4,
                max_pp_size=4,
                memory_profile=memory,
            )
            save(self.root / f"profile/{objective}.json", plan.to_dict())
            plans[objective] = list(plan.partitions)
            print("PLAN", objective, plan.partitions, plan.cost_ms, flush=True)
        save(self.root / "plans.json", plans)

    def evaluate(self):
        plans = json.loads((self.root / "plans.json").read_text())
        seen = set()
        for name, parts in plans.items():
            if tuple(parts) in seen:
                continue
            seen.add(tuple(parts))
            key = f"{self.args.mode}_{name}"
            with self.server(key, parts):
                if self.args.mode == "pilot":
                    self.load(
                        f"{key}/warmup.json",
                        "warmup",
                        8,
                        8,
                        self.args.warmup_output_tokens,
                    )
                    # Isolated TTFT probe only: never use its throughput as a result.
                    self.load(f"{key}/ttft_probe.json", "evaluation", 32, 1, 1)
                    self.load(
                        f"{key}/c8.json", "evaluation", self.args.pilot_requests, 8
                    )
                else:
                    points = []
                    for concurrency in self.args.concurrencies:
                        self.load(
                            f"{key}/warmup_c{concurrency}.json",
                            "warmup",
                            max(16, concurrency),
                            concurrency,
                        )
                        point = self.load(
                            f"{key}/c{concurrency}.json",
                            "evaluation",
                            self.args.requests,
                            concurrency,
                        )
                        points.append(point)
                        save(
                            self.root / key / "capacity.json", capacity_summary(points)
                        )
                save(
                    self.root / key / "memory_observations.json",
                    self.rpc("get_pp_memory_observation"),
                )


def main():
    def interrupt(_signum, _frame):
        # Background shells may inherit SIGINT=ignore. Always unwind the server
        # context so cancelling the driver also releases its NPU workers.
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=18762)
    parser.add_argument("--kv-blocks", type=int, default=1024)
    parser.add_argument(
        "--mode", choices=("profile", "pilot", "sweep"), default="pilot"
    )
    parser.add_argument("--profile-requests", type=int, default=32)
    parser.add_argument("--profile-concurrency", type=int, default=8)
    parser.add_argument("--pilot-requests", type=int, default=16)
    parser.add_argument("--warmup-output-tokens", type=int, default=16)
    parser.add_argument("--requests", type=int, default=2048)
    parser.add_argument(
        "--concurrencies",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
    )
    args = parser.parse_args()
    if args.kv_blocks < 32 or min(args.profile_requests, args.pilot_requests) < 8:
        parser.error("need at least 32 KV blocks and eight profile/pilot requests")
    if not 1 <= args.warmup_output_tokens <= 1024:
        parser.error("warmup output must be between 1 and 1024 tokens")
    if not 1 <= args.profile_concurrency <= min(256, args.profile_requests):
        parser.error("profiling concurrency must be 1..256 and <= profile requests")
    if not all(1 <= c <= 256 for c in args.concurrencies):
        parser.error("concurrency must be between 1 and 256")
    experiment = Experiment(args)
    experiment.root.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        serving=experiment.serving,
        kv_blocks=args.kv_blocks,
        compute_slowdown=[1, 1, 2, 4],
        network="native_single_host_HCCL",
        sample_manifest=json.loads((Path(args.data) / "manifest.json").read_text()),
    )
    protocol_path = experiment.root / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("protocol differs from existing run; use a new output folder")
    save(protocol_path, protocol)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True)
    save(
        experiment.root / f"invocation_{time.time_ns()}.json",
        dict(commit=commit.strip(), args=vars(args)),
    )
    if not (experiment.root / "plans.json").exists():
        experiment.profile()
    if args.mode != "profile":
        experiment.evaluate()


if __name__ == "__main__":
    main()
