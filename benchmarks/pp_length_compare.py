# SPDX-License-Identifier: Apache-2.0
"""Match saved request-length scans with uniform TP and measured latency-DP PP.

Formal serving uses the reference decode Graph configuration. Direct layer
profiling is eager because replay does not execute Python layer hooks. A single
profiling server measures multiple disjoint windows; no source serving changes
are required. Completed results are validated rather than blindly skipped.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from pp_sla import read_samples, run_load, save
from pp_sla_experiment import memory_bounds
from tp_pp_compare import successful

REPO = Path(__file__).resolve().parents[1]
GRAPH = dict(
    mode=0,
    cudagraph_mode="FULL_DECODE_ONLY",
    cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64],
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resize(rows, inputs, outputs):
    result = []
    for row in rows:
        tokens = row["prompt"][:inputs]
        if len(tokens) != inputs:
            raise ValueError("source prompt is shorter than requested input")
        result.append(
            dict(
                row,
                prompt=tokens,
                input_tokens=inputs,
                output_tokens=outputs,
                prompt_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            )
        )
    return result


def prepare(reference):
    cases = []
    data = {}
    with (reference / "metrics.csv").open(encoding="utf-8-sig") as file:
        specs = list(csv.DictReader(file))
    for spec in specs:
        model, directory = spec["model"], spec["network"]
        if model not in data:
            data[model] = {
                split: read_samples(reference / "data" / model / f"{split}.jsonl")
                for split in ("warmup", "evaluation")
            }
        folder = reference / model / directory
        path = folder / (spec["file"] + ".json")
        result = json.loads(path.read_text())
        c, i, o = (
            int(spec[k]) for k in ("concurrency", "input_tokens", "output_tokens")
        )
        ordered = sorted(result["records"], key=lambda r: r["index"])
        lookup = {r["id"]: r for r in data[model]["evaluation"]}
        samples = resize([lookup[r["id"]] for r in ordered], i, o)
        successful(result, samples, c)
        if len(samples) != int(spec["count"]):
            raise ValueError("CSV request count differs from original result")
        warm = resize(data[model]["warmup"][: max(8, c)], i, o)
        warm_path = folder / "warmup" / path.name
        successful(json.loads(warm_path.read_text()), warm, c)
        settings = json.loads((folder / "settings.json").read_text())
        if (
            json.loads(
                settings["command"][
                    settings["command"].index("--compilation-config") + 1
                ]
            )
            != GRAPH
        ):
            raise ValueError("reference Graph configuration changed")
        if "--enforce-eager" in settings["command"]:
            raise ValueError("reference formal serving unexpectedly eager")
        base_network = directory.split("_")[0]
        size = len(settings["partition"])
        group = f"{model}_{base_network}"
        key = f"{model}/{directory}/{spec['file']}"
        workload = f"c{c}_i{i}_o{o}"
        cases.append(
            dict(
                key=key,
                group=group,
                model=model,
                network=base_network,
                concurrency=c,
                input_tokens=i,
                output_tokens=o,
                samples=samples,
                warmup=warm,
                size=size,
                workload=workload,
                scales=settings["compute_scales"],
                pp_network=settings["network"],
                model_path=settings["command"][settings["command"].index("serve") + 1],
                reference=str(path.relative_to(reference)),
                reference_sha256=sha(path),
                warmup_sha256=sha(warm_path),
                reference_revision=settings["git_revision"],
                kind=spec["kind"],
            )
        )
    # Profile on unused source IDs, avoiding any formal request across all scans.
    for model in data:
        used = {
            r["id"] for case in cases if case["model"] == model for r in case["samples"]
        }
        held_out = [r for r in data[model]["evaluation"] if r["id"] not in used]
        for case in cases:
            if case["model"] == model:
                n = max(64, 2 * case["concurrency"])
                if len(held_out) < n:
                    raise ValueError("not enough held-out profiling requests")
                case["profile"] = resize(
                    held_out[:n], case["input_tokens"], case["output_tokens"]
                )
                if {r["id"] for r in case["warmup"]} & {
                    r["id"] for r in case["samples"] + case["profile"]
                }:
                    raise ValueError("warmup source IDs overlap formal/profile samples")
    return cases


def rpc(url, method, args=()):
    request = urllib.request.Request(
        url + "/collective_rpc",
        json.dumps(dict(method=method, args=list(args))).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    if "error" in result or "results" not in result:
        raise RuntimeError(result)
    return result["results"]


def serving_command(case, method, port, profile=False):
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        case["model_path"],
        "--served-model-name",
        "length-compare",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "float16",
        "--pipeline-parallel-size",
        str(case["size"] if method == "pp" else 1),
        "--tensor-parallel-size",
        str(case["size"] if method == "tp" else 1),
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
        "1024",
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--disable-log-requests",
        "--compilation-config",
        json.dumps(dict(mode=0) if profile else GRAPH),
        *(["--enforce-eager"] if profile else []),
    ]


@contextmanager
def server(args, case, method, parts=None, profile=False):
    folder = args.output / "servers" / f"{case['group']}_{method}_{time.time_ns()}"
    folder.mkdir(parents=True)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(
            ("VLLM_PP_", "VLLM_TP_", "VLLM_DEVICE_BENCH_", "VLLM_PROFILER")
        ):
            del env[key]
    size = case["size"]
    env.update(
        VLLM_SERVER_DEV_MODE="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_NO_USAGE_STATS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, range(size))),
        PYTORCH_NPU_ALLOC_CONF="expandable_segments:True",
    )
    if method == "pp":
        env.update(
            VLLM_PP_LAYER_PARTITION=",".join(map(str, parts)),
            VLLM_PP_HETERO=",".join(map(str, case["scales"])),
            VLLM_PP_COMPUTE_MODEL="layer-measured",
            VLLM_PP_DEVICE_ORDER=",".join(map(str, range(size))),
            VLLM_PP_NETWORK=json.dumps(case["pp_network"]),
        )
        if profile:
            env.update(
                VLLM_PP_STAGE_TRACE=str(folder / "traces"),
                VLLM_PP_TRACE_SESSION=uuid.uuid4().hex,
            )
    else:
        env.update(
            VLLM_TP_COMPUTE_SCALES=",".join(map(str, case["scales"])),
            VLLM_TP_CROSS_GROUP_SIZE=str(size // 2),
        )
        if case["network"] != "native":
            native_path = args.native_tp / f"native_tp{size}/native_calibration.json"
            native = json.loads(native_path.read_text())
            if native["tp_size"] != size:
                raise ValueError("native TP calibration rank count mismatch")
            env["VLLM_TP_CROSS_NETWORK"] = json.dumps(
                dict(
                    tp_size=size,
                    cross_group_size=size // 2,
                    bandwidth_gbps=int(case["network"][:-1]),
                    native_collectives=native["native_collectives"],
                )
            )
    command = serving_command(case, method, args.port, profile)
    settings = dict(
        command=command,
        profile=profile,
        parts=parts,
        started=time.time(),
        git_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        environment={
            k: v
            for k, v in env.items()
            if k.startswith(("VLLM_PP_", "VLLM_TP_", "VLLM_ASCEND_"))
        },
    )
    save(folder / "settings.json", settings)
    url = f"http://127.0.0.1:{args.port}"
    with (folder / "server.log").open("w") as log:
        proc = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        save(folder / "pid.json", dict(pid=proc.pid))
        try:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited: {folder}/server.log")
                try:
                    with urllib.request.urlopen(url + "/health", timeout=2):
                        break
                except (OSError, TimeoutError):
                    time.sleep(2)
            else:
                raise TimeoutError("server startup exceeded 900 seconds")
            print(
                "READY",
                folder.name,
                round(time.time() - settings["started"], 1),
                flush=True,
            )
            yield url, folder
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=15)
            save(
                folder / "finished.json",
                dict(time=time.time(), returncode=proc.returncode),
            )


def measure(args, case, method, url, folder, parts=None):
    path = args.output / "results" / method / (case["key"] + ".json")
    if path.exists():
        result = json.loads(path.read_text())
        successful(result, case["samples"], case["concurrency"])
        if (
            result["reference_sha256"] != case["reference_sha256"]
            or result.get("parts") != parts
        ):
            raise ValueError("completed result identity differs")
        return
    save(
        args.output / "status.json",
        dict(
            stage=method,
            case=case["key"],
            completed_tp=len(list((args.output / "results/tp").rglob("*.json"))),
            completed_edgeshard=len(
                list((args.output / "results/edgeshard").rglob("*.json"))
            ),
            time=time.time(),
        ),
    )
    warm = asyncio.run(
        run_load(case["warmup"], url, "length-compare", case["concurrency"])
    )
    successful(warm, case["warmup"], case["concurrency"])
    save(args.output / "warmup" / method / (case["key"] + ".json"), warm)
    result = asyncio.run(
        run_load(case["samples"], url, "length-compare", case["concurrency"])
    )
    result.update(
        reference_sha256=case["reference_sha256"],
        method=method,
        parts=parts,
        input_tokens=case["input_tokens"],
        output_tokens=case["output_tokens"],
        server=str(folder.relative_to(args.output)),
    )
    successful(result, case["samples"], case["concurrency"])
    save(path, result)
    backup(args)
    print("RESULT", method, case["key"], result["summary"], flush=True)


def backup(args):
    if args.backup:
        shutil.copytree(args.output, args.backup, dirs_exist_ok=True)


def plan(args, case, folder, bounds):
    from vllm.distributed.pp_layer_cost import measured_layer_rank_costs
    from vllm.distributed.pp_memory import PPMemoryProfile
    from vllm.distributed.pp_partition import load_trace_records, partition_layers

    traces = load_trace_records(folder / "traces")
    costs = measured_layer_rank_costs(
        [traces], workload="all", warmup_steps=0, num_layers=bounds["num_layers"]
    )
    memory = PPMemoryProfile.from_file(folder / "memory.json")
    layers = bounds["num_layers"]
    result = partition_layers(
        costs,
        objective="latency",
        num_layers=layers,
        min_pp_size=case["size"],
        max_pp_size=case["size"],
        memory_profile=memory,
    )
    save(folder / "costs.json", [asdict(c) for c in costs])
    save(folder / "latency.json", result.to_dict())
    return result


def select_window(raw, start, stop, size):
    """Select reserved host timestamps and reject partial/misaligned rank slices."""
    if set(raw) != set(range(size)):
        raise ValueError("missing raw trace ranks")
    selected = {
        rank: sorted(
            [r for r in rows if start <= r["ts_unix"] <= stop and not r["is_warmup"]],
            key=lambda r: r["step"],
        )
        for rank, rows in raw.items()
    }
    expected = [(r["step"], r["batch_id"]) for r in selected[0]]
    if not expected:
        raise ValueError("profiling window is empty")
    if any(
        [(r["step"], r["batch_id"]) for r in rows] != expected
        for rows in selected.values()
    ):
        raise ValueError("profiling window does not contain matching rank batches")
    return selected


def profile_group(args, group, attempt):
    first = group[0]
    config = json.loads((Path(first["model_path"]) / "config.json").read_text())
    layers = config["num_hidden_layers"]
    unique = {case["workload"]: case for case in group}
    pending = [
        c
        for c in unique.values()
        if not (
            args.output / "profiles" / c["group"] / c["workload"] / "latency.json"
        ).exists()
    ]
    if not pending:
        return []
    windows = []
    with server(
        args, first, "pp", [layers // first["size"]] * first["size"], profile=True
    ) as (url, server_dir):
        for case in pending:
            save(
                args.output / "status.json",
                dict(
                    stage="profiling",
                    case=case["key"],
                    attempt=attempt,
                    time=time.time(),
                ),
            )
            rpc(url, "set_pp_profile_warmup", [True])
            warmed = asyncio.run(
                run_load(case["warmup"], url, "length-compare", case["concurrency"])
            )
            successful(warmed, case["warmup"], case["concurrency"])
            rpc(url, "set_pp_profile_warmup", [False])
            start = time.time()
            result = asyncio.run(
                run_load(case["profile"], url, "length-compare", case["concurrency"])
            )
            stop = time.time()
            successful(result, case["profile"], case["concurrency"])
            rpc(url, "set_pp_profile_warmup", [True])
            folder = args.output / "profiles" / case["group"] / case["workload"]
            if folder.exists():
                folder.rename(
                    folder.with_name(folder.name + f"_failed_{time.time_ns()}")
                )
            save(folder / "requests.json", result)
            save(folder / "warmup.json", warmed)
            save(
                folder / "window.json",
                dict(
                    start=start,
                    stop=stop,
                    server=str(server_dir.relative_to(args.output)),
                    attempt=attempt,
                ),
            )
            windows.append((case, folder, start, stop))
            print("PROFILE", case["group"], case["workload"], flush=True)
        observations = rpc(url, "get_pp_memory_observation")
    serving = dict(
        dtype="float16",
        kv_cache_dtype="auto",
        block_size=128,
        max_model_len=4096,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.85,
    )
    bounds = memory_bounds(observations, config, serving, 1024)
    failed = []
    # Read unpaired raw rows only after graceful shutdown has drained the writer.
    raw = defaultdict(list)
    for path in (server_dir / "traces").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            raw[row["pp_rank"]].append(row)
    for case, folder, start, stop in windows:
        trace_dir = folder / "traces"
        trace_dir.mkdir()
        for rank, selected in select_window(raw, start, stop, first["size"]).items():
            (trace_dir / f"pp_stage_pp{rank}_tp0.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in selected)
            )
        save(folder / "memory_observations.json", observations)
        save(folder / "memory.json", bounds)
        try:
            plan(args, case, folder, bounds)
        except ValueError as error:
            save(folder / "error.json", dict(error=str(error), attempt=attempt))
            failed.append(case)
            print("PROFILE_INVALID", case["workload"], str(error), flush=True)
    backup(args)
    return failed


def run(args):
    cases = prepare(args.reference)
    manifest = dict(
        reference=str(args.reference),
        source_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        graph=GRAPH,
        profile_mode="eager direct layer events",
        delay_library_sha256=sha(os.environ["VLLM_ASCEND_DELAY_LIBRARY"]),
        delay_kernel_sha256=sha(
            Path(os.environ["VLLM_ASCEND_DELAY_LIBRARY"]).with_name(
                "libhetero_delay_kernels.so"
            )
        ),
        native_tp_sha256={
            str(n): sha(args.native_tp / f"native_tp{n}/native_calibration.json")
            for n in (2, 4)
        },
        cases=[
            {k: v for k, v in c.items() if k not in ("samples", "warmup", "profile")}
            for c in cases
        ],
        model_config_sha256={
            c["model"]: sha(Path(c["model_path"]) / "config.json") for c in cases
        },
        profile_signature_sha256={
            f"{c['group']}/{c['workload']}": hashlib.sha256(
                json.dumps(
                    [
                        [r["id"], r["prompt_sha256"], r["output_tokens"]]
                        for r in c["profile"]
                    ]
                ).encode()
            ).hexdigest()
            for c in cases
        },
        points_per_strategy=len(cases),
        requests_per_strategy=sum(len(c["samples"]) for c in cases),
    )
    manifest_path = args.output / "protocol.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("existing experiment identity changed")
    save(manifest_path, manifest)
    groups = defaultdict(list)
    for case in cases:
        groups[case["group"]].append(case)
    if args.validate_only:
        print("VALIDATED", len(cases), manifest["requests_per_strategy"], flush=True)
        return
    for group in groups.values():
        remaining = [
            c
            for c in group
            if not (args.output / "results/tp" / (c["key"] + ".json")).exists()
        ]
        if remaining:
            with server(args, group[0], "tp") as (url, folder):
                for case in group:
                    measure(args, case, "tp", url, folder)
    unresolved = []
    for group in groups.values():
        failed = profile_group(args, group, 1)
        if failed:
            failed = profile_group(args, failed, 2)
        unresolved.extend(c["key"] for c in failed)
        partitions = defaultdict(list)
        for case in group:
            plan_path = (
                args.output
                / "profiles"
                / case["group"]
                / case["workload"]
                / "latency.json"
            )
            if plan_path.exists():
                parts = tuple(json.loads(plan_path.read_text())["partitions"])
                partitions[parts].append(case)
        for parts, matching in partitions.items():
            remaining = [
                c
                for c in matching
                if not (
                    args.output / "results/edgeshard" / (c["key"] + ".json")
                ).exists()
            ]
            if remaining:
                with server(args, group[0], "pp", list(parts)) as (url, folder):
                    for case in matching:
                        measure(args, case, "edgeshard", url, folder, list(parts))
    # Validate every formal result at completion, including resumed points.
    for method in ("tp", "edgeshard"):
        for case in cases:
            path = args.output / "results" / method / (case["key"] + ".json")
            if path.exists():
                successful(
                    json.loads(path.read_text()), case["samples"], case["concurrency"]
                )
            elif case["key"] not in unresolved or method == "tp":
                raise ValueError(f"missing result {path}")
    save(
        args.output / ("blocked.json" if unresolved else "complete.json"),
        dict(
            time=time.time(), unresolved=unresolved, protocol_sha256=sha(manifest_path)
        ),
    )
    save(
        args.output / "status.json",
        dict(
            stage="blocked" if unresolved else "complete",
            unresolved=unresolved,
            time=time.time(),
        ),
    )
    backup(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-tp", type=Path, required=True)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--port", type=int, default=18845)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("experiment interrupted; completed points preserved")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        run(args)
    except BaseException as error:
        save(
            args.output / "status.json",
            dict(stage="failed", error=repr(error), time=time.time()),
        )
        backup(args)
        raise


if __name__ == "__main__":
    main()
