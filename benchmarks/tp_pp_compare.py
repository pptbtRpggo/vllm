# SPDX-License-Identifier: Apache-2.0
"""Compare uniform TP to saved PP results on exactly the same requests and SLOs."""

import argparse
import asyncio
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

from pp_sla import read_samples, run_load, save


def signature(rows, output_key):
    return [
        [r["id"], r["prompt_sha256"], r["input_tokens"], r[output_key]] for r in rows
    ]


def prepare_cases(reference_root, data):
    """Reject stale slices or changed data before spending any serving time."""
    protocol = json.loads((reference_root / "protocol.json").read_text())
    if (protocol.get("pp_size") or len(protocol["scales"])) not in (2, 4):
        raise ValueError("matched comparison requires two or four devices")
    evaluation = read_samples(data / "evaluation.jsonl")
    lookup = {r["id"]: r for r in evaluation}
    if len(lookup) != len(evaluation):
        raise ValueError("evaluation IDs must be unique")
    warm = read_samples(data / "warmup.jsonl")
    for row in evaluation + warm:
        digest = hashlib.sha256(json.dumps(row["prompt"]).encode()).hexdigest()
        if digest != row["prompt_sha256"]:
            raise ValueError("sample token IDs differ from their stored hash")
    expected_serving = dict(
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
    cases = []
    for spec in protocol["scenarios"]:
        folder = reference_root / spec["name"]
        if not (folder / "complete.json").exists():
            raise ValueError(f"reference PP scenario is incomplete: {folder}")
        settings = json.loads((folder / "settings.json").read_text())
        if any(settings["serving"].get(k) != v for k, v in expected_serving.items()):
            raise ValueError("PP serving parameters differ from matched TP protocol")
        if (
            Path(settings["serving"]["model"]).resolve()
            != Path(protocol["model"]).resolve()
        ):
            raise ValueError("PP reference model differs from its protocol")
        formal = [lookup[i] for i in settings["formal_ids"]]
        if len(formal) < spec["raised"]:
            raise ValueError("not enough requests to maintain raised concurrency")
        if {r["prompt_sha256"] for r in warm} & {r["prompt_sha256"] for r in formal}:
            raise ValueError("warmup and formal prompts overlap")
        references = {}
        for strategy in ("uniform", "latency_dp"):
            for c in (spec["base"], spec["raised"]):
                path = folder / strategy / f"round1_c{c}.json"
                ref = json.loads(path.read_text())
                try:
                    successful(ref, formal, c)
                except ValueError as error:
                    raise ValueError(f"invalid PP reference {path}: {error}") from error
                references[f"{strategy}_c{c}"] = dict(
                    path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
                )
        for c in (spec["base"], spec["raised"]):
            paths = sorted((folder / "uniform").glob(f"warmup_round1_c{c}_*.json"))
            if not paths:
                paths = sorted((folder / "uniform").glob(f"warmup_c{c}_*.json"))
            if not paths:
                raise ValueError(f"missing reference PP warmup at concurrency {c}")
            path = paths[0]
            successful(json.loads(path.read_text()), warm[: max(16, c)], c, override=16)
            references[f"warmup_c{c}"] = dict(
                path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
        cases.append(
            dict(
                spec=spec,
                samples=formal,
                slo=json.loads((folder / "slo.json").read_text()),
                references=references,
            )
        )
    return protocol, warm, cases


def command(model, size, port):
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model,
        "--served-model-name",
        "tp-matched",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "float16",
        "--tensor-parallel-size",
        str(size),
        "--pipeline-parallel-size",
        "1",
        "--distributed-executor-backend",
        "mp",
        "--enforce-eager",
        "--compilation-config",
        '{"mode":0}',
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


@contextmanager
def server(args, protocol, bandwidth, native, label):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("VLLM_PP_", "VLLM_TP_")):
            env.pop(key)
    size = protocol.get("pp_size") or len(protocol["scales"])
    env.update(
        VLLM_TP_COMPUTE_SCALES=",".join(map(str, protocol["scales"])),
        VLLM_TP_CROSS_GROUP_SIZE=str(size // 2),
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_NO_USAGE_STATS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTORCH_NPU_ALLOC_CONF="expandable_segments:True",
        ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, range(size))),
    )
    network = None
    if bandwidth is not None:
        network = dict(
            tp_size=size,
            cross_group_size=size // 2,
            bandwidth_gbps=bandwidth,
            native_collectives=native["native_collectives"],
        )
        env["VLLM_TP_CROSS_NETWORK"] = json.dumps(network)
    path = args.output / "servers" / f"{label}_{time.time_ns()}"
    path.mkdir(parents=True)
    cmd = command(args.model, size, args.port)
    save(
        path / "settings.json",
        dict(
            command=cmd,
            scales=protocol["scales"],
            network=network,
            git_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        ),
    )
    url = f"http://127.0.0.1:{args.port}"
    with (path / "server.log").open("w") as log:
        proc = subprocess.Popen(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited {proc.returncode}; see {path}")
                try:
                    with urllib.request.urlopen(url + "/health", timeout=2):
                        break
                except (OSError, TimeoutError):
                    time.sleep(2)
            else:
                raise TimeoutError(f"server startup timed out: {path}")
            yield url, path
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


def successful(result, samples, concurrency, override=None):
    if result["concurrency"] != concurrency or result["summary"]["failed"]:
        raise ValueError("load failed or concurrency changed")
    expected = (
        [dict(r, output_tokens=override) for r in samples]
        if override is not None
        else samples
    )
    records = sorted(result["records"], key=lambda r: r["index"])
    if signature(records, "requested_output_tokens") != signature(
        expected, "output_tokens"
    ):
        raise ValueError("saved request signature differs")
    if any(
        not r["success"] or r["output_tokens"] != r["requested_output_tokens"]
        for r in records
    ):
        raise ValueError("incomplete response or output length")


def run(args):
    protocol, warm, cases = prepare_cases(args.reference_root, args.data)
    size = protocol.get("pp_size") or len(protocol["scales"])
    if Path(args.model).resolve() != Path(protocol["model"]).resolve():
        raise ValueError("TP model differs from reference PP model")
    if len(warm) < max(16, max(c["spec"]["raised"] for c in cases)):
        raise ValueError("not enough warmup requests")
    if args.validate_only:
        print(
            "MATCHED_REQUESTS",
            len(cases),
            sum(2 * len(c["samples"]) for c in cases),
            flush=True,
        )
        return
    native = (
        json.loads(args.native_calibration.read_text())
        if args.native_calibration
        else None
    )
    if any(c["spec"]["bandwidth"] is not None for c in cases) and (
        native is None
        or native["tp_size"] != size
        or native["cross_group_size"] != size // 2
    ):
        raise ValueError("missing native TP calibration or topology differs")
    args.output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__).resolve(),
        repo / "benchmarks/pp_sla.py",
        repo / "vllm/pp_hetero_env.py",
        repo / "vllm/distributed/tp_hetero.py",
        repo / "vllm/v1/worker/tp_ascend_worker.py",
        repo / "vllm/distributed/ascend_device_delay.py",
        repo / "vllm/distributed/pp_hetero.py",
    ]
    hashes = {
        str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources
    }
    library_path = os.environ.get("VLLM_ASCEND_DELAY_LIBRARY")
    if not library_path:
        raise ValueError("VLLM_ASCEND_DELAY_LIBRARY is required for heterogeneity")
    library = Path(library_path)
    library_hashes = {library.name: hashlib.sha256(library.read_bytes()).hexdigest()}
    kernels = library.parent / "libhetero_delay_kernels.so"
    if kernels.exists():
        library_hashes[kernels.name] = hashlib.sha256(kernels.read_bytes()).hexdigest()
    manifest = dict(
        model=args.model,
        source_sha256=hashes,
        delay_library_sha256=library_hashes,
        warmup_file_sha256=hashlib.sha256(
            (args.data / "warmup.jsonl").read_bytes()
        ).hexdigest(),
        tp_size=size,
        scales=protocol["scales"],
        reference_root=str(args.reference_root),
        data_manifest_sha256=hashlib.sha256(
            (args.data / "manifest.json").read_bytes()
        ).hexdigest(),
        native_calibration_sha256=hashlib.sha256(
            args.native_calibration.read_bytes()
        ).hexdigest()
        if args.native_calibration
        else None,
        git_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        cases=[
            dict(
                spec=c["spec"],
                slo=c["slo"],
                references=c["references"],
                request_signature=signature(c["samples"], "output_tokens"),
            )
            for c in cases
        ],
    )
    previous = args.output / "protocol.json"
    if previous.exists() and json.loads(previous.read_text()) != manifest:
        raise ValueError("protocol changed; use a new output directory")
    save(previous, manifest)
    grouped = defaultdict(list)
    for case in cases:
        grouped[case["spec"]["bandwidth"]].append(case)
    for bandwidth, group in grouped.items():
        pending = []
        for case in group:
            for c in (case["spec"]["base"], case["spec"]["raised"]):
                path = args.output / case["spec"]["name"] / f"formal_c{c}.json"
                if path.exists():
                    saved = json.loads(path.read_text())
                    if saved["summary"]["failed"]:
                        failures = path.parent / "failures"
                        failures.mkdir(exist_ok=True)
                        path.rename(failures / f"{path.stem}_{time.time_ns()}.json")
                        pending.append((case, c, path))
                    else:
                        successful(saved, case["samples"], c)
                else:
                    pending.append((case, c, path))
        if not pending:
            continue
        label = "native" if bandwidth is None else f"{bandwidth}g"
        with server(args, protocol, bandwidth, native, label) as (url, server_path):
            for case, c, path in pending:
                save(
                    args.output / "status.json",
                    dict(
                        stage="running",
                        scenario=case["spec"]["name"],
                        concurrency=c,
                        time=time.time(),
                    ),
                )
                print(
                    "BEGIN", case["spec"]["name"], c, len(case["samples"]), flush=True
                )
                warmed = asyncio.run(
                    run_load(
                        warm[: max(16, c)], url, "tp-matched", c, output_override=16
                    )
                )
                save(path.with_name(f"warmup_c{c}_{time.time_ns()}.json"), warmed)
                successful(warmed, warm[: max(16, c)], c, override=16)
                result = asyncio.run(run_load(case["samples"], url, "tp-matched", c))
                result.update(
                    reference_pp=case["references"],
                    slo=case["slo"],
                    server=str(server_path),
                )
                save(path, result)
                successful(result, case["samples"], c)
                print(
                    "DONE",
                    case["spec"]["name"],
                    c,
                    json.dumps(result["summary"]),
                    flush=True,
                )
    save(
        args.output / "complete.json",
        dict(scenarios=len(cases), formal_points=2 * len(cases), time=time.time()),
    )
    save(args.output / "status.json", dict(stage="complete", time=time.time()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-calibration", type=Path)
    parser.add_argument("--port", type=int, default=18791)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        run(args)
    except BaseException as error:
        save(
            args.output / "status.json",
            dict(stage="failed", error=repr(error), time=time.time()),
        )
        raise
