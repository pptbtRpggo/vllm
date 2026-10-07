# SPDX-License-Identifier: Apache-2.0
"""One-variable CodeLlama PP experiments, paired with uniform PP serving.

Prepare data without a device runtime; run only after any other NPU experiments
have finished. Use the same workload inside each sweep; shortened outputs
apply to every arm.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from pp_sla import read_samples, save
from pp_sla_experiment import Experiment
from tp_pp_compare import signature, successful

REPO = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_samples(folder, splits):
    folder.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        path = folder / f"{split}.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(r) + "\n" for r in rows))
        temporary.replace(path)
    save(
        folder / "manifest.json",
        dict(
            counts={k: len(v) for k, v in splits.items()},
            signatures={k: signature(v, "output_tokens") for k, v in splits.items()},
        ),
    )


def prepare(args):
    source = {
        k: read_samples(args.source_data / f"{k}.jsonl")
        for k in ("warmup", "profile", "evaluation")
    }
    for rows in source.values():
        for r in rows:
            sha = hashlib.sha256(json.dumps(r["prompt"]).encode()).hexdigest()
            if sha != r["prompt_sha256"] or len(r["prompt"]) != r["input_tokens"]:
                raise ValueError("source prompt tokens or stored length changed")
    existing = dict(
        warmup=source["warmup"][:32],
        profile=source["profile"][:64],
        evaluation=source["evaluation"][64:192],
    )
    if [len(existing[k]) for k in ("warmup", "profile", "evaluation")] != [32, 64, 128]:
        raise ValueError("insufficient source samples")
    reference = args.reference_root / "p99_100g_c8"
    source_protocol = json.loads((args.reference_root / "protocol.json").read_text())
    settings = json.loads((reference / "settings.json").read_text())
    if (
        source_protocol["scales"] != [1, 1, 2, 4]
        or settings["serving"]["max_num_batched_tokens"] != 2048
    ):
        raise ValueError("reference compute scales or batch token budget differ")
    if "codellama-34b" not in source_protocol["model"].lower():
        raise ValueError("this sweep requires the CodeLlama34B reference")
    refs = {}
    for c in (8, 12):
        for strategy in ("uniform", "latency_dp"):
            path = reference / strategy / f"round1_c{c}.json"
            successful(json.loads(path.read_text()), existing["evaluation"][:96], c)
            refs[f"{strategy}_c{c}"] = digest(path)
    if not (reference / "complete.json").exists():
        raise ValueError("reference scenario is incomplete")
    existing = {
        k: [
            dict(r, original_output_tokens=r["output_tokens"], output_tokens=64)
            for r in rows
        ]
        for k, rows in existing.items()
    }
    datasets = {"original": existing}
    # Deduplicate at the shortest prefix so different splits remain disjoint at
    # all four input lengths, including prompts with shared instruction text.
    eligible, seen = [], set()
    for split in ("warmup", "profile", "evaluation"):
        for row in source[split]:
            prefix = tuple(row["prompt"][:128])
            if row["input_tokens"] >= 1024 and prefix not in seen:
                eligible.append(row)
                seen.add(prefix)
    if len(eligible) < 208:
        raise ValueError("need 208 distinct long prompts for the input sweep")
    long_splits = dict(
        warmup=eligible[:16], profile=eligible[16:80], evaluation=eligible[80:208]
    )
    for length in (128, 256, 512, 1024):
        splits = {}
        for split, rows in long_splits.items():
            splits[split] = []
            for row in rows:
                prompt = row["prompt"][:length]
                splits[split].append(
                    dict(
                        row,
                        prompt=prompt,
                        input_tokens=length,
                        original_input_tokens=row["input_tokens"],
                        original_output_tokens=row["output_tokens"],
                        output_tokens=64,
                        original_prompt_sha256=row["prompt_sha256"],
                        prompt_sha256=hashlib.sha256(
                            json.dumps(prompt).encode()
                        ).hexdigest(),
                    )
                )
        datasets[f"input{length}"] = splits
    cases, axes = {}, {}

    def case(bandwidth=100, concurrency=8, budget=2048, data="original"):
        name = f"b{bandwidth}_c{concurrency}_budget{budget}_{data}"
        cases[name] = dict(
            name=name,
            bandwidth=bandwidth,
            concurrency=concurrency,
            budget=budget,
            data=data,
        )
        return name

    axes["bandwidth"] = [case(bandwidth=b) for b in (100, 25, 5, 1)]
    axes["batch_token_budget"] = [case(budget=b) for b in (512, 1024, 2048, 4096)]
    axes["input_tokens"] = [case(data=f"input{n}") for n in (128, 256, 512, 1024)]
    axes["concurrency"] = [case(concurrency=c) for c in (8, 12, 16, 20)]
    matrix = dict(
        model=source_protocol["model"],
        scales=[1, 1, 2, 4],
        axes=axes,
        cases=list(cases.values()),
        profile_requests=64,
        formal_requests=128,
        output_tokens=64,
        reference_sha256=refs,
        reference_settings_sha256=digest(reference / "settings.json"),
        slo_sha256=digest(reference / "slo.json"),
        derived_data_sha256={
            f"{name}/{split}.jsonl": hashlib.sha256(
                "".join(json.dumps(r) + "\n" for r in rows).encode()
            ).hexdigest()
            for name, splits in datasets.items()
            for split, rows in splits.items()
        },
        source_sha256={k: digest(args.source_data / f"{k}.jsonl") for k in source},
    )
    path = args.output / "matrix.json"
    if path.exists():
        if json.loads(path.read_text()) != matrix:
            raise ValueError("input matrix changed; use a new output directory")
        for name, expected in matrix["derived_data_sha256"].items():
            stored = args.output / "data" / name
            if stored.exists() and digest(stored) != expected:
                raise ValueError("prepared data changed; refusing to overwrite it")
    for name, splits in datasets.items():
        write_samples(args.output / "data" / name, splits)
    (args.output / "slo.json").write_bytes((reference / "slo.json").read_bytes())
    save(path, matrix)
    return matrix


def run(args):
    matrix = prepare(args)
    native_path = args.reference_root / "native_calibration.json"
    native = json.loads(native_path.read_text())
    library = Path(os.environ["VLLM_ASCEND_DELAY_LIBRARY"])
    sources = [
        Path(__file__).resolve(),
        REPO / "benchmarks/pp_sla.py",
        REPO / "benchmarks/pp_sla_experiment.py",
        REPO / "benchmarks/tp_pp_compare.py",
        REPO / "vllm/pp_hetero_env.py",
        REPO / "vllm/distributed/async_trace.py",
        REPO / "vllm/v1/worker/pp_ascend_worker.py",
        REPO / "vllm/distributed/pp_partition.py",
        REPO / "vllm/distributed/pp_layer_cost.py",
        REPO / "vllm/distributed/pp_hetero.py",
        REPO / "vllm/distributed/ascend_device_delay.py",
    ]
    sources.extend(sorted((REPO / "vllm/distributed").glob("pp_*.py")))
    protocol = dict(
        matrix_sha256=digest(args.output / "matrix.json"),
        source_sha256={str(p.relative_to(REPO)): digest(p) for p in sources},
        library_sha256=digest(library),
        kernels_sha256=digest(library.parent / "libhetero_delay_kernels.so"),
        native_calibration_sha256=digest(native_path),
        git_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
    )
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("execution protocol changed; use a new output directory")
    save(path, protocol)
    for spec in matrix["cases"]:
        folder = args.output / "cases" / spec["name"]
        folder.mkdir(parents=True, exist_ok=True)
        data = args.output / "data" / spec["data"]
        formal = read_samples(data / "evaluation.jsonl")
        c = spec["concurrency"]
        values = dict(
            model=matrix["model"],
            data=str(data),
            output=str(folder),
            port=args.port,
            kv_blocks=1024,
            profile_requests=matrix["profile_requests"],
            profile_concurrency=c,
            warmup_output_tokens=16,
            max_num_batched_tokens=spec["budget"],
            network_mode="target_total",
            cross_bandwidth_gbps=spec["bandwidth"],
            cross_extra_latency_ms=0,
            native_bandwidth_gbps=native["native_bandwidth_gbps"],
            native_latency_ms=native["native_latency_ms"],
        )
        exp = Experiment(SimpleNamespace(**values))
        if not (folder / "plans.json").exists():
            if (folder / "profile").exists():
                (folder / "profile").rename(folder / f"failed_profile_{time.time_ns()}")
            save(
                args.output / "status.json",
                dict(stage="profile", case=spec["name"], time=time.time()),
            )
            exp.profile()
        parts = json.loads((folder / "plans.json").read_text())["latency"]
        save(
            folder / "settings.json",
            dict(
                **spec,
                reused=False,
                latency_partition=parts,
                serving=exp.serving,
                network=exp.network,
            ),
        )
        for label, partition in (
            ("uniform", exp.uniform_partition),
            ("latency_dp", parts),
        ):
            path = folder / f"{label}.json"
            if path.exists():
                result = json.loads(path.read_text())
                if result["summary"]["failed"]:
                    path.rename(path.with_name(f"{label}_failed_{time.time_ns()}.json"))
                else:
                    successful(result, formal, c)
                    continue
            save(
                args.output / "status.json",
                dict(
                    stage="serving", case=spec["name"], strategy=label, time=time.time()
                ),
            )
            with exp.server(f"{label}_server_{time.time_ns()}", partition):
                warmed = exp.load(
                    f"warmup_{label}_{time.time_ns()}.json", "warmup", max(16, c), c, 16
                )
                successful(warmed, exp.data["warmup"][: max(16, c)], c, override=16)
                result = exp.load(
                    f"{label}.json", "evaluation", matrix["formal_requests"], c
                )
                successful(result, formal, c)
        save(folder / "complete.json", dict(reused=False, time=time.time()))
        print("CASE_COMPLETE", spec["name"], flush=True)
    save(
        args.output / "complete.json",
        dict(conditions=len(matrix["cases"]), new_conditions=14, time=time.time()),
    )
    save(args.output / "status.json", dict(stage="complete", time=time.time()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18792)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if args.prepare_only:
            matrix = prepare(args)
            print("PREPARED", len(matrix["cases"]), "conditions")
        else:
            run(args)
    except BaseException as error:
        save(
            args.output / "status.json",
            dict(stage="failed", error=repr(error), time=time.time()),
        )
        raise
