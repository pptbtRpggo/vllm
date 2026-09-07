# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequential full ShareGPT collection against an ALREADY RUNNING server.

Never starts/restarts a server. Keeps a deterministic source-index manifest,
archives each validated range, and fits each shard before starting the next.
On any failed batch, stops without silently retrying partly completed requests.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from tau_batch_fit import fit


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def prepare_index(data, tokenizer, seed):
    """Match ShareGPTDataset's first-two-turn text sampling, without oversampling."""
    order = [
        i for i, item in enumerate(data) if len(item.get("conversations", [])) >= 2
    ]
    random.Random(seed).shuffle(order)
    eligible, lengths = [], []
    for offset in range(0, len(order), 128):
        ids = order[offset : offset + 128]
        prompts = [data[i]["conversations"][0]["value"] for i in ids]
        if not all(isinstance(p, str) for p in prompts):
            raise ValueError("Campaign supports text-only ShareGPT prompts")
        tokens = tokenizer(
            prompts, truncation=True, max_length=1025, return_attention_mask=False
        ).input_ids
        for i, token_ids in zip(ids, tokens):
            if 4 <= len(token_ids) <= 1024:
                eligible.append(i)
                lengths.append(len(token_ids))
    return eligible, lengths


def load_index(args, manifest, data):
    expected = {
        "source_sha256": digest(args.dataset),
        "seed": args.seed,
        "model": manifest["model"],
        "prompt_min": 4,
        "prompt_max": 1024,
    }
    summary = args.index_dir / "summary.json"
    if not args.index_dir.exists():
        from vllm.tokenizers import get_tokenizer

        tokenizer = get_tokenizer(manifest["model"], trust_remote_code=True)
        ids, lengths = prepare_index(data, tokenizer, args.seed)
        args.index_dir.mkdir(parents=True)
        save(args.index_dir / "indices.json", ids)
        save(args.index_dir / "prompt_lengths.json", lengths)
        save(summary, dict(expected, eligible=len(ids), total=len(data)))
    metadata = json.loads(summary.read_text())
    if any(metadata.get(key) != val for key, val in expected.items()):
        raise ValueError("Index provenance does not match source/model/seed")
    ids = json.loads((args.index_dir / "indices.json").read_text())
    lengths = json.loads((args.index_dir / "prompt_lengths.json").read_text())
    if (
        not ids
        or len(ids) != len(set(ids))
        or len(ids) != len(lengths)
        or any(type(i) is not int or not 0 <= i < len(data) for i in ids)
        or any(not 4 <= n <= 1024 for n in lengths)
    ):
        raise ValueError("Invalid or duplicated source indices")
    return ids, metadata


def archive(trace, report_path, destination, count):
    report = json.loads(report_path.read_text())
    if not report.get("passed") or report.get("completed") != count:
        raise ValueError(f"Unsuccessful benchmark: {report_path}")
    start, end = report["trace_start_offset"], report["trace_end_offset"]
    if not 0 <= start < end <= trace.stat().st_size:
        raise ValueError("Invalid archive byte range")
    destination.mkdir(parents=True)
    target = destination / "trace.jsonl"
    remaining = end - start
    h = hashlib.sha256()
    with trace.open("rb") as src, target.with_suffix(".partial").open("xb") as dst:
        src.seek(start)
        while remaining:
            block = src.read(min(4 * 1024 * 1024, remaining))
            if not block:
                raise ValueError("Trace truncated while copying")
            dst.write(block)
            h.update(block)
            remaining -= len(block)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(target.with_suffix(".partial"), target)
    save(destination / "original_report.json", report)
    save(
        destination / "archive.json",
        {
            "source": str(trace),
            "source_report": str(report_path),
            "source_start_offset": start,
            "source_end_offset": end,
            "bytes": end - start,
            "sha256": h.hexdigest(),
        },
    )
    # A range-local report must address the range-local file, not the live file.
    report.update(trace_start_offset=0, trace_end_offset=end - start)
    local_report = destination / "trace_check.json"
    save(local_report, report)
    return target, local_report


def ensure_idle(manifest):
    settings = manifest["settings"]
    host = settings["HOST"]
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    url = f"http://{host}:{settings['PORT']}/metrics"
    with urllib.request.urlopen(url, timeout=10) as response:
        lines = response.read().decode().splitlines()
    found = {"running": [], "waiting": []}
    for line in lines:
        for label in found:
            if line.startswith(f"vllm:num_requests_{label}{{"):
                found[label].append(float(line.rsplit(" ", 1)[1]))
    if any(not values or any(v != 0 for v in values) for values in found.values()):
        raise ValueError(f"Server must be idle before each shard: {found}")


def campaign(args):
    if min(args.shard_size, args.concurrency, args.stage_layers) < 1:
        raise ValueError("Shard size, concurrency and stage layers must be positive")
    if not 1 <= args.output_len <= 1024:
        raise ValueError("Output length must be 1..1024 to preserve sampler validity")
    if bool(args.adopt_report) != bool(args.adopt_count):
        raise ValueError("adopt-report and adopt-count must be provided together")
    args.output_dir.mkdir(parents=True)  # Refuse reuse, including concurrent launches.
    state = {
        "status": "preparing",
        "completed": 0,
        "batches": [],
        "started_ns": time.time_ns(),
        "pid": os.getpid(),
    }
    status_path = args.output_dir / "status.json"
    save(status_path, state)
    try:
        manifest = json.loads((args.run_dir / "run.json").read_text())
        trace = Path(manifest["trace"])
        initial = trace.stat()
        identity = (initial.st_dev, initial.st_ino)
        data = json.loads(args.dataset.read_text())
        ids, index_metadata = load_index(args, manifest, data)
        if not 0 <= args.adopt_count <= len(ids):
            raise ValueError("Invalid adopted count")
        save(args.output_dir / "source_indices.json", ids)
        save(
            args.output_dir / "manifest.json",
            {
                "server": manifest,
                "index": index_metadata,
                "index_sha256": digest(args.index_dir / "indices.json"),
                "output_len": args.output_len,
                "concurrency": args.concurrency,
                "stage_layers": args.stage_layers,
                "shard_size": args.shard_size,
                "tools": {
                    p.name: digest(p)
                    for p in Path(__file__).parent.glob("tau_batch_*.py")
                },
                "note": "First two turns; fixed output length; no oversampling. "
                "Each source row appears in one formal shard. Warmup excluded.",
            },
        )
        state.update(status="waiting_for_adopted_batch", total=len(ids))
        save(status_path, state)
        if args.await_exit:
            while not args.await_exit.exists():
                time.sleep(10)
            if json.loads(args.await_exit.read_text())["returncode"] != 0:
                raise ValueError("Adopted benchmark failed")
        reports = []

        def complete(start, end, report):
            destination = args.output_dir / f"shard_{start:06d}_{end:06d}"
            archived, local_report = archive(trace, report, destination, end - start)
            save(destination / "source_indices.json", ids[start:end])
            fitted = fit(archived, [local_report], args.stage_layers)
            save(destination / "parameters.json", fitted)
            reports.append(report)
            state["batches"].append(
                {
                    "start": start,
                    "end": end,
                    "report": str(report),
                    "archive": str(destination),
                }
            )
            state.update(
                completed=end, status="between_batches", updated_ns=time.time_ns()
            )
            save(status_path, state)
            print(f"COMPLETED {end}/{len(ids)} {destination}", flush=True)

        if args.adopt_report:
            command = json.loads(
                args.adopt_report.with_name("result_command.json").read_text()
            )["command"]
            for flag, value in {
                "--num-prompts": args.adopt_count,
                "--seed": args.seed,
                "--sharegpt-output-len": args.output_len,
                "--dataset-path": args.dataset,
            }.items():
                if command[command.index(flag) + 1] != str(value):
                    raise ValueError(f"Adopted benchmark mismatch: {flag}")
            complete(0, args.adopt_count, args.adopt_report)

        for start in range(args.adopt_count, len(ids), args.shard_size):
            if (args.output_dir / "STOP_AFTER_BATCH").exists():
                state["status"] = "stopped_between_batches"
                save(status_path, state)
                return
            current = trace.stat()
            if (
                current.st_dev,
                current.st_ino,
            ) != identity or current.st_size < initial.st_size:
                raise ValueError("Live trace replaced or truncated; server run changed")
            if args.server_pid and not Path(f"/proc/{args.server_pid}").exists():
                raise ValueError("Original server process exited")
            if (
                min(
                    shutil.disk_usage(trace.parent).free,
                    shutil.disk_usage(args.output_dir).free,
                )
                < args.reserve_gib * 2**30
            ):
                raise ValueError("Free disk space below reserve; collection stopped")
            ensure_idle(manifest)
            end = min(start + args.shard_size, len(ids))
            dataset = args.output_dir / f"input_{start:06d}_{end:06d}.json"
            save(dataset, [data[i] for i in ids[start:end]])
            target = args.output_dir / f"bench_{start:06d}_{end:06d}"
            command = [
                sys.executable,
                str(Path(__file__).with_name("tau_batch_run.py")),
                "bench",
                str(args.run_dir),
                "--mode",
                "smoke",
                "--dataset",
                str(dataset),
                "--num-prompts",
                str(end - start),
                "--output-len",
                str(args.output_len),
                "--concurrency",
                str(args.concurrency),
                "--seed",
                str(args.seed),
                "--result-dir",
                str(target),
            ]
            state.update(
                status="collecting",
                current_start=start,
                current_end=end,
                command=command,
                updated_ns=time.time_ns(),
            )
            save(status_path, state)
            # Server is already warm. Smoke mode means no additional warmup;
            # count and output length above explicitly define the formal workload.
            with (args.output_dir / f"bench_{start:06d}.log").open("x") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            complete(start, end, target / "result_trace_check.json")
        state["status"] = "fitting_all"
        save(status_path, state)
        save(
            args.output_dir / "parameters_all.json",
            fit(trace, reports, args.stage_layers),
        )
        state.update(status="complete", finished_ns=time.time_ns())
        save(status_path, state)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), updated_ns=time.time_ns())
        save(status_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--stage-layers", type=int, required=True)
    parser.add_argument("--adopt-report", type=Path)
    parser.add_argument("--adopt-count", type=int, default=0)
    parser.add_argument("--await-exit", type=Path)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--reserve-gib", type=float, default=10)
    campaign(parser.parse_args())


if __name__ == "__main__":
    main()
