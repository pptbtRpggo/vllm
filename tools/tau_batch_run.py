# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ascend trace metadata and benchmark helpers; uses only the Python stdlib.

serve_tau.sh owns all server defaults and launches vllm directly. prepare-serve
only records its configuration. bench_tau.sh runs the benchmark workflow.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/"
    "resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)


def stamp():
    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def capture(command):
    try:
        result = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=30, check=False
        )
        return result.stdout + result.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)


# Manifest field names, not defaults. serve_tau.sh owns the values.
SERVE_SETTING_NAMES = (
    "HOST",
    "PORT",
    "TP",
    "PP",
    "MAX_MODEL_LEN",
    "MAX_NUM_SEQS",
    "MAX_NUM_BATCHED_TOKENS",
    "MAX_MICROBATCHES",
    "MIN_WAITING",
    "GPU_MEM",
    "ASCEND_RT_VISIBLE_DEVICES",
)


def prepare_serve(args):
    """Record the shell's exact configuration without launching a server."""
    run_path = Path(args.run_dir).expanduser()
    trace_path = Path(args.trace).expanduser()
    run = run_path.resolve()
    trace = trace_path.resolve()
    settings = {key: os.environ[key] for key in SERVE_SETTING_NAMES}
    # Preserve the manifest field; the launcher exposes only MAX_NUM_SEQS.
    settings["MAX_REQS_PER_MB"] = settings["MAX_NUM_SEQS"]
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("prepare-serve requires the actual server command")
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    run_dir=str(run),
                    trace=str(trace),
                    settings=settings,
                    command=command,
                ),
                indent=2,
            )
        )
        return 0
    if any(p.exists() or p.is_symlink() for p in (run_path, trace_path)):
        raise ValueError(
            "Use a NEW RUN_DIR and TRACE path; existing runs are not overwritten"
        )
    if run == trace:
        raise ValueError("RUN_DIR and TRACE must be different paths")
    run.mkdir(parents=True)
    trace.parent.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name in ("vllm", "vllm-ascend", "torch", "torch-npu"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    write_json(
        run / "run.json",
        {
            "model": args.model,
            "trace": str(trace),
            "settings": settings,
            "command": command,
            "versions": versions,
            "python": sys.executable,
            "vllm_executable": shutil.which(command[0]),
            "commit": capture(["git", "rev-parse", "HEAD"]).strip(),
            "git_status": capture(["git", "status", "--short"]),
            "created_ns": time.time_ns(),
        },
    )
    (run / "packages.txt").write_text(capture([sys.executable, "-m", "pip", "freeze"]))
    (run / "npu.txt").write_text(capture(["npu-smi", "info"]))
    print(f"RUN_DIR={run}\nTRACE={trace}", flush=True)
    print(f"终端 B: bash {ROOT / 'bench_tau.sh'} {shlex.quote(str(run))}", flush=True)
    print("启动命令: " + shlex.join(command), flush=True)
    if int(settings["MIN_WAITING"]) != 0:
        print("注意：MIN_WAITING 非零可能阻塞小样本或尾部请求；bench 脚本会拒绝运行。")
    return 0


def serve(args):
    """Keep the old CLI entry point, with no second set of server defaults."""
    command = ["bash", str(ROOT / "serve_tau.sh")]
    if args.model:
        command.append(args.model)
    if args.run_dir:
        command.extend(["--run-dir", args.run_dir])
    if args.dry_run:
        command.append("--dry-run")
    os.execvpe("bash", command, {**os.environ, "PYTHON": sys.executable})


def check_trace(path, pp, offset=0, end_offset=None):
    """Check a closed byte range. Coverage is structural, not timing calibration."""
    if pp < 1 or offset < 0:
        raise ValueError("pp must be positive and offset nonnegative")
    end_offset = (
        (path.stat().st_size if path.exists() else 0)
        if end_offset is None
        else end_offset
    )
    counts, coverage = Counter(), Counter()
    emits, seen, valid = {}, set(), set()
    malformed = invalid = orphan = duplicates = 0
    if path.exists():
        with path.open("rb") as stream:
            stream.seek(offset)
            while stream.tell() < end_offset:
                line = stream.readline(end_offset - stream.tell())
                if not line:
                    malformed += 1
                    break
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict) or not line.endswith(b"\n"):
                        raise ValueError("Incomplete JSONL record")
                except (ValueError, UnicodeDecodeError):
                    malformed += 1
                    continue
                kind = event.get("event")
                if not isinstance(kind, str):
                    malformed += 1
                    continue
                counts[kind] += 1
                if kind not in ("emit", "compute"):
                    continue
                fid = event.get("fwd_id")
                if type(fid) is not int or fid < 1:
                    malformed += 1
                    continue
                if kind == "emit":
                    if fid in emits:
                        duplicates += 1
                    emits[fid] = event
                    continue
                phase, rank = event.get("phase"), event.get("pp_rank")
                if fid not in emits:
                    orphan += 1
                    continue
                if type(rank) is not int or not 0 <= rank < pp:
                    invalid += 1
                    continue
                key = (fid, rank)
                if key in seen:
                    duplicates += 1
                seen.add(key)
                lens = event.get("seq_lens")
                start, end = event.get("start_ts_ns"), event.get("end_ts_ns")
                fields = ("phase", "n", "s_sum", "s_max", "tokens", "seq_lens")
                if (
                    phase not in ("prefill", "decode")
                    or any(event.get(k) != emits[fid].get(k) for k in fields)
                    or type(start) is not int
                    or type(end) is not int
                    or end <= start
                    or not isinstance(lens, list)
                    or not lens
                    or any(type(n) is not int or n < 1 for n in lens)
                    or len(lens) != event.get("n")
                    or sum(lens) != event.get("s_sum")
                    or max(lens) != event.get("s_max")
                    or type(event.get("tokens")) is not int
                    or event["tokens"] <= 0
                ):
                    invalid += 1
                    continue
                valid.add(key)
                coverage[f"pp{rank}/{phase}"] += 1
    missing = [
        f"pp{r}/{p}"
        for r in range(pp)
        for p in ("prefill", "decode")
        if not coverage[f"pp{r}/{p}"]
    ]
    incomplete = sum(any((fid, r) not in valid for r in range(pp)) for fid in emits)
    return {
        "events": dict(counts),
        "compute_coverage": dict(coverage),
        "missing": missing,
        "malformed": malformed,
        "invalid_compute": invalid,
        "orphan_compute": orphan,
        "duplicate_compute": duplicates,
        "incomplete_forwards": incomplete,
        "trace_start_offset": offset,
        "trace_end_offset": end_offset,
        "passed": not (
            missing or malformed or invalid or orphan or duplicates or incomplete
        ),
        "scope": "structural check only; verify compute timing on device",
    }


def download_dataset(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    print(f"Downloading {SHAREGPT_URL}", flush=True)
    try:
        with (
            urllib.request.urlopen(SHAREGPT_URL, timeout=60) as response,
            temporary.open("wb") as stream,
        ):
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def wait_ready(base_url, model, seconds):
    deadline = time.monotonic() + seconds
    while True:
        try:
            with urllib.request.urlopen(base_url + "/v1/models", timeout=5) as response:
                models = json.load(response)
            if model not in [item["id"] for item in models["data"]]:
                raise ValueError("The endpoint model does not match this run.json")
            return
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise ValueError(f"Server not ready at {base_url}") from None
            time.sleep(2)


def prepare_bench(args):
    """Read manifest and prepare artifacts; traffic settings come from shell."""
    run = Path(args.run_dir).expanduser().resolve()
    manifest = json.loads((run / "run.json").read_text())
    settings = manifest["settings"]
    if int(settings["MIN_WAITING"]) != 0:
        raise ValueError("Restart the server with MIN_WAITING=0 before benchmarking")
    if int(settings["TP"]) != 1:
        raise ValueError(
            "Current trace lacks TP rank IDs; this collection requires TP=1"
        )
    if int(settings["PP"]) < 1:
        raise ValueError("PP must be positive")
    for key in ("NUM_PROMPTS", "OUTPUT_LEN", "CONCURRENCY", "WARMUP_REQUESTS"):
        if int(os.environ[key]) < 1:
            raise ValueError(f"{key} must be positive")
    for key in ("REQUEST_RATE", "BURSTINESS"):
        value = float(os.environ[key])
        if value <= 0 or math.isnan(value):
            raise ValueError(f"{key} must be positive (inf is allowed)")
    for key in ("IGNORE_EOS", "DOWNLOAD"):
        if os.environ[key] not in ("0", "1"):
            raise ValueError(f"{key} must be 0 or 1")
    int(os.environ["SEED"])
    timeout = float(os.environ["READY_TIMEOUT"])
    if timeout < 0 or not math.isfinite(timeout):
        raise ValueError("READY_TIMEOUT must be finite and nonnegative")
    mode = os.environ["MODE"]
    if mode not in ("smoke", "collect"):
        raise ValueError("MODE must be smoke or collect")
    host = settings["HOST"]
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    base_url = (
        os.environ.get("BASE_URL") or f"http://{host}:{settings['PORT']}"
    ).rstrip("/")
    dataset = Path(os.environ["DATASET"]).expanduser().resolve()
    target = (
        Path(os.environ.get("RESULT_DIR") or run / "bench" / (mode + "_" + stamp()))
        .expanduser()
        .resolve()
    )
    if not args.dry_run:
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Use a new result directory: {target}")
        if not dataset.exists():
            if os.environ["DOWNLOAD"] != "1":
                raise ValueError(
                    f"Dataset missing: {dataset}; supply --dataset or --download"
                )
            download_dataset(dataset)
        digest = hashlib.sha256()
        with dataset.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        target.mkdir(parents=True)
        write_json(
            target / "dataset.json",
            {
                "path": str(dataset),
                "sha256": digest.hexdigest(),
                "bytes": dataset.stat().st_size,
            },
        )
        wait_ready(base_url, manifest["model"], timeout)
    # Only fixed variable names and shlex-quoted values are sourced by the shell.
    context = dict(
        RUN_DIR=str(run),
        MODEL=manifest["model"],
        TRACE=str(Path(manifest["trace"]).resolve()),
        BASE_URL=base_url,
        DATASET=str(dataset),
        RESULT_DIR=str(target),
    )
    Path(args.config_file).write_text(
        "".join(f"{key}={shlex.quote(value)}\n" for key, value in context.items())
    )
    return 0


def begin_bench(args):
    manifest = json.loads((Path(args.run_dir) / "run.json").read_text())
    trace = Path(manifest["trace"])
    stat = trace.stat() if trace.exists() else None
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    write_json(
        Path(args.result_dir) / (args.name + "_command.json"),
        {
            "command": command,
            "trace": str(trace),
            "expected": args.count,
            "trace_start_offset": stat.st_size if stat else 0,
            "trace_identity": [stat.st_dev, stat.st_ino] if stat else None,
            "started_ns": time.time_ns(),
        },
    )
    return 0


def end_bench(args):
    target = Path(args.result_dir)
    manifest = json.loads((Path(args.run_dir) / "run.json").read_text())
    metadata = json.loads((target / (args.name + "_command.json")).read_text())
    result = json.loads((target / (args.name + ".json")).read_text())
    trace = Path(metadata["trace"])
    stat = trace.stat() if trace.exists() else None
    identity = metadata["trace_identity"]
    if identity is not None and (
        stat is None or identity != [stat.st_dev, stat.st_ino]
    ):
        raise ValueError("Trace file replaced during benchmark")
    end = stat.st_size if stat else 0
    if end < metadata["trace_start_offset"]:
        raise ValueError("Trace file truncated during benchmark")
    report = check_trace(
        trace, int(manifest["settings"]["PP"]), metadata["trace_start_offset"], end
    )
    report["completed"] = result.get("completed")
    report["expected"] = metadata["expected"]
    report["passed"] &= report["completed"] == report["expected"]
    write_json(target / (args.name + "_trace_check.json"), report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise ValueError(
            f"Trace is not ready for fitting; inspect {target}. "
            "If compute is missing, check TauAscendWorker in server.log."
        )
    return 0


def bench(args):
    """Compatibility entry point; the shell owns defaults and vllm invocation."""
    command = ["bash", str(ROOT / "bench_tau.sh")]
    if args.run_dir:
        command.append(args.run_dir)
    for name in (
        "mode",
        "dataset",
        "num_prompts",
        "result_dir",
        "output_len",
        "concurrency",
        "request_rate",
        "burstiness",
        "seed",
        "base_url",
        "ready_timeout",
    ):
        value = getattr(args, name)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    for name in ("download", "dry_run"):
        if getattr(args, name):
            command.append("--" + name.replace("_", "-"))
    os.execvpe("bash", command, {**os.environ, "PYTHON": sys.executable})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    server = subs.add_parser(
        "serve", help="Compatibility entry point: delegates to serve_tau.sh"
    )
    server.add_argument("model", nargs="?")
    server.add_argument("--run-dir")
    server.add_argument("--dry-run", action="store_true")
    metadata = subs.add_parser(
        "prepare-serve", help="Record serve_tau.sh configuration"
    )
    metadata.add_argument("--model", required=True)
    metadata.add_argument("--run-dir", required=True)
    metadata.add_argument("--trace", required=True)
    metadata.add_argument("--dry-run", action="store_true")
    metadata.add_argument("command", nargs=argparse.REMAINDER)
    benchmark = subs.add_parser(
        "bench", help="ShareGPT smoke check or measured collection"
    )
    benchmark.add_argument("run_dir", nargs="?")
    benchmark.add_argument("--mode", choices=("smoke", "collect"))
    benchmark.add_argument("--dataset")
    benchmark.add_argument("--download", action="store_true")
    benchmark.add_argument("--num-prompts", type=int)
    benchmark.add_argument("--result-dir", help="New directory for this benchmark")
    benchmark.add_argument("--output-len", type=int)
    benchmark.add_argument("--concurrency", type=int)
    benchmark.add_argument("--request-rate", type=float)
    benchmark.add_argument("--burstiness", type=float)
    benchmark.add_argument("--seed", type=int)
    benchmark.add_argument("--base-url")
    benchmark.add_argument("--ready-timeout", type=float)
    benchmark.add_argument("--dry-run", action="store_true")
    prepare = subs.add_parser(
        "prepare-bench", help="Prepare bench artifacts and context"
    )
    prepare.add_argument("run_dir")
    prepare.add_argument("--config-file", required=True)
    prepare.add_argument("--dry-run", action="store_true")
    for action in ("begin-bench", "end-bench"):
        phase = subs.add_parser(action, help="Record/check one benchmark trace range")
        phase.add_argument("run_dir")
        phase.add_argument("result_dir")
        phase.add_argument("name", choices=("warmup", "result"))
        if action == "begin-bench":
            phase.add_argument("count", type=int)
            phase.add_argument("command", nargs=argparse.REMAINDER)
    check = subs.add_parser("check-trace")
    check.add_argument("path", type=Path)
    check.add_argument("--pp", type=int, default=2)
    args = parser.parse_args()
    if args.action == "check-trace":
        report = check_trace(args.path, args.pp)
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 2
    if args.action == "prepare-serve":
        return prepare_serve(args)
    helpers = {
        "prepare-bench": prepare_bench,
        "begin-bench": begin_bench,
        "end-bench": end_bench,
    }
    if args.action in helpers:
        return helpers[args.action](args)
    return serve(args) if args.action == "serve" else bench(args)


if __name__ == "__main__":

    def terminate(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
