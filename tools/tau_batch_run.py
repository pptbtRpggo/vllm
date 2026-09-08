# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ascend trace metadata and benchmark helpers; uses only the Python stdlib.

configs/serve.yaml and configs/bench.yaml own experiment defaults. The shell
launchers run vllm directly; these helpers record metadata and validate traces.
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
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


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
    "TRACE_ENABLED",
)


def prepare_serve(args):
    """Record the shell's exact configuration without launching a server."""
    run_path = Path(args.run_dir).expanduser()
    trace_path = Path(args.trace).expanduser() if args.trace else None
    run = run_path.resolve()
    trace = trace_path.resolve() if trace_path else None
    settings = {key: os.environ[key] for key in SERVE_SETTING_NAMES}
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("prepare-serve requires the actual server command")
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    run_dir=str(run),
                    trace=str(trace) if trace else None,
                    settings=settings,
                    command=command,
                    launch_config=json.loads(
                        os.environ.get("TAU_LAUNCH_CONFIG", "null")
                    ),
                ),
                indent=2,
            )
        )
        return 0
    paths = [run_path] + ([trace_path] if trace_path else [])
    if any(p.exists() or p.is_symlink() for p in paths):
        raise ValueError(
            "Use a NEW RUN_DIR and TRACE path; existing runs are not overwritten"
        )
    if run == trace:
        raise ValueError("RUN_DIR and TRACE must be different paths")
    latest = ROOT / "output" / "latest"
    resolved_paths = [run] + ([trace] if trace else [])
    if any(latest == p or latest in p.parents for p in resolved_paths):
        raise ValueError("output/latest is reserved for the latest run link")
    if latest.exists() and not latest.is_symlink():
        raise ValueError(f"{latest} already exists and is not a symlink")
    run.mkdir(parents=True)
    if trace:
        trace.parent.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name in ("vllm", "vllm-ascend", "torch", "torch-npu"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    write_json(
        run / "server_meta.json",
        {
            "launch_config": json.loads(os.environ.get("TAU_LAUNCH_CONFIG", "null")),
            "model": args.model,
            "trace": str(trace) if trace else None,
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
    # Publish only complete metadata. Atomic replacement also handles a stale link.
    latest.parent.mkdir(parents=True, exist_ok=True)
    temporary = latest.with_name(".latest_" + uuid.uuid4().hex)
    try:
        temporary.symlink_to(run, target_is_directory=True)
        temporary.replace(latest)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"RUN_DIR={run}\nTRACE={trace or 'disabled'}", flush=True)
    print(
        f"终端 B: bash {shlex.quote(str(ROOT / 'bench_tau.sh'))} --mode smoke",
        flush=True,
    )
    print("启动命令: " + shlex.join(command), flush=True)
    if int(settings["MIN_WAITING"]) != 0:
        print("注意：MIN_WAITING 非零可能阻塞小样本或尾部请求；bench 脚本会拒绝运行。")
    return 0


def serve(args):
    """Keep the old CLI entry point, with no second set of server defaults."""
    command = ["bash", str(ROOT / "serve_tau.sh")]
    if args.config:
        command.extend(["--config", args.config])
    if args.model:
        command.append(args.model)
    if args.run_dir:
        command.extend(["--run-dir", args.run_dir])
    if args.dry_run:
        command.append("--dry-run")
    if args.trace_enabled is not None:
        command.append("--trace" if args.trace_enabled else "--no-trace")
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
                raise ValueError(
                    "The endpoint model does not match this server_meta.json"
                )
            return
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise ValueError(f"Server not ready at {base_url}") from None
            time.sleep(2)


def prepare_bench(args):
    """Read manifest and prepare artifacts; traffic settings come from shell."""
    run = Path(args.run_dir).expanduser().resolve()
    manifest = json.loads((run / "server_meta.json").read_text())
    settings = manifest["settings"]
    if int(settings["MIN_WAITING"]) != 0:
        raise ValueError("Restart the server with MIN_WAITING=0 before benchmarking")
    if manifest.get("trace") and int(settings["TP"]) != 1:
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
    if os.environ.get("SLO_CONFIG") or os.environ.get("SLO_INLINE"):
        int(os.environ["SLO_SEED"])
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
    slo_config = None
    if os.environ.get("SLO_CONFIG"):
        slo_config = Path(os.environ["SLO_CONFIG"]).expanduser().resolve()
        if not slo_config.is_file():
            raise ValueError(f"SLO config missing: {slo_config}")
    slo_source = None
    slo_content = None
    if slo_config:
        slo_content = slo_config.read_bytes()
        slo_source = {"path": str(slo_config)}
    elif os.environ.get("SLO_INLINE"):
        slo_content = os.environ["SLO_INLINE"].encode()
        source = json.loads(os.environ["TAU_LAUNCH_CONFIG"])
        slo_source = {"path": source["path"], "section": "SLO"}
        slo_config = Path(os.environ["BENCH_WORK_DIR"]) / "slo_config.json"
        # Dry-run also shows the effective inline SLO in its disposable config.
        if args.dry_run:
            slo_config.write_bytes(slo_content)
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
        metadata = {
            "launch_config": json.loads(os.environ.get("TAU_LAUNCH_CONFIG", "null")),
            "server_meta": str(run / "server_meta.json"),
            "created_ns": time.time_ns(),
            "settings": {
                k: os.environ.get(k, "")
                for k in (
                    "MODE",
                    "NUM_PROMPTS",
                    "OUTPUT_LEN",
                    "CONCURRENCY",
                    "WARMUP_REQUESTS",
                    "REQUEST_RATE",
                    "BURSTINESS",
                    "SEED",
                    "SLO_SEED",
                    "IGNORE_EOS",
                    "READY_TIMEOUT",
                )
            },
            "base_url": base_url,
            "dataset": {
                "path": str(dataset),
                "sha256": digest.hexdigest(),
                "bytes": dataset.stat().st_size,
            },
            "slo": None,
            "phases": {},
        }
        if slo_config is not None:
            content = slo_content
            metadata["slo"] = {
                **slo_source,
                "sha256": hashlib.sha256(content).hexdigest(),
                "slo_seed": int(os.environ["SLO_SEED"]),
                "config": json.loads(content),
            }
            # The persistent snapshot is embedded in bench_meta.json. The native
            # CLI reads an identical, disposable copy from local scratch space.
            slo_config = Path(os.environ["BENCH_WORK_DIR"]) / "slo_config.json"
            slo_config.write_bytes(content)
        write_json(target / "bench_meta.json", metadata)
    if args.dry_run and slo_content is not None:
        print("Effective SLO: " + slo_content.decode(), flush=True)
    # Only fixed variable names and shlex-quoted values are sourced by the shell.
    context = dict(
        RUN_DIR=str(run),
        MODEL=manifest["model"],
        TRACE=str(Path(manifest["trace"]).resolve()) if manifest.get("trace") else "",
        BASE_URL=base_url,
        DATASET=str(dataset),
        RESULT_DIR=str(target),
        SLO_CONFIG=str(slo_config) if slo_config else "",
    )
    Path(args.config_file).write_text(
        "".join(f"{key}={shlex.quote(value)}\n" for key, value in context.items())
    )
    if not args.dry_run:
        wait_ready(base_url, manifest["model"], timeout)
    return 0


def begin_bench(args):
    manifest = json.loads((Path(args.run_dir) / "server_meta.json").read_text())
    trace = Path(manifest["trace"]) if manifest.get("trace") else None
    stat = trace.stat() if trace and trace.exists() else None
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    path = Path(args.result_dir) / "bench_meta.json"
    metadata = json.loads(path.read_text())
    metadata["phases"][args.name] = {
        "command": command,
        "trace": str(trace) if trace else None,
        "expected": args.count,
        "trace_start_offset": stat.st_size if stat else 0,
        "trace_identity": [stat.st_dev, stat.st_ino] if stat else None,
        "started_ns": time.time_ns(),
    }
    write_json(path, metadata)
    return 0


def end_bench(args):
    target = Path(args.result_dir)
    manifest = json.loads((Path(args.run_dir) / "server_meta.json").read_text())
    meta_path = target / "bench_meta.json"
    bench_meta = json.loads(meta_path.read_text())
    metadata = bench_meta["phases"][args.name]
    result = json.loads(
        (Path(os.environ["BENCH_WORK_DIR"]) / (args.name + ".json")).read_text()
    )
    report = {
        "enabled": bool(metadata.get("trace")),
        "completed": result.get("completed"),
        "expected": metadata["expected"],
    }
    if not report["enabled"]:
        report["trace_check"] = "disabled"
    else:
        try:
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
            report.update(
                check_trace(
                    trace,
                    int(manifest["settings"]["PP"]),
                    metadata["trace_start_offset"],
                    end,
                )
            )
            report["path"] = str(trace)
            report["identity"] = [stat.st_dev, stat.st_ino] if stat else None
            report["passed"] &= report["completed"] == report["expected"]
        except (OSError, ValueError) as exc:
            report.update(passed=False, error=str(exc))
    metadata["finished_ns"] = time.time_ns()
    assignment = result.pop("slo_assignment", None)
    if assignment is not None:
        metadata["slo_assignment"] = {
            k: v
            for k, v in assignment.items()
            if k not in ("requests", "config", "dataset_path")
        }
    # Keep CLI/run configuration in metadata; only measurements in the summary.
    for key in (
        "date",
        "endpoint_type",
        "backend",
        "label",
        "model_id",
        "tokenizer_id",
        "num_prompts",
        "request_rate",
        "burstiness",
        "max_concurrency",
    ):
        if key in result:
            metadata[key] = result.pop(key)
    write_json(meta_path, bench_meta)
    summary_path = target / "summary.json"
    previous = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if args.name == "warmup":
        summary = {"warmup": {**result, "trace": report}}
    else:
        summary = {**result, "trace": report}
        if "warmup" in previous:
            summary["warmup"] = previous["warmup"]
    write_json(summary_path, summary)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if report["completed"] != report["expected"]:
        raise ValueError("Benchmark did not complete the expected requests")
    if report["enabled"] and not report["passed"]:
        raise ValueError(
            f"{report.get('error', 'Trace is not ready for fitting')}; "
            f"inspect {summary_path}. "
            "If compute is missing, check TauAscendWorker in server.log."
        )
    return 0


def bench(args):
    """Compatibility entry point; delegates to the configured shell launcher."""
    command = ["bash", str(ROOT / "bench_tau.sh")]
    if args.run_dir:
        command.append(args.run_dir)
    for name in (
        "config",
        "mode",
        "dataset",
        "num_prompts",
        "result_dir",
        "output_len",
        "concurrency",
        "request_rate",
        "burstiness",
        "seed",
        "slo_config",
        "slo_seed",
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
    server.add_argument("--config")
    server.add_argument("--run-dir")
    server.add_argument("--dry-run", action="store_true")
    server.add_argument(
        "--trace",
        dest="trace_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
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
    benchmark.add_argument("--config")
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
    benchmark.add_argument("--slo-config")
    benchmark.add_argument("--slo-seed", type=int)
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
