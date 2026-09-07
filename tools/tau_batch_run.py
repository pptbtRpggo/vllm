# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ascend fixed-wave trace collection helpers; uses only the Python stdlib.

Run through serve_tau.sh / bench_tau.sh, or use the check-trace subcommand.
Selects the repository's instrumented Ascend worker; does not modify the plugin
installation or install packages.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from contextlib import suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/"
    "resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)
CLI = [sys.executable, "-m", "vllm.entrypoints.cli.main"]


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


def run_logged(command, log):
    """Forward output and signals; the foreground wrapper owns its child group."""
    print("+ " + shlex.join(command), flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    with log.open("w") as stream:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            assert child.stdout is not None
            for line in child.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            return child.wait()
        except KeyboardInterrupt:
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            return 130


def serve_settings():
    defaults = {
        "HOST": "127.0.0.1",
        "PORT": "8000",
        "TP": "1",
        "PP": "2",
        "MAX_MODEL_LEN": "4096",
        "MAX_NUM_SEQS": "32",
        "MAX_NUM_BATCHED_TOKENS": "8192",
        "MAX_REQS_PER_MB": "4",
        "MAX_MICROBATCHES": "0",
        "MIN_WAITING": "0",
        "GPU_MEM": "0.90",
        "ASCEND_RT_VISIBLE_DEVICES": "0,1",
    }
    return {key: os.environ.get(key, value) for key, value in defaults.items()}


def serve_command(model, settings, trace):
    flags = {
        "HOST": "host",
        "PORT": "port",
        "TP": "tensor-parallel-size",
        "PP": "pipeline-parallel-size",
        "MAX_MODEL_LEN": "max-model-len",
        "MAX_NUM_SEQS": "max-num-seqs",
        "MAX_NUM_BATCHED_TOKENS": "max-num-batched-tokens",
        "GPU_MEM": "gpu-memory-utilization",
        "MAX_REQS_PER_MB": "tau-batch-max-reqs-per-microbatch",
        "MAX_MICROBATCHES": "tau-batch-max-microbatches",
        "MIN_WAITING": "tau-batch-min-waiting",
    }
    command = CLI + ["serve", model]
    for key, flag in flags.items():
        command.extend(["--" + flag, settings[key]])
    return command + [
        "--worker-cls",
        "vllm.v1.worker.tau_ascend_worker.TauAscendWorker",
        "--scheduler-cls",
        "vllm.v1.core.sched.tau_batch.TauScheduler",
        "--tau-batch-trace",
        str(trace),
        "--trust-remote-code",
    ]


def serve(args):
    model = args.model or os.environ.get("MODEL")
    if not model:
        raise ValueError("Provide the model path: bash serve_tau.sh /path/to/model")
    if Path(model).expanduser().exists():
        model = str(Path(model).expanduser().resolve())
    settings = serve_settings()
    run = (
        Path(
            args.run_dir
            or os.environ.get("RUN_DIR")
            or os.environ.get("RUN")
            or ROOT / "trace_runs" / stamp()
        )
        .expanduser()
        .resolve()
    )
    trace = (
        Path(os.environ.get("TRACE", str(run / "trace.jsonl"))).expanduser().resolve()
    )
    command = serve_command(model, settings, trace)
    if args.dry_run:
        print(
            json.dumps(
                {"run_dir": str(run), "settings": settings, "command": command},
                indent=2,
            )
        )
        return 0
    if run.exists() or trace.exists():
        raise ValueError(
            "Use a NEW RUN_DIR and TRACE path; existing runs are not overwritten"
        )
    run.mkdir(parents=True)
    trace.parent.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name in ("vllm", "vllm-ascend", "torch", "torch-npu"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    manifest = {
        "model": model,
        "trace": str(trace),
        "settings": settings,
        "command": command,
        "versions": versions,
        "python": sys.executable,
        "commit": capture(["git", "rev-parse", "HEAD"]).strip(),
        "git_status": capture(["git", "status", "--short"]),
        "created_ns": time.time_ns(),
    }
    write_json(run / "run.json", manifest)
    (run / "packages.txt").write_text(capture([sys.executable, "-m", "pip", "freeze"]))
    (run / "npu.txt").write_text(capture(["npu-smi", "info"]))
    print(f"RUN_DIR={run}\nTRACE={trace}", flush=True)
    print(f"终端 B: bash {ROOT / 'bench_tau.sh'} {shlex.quote(str(run))}", flush=True)
    if int(settings["MIN_WAITING"]) != 0:
        print("注意：MIN_WAITING 非零可能阻塞小样本或尾部请求；bench 脚本会拒绝运行。")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = settings["ASCEND_RT_VISIBLE_DEVICES"]
    os.environ["VLLM_USE_V1"] = "1"
    code = run_logged(command, run / "server.log")
    write_json(
        run / "server_exit.json", {"returncode": code, "ended_ns": time.time_ns()}
    )
    return code


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


def bench_command(manifest, args, target, count, name, output_len):
    return CLI + [
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--base-url",
        args.base_url,
        "--endpoint",
        "/v1/completions",
        "--model",
        manifest["model"],
        "--tokenizer",
        manifest["model"],
        "--dataset-name",
        "sharegpt",
        "--dataset-path",
        str(args.dataset),
        "--num-prompts",
        str(count),
        "--sharegpt-output-len",
        str(output_len),
        "--ignore-eos",
        "--request-rate",
        str(args.request_rate),
        "--max-concurrency",
        str(args.concurrency),
        "--seed",
        str(args.seed),
        "--ready-check-timeout-sec",
        "0",
        "--num-warmups",
        "0",
        "--no-oversample",
        "--trust-remote-code",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(target),
        "--result-filename",
        name + ".json",
    ]


def bench(args):
    run = Path(args.run_dir or os.environ.get("RUN_DIR", "")).expanduser().resolve()
    manifest = json.loads((run / "run.json").read_text())
    settings = manifest["settings"]
    if int(settings["MIN_WAITING"]) != 0:
        raise ValueError("Restart the server with MIN_WAITING=0 before benchmarking")
    if int(settings["TP"]) != 1:
        raise ValueError(
            "Current trace lacks TP rank IDs; this collection workflow requires TP=1"
        )
    if args.num_prompts is None:
        args.num_prompts = 32 if args.mode == "smoke" else 1000
    if args.output_len is None:
        args.output_len = 64 if args.mode == "smoke" else 256
    if int(settings["PP"]) < 1:
        raise ValueError("PP must be positive")
    if args.request_rate <= 0 or math.isnan(args.request_rate):
        raise ValueError("request-rate must be positive (inf is allowed)")
    if not math.isfinite(args.ready_timeout) or args.ready_timeout < 0:
        raise ValueError("ready-timeout must be finite and nonnegative")
    if min(args.num_prompts, args.output_len, args.concurrency) < 1:
        raise ValueError("num-prompts, output-len and concurrency must be positive")
    host = settings["HOST"]
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    args.base_url = (args.base_url or f"http://{host}:{settings['PORT']}").rstrip("/")
    args.dataset = Path(args.dataset).expanduser().resolve()
    trace = Path(manifest["trace"])
    target = run / "bench" / (args.mode + "_" + stamp())
    command = bench_command(
        manifest, args, target, args.num_prompts, "result", args.output_len
    )
    if args.dry_run:
        print(shlex.join(command))
        return 0
    if not args.dataset.exists():
        if not args.download:
            raise ValueError(
                f"Dataset missing: {args.dataset}; supply --dataset or --download"
            )
        download_dataset(args.dataset)
    digest = hashlib.sha256()
    with args.dataset.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    target.mkdir(parents=True)
    write_json(
        target / "dataset.json",
        {
            "path": str(args.dataset),
            "sha256": digest.hexdigest(),
            "bytes": args.dataset.stat().st_size,
        },
    )
    wait_ready(args.base_url, manifest["model"], args.ready_timeout)

    def execute(count, name, output_len):
        offset = trace.stat().st_size if trace.exists() else 0
        command = bench_command(manifest, args, target, count, name, output_len)
        write_json(
            target / (name + "_command.json"),
            {
                "command": command,
                "trace": str(trace),
                "trace_start_offset": offset,
                "started_ns": time.time_ns(),
            },
        )
        code = run_logged(command, target / (name + ".log"))
        if code:
            raise ValueError(f"Benchmark failed ({code}); see {target}")
        result = json.loads((target / (name + ".json")).read_text())
        report = check_trace(trace, int(settings["PP"]), offset)
        report["completed"] = result.get("completed")
        report["expected"] = count
        report["passed"] &= report["completed"] == count
        write_json(target / (name + "_trace_check.json"), report)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        if not report["passed"]:
            raise ValueError(
                f"Trace is not ready for fitting; inspect {target}. "
                "If compute is missing, verify that serve selected TauAscendWorker "
                "and its runner hook was installed (see server.log). "
                "Do not substitute stage or done/PP for compute duration."
            )

    if args.mode == "collect":
        # Separate warmup avoids mixing its samples into the measured byte range.
        execute(32, "warmup", args.output_len)
    execute(args.num_prompts, "result", args.output_len)
    print(
        f"采集及覆盖检查完成：{target}\n"
        "计时仍需在 NPU 实机验证；覆盖检查通过不代表计时口径已校准。"
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    server = subs.add_parser(
        "serve", help="Launch Ascend server and save a fresh run directory"
    )
    server.add_argument("model", nargs="?")
    server.add_argument("--run-dir")
    server.add_argument("--dry-run", action="store_true")
    benchmark = subs.add_parser(
        "bench", help="ShareGPT smoke check or measured collection"
    )
    benchmark.add_argument("run_dir", nargs="?")
    benchmark.add_argument("--mode", choices=("smoke", "collect"), default="smoke")
    benchmark.add_argument(
        "--dataset", default=str(ROOT / "datasets" / "sharegpt.json")
    )
    benchmark.add_argument("--download", action="store_true")
    benchmark.add_argument("--num-prompts", type=int)
    benchmark.add_argument("--output-len", type=int)
    benchmark.add_argument("--concurrency", type=int, default=32)
    benchmark.add_argument("--request-rate", type=float, default=float("inf"))
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--base-url")
    benchmark.add_argument("--ready-timeout", type=float, default=300)
    benchmark.add_argument("--dry-run", action="store_true")
    check = subs.add_parser("check-trace")
    check.add_argument("path", type=Path)
    check.add_argument("--pp", type=int, default=2)
    args = parser.parse_args()
    if args.action == "check-trace":
        report = check_trace(args.path, args.pp)
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 2
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
