# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read device-wide TorchNPU counters; keep CSV separate from native diagnostics."""

import argparse
import csv
import datetime
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

FIELDS = (
    "timestamp_utc",
    "elapsed_s",
    "node",
    "role",
    "logical_device",
    "device_name",
    "utilization_pct",
    "memory_used_bytes",
    "memory_total_bytes",
    "status",
    "error",
)


def collect(
    npu,
    stream,
    *,
    interval=2.0,
    samples=0,
    duration=0.0,
    node="",
    role="",
    stop_file=None,
):
    """Write visible-device counters, preserving missing values and failures."""
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    stream.flush()
    count = npu.device_count()
    if not count:
        raise RuntimeError("容器内没有可见 NPU")
    started = time.monotonic()
    disabled = {}
    iteration = 0
    while True:
        if stop_file is not None and stop_file.exists():
            break
        for device in range(count):
            if device in disabled:
                continue
            row = dict.fromkeys(FIELDS, "")
            row.update(
                timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                elapsed_s=f"{time.monotonic() - started:.3f}",
                node=node,
                role=role,
                logical_device=device,
                status="ok",
            )
            errors = []
            try:
                npu.set_device(device)
            except Exception as exc:
                disabled[device] = str(exc)
                row.update(status="error", error=f"set_device: {exc}")
                writer.writerow(row)
                stream.flush()
                continue
            for field, query in (
                ("device_name", lambda device=device: npu.get_device_name(device)),
                (
                    "utilization_pct",
                    lambda device=device: float(npu.utilization(device)),
                ),
                ("memory", lambda device=device: npu.mem_get_info(device)),
            ):
                try:
                    value = query()
                    if field == "memory":
                        free, total = value
                        row["memory_used_bytes"] = int(total - free)
                        row["memory_total_bytes"] = int(total)
                    elif field == "utilization_pct":
                        if not math.isfinite(value) or not 0 <= value <= 100:
                            raise ValueError(f"无效利用率 {value}")
                        row[field] = f"{value:.1f}"
                    else:
                        row[field] = value
                except Exception as exc:
                    errors.append(f"{field}: {exc}")
            if errors:
                row.update(status="partial", error="; ".join(errors))
            writer.writerow(row)
            stream.flush()
        iteration += 1
        elapsed = time.monotonic() - started
        if len(disabled) == count or (samples and iteration >= samples):
            break
        if duration and elapsed >= duration:
            break
        next_sample = time.monotonic() + (
            min(interval, duration - elapsed) if duration else interval
        )
        while time.monotonic() < next_sample:
            if stop_file is not None and stop_file.exists():
                break
            time.sleep(min(0.2, max(0, next_sample - time.monotonic())))
        if duration and time.monotonic() - started >= duration:
            break
    return 1 if disabled else 0


def bootstrap(args):
    """Reuse the serving interpreter/SDK without probing or restarting the model."""
    from check_cpu_gloo import capture, load_sdk_environment, prepare_environment

    record = {}
    if args.runtime.is_file():
        record = json.loads(args.runtime.read_text())
    python = args.python or record.get("python") or sys.executable
    environment = dict(os.environ)
    sdk = args.cann_env or record.get("sdk_environment", "current")
    deadline = time.monotonic() + 45
    for script in sdk.split(" + "):
        if script != "current":
            if not Path(script).is_file():
                raise ValueError(f"SDK 脚本不存在：{script}")
            environment = load_sdk_environment(script, environment, deadline)
    root = Path(__file__).resolve().parents[3]
    environment = prepare_environment(root, environment)
    rc, output = capture(
        [python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        environment,
        10,
    )
    if rc:
        raise RuntimeError(f"无法启动监控解释器：{output[-1000:]}")
    plugin = Path(output.strip().splitlines()[-1]) / "vllm_ascend"
    if plugin.is_dir():
        environment["LD_LIBRARY_PATH"] = (
            str(plugin) + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
        )
    os.execvpe(
        python,
        [python, "-u", str(Path(__file__).resolve()), "--collect", *sys.argv[1:]],
        environment,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--python")
    parser.add_argument("--cann-env")
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path("/work/hetero_logs/start/local_preflight.json"),
    )
    parser.add_argument("--collect", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stop-directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        not math.isfinite(args.interval)
        or args.interval <= 0
        or not math.isfinite(args.duration)
        or args.duration < 0
        or args.samples < 0
    ):
        parser.error("interval 必须大于 0；duration/samples 必须非负")
    if not args.collect:
        bootstrap(args)
    # Torch/CANN may print to fd 1, including from C++. Preserve a CSV-only fd.
    stream = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    stop_file = args.stop_directory / "stop" if args.stop_directory else None
    try:
        if stop_file is not None and stop_file.exists():
            return 0
        import torch
        import torch_npu  # noqa: F401

        return collect(
            torch.npu,
            stream,
            interval=args.interval,
            samples=args.samples,
            duration=args.duration,
            node=os.environ.get("VLLM_HOST_IP", socket.gethostname()),
            role=os.environ.get("HETERO_PP_ROLE", ""),
            stop_file=stop_file,
        )
    except KeyboardInterrupt:
        return 0
    finally:
        stream.close()
        if args.stop_directory:
            if stop_file is not None:
                stop_file.unlink(missing_ok=True)
            args.stop_directory.rmdir()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"监控退出：{exc}", file=sys.stderr)
        sys.exit(1)
