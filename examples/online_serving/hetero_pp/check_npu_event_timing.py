# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare owner/background readings of the same NPU events; no model required.

Run inside the existing Ascend container with its working Python/CANN setup:
    python check_npu_event_timing.py --device 1
Device numbers are container-visible logical indices; check the mapping first.
The parent enforces a 90-second deadline. This is a diagnostic, not a profile
release gate or evidence that full model timing is correct.
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path


def consistent(row: dict) -> bool:
    """Require finite, contained and additive intervals within 0.01 ms."""
    total, start, end, inner = (
        row[key] for key in ("total_ms", "start_ms", "end_ms", "inner_ms")
    )
    return (
        all(math.isfinite(v) for v in (total, start, end, inner))
        and min(total, start, end, inner) >= -0.01
        and start <= end + 0.01
        and end <= total + 0.01
        and inner <= total + 0.01
        and abs((end - start) - inner) <= 0.01
    )


def read_events(events) -> dict:
    """Read the exact same event objects from either calling thread."""
    start, inner_start, inner_end, end = events
    row = {
        "total_ms": start.elapsed_time(end),
        "inner_ms": inner_start.elapsed_time(inner_end),
        "start_ms": start.elapsed_time(inner_start),
        "end_ms": start.elapsed_time(inner_end),
    }
    row["consistent"] = consistent(row)
    return row


def background_read(torch, device, events) -> dict:
    """Match the profiler's background set_device plus elapsed_time path."""
    result, errors = [], []

    def run():
        try:
            torch.npu.set_device(device)
            result.append(read_events(events))
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=10)
    if thread.is_alive():
        raise TimeoutError("background event read did not finish within 10 seconds")
    if errors:
        raise errors[0]
    return result[0]


def probe(device: int, repeats: int) -> dict:
    import torch
    import torch_npu

    torch.npu.set_device(device)
    stream = torch.npu.current_stream(device)
    x = torch.ones((64, 64), dtype=torch.float16, device=f"npu:{device}")
    for _ in range(2):
        torch.mm(x, x)
    stream.synchronize()
    rows = []
    for sync in ("event", "stream"):
        for first in ("owner", "background"):
            group = []
            for repeat in range(repeats):
                events = [torch.npu.Event(enable_timing=True) for _ in range(4)]
                events[0].record(stream)
                events[1].record(stream)
                y = torch.mm(x, x)
                events[2].record(stream)
                events[3].record(stream)
                if sync == "event":
                    events[3].synchronize()
                else:
                    stream.synchronize()
                row = {"sync": sync, "first": first, "repeat": repeat}
                for reader in (first, "background" if first == "owner" else "owner"):
                    row[reader] = (
                        read_events(events)
                        if reader == "owner"
                        else background_read(torch, device, events)
                    )
                if not bool((y.cpu() == 64).all()):
                    raise ValueError("matmul content check failed")
                group.append(row)
                rows.append(row)
            counts = {
                reader: sum(not r[reader]["consistent"] for r in group)
                for reader in ("owner", "background")
            }
            delta = max(
                abs(r["owner"][key] - r["background"][key])
                for r in group
                for key in ("total_ms", "inner_ms", "start_ms", "end_ms")
            )
            print(
                f"{sync=} {first=}: inconsistent={counts}/{repeats}, "
                f"max_reader_delta_ms={delta:.6f}",
                flush=True,
            )
            bad = next(
                (
                    r
                    for r in group
                    if not all(
                        r[reader]["consistent"] for reader in ("owner", "background")
                    )
                ),
                None,
            )
            if bad:
                print("first_inconsistent=" + json.dumps(bad), flush=True)
    stream.synchronize()
    return {
        "diagnostic_completed": True,
        "device": device,
        "device_name": torch.npu.get_device_name(device),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "rows": rows,
        "all_intervals_consistent": all(
            r[reader]["consistent"] for r in rows for reader in ("owner", "background")
        ),
        "full_model_timing_validated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.device < 0 or not 1 <= args.repeats <= 20:
        parser.error("device must be nonnegative; repeats must be 1..20")
    if args.output is None:
        args.output = Path(tempfile.gettempdir()) / f"npu_events_{time.time_ns()}.json"
    if args.output.exists():
        parser.error("output already exists; preserve it and choose a new path")
    if not args.worker:
        print(f"Result: {args.output}", flush=True)
        try:
            return subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--device",
                    str(args.device),
                    "--repeats",
                    str(args.repeats),
                    "--output",
                    str(args.output),
                ],
                timeout=90,
            ).returncode
        except subprocess.TimeoutExpired:
            print("Probe timed out after 90 seconds; worker stopped", file=sys.stderr)
            return 1
    try:
        result = probe(args.device, args.repeats)
    except Exception:
        result = {"diagnostic_completed": False, "error": traceback.format_exc()}
        print(result["error"], file=sys.stderr)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
    return 0 if result["diagnostic_completed"] else 1


if __name__ == "__main__":
    sys.exit(main())
