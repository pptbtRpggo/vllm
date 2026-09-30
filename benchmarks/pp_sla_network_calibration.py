"""Estimate native PP communication cost from a no-delay serving trace.

The affine estimate is used only to subtract native cost when simulating a
slower target-total link. It is not a physical wire-bandwidth measurement.
"""

import argparse
import json
import statistics
from pathlib import Path

from vllm.distributed.pp_comm_trace import paired_transfer


def records(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def calibrate(trace_dir: Path, source_rank: int = 1):
    send = records(trace_dir / f"pp_stage_pp{source_rank}_tp0.jsonl")
    recv = records(trace_dir / f"pp_stage_pp{source_rank + 1}_tp0.jsonl")
    peers = {(row["trace_session"], row["step"]): row for row in recv}
    pairs = []
    for row in send:
        if row.get("is_warmup"):
            continue
        peer = peers.get((row["trace_session"], row["step"]))
        if peer is None:
            continue
        status, values = paired_transfer(row, peer)
        if status == "paired":
            pairs.append((row["send_bytes"], values["send_overlap_ms"]))
    small = [time_ms for size, time_ms in pairs if size < 1024**2]
    large = [(size, time_ms) for size, time_ms in pairs if size >= 16 * 1024**2]
    if len(small) < 20 or len(large) < 3:
        raise ValueError(
            f"insufficient native pairs: {len(small)} under 1 MiB, "
            f"{len(large)} at least 16 MiB"
        )
    latency_ms = statistics.median(small)
    large_bytes = statistics.mean(size for size, _ in large)
    large_ms = statistics.mean(time_ms for _, time_ms in large)
    if large_ms <= latency_ms:
        raise ValueError("large-transfer time must exceed the small-transfer baseline")
    bandwidth_gbps = 8 * large_bytes / ((large_ms - latency_ms) * 1_000_000)
    return {
        "source_rank": source_rank,
        "small_samples": len(small),
        "large_samples": len(large),
        "large_mean_mib": large_bytes / 1024**2,
        "large_mean_ms": large_ms,
        "native_latency_ms": latency_ms,
        "native_bandwidth_gbps": bandwidth_gbps,
        "effective_large_gbps": 8 * large_bytes / (large_ms * 1_000_000),
        "trace_dir": str(trace_dir.resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = calibrate(args.trace_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
