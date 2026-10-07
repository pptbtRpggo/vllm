"""Estimate native PP communication cost from a no-delay serving trace.

The affine estimate is used only to subtract native cost when simulating a
slower target-total link. It is not a physical wire-bandwidth measurement.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

from vllm.distributed.pp_comm_trace import paired_transfer


def records(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def calibrate(trace_dir: Path, source_rank: int = 1, *, large_min_bytes=16 * 1024**2):
    if type(large_min_bytes) is not int or large_min_bytes < 1024**2:
        raise ValueError("large payload threshold must be at least 1 MiB")
    send = records(trace_dir / f"pp_stage_pp{source_rank}_tp0.jsonl")
    recv = records(trace_dir / f"pp_stage_pp{source_rank + 1}_tp0.jsonl")
    peers = {(row["trace_session"], row["step"]): row for row in recv}
    pairs = []
    for row in send:
        if row.get("is_warmup"):
            continue
        size = row.get("send_bytes")
        if type(size) is not int or size <= 0:
            raise ValueError("native trace payload must be a positive integer")
        peer = peers.get((row["trace_session"], row["step"]))
        if peer is None:
            continue
        status, values = paired_transfer(row, peer)
        if status == "paired":
            duration = values["send_overlap_ms"]
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("native transfer time must be positive and finite")
            pairs.append((size, duration))
    small = [time_ms for size, time_ms in pairs if size < 1024**2]
    large = [(size, time_ms) for size, time_ms in pairs if size >= large_min_bytes]
    if len(small) < 20 or len(large) < 3:
        raise ValueError(
            f"insufficient native pairs: {len(small)} under 1 MiB, "
            f"{len(large)} at least {large_min_bytes / 1024**2:g} MiB"
        )
    # Small serving transfers may include a different scheduling/metadata cost
    # from large prefill transfers. Subtracting their median from the large
    # mean can leave a near-zero denominator and an arbitrary bandwidth.
    # Fit the large-transfer observations together instead. The resulting
    # affine curve is an empirical large-payload cost, not physical bandwidth.
    sizes = [size for size, _ in large]
    times = [time_ms for _, time_ms in large]
    if len(set(sizes)) < 3 or max(sizes) < 2 * min(sizes):
        raise ValueError("native calibration needs three varied large payload sizes")
    slope, latency_ms = statistics.linear_regression(sizes, times)
    if not math.isfinite(slope) or not math.isfinite(latency_ms) or slope <= 0:
        raise ValueError("large-transfer time must increase with payload size")
    if latency_ms < 0:
        # Nonnegative affine fit: the optimum on the intercept=0 boundary.
        latency_ms = 0.0
        slope = sum(s * t for s, t in large) / sum(s * s for s, _ in large)
    residuals = [t - (latency_ms + slope * s) for s, t in large]
    variation = sum((t - statistics.mean(times)) ** 2 for t in times)
    r_squared = 1 - sum(r * r for r in residuals) / variation
    if not math.isfinite(r_squared) or r_squared < 0.8:
        raise ValueError("large-transfer observations do not support an affine fit")
    large_bytes = statistics.mean(size for size, _ in large)
    large_ms = statistics.mean(time_ms for _, time_ms in large)
    bandwidth_gbps = 8 / (slope * 1_000_000)
    return {
        "source_rank": source_rank,
        "small_samples": len(small),
        "large_samples": len(large),
        "large_mean_mib": large_bytes / 1024**2,
        "large_mean_ms": large_ms,
        "native_latency_ms": latency_ms,
        "native_bandwidth_gbps": bandwidth_gbps,
        "small_payload_median_ms": statistics.median(small),
        "fit_method": "nonnegative_affine_large_payloads",
        "fit_scope_min_bytes": large_min_bytes,
        "fit_r_squared": r_squared,
        "fit_max_residual_ms": max(map(abs, residuals)),
        "effective_large_gbps": 8 * large_bytes / (large_ms * 1_000_000),
        "trace_dir": str(trace_dir.resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-rank", type=int, default=1)
    parser.add_argument("--large-min-mib", type=int, default=16)
    args = parser.parse_args()
    result = calibrate(
        args.trace_dir, args.source_rank, large_min_bytes=args.large_min_mib * 1024**2
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
