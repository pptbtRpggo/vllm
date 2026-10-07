# SPDX-License-Identifier: Apache-2.0
"""Measure the native eager collectives used by NPUCommunicator, without a model.

Run with torchrun, two or four ranks. Synchronization occurs only at measurement
boundaries; these checks are not inserted into formal serving.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from vllm_ascend.distributed.communicator import NPUCommunicator

from vllm.distributed.tp_hetero import TPHeteroConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    size = int(os.environ["WORLD_SIZE"])
    if size not in (2, 4):
        raise ValueError("calibration requires two or four ranks on one host")
    torch.npu.set_device(rank)
    dist.init_process_group("gloo")
    group = dist.new_group(backend="hccl")
    comm = NPUCommunicator(dist.group.WORLD, device_group=group, unique_name="tp")
    cut = TPHeteroConfig((1,) * size, size // 2, 100)
    rows = []
    # Cover small decode payloads and large prefill payloads. Counts are divisible
    # by TP size for reduce_scatter; the collective consumes contiguous FP16.
    payloads = (
        16 * 1024,
        512 * 1024,
        4 * 1024**2,
        8 * 1024**2,
        16 * 1024**2,
        32 * 1024**2,
    )
    for op in ("all_reduce", "all_gather", "reduce_scatter"):
        method = getattr(comm, op)
        for nbytes in payloads:
            value = torch.empty(nbytes // 2, device="npu", dtype=torch.float16)
            for repeat in range(5):
                value.fill_(rank + 1)
                torch.npu.synchronize()
                dist.barrier()
                start, end = (
                    torch.npu.Event(enable_timing=True),
                    torch.npu.Event(enable_timing=True),
                )
                start.record()
                result = method(value)
                end.record()
                end.synchronize()
                elapsed = start.elapsed_time(end)
                if repeat < 2:
                    continue
                if op == "all_gather":
                    actual = result.reshape(size, -1)[:, 0].cpu()
                    expected = torch.arange(1, size + 1, dtype=torch.float16)
                else:
                    actual = result[[0, result.numel() - 1]].cpu()
                    expected = torch.full(
                        (2,), size * (size + 1) / 2, dtype=torch.float16
                    )
                torch.testing.assert_close(actual, expected)
                rows.append(
                    dict(
                        op=op,
                        input_bytes=nbytes,
                        cut_bytes=cut.cross_bytes(op, nbytes, size),
                        repeat=repeat - 2,
                        elapsed_ms=elapsed,
                        rank=rank,
                    )
                )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(json.dumps(rows, indent=2) + "\n")
    all_rows = [None] * size
    dist.all_gather_object(all_rows, rows)
    if rank == 0:
        measurements = []
        for index, row in enumerate(rows):
            # All participants were ready before the CPU barrier. Use the shortest
            # rank interval as a proxy with less early-arrival waiting.
            measurements.append(
                dict(row, elapsed_ms=min(r[index]["elapsed_ms"] for r in all_rows))
            )
        curves = {}
        for op in ("all_reduce", "all_gather", "reduce_scatter"):
            points = [
                r
                for r in measurements
                if r["op"] == op and r["input_bytes"] >= 4 * 1024**2
            ]
            x = np.array([r["cut_bytes"] / 1024**2 for r in points])
            y = np.array([r["elapsed_ms"] for r in points])
            slope, intercept = np.polyfit(x, y, 1)
            if intercept < 0:
                intercept = 0.0
                slope = float(x @ y / (x @ x))
            if not np.isfinite(slope) or slope <= 0:
                raise ValueError(f"invalid native collective fit: {op}")
            curves[op] = dict(
                bandwidth_gbps=8 * 1024**2 / (slope * 1_000_000),
                latency_ms=float(intercept),
            )
        report = dict(
            tp_size=size,
            cross_group_size=size // 2,
            dtype="float16",
            native_collectives=curves,
            measurements=measurements,
            note="Logical two-group cut bytes, not physical HCCL wire bandwidth",
        )
        (args.output / "native_calibration.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print("TP_NATIVE_CALIBRATION", json.dumps(curves), flush=True)
    dist.barrier()
    dist.destroy_process_group(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
