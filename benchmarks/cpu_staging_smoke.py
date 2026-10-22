# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Functional CPU-staged P2P checks on one or two hosts; no model is loaded.

Local: python benchmarks/cpu_staging_smoke.py --device npu --spawn
Two hosts: use torchrun, one rank/NPU per host, without --spawn.
The selected CPU interface must be reachable between hosts. No HCCL group is
created. This is a correctness check, not a model latency benchmark.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import socket
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "vllm/distributed/device_communicators/cpu_staging.py"
)


def _run(rank, args, rendezvous=None):
    torch.set_num_threads(1)
    index = rank if args.spawn else int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cpu")
    device_name = platform.processor()
    if args.device == "npu":
        import torch_npu

        torch.npu.set_device(index)
        device = torch.device("npu", index)
        device_name = torch.npu.get_device_name(index)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous or "env://",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    spec = importlib.util.spec_from_file_location("cpu_staging", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    transport = module.CpuStagingTransport(dist.group.WORLD)
    rows = []
    try:
        with torch.inference_mode():
            # FP16 activations, BF16, scalar tokens, empty and strided views.
            cases = [
                ("prefill", (128, 512), torch.float16),
                ("decode", (1, 512), torch.float16),
                ("token", (), torch.int64),
                ("empty", (0, 16), torch.float16),
                ("large", (1024, 1024), torch.float16),
            ]
            if args.bf16:
                cases.append(("bf16", (32, 64), torch.bfloat16))
            for step in range(args.iterations):
                for case, shape, dtype in cases:
                    count = 1
                    for dimension in shape:
                        count *= dimension
                    sequence = torch.arange(count)
                    local = ((sequence + step + rank * 7) % 251).to(dtype)
                    local = local.reshape(shape).to(device)
                    expected = ((sequence + step + (1 - rank) * 7) % 251).to(dtype)
                    expected = expected.reshape(shape)
                    destination = torch.empty_like(local)
                    if local.ndim == 2 and count:
                        local = local.T
                        destination = destination.T
                        expected = expected.T.contiguous()
                    tag = 100 + step * 10 + cases.index((case, shape, dtype))
                    received = transport.irecv(destination, src=1 - rank, tag=tag)
                    sent = transport.isend(local, dst=1 - rank, tag=tag)
                    local.fill_(-1)
                    assert received.wait(timedelta(seconds=30))
                    assert sent.wait(timedelta(seconds=30))
                    assert torch.equal(destination.cpu(), expected), (step, case)
                    rows.append(
                        {
                            "case": case,
                            "step": step,
                            "correct": True,
                            "numel": count,
                            "dtype": str(dtype),
                        }
                    )

            if args.device == "npu":
                # Exercise NPU format conversion and a nondefault posting stream.
                stream = torch.npu.Stream(device=device)
                with torch.npu.stream(stream):
                    sequence = torch.arange(1024).reshape(32, 32)
                    expected = (sequence + (1 - rank) * 7).half()
                    local = torch_npu.npu_format_cast(
                        (sequence + rank * 7).half().to(device), 29
                    )
                    destination = torch_npu.npu_format_cast(
                        torch.zeros((32, 32), device=device).half(), 29
                    )
                    received = transport.irecv(destination, src=1 - rank, tag=9000)
                    sent = transport.isend(local, dst=1 - rank, tag=9000)
                assert received.wait(timedelta(seconds=30))
                assert sent.wait(timedelta(seconds=30))
                assert torch.equal(destination.cpu(), expected)
                rows.append({"case": "format29_custom_stream", "correct": True})

        transport.close(timedelta(seconds=30))
        report = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "rank": rank,
            "device": str(device),
            "device_name": device_name,
            "torch": torch.__version__,
            "backend": str(dist.get_backend()),
            "transport_source": str(SOURCE),
            "transport_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "created_hccl_group": False,
            "checks_passed": len(rows),
            "checks": rows,
            "scope": "P2P transport only; no model or heterogeneous-SoC validation",
        }
        if args.device == "npu":
            report["torch_npu"] = torch_npu.__version__
        args.output.mkdir(parents=True, exist_ok=True)
        target = args.output / f"rank{rank}.json"
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "rank": rank,
                    "checks_passed": len(rows),
                    "device_name": device_name,
                    "report": str(target),
                }
            ),
            flush=True,
        )
        dist.barrier()
    finally:
        transport.close(timedelta(seconds=60))
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    parser.add_argument("--spawn", action="store_true", help="start two local ranks")
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="also check BF16; enable only on supporting devices",
    )
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 100:
        parser.error("iterations must be in [1, 100]")
    if args.spawn:
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_run, args=(args, f"file://{directory}/store"), nprocs=2)
    else:
        if os.environ.get("WORLD_SIZE") != "2":
            parser.error("use torchrun with exactly two ranks, or use --spawn")
        _run(int(os.environ["RANK"]), args)


if __name__ == "__main__":
    main()
