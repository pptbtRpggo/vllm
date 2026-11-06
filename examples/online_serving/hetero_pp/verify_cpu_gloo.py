# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify real NPU TensorDict values and the CPU/Gloo payload boundary.

Run with torchrun: two nodes, two ranks per node, using the same rendezvous.
This diagnostic instruments Gloo calls only in its own process. It does not
modify the serving transport, installed packages, or a running model server.
"""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from vllm.distributed.hetero_pp.channel import TensorDictChannel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    if dist.get_world_size() != 4:
        raise ValueError("this diagnostic requires exactly four ranks")
    device = torch.device("npu", local_rank)
    channel = TensorDictChannel(dist.group.WORLD, device, timeout_s=60)
    wire_calls = []
    checks = []

    def audit(original, operation):
        def checked(tensor, *a, **kwargs):
            backend = str(dist.get_backend(kwargs["group"]))
            observation = {
                "operation": operation,
                "peer": kwargs.get("dst", kwargs.get("src")),
                "tag": kwargs.get("tag", 0),
                "device": tensor.device.type,
                "backend": backend,
                "dtype": str(tensor.dtype),
                "bytes": tensor.numel() * tensor.element_size(),
            }
            assert observation["device"] == "cpu", observation
            assert backend == "gloo", observation
            wire_calls.append(observation)
            return original(tensor, *a, **kwargs)

        return checked

    try:
        with (
            patch.object(dist, "isend", audit(dist.isend, "isend")),
            patch.object(dist, "irecv", audit(dist.irecv, "irecv")),
        ):
            for sequence, m in enumerate([1, 64, 256, 2048]):
                base = (
                    (torch.arange(m * 128).reshape(128, m) % 101)
                    .to(torch.float16)
                    .T
                )
                if rank == 0:
                    x = base.to(device)
                    values = {
                        "hidden_states": x,
                        "residual": x * 2,
                        "optional": None,
                        "empty": torch.empty((0, 3), device=device),
                        "ids": torch.arange(m, dtype=torch.int64, device=device),
                    }
                else:
                    values = channel.recv(rank - 1, sequence)
                values["hidden_states"] += rank
                values["residual"] += rank * 2
                want = base + sum(range(rank + 1))
                assert values["hidden_states"].device.type == "npu"
                assert torch.equal(values["hidden_states"].cpu(), want)
                assert torch.equal(values["residual"].cpu(), want * 2)
                assert values["optional"] is None
                assert values["empty"].shape == (0, 3)
                assert torch.equal(values["ids"].cpu(), torch.arange(m))
                if rank < 3:
                    works = channel.send(values, rank + 1, sequence)
                    # Test ownership: a send snapshots values before reuse.
                    values["hidden_states"].fill_(-99)
                    for work in works:
                        work.wait(timedelta(seconds=60))
                else:
                    works = channel.send(
                        {"token": torch.tensor(17 + sequence, device=device)},
                        0,
                        sequence + 100,
                    )
                    for work in works:
                        work.wait(timedelta(seconds=60))
                if rank == 0:
                    assert channel.recv(3, sequence + 100)["token"].item() == (
                        17 + sequence
                    )
                checks.append({"M": m, "all_values_equal": True})
        assert wire_calls, "no actual Gloo payload calls observed"
        channel.close()
        dist.barrier()
        result = {
            "rank": rank,
            "local_rank": local_rank,
            "backend": str(dist.get_backend()),
            "destination_device": "npu",
            "wire_calls": wire_calls,
            "checks": checks,
            "passed": True,
        }
        (args.output / f"cpu_gloo_rank{rank}.json").write_text(
            json.dumps(result, indent=2)
        )
    finally:
        channel.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
