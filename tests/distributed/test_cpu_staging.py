# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone CPU/Gloo contract tests; no vLLM model or accelerator needed.

Run: python tests/distributed/test_cpu_staging.py
"""

import gc
import importlib.util
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "vllm/distributed/device_communicators/cpu_staging.py"
)


def load_transport():
    spec = importlib.util.spec_from_file_location("cpu_staging", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CpuStagingTransport


def _roundtrip(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    transport = load_transport()(dist.group.WORLD)
    try:
        # A view must arrive as logical values, not the sender's storage order.
        source = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).T
        if rank == 0:
            work = transport.isend(source, dst=1, tag=10)
            source.fill_(-100)  # isend owns an independent CPU snapshot.
            del work
            gc.collect()  # Dropping a Work must not free an in-flight buffer.
        else:
            backing = torch.full((2, 3), -1.0)
            destination = backing.T
            work = transport.irecv(destination, src=0, tag=10)
            assert work.wait(timedelta(seconds=5)) is True
            assert work.wait() is True
            assert work.is_completed()
            assert torch.equal(
                backing, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            )

        # Bit transport preserves FP16/BF16, booleans, integer tokens and scalars.
        for tag, dtype in enumerate(
            [torch.float16, torch.bfloat16, torch.bool, torch.int64], start=20
        ):
            value = torch.tensor([0, 1, 1, 0], dtype=dtype)
            if rank == 1:
                assert transport.isend(value, dst=0, tag=tag).wait()
            else:
                result = torch.empty_like(value)
                assert transport.irecv(result, src=1, tag=tag).wait()
                assert torch.equal(result, value)
        for tag, shape in [(30, ()), (31, (0, 3))]:
            value = torch.full(shape, 7.0)
            if rank == 0:
                assert transport.isend(value, dst=1, tag=tag).wait()
            else:
                result = torch.empty_like(value)
                assert transport.irecv(result, src=0, tag=tag).wait()
                assert torch.equal(result, value)

        # Caller timeout must not cancel a receive or mark it complete.
        if rank == 1:
            with torch.inference_mode():
                result = torch.empty(2)
            work = transport.irecv(result, src=0, tag=40)
            assert not work.is_completed()
            try:
                work.wait(timedelta(milliseconds=10))
            except TimeoutError:
                pass
            else:
                raise AssertionError("unmatched receive completed")
            assert not work.is_completed()
            dist.barrier()
            assert work.wait(timedelta(seconds=5))
            assert torch.equal(result, torch.tensor([11.0, 12.0]))
        else:
            dist.barrier()
            assert transport.isend(torch.tensor([11.0, 12.0]), dst=1, tag=40).wait()

        # Tags must separate outstanding messages, including full duplex traffic.
        values = [torch.tensor([101]), torch.tensor([202])]
        received = [torch.empty(1, dtype=torch.int64) for _ in values]
        reads = [
            transport.irecv(t, src=1 - rank, tag=51 - i) for i, t in enumerate(received)
        ]
        writes = [
            transport.isend(t, dst=1 - rank, tag=50 + i) for i, t in enumerate(values)
        ]
        for work in reads + writes:
            assert work.wait(timedelta(seconds=5))
        assert received[0].item() == 202
        assert received[1].item() == 101

        for peer in [-1, rank, 99]:
            try:
                transport.isend(torch.ones(1), dst=peer)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid peer accepted")
        transport.close()
        try:
            transport.isend(torch.ones(1), dst=1 - rank)
        except RuntimeError:
            pass
        else:
            raise AssertionError("closed transport accepted work")
    finally:
        transport.close()
        dist.destroy_process_group()


def _peer_failure(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=2),
    )
    transport = load_transport()(dist.group.WORLD)
    if rank == 0:
        work = transport.irecv(torch.empty(4), src=1)
        dist.barrier()
        try:
            work.wait(timedelta(seconds=8))
        except RuntimeError:
            assert work.is_completed()
        else:
            raise AssertionError("peer failure was silently accepted")
        # close must also report the transfer error while draining the transport.
        try:
            transport.close()
        except RuntimeError:
            pass
        else:
            raise AssertionError("close silently accepted a failed transfer")
    else:
        dist.barrier()
    dist.destroy_process_group()


def _subgroup(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=20),
    )
    group = dist.new_group([0, 2], backend="gloo")
    if rank != 1:
        transport = load_transport()(group)
        if rank == 0:
            transport.send(torch.tensor([13, 17]), dst=2)
        else:
            result = torch.empty(2, dtype=torch.int64)
            transport.recv(result, src=0)
            assert torch.equal(result, torch.tensor([13, 17]))
        try:
            transport.isend(torch.ones(1), dst=1)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted a peer outside the subgroup")
        transport.close()
        dist.destroy_process_group(group)
    dist.barrier()
    dist.destroy_process_group()


class TestCpuStaging(unittest.TestCase):
    def test_real_gloo_roundtrip_and_async_lifetime(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_roundtrip, args=(f"file://{directory}/store",), nprocs=2)

    def test_peer_failure_reaches_waiter(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_peer_failure, args=(f"file://{directory}/store",), nprocs=2)

    def test_subgroup_uses_global_peer_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_subgroup, args=(f"file://{directory}/store",), nprocs=3)


if __name__ == "__main__":
    unittest.main()
