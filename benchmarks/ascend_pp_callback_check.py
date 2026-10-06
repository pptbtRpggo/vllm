# SPDX-License-Identifier: Apache-2.0
"""Check native HCCL point-to-point dependencies without per-step device sync.

Run on two free Ascend devices. Only final validation synchronizes the device;
callbacks perform CPU timing/sleep and never access tensors or device APIs.
"""

import argparse
import json
import socket
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp


def worker(rank, devices, port, root):
    import torch_npu  # noqa: F401

    from vllm.distributed.tp_stream_delay import TPStreamDelay

    torch.npu.set_device(devices[rank])
    torch.distributed.init_process_group(
        "hccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )
    callbacks = TPStreamDelay()
    rows, received = [], []
    try:
        for batch in range(3):
            row = {"batch": batch, "rank": rank}
            rows.append(row)

            def begin(r=row):
                r["start_ns"] = time.perf_counter_ns()

            callbacks.enqueue(begin)
            if rank == 0:
                tensor = torch.arange(1024, device="npu", dtype=torch.float32) + batch

                def compute_delay(r=row):
                    time.sleep(0.02)
                    r["compute_delay_end_ns"] = time.perf_counter_ns()

                callbacks.enqueue(compute_delay)
                torch.distributed.send(tensor, dst=1)
            else:
                tensor = torch.empty(1024, device="npu", dtype=torch.float32)
                torch.distributed.recv(tensor, src=0)

            def transfer_end(r=row):
                r["transfer_end_ns"] = time.perf_counter_ns()
                time.sleep(0.01)
                r["network_delay_end_ns"] = time.perf_counter_ns()

            callbacks.enqueue(transfer_end)
            if rank == 1:
                # This clone must execute after the receive and callback delay.
                received.append(tensor.clone())

            def after_work(r=row):
                r["next_work_ns"] = time.perf_counter_ns()

            callbacks.enqueue(after_work)

        torch.npu.synchronize()  # Final assertion/shutdown only.
        callbacks.close()
        for batch, tensor in enumerate(received):
            expected = torch.arange(1024, dtype=torch.float32) + batch
            torch.testing.assert_close(tensor.cpu(), expected)
        for row in rows:
            assert row["next_work_ns"] >= row["network_delay_end_ns"]
            if rank == 0:
                assert row["transfer_end_ns"] >= row["compute_delay_end_ns"]
        for previous, current in zip(rows, rows[1:]):
            assert current["start_ns"] >= previous["next_work_ns"]
        (Path(root) / f"rank{rank}.json").write_text(json.dumps(rows, indent=2))
    finally:
        torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    devices = [int(value) for value in args.devices.split(",")]
    if len(devices) != 2 or devices[0] == devices[1]:
        parser.error("provide two distinct free NPU devices")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(devices, port, str(root)), nprocs=2, join=True)
    print("PASS: three transfers, correct tensors, ordered callbacks, no per-step sync")


if __name__ == "__main__":
    main()
