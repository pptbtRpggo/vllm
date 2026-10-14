# SPDX-License-Identifier: Apache-2.0
"""Check that current-stream NPU event completion covers HCCL all_reduce."""

import json
import os
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    expected = dist.get_world_size()
    tensor = torch.ones(16_000_000, dtype=torch.float16, device=f"npu:{rank}")
    event = torch.npu.Event(enable_timing=False)
    rows = []
    for index in range(15):
        tensor.fill_(1)
        torch.npu.synchronize()
        started = time.perf_counter()
        dist.all_reduce(tensor)
        returned = time.perf_counter()
        event.record()
        event.synchronize()
        event_done = time.perf_counter()
        torch.npu.synchronize()
        device_done = time.perf_counter()
        if index >= 5:
            rows.append(
                {
                    "all_reduce_call_ms": (returned - started) * 1000,
                    "event_wait_ms": (event_done - returned) * 1000,
                    "device_after_event_ms": (device_done - event_done) * 1000,
                }
            )
        if tensor[0].item() != expected:
            raise AssertionError("wrong all_reduce output")
    print(json.dumps({"rank": rank, "rows": rows}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
