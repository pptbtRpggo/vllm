# SPDX-License-Identifier: Apache-2.0
"""Validate current-execution proportional waits in eager and NPU Graph."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm.distributed.ascend_device_delay import AscendDeviceDelay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    runtime = AscendDeviceDelay()
    x = torch.ones((64, 1024), device="npu", dtype=torch.float16)
    weight = torch.eye(1024, device="npu", dtype=torch.float16)
    buffer = runtime.buffer()

    def work():
        runtime.begin_interval(buffer)
        result = x
        for _ in range(32):
            result = result @ weight
        runtime.end_interval(buffer, factor=3)
        return result

    for _ in range(3):
        work()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_output = work()
    rows = []
    for label in ("eager", "graph", "eager", "graph"):
        # Use different data on each execution to ensure graph input freshness.
        x.fill_(len(rows) + 1)
        if label == "graph":
            graph.replay()
            result = graph_output
        else:
            result = work()
        host = runtime.snapshot(buffer)
        complete = runtime.event()
        complete.synchronize()  # Benchmark boundary only.
        base, delay = runtime.durations(host)
        torch.testing.assert_close(
            result.cpu(), torch.full((64, 1024), len(rows) + 1, dtype=torch.float16)
        )
        assert base > 0 and 2.9 <= delay / base <= 3.5, (label, base, delay)
        rows.append(dict(mode=label, base_ms=base, wait_ms=delay, ratio=delay / base))

    def ordered_wait():
        runtime.begin_interval(buffer)
        runtime.wait_ms(20)
        runtime.end_interval(buffer, factor=1)

    # A known device interval checks queue ordering independently of matmul
    # submission latency. Repeat after capture to check current replay timing.
    for _ in range(3):
        ordered_wait()
    torch.npu.synchronize()
    wait_graph = torch.npu.NPUGraph()
    with torch.npu.graph(wait_graph):
        ordered_wait()
    ordered_rows = []
    for label in ("eager", "graph", "eager", "graph"):
        (ordered_wait if label == "eager" else wait_graph.replay)()
        host = runtime.snapshot(buffer)
        runtime.event().synchronize()  # Benchmark boundary only.
        base, delay = runtime.durations(host)
        assert 19.5 <= base <= 24 and 0.99 <= delay / base <= 1.05, (
            label,
            base,
            delay,
        )
        ordered_rows.append(dict(mode=label, base_ms=base, wait_ms=delay))
    runtime.calibrate_clock()
    start = runtime.event()
    runtime.wait_ms(20)
    end = runtime.event()
    end.synchronize()  # No other work pending at this test boundary.
    event_read = runtime.elapsed_time(start, end)
    reference = start.elapsed_time(end)
    assert abs(event_read - reference) < 0.01 and 19.5 <= event_read <= 24
    print(json.dumps(dict(rows=rows, ordered_rows=ordered_rows)), flush=True)
    Path(args.output).write_text(
        json.dumps(
            dict(
                rows=rows,
                ordered_rows=ordered_rows,
                clock_error_ns=runtime.clock_error_ns,
                event_elapsed_ms=event_read,
                event_reference_ms=reference,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
