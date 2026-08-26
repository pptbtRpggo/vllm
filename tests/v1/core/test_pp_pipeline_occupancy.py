# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drive the real V1 Scheduler with mocked GPU output and record PP occupancy.

Mimics EngineCore.step_with_batch_queue: keep up to ``pp_size`` in-flight
SchedulerOutputs, then apply a fake ModelRunnerOutput in FIFO order.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import pytest
except ImportError:
    pytest = None  # type: ignore[assignment]

from vllm.v1.outputs import ModelRunnerOutput

try:
    from .utils import create_requests, create_scheduler
except ImportError:
    from tests.v1.core.utils import create_requests, create_scheduler

if pytest is not None:
    pytestmark = pytest.mark.cpu_test

TRACE_PATH = (
    Path(__file__).resolve().parents[3] / "figures" / "vllm-pp-occupancy-trace.json"
)

_DUMMY_TOKEN = 1


@dataclass
class BatchItem:
    req_id: str
    num_tokens: int
    kind: str
    computed_after: int
    prompt_len: int


@dataclass
class IssuedBatch:
    batch_id: int
    pp0_tick: int
    pp1_tick: int
    items: list[BatchItem]


def _describe_items(scheduler, scheduler_output) -> list[BatchItem]:
    items: list[BatchItem] = []
    for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
        request = scheduler.requests[req_id]
        start = request.num_computed_tokens - num_tokens
        kind = "prefill" if start < request.num_prompt_tokens else "decode"
        items.append(
            BatchItem(
                req_id=req_id,
                num_tokens=num_tokens,
                kind=kind,
                computed_after=request.num_computed_tokens,
                prompt_len=request.num_prompt_tokens,
            )
        )
    return items


def _fake_model_output(
    scheduler_output, sample_on_complete: dict[str, bool]
) -> ModelRunnerOutput:
    req_ids = [
        req_id
        for req_id in scheduler_output.num_scheduled_tokens
        if req_id in sample_on_complete
    ]
    sampled = [
        [_DUMMY_TOKEN] if sample_on_complete[req_id] else [] for req_id in req_ids
    ]
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def simulate_pp_occupancy(
    *,
    pp_size: int,
    max_num_batched_tokens: int,
    async_scheduling: bool,
    arrivals: list[tuple[int, str, int, int]],
) -> dict:
    """Run a discrete-event PP occupancy simulation.

    Args:
        pp_size: Pipeline parallel size / in-flight batch depth.
        max_num_batched_tokens: Scheduler token budget per step.
        async_scheduling: Whether to use AsyncScheduler.
        arrivals: (after_n_issued_batches, req_id, prompt_len, max_tokens).
            ``after_n_issued_batches=0`` means present before the first schedule.

    Returns:
        JSON-serializable occupancy trace.
    """
    scheduler = create_scheduler(
        max_num_seqs=8,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=128,
        enable_chunked_prefill=True,
        pipeline_parallel_size=pp_size,
        block_size=16,
        num_blocks=10000,
        async_scheduling=async_scheduling,
        use_v2_model_runner=False,
    )

    pending_arrivals = deque(sorted(arrivals, key=lambda row: row[0]))

    def admit_ready(issued_count: int) -> list[str]:
        admitted: list[str] = []
        while pending_arrivals and pending_arrivals[0][0] <= issued_count:
            _, req_id, prompt_len, max_tokens = pending_arrivals.popleft()
            (request,) = create_requests(
                num_requests=1,
                num_tokens=prompt_len,
                max_tokens=max_tokens,
                ignore_eos=True,
                req_ids=[req_id],
            )
            scheduler.add_request(request)
            admitted.append(req_id)
        return admitted

    admit_ready(0)

    pending: deque[tuple[IssuedBatch, object, dict[str, bool]]] = deque()
    issued: list[IssuedBatch] = []
    next_pp0_tick = 0
    max_steps = 64

    for _ in range(max_steps):
        while len(pending) < pp_size:
            if not scheduler.has_unfinished_requests() and not pending_arrivals:
                break
            output = scheduler.schedule()
            if output.total_num_scheduled_tokens == 0:
                break
            sample_on_complete = {
                req_id: not scheduler.requests[req_id].is_prefill_chunk
                for req_id in output.num_scheduled_tokens
                if req_id in scheduler.requests
            }
            batch = IssuedBatch(
                batch_id=len(issued),
                pp0_tick=next_pp0_tick,
                pp1_tick=next_pp0_tick + 1 if pp_size > 1 else next_pp0_tick,
                items=_describe_items(scheduler, output),
            )
            issued.append(batch)
            pending.append((batch, output, sample_on_complete))
            next_pp0_tick += 1
            admit_ready(len(issued))

        if not pending:
            break

        batch, scheduler_output, sample_on_complete = pending.popleft()
        scheduler.update_from_output(
            scheduler_output,
            _fake_model_output(scheduler_output, sample_on_complete),
        )
        complete_tick = batch.pp0_tick + pp_size
        next_pp0_tick = max(next_pp0_tick, complete_tick)
        admit_ready(len(issued))

    last_tick = 0
    if issued:
        last_tick = max(batch.pp0_tick + pp_size for batch in issued)
    devices = [f"GPU{i} PP{i}" for i in range(pp_size)]
    occupied: dict[tuple[int, int], IssuedBatch] = {}
    for batch in issued:
        for stage in range(pp_size):
            occupied[(batch.pp0_tick + stage, stage)] = batch

    cells = []
    for tick in range(last_tick):
        for device in range(pp_size):
            batch = occupied.get((tick, device))
            cells.append(
                {
                    "tick": tick,
                    "device": device,
                    "batch_id": None if batch is None else batch.batch_id,
                    "label": "" if batch is None else _cell_label(batch),
                    "bubble": batch is None,
                }
            )

    return {
        "pp_size": pp_size,
        "max_num_batched_tokens": max_num_batched_tokens,
        "async_scheduling": async_scheduling,
        "arrivals": [
            {
                "after_n_issued_batches": after,
                "req_id": req_id,
                "prompt_len": prompt_len,
                "max_tokens": max_tokens,
            }
            for after, req_id, prompt_len, max_tokens in arrivals
        ],
        "batches": [asdict(batch) for batch in issued],
        "last_tick": last_tick,
        "devices": devices,
        "cells": cells,
        "num_bubbles": sum(1 for cell in cells if cell["bubble"]),
        "num_busy": sum(1 for cell in cells if not cell["bubble"]),
    }


def _cell_label(batch: IssuedBatch) -> str:
    parts = []
    for item in batch.items:
        tag = "P" if item.kind == "prefill" else "D"
        parts.append(f"{item.req_id} {tag}{item.num_tokens}")
    return f"B{batch.batch_id} " + "+".join(parts)


def _text_grid(trace: dict) -> str:
    last_tick = trace["last_tick"]
    pp_size = trace["pp_size"]
    by_pos = {(c["tick"], c["device"]): c for c in trace["cells"]}
    header = "      " + " | ".join(f"t{t:<14}" for t in range(last_tick))
    rows = [header]
    for device, name in enumerate(trace["devices"]):
        cells = []
        for tick in range(last_tick):
            cell = by_pos[(tick, device)]
            text = "空" if cell["bubble"] else cell["label"]
            cells.append(f"{text:<16}")
        rows.append(f"{name:<5} " + " | ".join(cells))
    return "\n".join(rows)


def run_occupancy_trace() -> dict:
    """A (24-token prompt, 2 decode) + B (8, 2); C joins after the first batch."""
    arrivals = [
        (0, "A", 24, 2),
        (0, "B", 8, 2),
        (1, "C", 8, 2),
    ]
    kwargs = dict(
        max_num_batched_tokens=16,
        async_scheduling=False,
    )
    mixed = simulate_pp_occupancy(pp_size=2, arrivals=arrivals, **kwargs)
    solo = simulate_pp_occupancy(
        pp_size=2, arrivals=[(0, "A", 24, 2)], **kwargs
    )
    mixed4 = simulate_pp_occupancy(pp_size=4, arrivals=arrivals, **kwargs)
    solo4 = simulate_pp_occupancy(
        pp_size=4, arrivals=[(0, "A", 24, 2)], **kwargs
    )
    mixed4_again = simulate_pp_occupancy(pp_size=4, arrivals=arrivals, **kwargs)
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mixed": mixed,
        "solo": solo,
        "mixed_pp4": mixed4,
        "solo_pp4": solo4,
        "pp4_repeat_identical": mixed4 == mixed4_again,
    }
    TRACE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def test_pp_occupancy_mixed_requests_sync_scheduler():
    payload = run_occupancy_trace()
    mixed = payload["mixed"]
    solo = payload["solo"]

    print("\n=== mixed PP=2 ===")
    print(_text_grid(mixed))
    print("\n=== solo PP=2 ===")
    print(_text_grid(solo))
    print("\n=== mixed PP=4 ===")
    print(_text_grid(payload["mixed_pp4"]))
    print("\n=== solo PP=4 ===")
    print(_text_grid(payload["solo_pp4"]))
    print(f"\nPP=4 second run identical: {payload['pp4_repeat_identical']}")
    print(f"\ntrace written to {TRACE_PATH}")

    assert mixed["batches"], "scheduler issued no batches"
    first = mixed["batches"][0]
    assert first["items"][0]["req_id"] == "A"
    assert first["items"][0]["kind"] == "prefill"
    assert first["items"][0]["num_tokens"] == 16

    # Running-first: A's leftover prefill is scheduled before waiting B.
    second_ids = [item["req_id"] for item in mixed["batches"][1]["items"]]
    assert "A" in second_ids

    # GPU1 is idle on the first tick (pipeline fill).
    assert any(
        c["tick"] == 0 and c["device"] == 1 and c["bubble"] for c in mixed["cells"]
    )

    # Mixed serving keeps the GPUs busier over the run.
    assert mixed["num_busy"] > solo["num_busy"]
    mixed_rate = mixed["num_bubbles"] / max(len(mixed["cells"]), 1)
    solo_rate = solo["num_bubbles"] / max(len(solo["cells"]), 1)
    assert mixed_rate <= solo_rate

    mixed4 = payload["mixed_pp4"]
    assert mixed4["pp_size"] == 4
    assert mixed4["devices"] == ["GPU0 PP0", "GPU1 PP1", "GPU2 PP2", "GPU3 PP3"]
    assert any(
        c["tick"] == 0 and c["device"] == 3 and c["bubble"] for c in mixed4["cells"]
    )
    assert payload["pp4_repeat_identical"] is True


if __name__ == "__main__":
    test_pp_occupancy_mixed_requests_sync_scheduler()
    print("ok")
