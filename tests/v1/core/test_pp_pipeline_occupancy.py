# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trace PP request dependencies using the real V1 Scheduler.

The Scheduler decides every non-empty batch. A deterministic runner completes
those batches in FIFO order and returns unique sampled tokens. The PP grid is
an explanatory unit-time model, not a GPU performance model.
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

from vllm.v1.core.sched.output import SchedulerOutput
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

PARTIAL_PREFILL = "partial_prefill"
FINAL_PREFILL = "final_prefill"
DECODE = "decode"
STAGE_ORDER = "stage_order"
SAMPLE_TOKEN = "sample_token"


@dataclass
class BatchItem:
    req_id: str
    num_tokens: int
    phase: str
    token_start: int
    token_end: int
    prompt_len: int
    decode_step: int | None
    consumed_token_id: int | None
    dependency_kind: str | None
    depends_on_batch_id: int | None
    emitted_token_id: int | None = None


@dataclass
class IssuedBatch:
    batch_id: int
    issue_tick: int
    completion_tick: int
    items: list[BatchItem]


@dataclass
class RunnerRequestState:
    sequence_length: int
    num_completed_tokens: int = 0
    last_emitted_token_id: int | None = None


class MockPipelineRunner:
    """Complete SchedulerOutputs from an independent runner-side view."""

    def __init__(self) -> None:
        self.requests: dict[str, RunnerRequestState] = {}
        self.next_token_id = 1000

    def add_request(self, req_id: str, prompt_len: int) -> None:
        assert req_id not in self.requests
        self.requests[req_id] = RunnerRequestState(sequence_length=prompt_len)

    def complete(
        self, batch: IssuedBatch, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput:
        items_by_req_id = {item.req_id: item for item in batch.items}
        req_ids = list(scheduler_output.num_scheduled_tokens)
        sampled_token_ids: list[list[int]] = []

        for req_id in req_ids:
            item = items_by_req_id[req_id]
            state = self.requests[req_id]
            assert item.token_start == state.num_completed_tokens
            assert item.token_end <= state.sequence_length
            if item.phase == DECODE:
                assert item.consumed_token_id == state.last_emitted_token_id
            state.num_completed_tokens = item.token_end

            if item.phase == PARTIAL_PREFILL:
                sampled_token_ids.append([])
                continue

            token_id = self.next_token_id
            self.next_token_id += 1
            item.emitted_token_id = token_id
            state.last_emitted_token_id = token_id
            state.sequence_length += 1
            sampled_token_ids.append([token_id])

        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled_token_ids,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )


def _snapshot_items(
    scheduler,
    scheduler_output: SchedulerOutput,
    batch_id: int,
    last_compute_batch: dict[str, int],
    last_sample_batch: dict[str, int],
) -> list[BatchItem]:
    items: list[BatchItem] = []
    for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
        request = scheduler.requests[req_id]
        token_end = request.num_computed_tokens
        token_start = token_end - num_tokens
        prompt_len = request.num_prompt_tokens

        if token_start < prompt_len:
            phase = (
                PARTIAL_PREFILL if token_end < prompt_len else FINAL_PREFILL
            )
            dependency_kind = (
                STAGE_ORDER if req_id in last_compute_batch else None
            )
            depends_on_batch_id = last_compute_batch.get(req_id)
            decode_step = None
            consumed_token_id = None
        else:
            phase = DECODE
            dependency_kind = SAMPLE_TOKEN
            depends_on_batch_id = last_sample_batch.get(req_id)
            assert depends_on_batch_id is not None
            decode_step = token_start - prompt_len + 1
            consumed_token_id = request.all_token_ids[token_start]

        items.append(
            BatchItem(
                req_id=req_id,
                num_tokens=num_tokens,
                phase=phase,
                token_start=token_start,
                token_end=token_end,
                prompt_len=prompt_len,
                decode_step=decode_step,
                consumed_token_id=consumed_token_id,
                dependency_kind=dependency_kind,
                depends_on_batch_id=depends_on_batch_id,
            )
        )
        if phase != DECODE:
            last_compute_batch[req_id] = batch_id

    return items


def _build_dependencies(batches: list[IssuedBatch]) -> list[dict]:
    batches_by_id = {batch.batch_id: batch for batch in batches}
    dependencies = []
    for target_batch in batches:
        for target_item in target_batch.items:
            source_batch_id = target_item.depends_on_batch_id
            if source_batch_id is None:
                continue
            source_batch = batches_by_id[source_batch_id]
            source_item = next(
                item
                for item in source_batch.items
                if item.req_id == target_item.req_id
            )
            dependencies.append(
                {
                    "req_id": target_item.req_id,
                    "kind": target_item.dependency_kind,
                    "from_batch_id": source_batch_id,
                    "to_batch_id": target_batch.batch_id,
                    "token_id": (
                        source_item.emitted_token_id
                        if target_item.dependency_kind == SAMPLE_TOKEN
                        else None
                    ),
                }
            )
    return dependencies


def simulate_pp_schedule_trace(
    *,
    pp_size: int,
    max_num_batched_tokens: int,
    arrivals: list[tuple[int, str, int, int]],
) -> dict:
    """Run a dependency-aware synchronous PP scheduling trace.

    Args:
        pp_size: Number of PP stages and synchronous in-flight batch slots.
        max_num_batched_tokens: Scheduler token budget per batch.
        arrivals: ``(after_n_issued_batches, req_id, prompt_len, max_tokens)``.

    Returns:
        A JSON-serializable trace of Scheduler decisions and PP stage slots.
    """
    scheduler = create_scheduler(
        max_num_seqs=8,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=128,
        enable_chunked_prefill=True,
        pipeline_parallel_size=pp_size,
        block_size=16,
        num_blocks=10000,
        async_scheduling=False,
        use_v2_model_runner=False,
    )
    runner = MockPipelineRunner()
    pending_arrivals = deque(sorted(arrivals, key=lambda row: row[0]))

    def admit_ready(issued_count: int) -> None:
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
            runner.add_request(req_id, prompt_len)

    admit_ready(0)

    pending: deque[tuple[IssuedBatch, SchedulerOutput]] = deque()
    batches: list[IssuedBatch] = []
    scheduler_waits: list[dict] = []
    last_compute_batch: dict[str, int] = {}
    last_sample_batch: dict[str, int] = {}
    next_issue_tick = 0
    max_in_flight_batches = 0

    for _ in range(128):
        while len(pending) < pp_size and scheduler.has_requests():
            scheduler_output = scheduler.schedule()
            if scheduler_output.total_num_scheduled_tokens == 0:
                if pending:
                    scheduler_waits.append(
                        {
                            "tick": next_issue_tick,
                            "waiting_for_batch_ids": [
                                batch.batch_id for batch, _ in pending
                            ],
                        }
                    )
                break

            batch_id = len(batches)
            batch = IssuedBatch(
                batch_id=batch_id,
                issue_tick=next_issue_tick,
                completion_tick=next_issue_tick + pp_size,
                items=_snapshot_items(
                    scheduler,
                    scheduler_output,
                    batch_id,
                    last_compute_batch,
                    last_sample_batch,
                ),
            )
            batches.append(batch)
            pending.append((batch, scheduler_output))
            max_in_flight_batches = max(max_in_flight_batches, len(pending))
            next_issue_tick += 1
            admit_ready(len(batches))

        if not pending:
            if scheduler.has_requests() or pending_arrivals:
                raise AssertionError("simulation stalled before all requests arrived")
            break

        batch, scheduler_output = pending.popleft()
        model_runner_output = runner.complete(batch, scheduler_output)
        for item in batch.items:
            if item.emitted_token_id is not None:
                last_sample_batch[item.req_id] = batch.batch_id
        scheduler.update_from_output(scheduler_output, model_runner_output)
        next_issue_tick = max(next_issue_tick, batch.completion_tick)
        admit_ready(len(batches))
    else:
        raise AssertionError("simulation did not converge")

    last_tick = max((batch.completion_tick for batch in batches), default=0)
    devices = [f"GPU{stage} PP{stage}" for stage in range(pp_size)]
    occupied: dict[tuple[int, int], IssuedBatch] = {}
    for batch in batches:
        for stage in range(pp_size):
            position = (batch.issue_tick + stage, stage)
            assert position not in occupied
            occupied[position] = batch

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

    completion_events = [
        {
            "tick": batch.completion_tick,
            "batch_id": batch.batch_id,
            "samples": [
                {"req_id": item.req_id, "token_id": item.emitted_token_id}
                for item in batch.items
                if item.emitted_token_id is not None
            ],
        }
        for batch in batches
    ]

    return {
        "model": "unit-time PP stage model",
        "scheduler_mode": "sync",
        "pp_size": pp_size,
        "queue_size": pp_size,
        "max_num_batched_tokens": max_num_batched_tokens,
        "arrivals": [
            {
                "after_n_issued_batches": after,
                "req_id": req_id,
                "prompt_len": prompt_len,
                "max_tokens": max_tokens,
            }
            for after, req_id, prompt_len, max_tokens in arrivals
        ],
        "batches": [asdict(batch) for batch in batches],
        "dependencies": _build_dependencies(batches),
        "scheduler_waits": scheduler_waits,
        "completion_events": completion_events,
        "max_in_flight_batches": max_in_flight_batches,
        "last_tick": last_tick,
        "devices": devices,
        "cells": cells,
        "num_bubbles": sum(1 for cell in cells if cell["bubble"]),
        "num_busy": sum(1 for cell in cells if not cell["bubble"]),
    }


def _item_label(item: BatchItem) -> str:
    if item.phase == PARTIAL_PREFILL:
        return f"{item.req_id} P[{item.token_start}:{item.token_end}]"
    if item.phase == FINAL_PREFILL:
        return f"{item.req_id} PF[{item.token_start}:{item.token_end}]"
    return f"{item.req_id} D{item.decode_step}(y{item.consumed_token_id})"


def _cell_label(batch: IssuedBatch) -> str:
    return f"B{batch.batch_id} " + "+".join(
        _item_label(item) for item in batch.items
    )


def _text_grid(trace: dict) -> str:
    last_tick = trace["last_tick"]
    pp_size = trace["pp_size"]
    by_pos = {(cell["tick"], cell["device"]): cell for cell in trace["cells"]}
    header = "      " + " | ".join(f"t{tick:<24}" for tick in range(last_tick))
    rows = [header]
    for device, name in enumerate(trace["devices"]):
        cells = []
        for tick in range(last_tick):
            cell = by_pos[(tick, device)]
            text = "空" if cell["bubble"] else cell["label"]
            cells.append(f"{text:<26}")
        rows.append(f"{name:<8} " + " | ".join(cells))
    return "\n".join(rows)


def build_dependency_traces() -> dict:
    single_arrivals = [(0, "A", 24, 3)]
    mixed_arrivals = [
        (0, "A", 24, 3),
        (0, "B", 8, 3),
        (1, "C", 8, 3),
    ]
    kwargs = {"pp_size": 2, "max_num_batched_tokens": 16}
    return {
        "single_request": simulate_pp_schedule_trace(
            arrivals=single_arrivals, **kwargs
        ),
        "mixed_requests": simulate_pp_schedule_trace(
            arrivals=mixed_arrivals, **kwargs
        ),
    }


def write_dependency_trace(path: Path = TRACE_PATH) -> dict:
    payload = build_dependency_traces()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def test_pp_schedule_trace_models_request_dependencies():
    trace = simulate_pp_schedule_trace(
        pp_size=2,
        max_num_batched_tokens=16,
        arrivals=[(0, "A", 24, 3)],
    )
    batches = trace["batches"]

    assert [batch["items"][0]["phase"] for batch in batches] == [
        PARTIAL_PREFILL,
        FINAL_PREFILL,
        DECODE,
        DECODE,
    ]
    assert batches[1]["issue_tick"] < batches[0]["completion_tick"]

    stage_dependency = batches[1]["items"][0]
    assert stage_dependency["dependency_kind"] == STAGE_ORDER
    assert stage_dependency["depends_on_batch_id"] == 0
    for stage in range(trace["pp_size"]):
        source_end = batches[0]["issue_tick"] + stage + 1
        target_start = batches[1]["issue_tick"] + stage
        assert source_end <= target_start

    first_decode = batches[2]["items"][0]
    second_decode = batches[3]["items"][0]
    assert first_decode["dependency_kind"] == SAMPLE_TOKEN
    assert first_decode["depends_on_batch_id"] == 1
    assert first_decode["consumed_token_id"] == batches[1]["items"][0][
        "emitted_token_id"
    ]
    assert batches[2]["issue_tick"] >= batches[1]["completion_tick"]
    assert second_decode["depends_on_batch_id"] == 2
    assert second_decode["consumed_token_id"] == batches[2]["items"][0][
        "emitted_token_id"
    ]
    assert batches[3]["issue_tick"] >= batches[2]["completion_tick"]
    assert [wait["tick"] for wait in trace["scheduler_waits"]] == [2, 4, 6]


def test_pp_schedule_trace_batches_other_requests_during_decode_waits():
    trace = simulate_pp_schedule_trace(
        pp_size=2,
        max_num_batched_tokens=16,
        arrivals=[
            (0, "A", 24, 3),
            (0, "B", 8, 3),
            (1, "C", 8, 3),
        ],
    )
    actual_batches = [
        [(item["req_id"], item["phase"]) for item in batch["items"]]
        for batch in trace["batches"]
    ]
    assert actual_batches == [
        [("A", PARTIAL_PREFILL)],
        [("A", FINAL_PREFILL), ("B", FINAL_PREFILL)],
        [("C", FINAL_PREFILL)],
        [("A", DECODE), ("B", DECODE)],
        [("C", DECODE)],
        [("A", DECODE), ("B", DECODE)],
        [("C", DECODE)],
    ]
    assert trace["max_in_flight_batches"] == trace["queue_size"] == 2
    assert [wait["tick"] for wait in trace["scheduler_waits"]] == [7]

    pp0_by_tick = {
        cell["tick"]: cell
        for cell in trace["cells"]
        if cell["device"] == 0
    }
    assert all(not pp0_by_tick[tick]["bubble"] for tick in range(7))
    assert pp0_by_tick[2]["label"].startswith("B2 C PF")

    batches = {batch["batch_id"]: batch for batch in trace["batches"]}
    for dependency in trace["dependencies"]:
        if dependency["kind"] != SAMPLE_TOKEN:
            continue
        source = batches[dependency["from_batch_id"]]
        target = batches[dependency["to_batch_id"]]
        assert target["issue_tick"] >= source["completion_tick"]


if __name__ == "__main__":
    test_pp_schedule_trace_models_request_dependencies()
    test_pp_schedule_trace_batches_other_requests_during_decode_waits()
    traces = write_dependency_trace()
    print("\n=== single request, PP=2 ===")
    print(_text_grid(traces["single_request"]))
    print("\n=== mixed requests, PP=2 ===")
    print(_text_grid(traces["mixed_requests"]))
    print(f"\ntrace written to {TRACE_PATH}")
