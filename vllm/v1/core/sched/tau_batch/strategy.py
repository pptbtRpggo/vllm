# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Protocol

from vllm.v1.core.sched.tau_batch.types import (
    MicroBatchList,
    MicroBatchTask,
    PackContext,
    TauRequestSnapshot,
    estimate_kv_blocks,
)


class ListPackingStrategy(Protocol):
    """Packs a waiting snapshot into a micro-batch-task list.

    The returned list order is the dispatch order. The scheduler stamps a
    wave id when it starts dispatching this list.
    """

    def pack(
        self,
        requests: Sequence[TauRequestSnapshot],
        ctx: PackContext,
    ) -> MicroBatchList:
        """Pack requests into a micro-batch-task list.

        Args:
            requests: Validated waiting snapshot. Ids are unique.
            ctx: Shared packing limits for this call.

        Returns:
            A MicroBatchList. Empty ``tasks`` if nothing can be admitted.
        """
        ...


class GreedyListStrategy:
    """Take up to ``max_num_seqs``, then split; overflow deferred, never pad.

    Sort by ``(tpot_slo_ms, arrival_time, request_id)``. First take at most
    ``max_num_seqs`` requests whose reserved KV fits. Then split that take
    by ``max_reqs_per_microbatch``, at most ``max_microbatches`` tasks.
    Requests that do not fit KV, exceed the take, or do not fit the remaining
    tasks are deferred. A short take is packed as-is.

    This is the current default. The paper dual-ceiling packer is not here yet.
    """

    def pack(
        self,
        requests: Sequence[TauRequestSnapshot],
        ctx: PackContext,
    ) -> MicroBatchList:
        input_ids = frozenset(r.request_id for r in requests)
        if not requests:
            return _empty_list(input_ids)

        ordered = sorted(
            requests,
            key=lambda r: (r.tpot_slo_ms, r.arrival_time, r.request_id),
        )
        taken = _take_requests(ordered, ctx)
        batches = _split_taken(taken, ctx)
        if not batches:
            return _empty_list(input_ids)

        tasks = tuple(
            MicroBatchTask(
                req_ids=tuple(r.request_id for r in batch),
                index=i,
            )
            for i, batch in enumerate(batches)
        )
        admitted_ids = frozenset(
            req_id for task in tasks for req_id in task.req_ids
        )
        return MicroBatchList(
            tasks=tasks,
            admitted_ids=admitted_ids,
            deferred_ids=input_ids - admitted_ids,
            extra={"strategy": "greedy"},
        )


def _take_requests(
    ordered: Sequence[TauRequestSnapshot],
    ctx: PackContext,
) -> list[TauRequestSnapshot]:
    """Select at most ``max_num_seqs`` requests that fit remaining KV."""
    taken: list[TauRequestSnapshot] = []
    remaining_kv = ctx.kv_free_blocks
    for req in ordered:
        if len(taken) >= ctx.max_num_seqs:
            break
        need = _kv_blocks_if_fits(req, remaining_kv, ctx.block_size)
        if need is None:
            continue
        taken.append(req)
        if remaining_kv is not None:
            remaining_kv -= need
    return taken


def _split_taken(
    taken: Sequence[TauRequestSnapshot],
    ctx: PackContext,
) -> list[list[TauRequestSnapshot]]:
    """Split the take into micro-batch tasks. Overflow is deferred."""
    batches: list[list[TauRequestSnapshot]] = []
    current: list[TauRequestSnapshot] = []
    for req in taken:
        if current and len(current) >= ctx.max_reqs_per_microbatch:
            if len(batches) >= ctx.max_microbatches:
                break
            batches.append(current)
            current = []
        if not current and len(batches) >= ctx.max_microbatches:
            break
        current.append(req)
    if current and len(batches) < ctx.max_microbatches:
        batches.append(current)
    return batches


def _kv_blocks_if_fits(
    req: TauRequestSnapshot,
    remaining_kv: int | None,
    block_size: int | None,
) -> int | None:
    if remaining_kv is None or block_size is None:
        return 0
    need = estimate_kv_blocks(req.prompt_len, req.max_new_tokens, block_size)
    if need > remaining_kv:
        return None
    return need


def _empty_list(input_ids: frozenset[str]) -> MicroBatchList:
    return MicroBatchList(
        tasks=(),
        admitted_ids=frozenset(),
        deferred_ids=input_ids,
        extra={"strategy": "greedy"},
    )
