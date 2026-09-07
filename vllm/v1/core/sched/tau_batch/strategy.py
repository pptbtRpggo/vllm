# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Protocol

from vllm.v1.core.sched.tau_batch.types import (
    EosEvent,
    MicroBatchList,
    MicroBatchTask,
    PackContext,
    TauRequestSnapshot,
    estimate_kv_blocks,
)


class ListPackingStrategy(Protocol):
    """Packs the waiting pool into a micro-batch-task list.

    The strategy sees the whole snapshot and must not first cut it down to
    ``max_num_seqs``. The returned list order is the dispatch order. The
    scheduler stamps a wave id when it starts dispatching this list.
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


class EosStrategy(Protocol):
    """Runs when admitted requests finish (EOS or length cap).

    The current implementation is a no-op. A later strategy can refill
    the leftover list from waiting without exceeding the list ceiling.
    """

    def on_eos(self, event: EosEvent) -> None:
        """Handle one model step that finished one or more admitted requests.

        Args:
            event: Who finished, who remains on the list, and who is waiting.
        """
        ...


class NoOpEosStrategy:
    """Placeholder EOS hook. Does not refill or mutate the list."""

    def on_eos(self, event: EosEvent) -> None:
        return


class GreedyListStrategy:
    """Walk the whole waiting pool, fill micro-batch tasks, and never pad.

    Sort by ``(tpot_slo_ms, arrival_time, request_id)``. Walk that order and
    append into the current task while reserved KV fits. A full task starts
    the next one until ``max_microbatches``; 0 means no list-length cap.
    KV- or token-unfit requests are skipped so a later request can still
    enter. Token-unfit requests are reconsidered for the next task.

    This is the current default. Snapshots carry wait/slack; this strategy
    does not use them. The paper dual-ceiling packer is not here yet.
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
        batches = _pack_pool(ordered, ctx)
        if not batches:
            return _empty_list(input_ids)

        tasks = tuple(
            MicroBatchTask(
                req_ids=tuple(r.request_id for r in batch),
                index=i,
            )
            for i, batch in enumerate(batches)
        )
        admitted_ids = frozenset(req_id for task in tasks for req_id in task.req_ids)
        return MicroBatchList(
            tasks=tasks,
            admitted_ids=admitted_ids,
            deferred_ids=input_ids - admitted_ids,
            extra={"strategy": "greedy"},
        )


def _pack_pool(
    ordered: Sequence[TauRequestSnapshot],
    ctx: PackContext,
) -> list[list[TauRequestSnapshot]]:
    """Fill micro-batch tasks from the full ordered waiting pool."""
    batches: list[list[TauRequestSnapshot]] = []
    remaining_kv = ctx.kv_free_blocks
    cap = ctx.max_microbatches
    pending = list(ordered)
    while pending and (cap == 0 or len(batches) < cap):
        current: list[TauRequestSnapshot] = []
        skipped: list[TauRequestSnapshot] = []
        tokens = 0
        for req in pending:
            need = _kv_blocks_if_fits(req, remaining_kv, ctx.block_size)
            if need is None:
                # Free capacity only decreases during this pack call.
                continue
            if len(current) >= ctx.max_reqs_per_microbatch or (
                ctx.max_num_batched_tokens is not None
                and tokens + req.prompt_len > ctx.max_num_batched_tokens
            ):
                skipped.append(req)
                continue
            current.append(req)
            tokens += req.prompt_len
            if remaining_kv is not None:
                remaining_kv -= need
        if not current:
            break
        batches.append(current)
        pending = skipped
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
