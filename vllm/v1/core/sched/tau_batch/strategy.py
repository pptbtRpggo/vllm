# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Protocol

from vllm.v1.core.sched.tau_batch.types import (
    MicroBatchPlan,
    PackContext,
    TauRequestSnapshot,
    WavePlan,
    estimate_kv_blocks,
)


class WavePackingStrategy(Protocol):
    """Selects a subset of a waiting snapshot and packs it into micro-batches.

    Implementations must return a WavePlan whose micro-batch order is the
    dispatch order. TauBatchPlanner assigns ``wave_id`` after pack() returns.
    """

    def pack(
        self,
        requests: Sequence[TauRequestSnapshot],
        ctx: PackContext,
    ) -> WavePlan:
        """Pack requests into a wave.

        Args:
            requests: Validated waiting snapshot. Ids are unique.
            ctx: Shared capacity limits for this call.

        Returns:
            A WavePlan. Use ``wave_id=0``; the planner overwrites it.
            Return empty ``microbatches`` if nothing can be admitted.
        """
        ...


class GreedyWaveStrategy:
    """Placeholder packer: tight TPOT first, skip what does not fit.

    Sort by ``(tpot_slo_ms, arrival_time, request_id)``. Admit a request
    only if its reserved KV (prompt + max generate) fits the remaining
    free blocks. Split only by ``max_reqs_per_microbatch``. A request
    that does not fit is deferred; the next request is tried.
    This is not the paper algorithm.
    """

    def pack(
        self,
        requests: Sequence[TauRequestSnapshot],
        ctx: PackContext,
    ) -> WavePlan:
        input_ids = frozenset(r.request_id for r in requests)
        if not requests:
            return _empty_plan(input_ids)

        ordered = sorted(
            requests,
            key=lambda r: (r.tpot_slo_ms, r.arrival_time, r.request_id),
        )
        remaining_kv = ctx.kv_free_blocks
        batches: list[list[TauRequestSnapshot]] = []
        current: list[TauRequestSnapshot] = []
        admitted_count = 0

        for req in ordered:
            if admitted_count >= ctx.max_num_seqs:
                break

            need = _kv_blocks_if_fits(req, remaining_kv, ctx.block_size)
            if need is None:
                continue
            if _needs_new_batch(current, ctx):
                if len(batches) >= ctx.max_microbatches:
                    continue
                batches.append(current)
                current = []
            if not current and len(batches) >= ctx.max_microbatches:
                continue

            current.append(req)
            admitted_count += 1
            if remaining_kv is not None:
                remaining_kv -= need

        if current and len(batches) < ctx.max_microbatches:
            batches.append(current)

        if not batches:
            return _empty_plan(input_ids)

        microbatches = tuple(
            MicroBatchPlan(
                req_ids=tuple(r.request_id for r in batch),
                index=i,
            )
            for i, batch in enumerate(batches)
        )
        admitted_ids = frozenset(
            req_id for batch in microbatches for req_id in batch.req_ids
        )
        return WavePlan(
            wave_id=0,
            microbatches=microbatches,
            admitted_ids=admitted_ids,
            deferred_ids=input_ids - admitted_ids,
            extra={"strategy": "greedy"},
        )


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


def _needs_new_batch(
    current: list[TauRequestSnapshot],
    ctx: PackContext,
) -> bool:
    return bool(current) and len(current) >= ctx.max_reqs_per_microbatch


def _empty_plan(input_ids: frozenset[str]) -> WavePlan:
    return WavePlan(
        wave_id=0,
        microbatches=(),
        admitted_ids=frozenset(),
        deferred_ids=input_ids,
        extra={"strategy": "greedy"},
    )
