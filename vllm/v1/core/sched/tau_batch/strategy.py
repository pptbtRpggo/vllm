# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Protocol

from vllm.v1.core.sched.tau_batch.types import (
    MicroBatchPlan,
    PackContext,
    TauRequestSnapshot,
    WavePlan,
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
    """Placeholder packer: tight TPOT first, then fill capacity.

    Sort by ``(tpot_slo_ms, arrival_time, request_id)``, admit at most
    ``min(max_num_seqs, max_microbatches * max_reqs_per_microbatch)``
    requests, and split them into non-empty micro-batches of size
    ``max_reqs_per_microbatch``. This is not the paper algorithm.
    """

    def pack(
        self,
        requests: Sequence[TauRequestSnapshot],
        ctx: PackContext,
    ) -> WavePlan:
        input_ids = frozenset(r.request_id for r in requests)
        if not requests:
            return _empty_plan(input_ids)

        capacity = min(
            ctx.max_num_seqs,
            ctx.max_microbatches * ctx.max_reqs_per_microbatch,
        )
        if capacity <= 0:
            return _empty_plan(input_ids)

        ordered = sorted(
            requests,
            key=lambda r: (r.tpot_slo_ms, r.arrival_time, r.request_id),
        )
        picked = ordered[:capacity]
        batch_size = ctx.max_reqs_per_microbatch
        microbatches: list[MicroBatchPlan] = []
        for start in range(0, len(picked), batch_size):
            if len(microbatches) >= ctx.max_microbatches:
                break
            chunk = picked[start : start + batch_size]
            microbatches.append(
                MicroBatchPlan(
                    req_ids=tuple(r.request_id for r in chunk),
                    index=len(microbatches),
                )
            )

        if not microbatches:
            return _empty_plan(input_ids)

        admitted_ids = frozenset(
            req_id for batch in microbatches for req_id in batch.req_ids
        )
        return WavePlan(
            wave_id=0,
            microbatches=tuple(microbatches),
            admitted_ids=admitted_ids,
            deferred_ids=input_ids - admitted_ids,
            extra={"strategy": "greedy"},
        )


def _empty_plan(input_ids: frozenset[str]) -> WavePlan:
    return WavePlan(
        wave_id=0,
        microbatches=(),
        admitted_ids=frozenset(),
        deferred_ids=input_ids,
        extra={"strategy": "greedy"},
    )
