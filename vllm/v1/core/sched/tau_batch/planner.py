# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import replace

from vllm.v1.core.sched.tau_batch.strategy import (
    GreedyWaveStrategy,
    WavePackingStrategy,
)
from vllm.v1.core.sched.tau_batch.types import (
    PackContext,
    TauRequestSnapshot,
    WavePlan,
)


class TauBatchPlanner:
    """Plans a wave from a waiting-queue snapshot.

    The planner is stateless with respect to previous deferred ids: each call
    takes the current snapshot plus PackContext. admitted/deferred on the
    returned WavePlan are a record of this call only.
    """

    def __init__(
        self,
        strategy: WavePackingStrategy | None = None,
        *,
        wave_id_start: int = 0,
    ) -> None:
        """Initialize the planner.

        Args:
            strategy: Packing strategy. Defaults to GreedyWaveStrategy.
            wave_id_start: First wave_id to assign on a successful plan.
        """
        self.strategy = strategy if strategy is not None else GreedyWaveStrategy()
        self._next_wave_id = wave_id_start

    def plan_wave(
        self,
        requests: Sequence[TauRequestSnapshot],
        ctx: PackContext,
    ) -> WavePlan | None:
        """Select a subset and pack it into ordered micro-batches.

        Args:
            requests: Current waiting snapshot. Each request appears once.
            ctx: Shared packing limits for this call.

        Returns:
            A WavePlan, or None if the snapshot is empty or nothing was
            admitted.

        Raises:
            ValueError: Invalid snapshot or PackContext, or the strategy
                returned a plan that violates invariants.
        """
        self._validate_context(ctx)
        self._validate_requests(requests)
        if not requests:
            return None

        raw = self.strategy.pack(requests, ctx)
        if not raw.microbatches:
            return None

        plan = replace(raw, wave_id=self._next_wave_id)
        self._validate_plan(plan, requests, ctx)
        self._next_wave_id += 1
        return plan

    @staticmethod
    def _validate_context(ctx: PackContext) -> None:
        if ctx.max_num_seqs < 1:
            raise ValueError(f"max_num_seqs must be >= 1, got {ctx.max_num_seqs}")
        if ctx.max_microbatches < 1:
            raise ValueError(
                f"max_microbatches must be >= 1, got {ctx.max_microbatches}"
            )
        if ctx.max_reqs_per_microbatch < 1:
            raise ValueError(
                "max_reqs_per_microbatch must be >= 1, got "
                f"{ctx.max_reqs_per_microbatch}"
            )
        if ctx.kv_free_blocks is not None and ctx.kv_free_blocks < 0:
            raise ValueError(
                f"kv_free_blocks must be >= 0, got {ctx.kv_free_blocks}"
            )
        if ctx.kv_free_blocks is not None and (
            ctx.block_size is None or ctx.block_size < 1
        ):
            raise ValueError(
                "block_size must be >= 1 when kv_free_blocks is set"
            )

    @staticmethod
    def _validate_requests(requests: Sequence[TauRequestSnapshot]) -> None:
        seen: set[str] = set()
        for req in requests:
            if not req.request_id:
                raise ValueError("request_id must be non-empty")
            if req.request_id in seen:
                raise ValueError(f"duplicate request_id: {req.request_id}")
            seen.add(req.request_id)
            if req.prompt_len < 1:
                raise ValueError(
                    f"prompt_len must be >= 1 for {req.request_id}, "
                    f"got {req.prompt_len}"
                )
            if req.ttft_slo_ms <= 0:
                raise ValueError(
                    f"ttft_slo_ms must be > 0 for {req.request_id}, "
                    f"got {req.ttft_slo_ms}"
                )
            if req.tpot_slo_ms <= 0:
                raise ValueError(
                    f"tpot_slo_ms must be > 0 for {req.request_id}, "
                    f"got {req.tpot_slo_ms}"
                )
            if req.max_new_tokens < 0:
                raise ValueError(
                    f"max_new_tokens must be >= 0 for {req.request_id}, "
                    f"got {req.max_new_tokens}"
                )

    @staticmethod
    def _validate_plan(
        plan: WavePlan,
        requests: Sequence[TauRequestSnapshot],
        ctx: PackContext,
    ) -> None:
        input_ids = {req.request_id for req in requests}
        admitted = plan.admitted_ids
        deferred = plan.deferred_ids

        if not plan.microbatches:
            raise ValueError("WavePlan.microbatches must be non-empty")
        extra = admitted - input_ids
        if extra:
            raise ValueError(f"admitted_ids contains unknown ids: {sorted(extra)}")
        if admitted | deferred != input_ids:
            raise ValueError(
                "admitted_ids and deferred_ids must partition the input ids"
            )
        if admitted & deferred:
            raise ValueError("admitted_ids and deferred_ids must be disjoint")

        packed: list[str] = []
        packed_set: set[str] = set()
        for i, batch in enumerate(plan.microbatches):
            if batch.index != i:
                raise ValueError(
                    f"microbatches[{i}].index must be {i}, got {batch.index}"
                )
            if not batch.req_ids:
                raise ValueError(f"microbatches[{i}] must be non-empty")
            if len(batch.req_ids) > ctx.max_reqs_per_microbatch:
                raise ValueError(
                    f"microbatches[{i}] has {len(batch.req_ids)} reqs, "
                    f"max is {ctx.max_reqs_per_microbatch}"
                )
            for req_id in batch.req_ids:
                if req_id in packed_set:
                    raise ValueError(f"request {req_id} appears in multiple batches")
                packed_set.add(req_id)
                packed.append(req_id)

        if frozenset(packed) != admitted:
            raise ValueError("union of microbatch req_ids must equal admitted_ids")
        if len(admitted) > ctx.max_num_seqs:
            raise ValueError(
                f"admitted {len(admitted)} requests, max_num_seqs is {ctx.max_num_seqs}"
            )
        if len(plan.microbatches) > ctx.max_microbatches:
            raise ValueError(
                f"{len(plan.microbatches)} microbatches, max is {ctx.max_microbatches}"
            )
