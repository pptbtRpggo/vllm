# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.v1.core.sched.tau_batch.strategy import (
    EosStrategy,
    GreedyListStrategy,
    ListPackingStrategy,
    NoOpEosStrategy,
)
from vllm.v1.core.sched.tau_batch.types import (
    EosEvent,
    MicroBatchList,
    PackContext,
    TauRequestSnapshot,
    annotate_request_budget,
)


class TauBatchPlanner:
    """Packs a waiting snapshot into a micro-batch-task list.

    The planner is stateless with respect to previous deferred ids: each call
    takes the current snapshot plus PackContext. admitted/deferred on the
    returned list record this call only. No wave id is assigned here.

    ``on_eos`` forwards to ``eos_strategy``. The default hook is empty.
    """

    def __init__(
        self,
        strategy: ListPackingStrategy | None = None,
        eos_strategy: EosStrategy | None = None,
    ) -> None:
        """Initialize the planner.

        Args:
            strategy: Packing strategy. Defaults to GreedyListStrategy.
            eos_strategy: Hook after admitted requests finish. Defaults
                to NoOpEosStrategy.
        """
        self.strategy = strategy if strategy is not None else GreedyListStrategy()
        self.eos_strategy = (
            eos_strategy if eos_strategy is not None else NoOpEosStrategy()
        )

    def on_eos(self, event: EosEvent) -> None:
        """Forward an EOS event to ``eos_strategy``. Currently a no-op."""
        self.eos_strategy.on_eos(event)

    def plan(
        self,
        requests: Sequence[TauRequestSnapshot],
        ctx: PackContext,
    ) -> MicroBatchList | None:
        """Pack a micro-batch-task list from the waiting snapshot.

        Args:
            requests: Current waiting snapshot. Each request appears once.
            ctx: System caps and clock for this call.

        Returns:
            A MicroBatchList, or None if the snapshot is empty or nothing
            was admitted.

        Raises:
            ValueError: Invalid snapshot or PackContext, or the strategy
                returned a list that violates invariants.
        """
        self._validate_context(ctx)
        self._validate_requests(requests)
        if not requests:
            return None

        requests = [
            annotate_request_budget(req, ctx.now, ctx.pp_size) for req in requests
        ]
        packed = self.strategy.pack(requests, ctx)
        if not packed.tasks:
            return None

        self._validate_list(packed, requests, ctx)
        return packed

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
    def _validate_list(
        packed: MicroBatchList,
        requests: Sequence[TauRequestSnapshot],
        ctx: PackContext,
    ) -> None:
        input_ids = {req.request_id for req in requests}
        admitted = packed.admitted_ids
        deferred = packed.deferred_ids

        if not packed.tasks:
            raise ValueError("MicroBatchList.tasks must be non-empty")
        extra = admitted - input_ids
        if extra:
            raise ValueError(f"admitted_ids contains unknown ids: {sorted(extra)}")
        if admitted | deferred != input_ids:
            raise ValueError(
                "admitted_ids and deferred_ids must partition the input ids"
            )
        if admitted & deferred:
            raise ValueError("admitted_ids and deferred_ids must be disjoint")

        seen: list[str] = []
        seen_set: set[str] = set()
        for i, task in enumerate(packed.tasks):
            if task.index != i:
                raise ValueError(
                    f"tasks[{i}].index must be {i}, got {task.index}"
                )
            if not task.req_ids:
                raise ValueError(f"tasks[{i}] must be non-empty")
            if len(task.req_ids) > ctx.max_reqs_per_microbatch:
                raise ValueError(
                    f"tasks[{i}] has {len(task.req_ids)} reqs, "
                    f"max is {ctx.max_reqs_per_microbatch}"
                )
            for req_id in task.req_ids:
                if req_id in seen_set:
                    raise ValueError(f"request {req_id} appears in multiple tasks")
                seen_set.add(req_id)
                seen.append(req_id)

        if frozenset(seen) != admitted:
            raise ValueError("union of task req_ids must equal admitted_ids")
        if len(admitted) > ctx.max_num_seqs:
            raise ValueError(
                f"admitted {len(admitted)} requests, max_num_seqs is {ctx.max_num_seqs}"
            )
        if len(packed.tasks) > ctx.max_microbatches:
            raise ValueError(
                f"{len(packed.tasks)} tasks, max is {ctx.max_microbatches}"
            )
