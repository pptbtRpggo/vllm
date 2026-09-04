# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.core.sched.tau_batch.dispatch import (
    DispatchPhase,
    DispatchPolicy,
    DispatchSlot,
    ListDispatcher,
)
from vllm.v1.core.sched.tau_batch.planner import TauBatchPlanner
from vllm.v1.core.sched.tau_batch.strategy import (
    GreedyListStrategy,
    ListPackingStrategy,
)
from vllm.v1.core.sched.tau_batch.types import (
    MicroBatchList,
    MicroBatchPlan,
    MicroBatchTask,
    PackContext,
    TaskFeatures,
    TauRequestSnapshot,
    annotate_request_budget,
    estimate_kv_blocks,
    request_budget_dict,
    tau_max_ms,
    ttft_slack_ms,
    wait_ms,
)

__all__ = [
    "DispatchPhase",
    "DispatchPolicy",
    "DispatchSlot",
    "GreedyListStrategy",
    "ListDispatcher",
    "ListPackingStrategy",
    "MicroBatchList",
    "MicroBatchPlan",
    "MicroBatchTask",
    "PackContext",
    "TaskFeatures",
    "TauBatchPlanner",
    "TauRequestSnapshot",
    "annotate_request_budget",
    "estimate_kv_blocks",
    "request_budget_dict",
    "tau_max_ms",
    "ttft_slack_ms",
    "wait_ms",
    "TauScheduler",
    "snapshot_from_request",
]


def __getattr__(name: str):
    # TauScheduler imports the vLLM Request/Scheduler stack; keep that lazy so
    # planner/dispatch unit tests do not pull optional runtime deps.
    if name in {"TauScheduler", "snapshot_from_request"}:
        from vllm.v1.core.sched.tau_batch.scheduler import (
            TauScheduler,
            snapshot_from_request,
        )

        exports = {
            "TauScheduler": TauScheduler,
            "snapshot_from_request": snapshot_from_request,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
