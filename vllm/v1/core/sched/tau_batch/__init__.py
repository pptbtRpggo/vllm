# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.core.sched.tau_batch.dispatch import (
    DispatchPhase,
    DispatchSlot,
    WaveDispatcher,
    WaveDispatchPolicy,
)
from vllm.v1.core.sched.tau_batch.planner import TauBatchPlanner
from vllm.v1.core.sched.tau_batch.strategy import (
    GreedyWaveStrategy,
    WavePackingStrategy,
)
from vllm.v1.core.sched.tau_batch.types import (
    MicroBatchPlan,
    PackContext,
    TauRequestSnapshot,
    WavePlan,
    estimate_kv_blocks,
)

__all__ = [
    "DispatchPhase",
    "DispatchSlot",
    "GreedyWaveStrategy",
    "MicroBatchPlan",
    "PackContext",
    "TauBatchPlanner",
    "TauRequestSnapshot",
    "estimate_kv_blocks",
    "TauScheduler",
    "WaveDispatcher",
    "WaveDispatchPolicy",
    "WavePackingStrategy",
    "WavePlan",
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
