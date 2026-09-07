# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Contracts for future online policies; no paper algorithm lives here.

The current scheduler exposes ``pipeline_snapshot`` and oracle injection.
OnlinePolicy/PlanUpdate describe the subsequent event-to-action adapter;
they are not yet executed by the fixed-list scheduler. See INTERFACES.md.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

from vllm.v1.core.sched.tau_batch.types import (
    MicroBatchList,
    PackContext,
    TaskFeatures,
    TauRequestSnapshot,
)


@dataclass(frozen=True)
class LatencyEstimate:
    """Per-stage forward prediction in milliseconds, excluding PP recv/send.

    upper_ms is an optional calibrated upper estimate, not a hard guarantee.
    Device-only and host-forward measurements must use different profiles.
    """

    mean_ms: float
    upper_ms: float | None = None


class LatencyOracle(Protocol):
    """Load fitted parameters outside the scheduling hot path.

    An oracle instance is bound to one model/hardware/PP/TP/dtype/backend
    configuration and one measurement scope. seq_lens includes this
    forward's input token in decode, and the full prompt in prefill.
    """

    def predict(self, features: TaskFeatures, *, pp_rank: int) -> LatencyEstimate: ...


@dataclass(frozen=True)
class ActiveRequestSnapshot:
    request: TauRequestSnapshot
    computed_tokens: int
    output_tokens: int
    kv_block_ids: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ActiveMicroBatchSnapshot:
    index: int
    requests: tuple[ActiveRequestSnapshot, ...]
    prefill_dispatched: bool
    prefill_completed: bool
    retired: bool
    in_flight: int


@dataclass(frozen=True)
class PipelineSnapshot:
    """Copied state; policies must not mutate Request/KV/dispatcher objects.

    Requests admitted but not dispatched appear in active, not waiting.
    computed_tokens includes scheduled work still in flight; completion
    flags/in_flight must be checked before reusing the corresponding KV.
    active_plan retains strategy ceilings/metadata in its extra field.
    """

    context: PackContext
    wave_id: int | None
    active_plan: MicroBatchList | None
    active: tuple[ActiveMicroBatchSnapshot, ...]
    waiting: tuple[TauRequestSnapshot, ...]
    pending_plan: MicroBatchList | None = None


@dataclass(frozen=True)
class SchedulingEvent:
    kind: Literal["arrival", "eos", "microbatch_done", "cancel"]
    request_ids: tuple[str, ...]
    wave_id: int | None
    microbatch_index: int | None = None


@dataclass(frozen=True)
class RefillProposal:
    """Prefill these waiting requests before joining the target Decode task.

    The executor must not insert their IDs directly into a Decode slot.
    Ceilings, token/KV capacity and target liveness are checked again at
    application time. Existing in-flight outputs retain their original plan.
    """

    microbatch_index: int
    request_ids: tuple[str, ...]


@dataclass(frozen=True)
class PlanUpdate:
    """Proposal only; application belongs to the scheduler adapter.

    None pending_plan preserves the previous pending plan; an empty plan
    explicitly clears it. It must never replace the active plan in place.
    Revalidate against live state even if based_on_wave_id still matches.
    """

    based_on_wave_id: int | None
    pending_plan: MicroBatchList | None = None
    refills: tuple[RefillProposal, ...] = ()


class OnlinePolicy(Protocol):
    def on_event(
        self, event: SchedulingEvent, state: PipelineSnapshot
    ) -> PlanUpdate | None: ...
