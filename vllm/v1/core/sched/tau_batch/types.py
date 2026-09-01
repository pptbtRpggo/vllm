# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TauRequestSnapshot:
    """Waiting-queue snapshot of one request for wave planning.

    This is independent of ``vllm.v1.request.Request`` so the planner can be
    unit-tested and later reused by a scheduler adapter.

    Attributes:
        request_id: Unique request id.
        arrival_time: Arrival timestamp in seconds (same clock as PackContext.now).
        prompt_len: Number of prompt tokens.
        ttft_slo_ms: Time-to-first-token SLO in milliseconds.
        tpot_slo_ms: Time-per-output-token SLO in milliseconds.
    """

    request_id: str
    arrival_time: float
    prompt_len: int
    ttft_slo_ms: float
    tpot_slo_ms: float


@dataclass(frozen=True)
class MicroBatchPlan:
    """One ordered micro-batch in a wave.

    ``index`` is the dispatch position. After construction the request order
    inside ``req_ids`` must not be reshuffled.

    Attributes:
        req_ids: Request ids in this micro-batch, in stable order.
        index: Zero-based dispatch index in the wave.
    """

    req_ids: tuple[str, ...]
    index: int


@dataclass(frozen=True)
class WavePlan:
    """Result of one ``plan_wave`` call on a waiting snapshot.

    ``admitted_ids`` / ``deferred_ids`` record this wave only. The next wave
    is planned from a fresh waiting snapshot; do not concatenate deferred ids
    with new arrivals.

    Attributes:
        wave_id: Monotonic id assigned by TauBatchPlanner.
        microbatches: Dispatch-ordered micro-batches. Union of req_ids equals
            admitted_ids.
        admitted_ids: Requests packed into this wave.
        deferred_ids: Requests seen in the snapshot but not packed.
        extra: Strategy-private metadata (e.g. strategy name). Not an input
            to the next plan_wave call.
    """

    wave_id: int
    microbatches: tuple[MicroBatchPlan, ...]
    admitted_ids: frozenset[str]
    deferred_ids: frozenset[str]
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PackContext:
    """Shared packing limits for one plan_wave call.

    Request-level SLOs live on TauRequestSnapshot. PackContext is the
    system-level capacity for this snapshot.

    Attributes:
        now: Current timestamp in seconds (same clock as arrival_time).
        max_num_seqs: Max admitted requests in this wave.
        max_microbatches: Max number of micro-batches P.
        max_reqs_per_microbatch: Max requests in one micro-batch.
        pp_size: Pipeline-parallel size. Reserved for later strategies.
        kv_free_blocks: Free KV blocks. Reserved for later strategies.
        oracle: Latency oracle. Reserved for later strategies.
    """

    now: float
    max_num_seqs: int
    max_microbatches: int
    max_reqs_per_microbatch: int
    pp_size: int | None = None
    kv_free_blocks: int | None = None
    oracle: Any | None = None
