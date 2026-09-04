# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class TauRequestSnapshot:
    """Waiting-queue snapshot of one request for list packing.

    This is independent of ``vllm.v1.request.Request`` so the planner can be
    unit-tested and later reused by a scheduler adapter.

    Arrival is recorded by the engine on ``Request.arrival_time``. Wait and
    slack are filled at plan time by ``annotate_request_budget`` using
    ``PackContext.now`` and ``pp_size``. They are not maintained as running
    counters on the live Request.

    Attributes:
        request_id: Unique request id.
        arrival_time: Arrival timestamp in seconds (same clock as PackContext.now).
        prompt_len: Number of prompt tokens.
        max_new_tokens: Max new tokens to generate. Used with prompt_len to
            reserve KV for the whole Prefill+Decode lifetime.
        ttft_slo_ms: Time-to-first-token SLO in milliseconds.
        tpot_slo_ms: Time-per-output-token SLO in milliseconds.
        wait_ms: Queue wait at the snapshot clock.
        ttft_slack_ms: Remaining TTFT ``T_ttft - wait`` (paper ``b_i``).
            May be negative if the deadline is already missed.
        tau_max_ms: Per-stage decode budget ``T_tpot / M`` (paper ``τ_max``).
    """

    request_id: str
    arrival_time: float
    prompt_len: int
    ttft_slo_ms: float
    tpot_slo_ms: float
    max_new_tokens: int = 1
    wait_ms: float = 0.0
    ttft_slack_ms: float = 0.0
    tau_max_ms: float = 0.0


@dataclass(frozen=True)
class MicroBatchTask:
    """One micro-batch task: n request tasks of the same phase.

    At dispatch time this becomes n prefills or n decodes together. ``index``
    is the position in the list. After construction ``req_ids`` must not be
    reshuffled.

    Attributes:
        req_ids: The n request ids in this micro-batch task, in stable order.
        index: Zero-based index in the micro-batch-task list.
    """

    req_ids: tuple[str, ...]
    index: int


@dataclass(frozen=True)
class MicroBatchList:
    """Packed micro-batch-task list.

    ``admitted_ids`` / ``deferred_ids`` record this pack call only.
    A wave id is stamped later, when dispatch starts.

    Attributes:
        tasks: Dispatch-ordered micro-batch tasks. Union of req_ids equals
            admitted_ids.
        admitted_ids: Requests packed into this list.
        deferred_ids: Requests seen in the snapshot but not packed.
        extra: Strategy-private metadata. Not an input to the next pack call.
    """

    tasks: tuple[MicroBatchTask, ...]
    admitted_ids: frozenset[str]
    deferred_ids: frozenset[str]
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EosEvent:
    """Admitted requests that finished in this model step.

    Fired from ``update_from_output`` after the parent scheduler frees
    those requests, while the active list is still set if anyone remains.
    The default EOS strategy is a no-op; later refill can use
    ``remaining_ids`` / ``waiting_ids`` without changing dispatch yet.
    """

    finished_ids: tuple[str, ...]
    wave_id: int | None
    batch_idx: int | None
    phase: str | None
    remaining_ids: tuple[str, ...]
    waiting_ids: tuple[str, ...]


# Older name kept so existing imports keep working.
MicroBatchPlan = MicroBatchTask


@dataclass(frozen=True)
class PackContext:
    """Shared packing limits for one pack call.

    Request-level SLOs live on TauRequestSnapshot. The default greedy
    strategy uses the caps below (take then split). A later paper
    strategy can use the SLOs, wait/slack, and ``oracle``.

    Attributes:
        now: Current timestamp in seconds (same clock as arrival_time).
            Used to compute wait and TTFT slack at plan time.
        max_num_seqs: Take at most this many from the waiting snapshot.
        max_microbatches: Then pack at most this many micro-batch tasks.
        max_reqs_per_microbatch: Max n in one micro-batch task. Not
            derived from max_num_seqs or max_microbatches. Overflow is
            deferred; a short take is packed as-is.
        pp_size: Pipeline-parallel size M. Used for ``τ_max = TPOT / M``.
        kv_free_blocks: Free KV blocks at pack time. None disables the
            KV filter.
        block_size: Tokens per KV block. Required when kv_free_blocks is set.
        max_num_batched_tokens: Unused. Kept so older callers still construct
            PackContext.
        oracle: Latency oracle. Reserved for later strategies.
    """

    now: float
    max_num_seqs: int
    max_microbatches: int
    max_reqs_per_microbatch: int
    pp_size: int | None = None
    kv_free_blocks: int | None = None
    block_size: int | None = None
    max_num_batched_tokens: int | None = None
    oracle: Any | None = None


@dataclass(frozen=True)
class TaskFeatures:
    """Affine latency factors for one micro-batch task.

    SCLS (arXiv:2406.13511) fits

        T_prefill(N, L) = p1*N*L + p2*N + p3*L + p4
        τ_decode(N, l)  = d1*N*l + d2*N + d3*l + d4

    τ-Batch uses the same (B, s) pair. Here ``n`` is N/B and ``s_max`` is
    L (max prompt) or l (max context). ``s_sum`` is the unpadded token
    count vLLM actually computes.
    """

    phase: str
    n: int
    s_max: int
    s_sum: int
    tokens: int
    pp_size: int
    seq_lens: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "n": self.n,
            "s_max": self.s_max,
            "s_sum": self.s_sum,
            "tokens": self.tokens,
            "pp_size": self.pp_size,
            "seq_lens": list(self.seq_lens),
        }


def pipeline_stages(pp_size: int | None) -> int:
    """Treat a missing or non-positive pp_size as a single stage."""
    if pp_size is None or pp_size < 1:
        return 1
    return pp_size


def wait_ms(arrival_time: float, now: float) -> float:
    """Queue wait in milliseconds. Clamped at 0 if ``now`` is before arrival."""
    return max(0.0, (now - arrival_time) * 1000.0)


def ttft_slack_ms(ttft_slo_ms: float, wait_ms_value: float) -> float:
    """Remaining TTFT budget (paper ``b_i = T_ttft - w``). May be negative."""
    return ttft_slo_ms - wait_ms_value


def tau_max_ms(tpot_slo_ms: float, pp_size: int | None) -> float:
    """Per-stage decode budget (paper ``τ_max = T_tpot / M``)."""
    return tpot_slo_ms / pipeline_stages(pp_size)


def annotate_request_budget(
    req: TauRequestSnapshot, now: float, pp_size: int | None
) -> TauRequestSnapshot:
    """Fill wait and slack from ``now`` and ``pp_size``. Does not select."""
    waited = wait_ms(req.arrival_time, now)
    return replace(
        req,
        wait_ms=waited,
        ttft_slack_ms=ttft_slack_ms(req.ttft_slo_ms, waited),
        tau_max_ms=tau_max_ms(req.tpot_slo_ms, pp_size),
    )


def request_budget_dict(req: TauRequestSnapshot) -> dict[str, Any]:
    """JSON-friendly wait/slack fields for a trace line."""
    return {
        "req_id": req.request_id,
        "arrival_time": req.arrival_time,
        "wait_ms": req.wait_ms,
        "ttft_slack_ms": req.ttft_slack_ms,
        "tau_max_ms": req.tau_max_ms,
        "ttft_slo_ms": req.ttft_slo_ms,
        "tpot_slo_ms": req.tpot_slo_ms,
        "prompt_len": req.prompt_len,
    }


def estimate_kv_blocks(
    prompt_len: int, max_new_tokens: int, block_size: int
) -> int:
    """Blocks needed for prompt plus max generate length.

    Args:
        prompt_len: Prompt tokens.
        max_new_tokens: Max tokens that may be generated.
        block_size: Tokens stored in one KV block.

    Returns:
        Ceiling of ``(prompt_len + max_new_tokens) / block_size``.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    tokens = prompt_len + max_new_tokens
    return (tokens + block_size - 1) // block_size
