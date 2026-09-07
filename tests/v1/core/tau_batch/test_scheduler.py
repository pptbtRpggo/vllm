# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.tau_batch.dispatch import (
    DispatchPhase,
    DispatchPolicy,
    ListDispatcher,
)
from vllm.v1.core.sched.tau_batch.scheduler import TauScheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test

EOS_TOKEN_ID = 50256
TEST_MODEL = "facebook/opt-125m"
_none_hash_initialized = False


def _tau_scheduler(
    *,
    max_microbatches: int = 2,
    max_reqs_per_microbatch: int = 2,
    max_num_seqs: int = 16,
    tau_batch_max_reqs_per_microbatch: int = 4,
    tau_batch_max_microbatches: int = 0,
    override_pack_limits: bool = True,
    pipeline_parallel_size: int = 2,
    enable_chunked_prefill: bool = False,
    enable_prefix_caching: bool = False,
    async_scheduling: bool = False,
    long_prefill_token_threshold: int = 0,
    tau_batch_min_waiting: int = 0,
    tau_batch_trace: str = "",
) -> TauScheduler:
    model_config = ModelConfig(
        model=TEST_MODEL,
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=8192,
        max_model_len=8192,
        long_prefill_token_threshold=long_prefill_token_threshold,
        enable_chunked_prefill=enable_chunked_prefill,
        async_scheduling=async_scheduling,
        scheduler_cls="vllm.v1.core.sched.tau_batch.TauScheduler",
        is_encoder_decoder=model_config.is_encoder_decoder,
        tau_batch_min_waiting=tau_batch_min_waiting,
        tau_batch_max_reqs_per_microbatch=tau_batch_max_reqs_per_microbatch,
        tau_batch_max_microbatches=tau_batch_max_microbatches,
        tau_batch_trace=tau_batch_trace,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(pipeline_parallel_size=pipeline_parallel_size),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            )
        ],
    )
    cache_config.num_gpu_blocks = 10000
    sched = TauScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=16,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )
    if override_pack_limits:
        sched.max_microbatches = max_microbatches
        sched.max_reqs_per_microbatch = max_reqs_per_microbatch
    return sched


def _req(
    req_id: str,
    *,
    tpot_slo_ms: float,
    prompt_len: int = 8,
    arrival_time: float | None = None,
    max_tokens: int = 4,
) -> Request:
    global _none_hash_initialized
    if not _none_hash_initialized:
        init_none_hash(sha256)
        _none_hash_initialized = True
    sampling_params = SamplingParams(
        ignore_eos=True,
        max_tokens=max_tokens,
        extra_args={
            "ttft_slo_ms": 10_000.0,
            "tpot_slo_ms": tpot_slo_ms,
        },
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=req_id,
        prompt_token_ids=[0] * prompt_len,
        sampling_params=sampling_params,
        pooling_params=None,
        eos_token_id=EOS_TOKEN_ID,
        arrival_time=time.time() if arrival_time is None else arrival_time,
        block_hasher=get_request_block_hasher(16, sha256),
    )


def _add_requests(
    sched: TauScheduler, n: int = 4, max_tokens: int = 4
) -> list[Request]:
    reqs = [
        _req(
            f"r{i}",
            tpot_slo_ms=10.0 * (i + 1),
            max_tokens=max_tokens,
        )
        for i in range(n)
    ]
    for req in reqs:
        sched.add_request(req)
    return reqs


def _sampled(scheduler_output, token_id: int = 1) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=[[token_id] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_prefill_holdback_exposes_one_microbatch():
    sched = _tau_scheduler()
    _add_requests(sched)
    out = sched.schedule()
    assert set(out.num_scheduled_tokens) == {"r0", "r1"}
    assert out.total_num_scheduled_tokens == 16
    assert {r.request_id for r in sched.running} == {"r0", "r1"}
    assert {r.request_id for r in sched.waiting} == {"r2", "r3"}
    assert sched.dispatcher.peek_slot() is not None
    assert sched.dispatcher.peek_slot().phase is DispatchPhase.PREFILL
    assert sched.dispatcher.peek_slot().microbatch_index == 1


def test_second_prefill_does_not_mix_first_batch():
    sched = _tau_scheduler()
    _add_requests(sched)
    first = sched.schedule()
    second = sched.schedule()
    assert set(first.num_scheduled_tokens) == {"r0", "r1"}
    assert set(second.num_scheduled_tokens) == {"r2", "r3"}
    assert {r.request_id for r in sched.running} == {"r0", "r1", "r2", "r3"}
    assert len(sched.waiting) == 0


def test_peek_none_before_prefill_complete_does_not_commit_decode():
    sched = _tau_scheduler()
    _add_requests(sched)
    sched.schedule()
    sched.schedule()
    empty = sched.schedule()
    assert empty.total_num_scheduled_tokens == 0
    assert empty.num_scheduled_tokens == {}
    assert sched.dispatcher.peek_slot() is None


def test_overlap_decode_after_own_prefill_complete():
    sched = _tau_scheduler()
    _add_requests(sched)
    pre0 = sched.schedule()
    pre1 = sched.schedule()
    sched.update_from_output(pre0, _sampled(pre0))
    dec = sched.schedule()
    assert set(dec.num_scheduled_tokens) == {"r0", "r1"}
    assert all(n == 1 for n in dec.num_scheduled_tokens.values())
    assert "r2" not in dec.num_scheduled_tokens
    assert "r3" not in dec.num_scheduled_tokens
    running_ids = {r.request_id for r in sched.running}
    assert running_ids == {"r0", "r1", "r2", "r3"}
    _ = pre1


def _assert_running_sends_block_delta(sched: TauScheduler, out, req_id: str) -> None:
    cached = out.scheduled_cached_reqs
    assert req_id in cached.req_ids
    new_ids = cached.new_block_ids[cached.req_ids.index(req_id)]
    full = list(sched.kv_cache_manager.get_block_ids(req_id)[0])
    new_list = [] if new_ids is None else list(new_ids[0])
    # Worker appends these ids, then re-adds the row when B0/B1 alternate.
    # Resending the full table grows it until add_row overflows (33 into 32).
    assert len(new_list) <= 1
    if len(full) > 1:
        assert new_list != full
    if new_list:
        assert full[-len(new_list) :] == new_list


def test_cached_decode_sends_only_new_kv_blocks():
    sched = _tau_scheduler()
    _add_requests(sched, max_tokens=32)
    pre0 = sched.schedule()
    pre1 = sched.schedule()
    sched.update_from_output(pre0, _sampled(pre0))
    dec0 = sched.schedule()
    assert set(dec0.num_scheduled_tokens) == {"r0", "r1"}
    _assert_running_sends_block_delta(sched, dec0, "r0")
    sched.update_from_output(pre1, _sampled(pre1))
    sched.update_from_output(dec0, _sampled(dec0))
    dec1 = sched.schedule()
    sched.update_from_output(dec1, _sampled(dec1))
    dec0_again = sched.schedule()
    assert "r0" in dec0_again.num_scheduled_tokens
    _assert_running_sends_block_delta(sched, dec0_again, "r0")


def test_drain_waits_for_all_prefill_completes():
    sched = _tau_scheduler()
    sched.dispatcher = ListDispatcher(DispatchPolicy.DRAIN)
    _add_requests(sched)
    pre0 = sched.schedule()
    pre1 = sched.schedule()
    sched.update_from_output(pre0, _sampled(pre0))
    empty = sched.schedule()
    assert empty.total_num_scheduled_tokens == 0
    sched.update_from_output(pre1, _sampled(pre1))
    dec = sched.schedule()
    assert set(dec.num_scheduled_tokens) == {"r0", "r1"}


def test_new_arrival_does_not_join_active_list():
    sched = _tau_scheduler()
    _add_requests(sched)
    pre0 = sched.schedule()
    extra = _req("late", tpot_slo_ms=1.0)
    sched.add_request(extra)
    pre1 = sched.schedule()
    assert "late" not in pre1.num_scheduled_tokens
    assert extra.request_id in {r.request_id for r in sched.waiting}
    sched.update_from_output(pre0, _sampled(pre0))
    dec = sched.schedule()
    assert "late" not in dec.num_scheduled_tokens
    assert set(dec.num_scheduled_tokens) == {"r0", "r1"}
    for rid in ("r0", "r1", "r2", "r3"):
        sched.finish_requests(rid, RequestStatus.FINISHED_ABORTED)
    cleanup = sched.schedule()
    assert not cleanup.num_scheduled_tokens
    assert cleanup.finished_req_ids == {"r0", "r1", "r2", "r3"}
    sched.update_from_output(pre1, _sampled(pre1))
    sched.update_from_output(dec, _sampled(dec))
    sched.update_from_output(cleanup, _sampled(cleanup))
    nxt = sched.schedule()
    assert set(nxt.num_scheduled_tokens) == {"late"}


def test_allocate_fail_drops_request_and_continues():
    sched = _tau_scheduler()
    _add_requests(sched, n=2)
    real = sched.kv_cache_manager.allocate_slots
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        return real(*args, **kwargs)

    with patch.object(sched.kv_cache_manager, "allocate_slots", side_effect=flaky):
        out = sched.schedule()
    assert "r0" not in out.num_scheduled_tokens
    assert set(out.num_scheduled_tokens) == {"r1"}
    assert "r0" not in sched.requests
    assert {r.request_id for r in sched.running} == {"r1"}


def test_allocate_fail_all_returns_empty():
    sched = _tau_scheduler()
    _add_requests(sched, n=2)
    with patch.object(sched.kv_cache_manager, "allocate_slots", return_value=None):
        empty = sched.schedule()
    assert empty.total_num_scheduled_tokens == 0
    assert len(sched.running) == 0
    assert "r0" not in sched.requests
    assert "r1" not in sched.requests


def test_init_disables_features_that_split_prefill():
    sched = _tau_scheduler(
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        long_prefill_token_threshold=4,
    )
    assert sched.scheduler_config.enable_chunked_prefill is False
    assert sched.scheduler_config.long_prefill_token_threshold == 0
    assert sched.cache_config.enable_prefix_caching is False
    assert sched.kv_cache_manager.enable_caching is False
    assert sched.vllm_config.speculative_config is None
    assert sched.num_spec_tokens == 0


def test_init_disables_async_scheduling():
    # Normalize before VllmConfig would reject async scheduling with PP.
    sched = _tau_scheduler(
        pipeline_parallel_size=2,
        async_scheduling=True,
    )
    assert sched.scheduler_config.async_scheduling is False


def test_disabled_long_prefill_does_not_cap_tokens():
    sched = _tau_scheduler(
        enable_chunked_prefill=True,
        long_prefill_token_threshold=4,
    )
    _add_requests(sched, n=2)
    out = sched.schedule()
    assert set(out.num_scheduled_tokens) == {"r0", "r1"}
    assert all(n == 8 for n in out.num_scheduled_tokens.values())
    assert out.total_num_scheduled_tokens == 16


def test_list_end_resets_dispatcher():
    sched = _tau_scheduler()
    _add_requests(sched, n=2)
    out = sched.schedule()
    assert sched._list is not None
    assert sched._wave_id == 0
    for rid in out.num_scheduled_tokens:
        sched.finish_requests(rid, RequestStatus.FINISHED_ABORTED)
    empty = sched.schedule()
    assert empty.total_num_scheduled_tokens == 0
    assert sched._list is not None
    sched.update_from_output(out, _sampled(out))
    sched.update_from_output(empty, _sampled(empty))
    assert sched._list is None
    assert sched._wave_id is None
    assert sched.dispatcher.active_list is None
    assert sched.dispatcher.peek_slot() is None


def test_eos_hook_runs_when_decode_finishes():
    seen: list = []

    class Capture:
        def on_eos(self, event) -> None:
            seen.append(event)

    sched = _tau_scheduler()
    sched.planner.eos_strategy = Capture()
    _add_requests(sched, n=2, max_tokens=2)
    pre = sched.schedule()
    sched.update_from_output(pre, _sampled(pre))
    assert seen == []
    assert sched._list is not None
    dec = sched.schedule()
    assert set(dec.num_scheduled_tokens) == {"r0", "r1"}
    sched.update_from_output(dec, _sampled(dec))
    assert len(seen) == 1
    assert set(seen[0].finished_ids) == {"r0", "r1"}
    assert seen[0].remaining_ids == ()
    assert seen[0].phase == "decode"


def test_min_waiting_to_plan_holds_until_threshold():
    sched = _tau_scheduler(tau_batch_min_waiting=8)
    _add_requests(sched, n=4)
    empty = sched.schedule()
    assert empty.total_num_scheduled_tokens == 0
    assert sched._list is None
    assert sched._wave_id is None
    assert len(sched.waiting) == 4
    extra = [_req(f"r{i}", tpot_slo_ms=10.0 * (i + 1)) for i in range(4, 8)]
    for req in extra:
        sched.add_request(req)
    out = sched.schedule()
    assert out.total_num_scheduled_tokens > 0
    assert sched._list is not None
    assert sched._wave_id == 0
    assert len(sched._list.admitted_ids) + len(sched._list.deferred_ids) == 8


def test_pack_context_zero_list_cap_is_unlimited():
    sched = _tau_scheduler(
        max_num_seqs=32,
        tau_batch_max_reqs_per_microbatch=4,
        tau_batch_max_microbatches=0,
        override_pack_limits=False,
    )
    ctx = sched._pack_context()
    assert ctx.max_num_seqs == 32
    assert ctx.max_reqs_per_microbatch == 4
    assert ctx.max_microbatches == 0


def test_pack_context_honors_explicit_p():
    sched = _tau_scheduler(
        max_num_seqs=32,
        tau_batch_max_reqs_per_microbatch=4,
        tau_batch_max_microbatches=2,
        override_pack_limits=False,
    )
    ctx = sched._pack_context()
    assert ctx.max_reqs_per_microbatch == 4
    assert ctx.max_microbatches == 2


def test_scheduler_packs_full_prompts_within_token_budget():
    sched = _tau_scheduler(max_reqs_per_microbatch=3)
    sched.max_num_scheduled_tokens = 10
    for rid, length in [("a", 7), ("b", 6), ("c", 3)]:
        sched.add_request(_req(rid, tpot_slo_ms=100, prompt_len=length))
    first = sched.schedule()
    second = sched.schedule()
    assert first.num_scheduled_tokens == {"a": 7, "c": 3}
    assert second.num_scheduled_tokens == {"b": 6}
    assert not sched.scheduler_config.enable_chunked_prefill


def test_emit_validation_precedes_allocation_and_state_mutation():
    sched = _tau_scheduler()
    requests = _add_requests(sched, n=2)
    sched._ensure_active_list()
    # Simulate a stale/custom plan after runtime capacity changes.
    sched.max_num_scheduled_tokens = 12
    with patch.object(sched.kv_cache_manager, "allocate_slots") as allocate:
        with pytest.raises(ValueError, match="forward has 16 tokens"):
            sched.schedule()
        allocate.assert_not_called()
    assert all(req.num_computed_tokens == 0 for req in requests)
    assert len(sched.waiting) == 2
    assert not sched.running
    assert sched.dispatcher.peek_slot().phase is DispatchPhase.PREFILL


@pytest.mark.parametrize("policy", list(DispatchPolicy))
def test_cancelled_prefill_batch_cannot_stall_surviving_requests(policy):
    sched = _tau_scheduler()
    sched.dispatcher = ListDispatcher(policy)
    _add_requests(sched, max_tokens=4)
    pre = sched.schedule()
    sched.update_from_output(pre, _sampled(pre))
    sched.finish_requests(["r2", "r3"], RequestStatus.FINISHED_ABORTED)
    for _ in range(3):
        out = sched.schedule()
        assert out.num_scheduled_tokens == {"r0": 1, "r1": 1}
        sched.update_from_output(out, _sampled(out))
    assert not sched.requests
    assert sched._list is None


def test_empty_allocation_batch_cannot_stall_other_batches():
    sched = _tau_scheduler()
    _add_requests(sched, max_tokens=2)
    pre = sched.schedule()
    sched.update_from_output(pre, _sampled(pre))
    with patch.object(sched.kv_cache_manager, "allocate_slots", return_value=None):
        assert sched.schedule().total_num_scheduled_tokens == 0
    out = sched.schedule()
    assert out.num_scheduled_tokens == {"r0": 1, "r1": 1}


def test_config_normalized_before_vllm_validation():
    original = VllmConfig.try_verify_and_update_config
    seen = []

    def check(config):
        seen.append(
            (
                config.scheduler_config.enable_chunked_prefill,
                config.scheduler_config.async_scheduling,
                config.cache_config.enable_prefix_caching,
            )
        )
        return original(config)

    with patch.object(VllmConfig, "try_verify_and_update_config", check):
        _tau_scheduler(
            enable_chunked_prefill=True,
            async_scheduling=True,
            enable_prefix_caching=True,
        )
    assert seen and all(flags == (False, False, False) for flags in seen)


def test_config_clears_speculation_before_worker_initialization():
    from vllm.v1.core.sched.tau_batch.config import configure_tau_batch

    sched = _tau_scheduler()
    config = sched.vllm_config
    config.speculative_config = SimpleNamespace(num_speculative_tokens=4)
    configure_tau_batch(config)
    assert config.speculative_config is None


def test_late_config_change_rejected_instead_of_silently_mutated():
    sched = _tau_scheduler()
    sched.vllm_config.cache_config.enable_prefix_caching = True
    with pytest.raises(ValueError, match="before worker startup"):
        TauScheduler(vllm_config=sched.vllm_config)
    assert sched.vllm_config.cache_config.enable_prefix_caching is True


def test_pipeline_snapshot_separates_waiting_and_admitted():
    sched = _tau_scheduler()
    _add_requests(sched)
    first = sched.schedule()
    sched.add_request(_req("late", tpot_slo_ms=100))
    snapshot = sched.pipeline_snapshot()
    assert [r.request_id for r in snapshot.waiting] == ["late"]
    assert snapshot.active[0].in_flight == 1
    assert snapshot.active[0].prefill_dispatched
    assert not snapshot.active[0].prefill_completed

    assert not snapshot.active[1].prefill_dispatched
    assert snapshot.active[1].requests[0].kv_block_ids == ((),)
    sched.update_from_output(first, _sampled(first))
    updated = sched.pipeline_snapshot()
    assert updated.active[0].prefill_completed
    assert updated.active[0].in_flight == 0
    assert updated.active[0].requests[0].output_tokens == 1
    assert not snapshot.active[0].prefill_completed


def test_pipeline_snapshot_does_not_share_nested_strategy_metadata():
    sched = _tau_scheduler()
    _add_requests(sched)
    sched.schedule()
    sched._list.extra["ceiling"] = {"decode_ms": [10.0]}
    snapshot = sched.pipeline_snapshot()
    snapshot.active_plan.extra["ceiling"]["decode_ms"][0] = 20.0
    assert sched._list.extra["ceiling"]["decode_ms"] == [10.0]


def test_latency_oracle_reaches_packing_context():
    class Oracle:
        def predict(self, features, *, pp_rank):
            from vllm.v1.core.sched.tau_batch.interfaces import LatencyEstimate

            return LatencyEstimate(mean_ms=features.n + pp_rank)

    sched = _tau_scheduler()
    sched.latency_oracle = Oracle()
    assert sched._pack_context().oracle is sched.latency_oracle
