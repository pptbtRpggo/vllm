# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.tau_batch.dispatch import (
    DispatchPhase,
    WaveDispatcher,
    WaveDispatchPolicy,
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
_none_hash_initialized = False


def _tau_scheduler(
    *,
    max_microbatches: int = 2,
    max_reqs_per_microbatch: int = 2,
    pipeline_parallel_size: int = 2,
) -> TauScheduler:
    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        max_model_len=8192,
        long_prefill_token_threshold=0,
        enable_chunked_prefill=False,
        async_scheduling=False,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(
            pipeline_parallel_size=pipeline_parallel_size
        ),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"], FullAttentionSpec(16, 1, 1, torch.float32, False)
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
    sched.max_microbatches = max_microbatches
    sched.max_reqs_per_microbatch = max_reqs_per_microbatch
    return sched


def _req(
    req_id: str,
    *,
    tpot_slo_ms: float,
    prompt_len: int = 8,
    arrival_time: float = 0.0,
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
        arrival_time=arrival_time,
        block_hasher=get_request_block_hasher(16, sha256),
    )


def _add_wave(
    sched: TauScheduler, n: int = 4, max_tokens: int = 4
) -> list[Request]:
    reqs = [
        _req(
            f"r{i}",
            tpot_slo_ms=10.0 * (i + 1),
            arrival_time=float(i),
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
    _add_wave(sched)
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
    _add_wave(sched)
    first = sched.schedule()
    second = sched.schedule()
    assert set(first.num_scheduled_tokens) == {"r0", "r1"}
    assert set(second.num_scheduled_tokens) == {"r2", "r3"}
    assert {r.request_id for r in sched.running} == {"r0", "r1", "r2", "r3"}
    assert len(sched.waiting) == 0


def test_peek_none_before_prefill_complete_does_not_commit_decode():
    sched = _tau_scheduler()
    _add_wave(sched)
    sched.schedule()
    sched.schedule()
    empty = sched.schedule()
    assert empty.total_num_scheduled_tokens == 0
    assert empty.num_scheduled_tokens == {}
    assert sched.dispatcher.peek_slot() is None


def test_overlap_decode_after_own_prefill_complete():
    sched = _tau_scheduler()
    _add_wave(sched)
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


def test_drain_waits_for_all_prefill_completes():
    sched = _tau_scheduler()
    sched.dispatcher = WaveDispatcher(WaveDispatchPolicy.DRAIN)
    _add_wave(sched)
    pre0 = sched.schedule()
    pre1 = sched.schedule()
    sched.update_from_output(pre0, _sampled(pre0))
    empty = sched.schedule()
    assert empty.total_num_scheduled_tokens == 0
    sched.update_from_output(pre1, _sampled(pre1))
    dec = sched.schedule()
    assert set(dec.num_scheduled_tokens) == {"r0", "r1"}


def test_new_arrival_does_not_join_active_wave():
    sched = _tau_scheduler()
    _add_wave(sched)
    pre0 = sched.schedule()
    extra = _req("late", tpot_slo_ms=1.0, arrival_time=99.0)
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
    nxt = sched.schedule()
    assert set(nxt.num_scheduled_tokens) == {"late"}


def test_failed_schedule_does_not_commit():
    sched = _tau_scheduler()
    _add_wave(sched)
    empty_out = SchedulerOutput.make_empty()
    with patch.object(Scheduler, "schedule", return_value=empty_out):
        empty = sched.schedule()
    assert empty.total_num_scheduled_tokens == 0
    assert len(sched.running) == 0
    assert len(sched.waiting) == 4
    slot = sched.dispatcher.peek_slot()
    assert slot is not None
    assert slot.microbatch_index == 0
    assert slot.phase is DispatchPhase.PREFILL
    out = sched.schedule()
    assert set(out.num_scheduled_tokens) == {"r0", "r1"}
