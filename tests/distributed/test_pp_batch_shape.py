# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace as NS

from vllm.distributed.pp_batch_shape import scheduled_batch_shape


def test_mixed_batch_uses_progress_not_request_newness():
    runner = NS(
        requests={
            "prefill": NS(prompt_token_ids=[0] * 128, num_computed_tokens=99),
            "decode": NS(prompt_token_ids=[0] * 16),
        }
    )
    scheduler = NS(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=NS(
            req_ids=["prefill", "decode"], num_computed_tokens=[64, 1024]
        ),
        num_scheduled_tokens={"prefill": 32, "decode": 1},
    )
    result = scheduled_batch_shape(scheduler, runner)
    assert result["batch_shape"] == [
        dict(query_tokens=32, context_tokens=64, prompt_tokens=32),
        dict(query_tokens=1, context_tokens=1024, prompt_tokens=0),
    ]
    assert result["num_ctx_tokens"] == 32
    assert result["num_generation_tokens"] == 1


def test_prefix_hit_resume_and_prompt_boundary():
    # Cached progress may reset on resume; never reuse runner's stale position.
    runner = NS(
        requests={"resume": NS(prompt_token_ids=[0] * 10, num_computed_tokens=100)}
    )
    scheduler = NS(
        scheduled_new_reqs=[
            NS(req_id="new", prompt_token_ids=[0] * 16, num_computed_tokens=15)
        ],
        scheduled_cached_reqs=NS(req_ids=["resume"], num_computed_tokens=[8]),
        num_scheduled_tokens={"new": 1, "resume": 3},
    )
    result = scheduled_batch_shape(scheduler, runner)
    assert result["batch_shape"][0]["context_tokens"] == 15
    assert result["batch_shape"][1] == dict(
        query_tokens=3, context_tokens=8, prompt_tokens=2
    )
    assert result["num_ctx_tokens"] == 3
    assert result["num_generation_tokens"] == 1
    # A request crossing the prompt boundary contributes to both counters.
    assert result["num_ctx_requests"] == 2
    assert result["num_generation_requests"] == 1


def test_missing_state_and_speculation_are_not_marked_decode():
    scheduler = NS(scheduled_new_reqs=[], num_scheduled_tokens={"unknown": 1})
    result = scheduled_batch_shape(scheduler, NS())
    assert result["batch_shape"] is None
    assert result["num_generation_tokens"] == 0
    scheduler.scheduled_spec_decode_tokens = {"unknown": [1, 2]}
    assert scheduled_batch_shape(scheduler, NS())["batch_shape"] is None


def test_prompt_embeds_lengths():
    scheduler = NS(
        scheduled_new_reqs=[
            NS(
                req_id="r",
                prompt_token_ids=None,
                prompt_embeds=NS(shape=(10, 4096)),
                num_computed_tokens=5,
            )
        ],
        num_scheduled_tokens={"r": 3},
    )
    assert scheduled_batch_shape(scheduler, NS())["batch_shape"] == [
        dict(query_tokens=3, context_tokens=5, prompt_tokens=3)
    ]


def test_async_positions_are_explicitly_unknown():
    runner = NS(vllm_config=NS(scheduler_config=NS(async_scheduling=True)))
    assert scheduled_batch_shape(NS(), runner)["batch_shape"] is None
