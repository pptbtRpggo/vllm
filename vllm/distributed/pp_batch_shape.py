# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Describe a scheduled PP batch without inventing a batch-wide phase.

Only lengths are recorded, never request IDs or token contents. Context means
the computed prefix before this step, not the total prompt length.
"""

import hashlib
import json
from typing import Any


def scheduled_batch_id(scheduler: Any) -> str:
    """Hash request identities/positions, without storing IDs or token contents.

    Pair this with the common session ID and step ordinal. The ordinal
    distinguishes repeated work after preemption; the hash detects misalignment.
    """
    positions = {r.req_id: r.num_computed_tokens for r in scheduler.scheduled_new_reqs}
    cached = scheduler.scheduled_cached_reqs
    positions.update(zip(cached.req_ids, cached.num_computed_tokens))
    items = sorted(
        (key, int(count), positions.get(key))
        for key, count in scheduler.num_scheduled_tokens.items()
    )
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


def scheduled_batch_shape(scheduler: Any, runner: Any) -> dict[str, Any]:
    """Snapshot before execute_model mutates request state.

    CachedRequestData carries the current computed position (including resets
    on resume); the runner cache is only used for the prompt length. Missing
    state is explicitly unknown, never implicitly decode. Speculative/async
    execution needs runner-resolved positions and is not supported here yet.
    """
    unknown = dict(
        batch_shape=None,
        num_ctx_requests=0,
        num_ctx_tokens=0,
        num_generation_requests=0,
        num_generation_tokens=0,
    )
    config = getattr(getattr(runner, "vllm_config", None), "scheduler_config", None)
    if getattr(config, "async_scheduling", False) or getattr(
        scheduler, "scheduled_spec_decode_tokens", None
    ):
        return unknown
    new = {req.req_id: req for req in scheduler.scheduled_new_reqs}
    cached = getattr(scheduler, "scheduled_cached_reqs", None)
    positions = dict(zip(cached.req_ids, cached.num_computed_tokens)) if cached else {}
    states = getattr(runner, "requests", {})
    shape = []
    for req_id, query in scheduler.num_scheduled_tokens.items():
        state = new.get(req_id) or states.get(req_id)
        context = (
            getattr(state, "num_computed_tokens", None)
            if req_id in new
            else positions.get(req_id)
        )
        if state is None or context is None or getattr(state, "mm_features", None):
            return unknown
        prompt = getattr(state, "prompt_token_ids", None)
        embeds = getattr(state, "prompt_embeds", None)
        if prompt is not None:
            prompt_len = len(prompt)
        elif embeds is not None:
            prompt_len = int(embeds.shape[0])
        else:
            return unknown
        if query <= 0 or context < 0:
            return unknown
        shape.append(
            dict(
                query_tokens=int(query),
                context_tokens=int(context),
                prompt_tokens=min(int(query), max(0, prompt_len - int(context))),
            )
        )
    return dict(
        batch_shape=shape,
        num_ctx_requests=sum(s["prompt_tokens"] > 0 for s in shape),
        num_ctx_tokens=sum(s["prompt_tokens"] for s in shape),
        num_generation_requests=sum(
            s["query_tokens"] > s["prompt_tokens"] for s in shape
        ),
        num_generation_tokens=sum(
            s["query_tokens"] - s["prompt_tokens"] for s in shape
        ),
    )
