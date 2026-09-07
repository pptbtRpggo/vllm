# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.v1.core.tau_batch.test_scheduler import _req, _sampled, _tau_scheduler
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.core.sched.tau_batch.config import (
    configure_tau_batch,
    validate_tau_batch_config,
    validate_tau_kv_layout,
)
from vllm.v1.core.sched.tau_batch.scheduler import TauScheduler
from vllm.v1.engine import FinishReason
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output.request import StructuredOutputRequest

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "field", ["lora_config", "kv_transfer_config", "ec_transfer_config"]
)
@pytest.mark.parametrize("validate", [configure_tau_batch, validate_tau_batch_config])
def test_unsupported_config_is_rejected_in_early_and_late_checks(field, validate):
    config = _tau_scheduler().vllm_config
    setattr(config, field, SimpleNamespace())
    with pytest.raises(ValueError, match=field):
        validate(config)


@pytest.mark.parametrize(
    "field",
    [
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ],
)
def test_context_or_data_parallel_requires_a_different_execution_contract(field):
    config = _tau_scheduler().vllm_config
    setattr(config.parallel_config, field, 2)
    with pytest.raises(ValueError, match=field):
        configure_tau_batch(config)


@pytest.mark.parametrize(
    "model,reason",
    [
        (
            SimpleNamespace(
                runner_type="pooling",
                is_encoder_decoder=False,
                is_multimodal_model=False,
            ),
            "non-generation",
        ),
        (
            SimpleNamespace(
                runner_type="generate",
                is_encoder_decoder=True,
                is_multimodal_model=False,
            ),
            "encoder-decoder",
        ),
        (
            SimpleNamespace(
                runner_type="generate",
                is_encoder_decoder=False,
                is_multimodal_model=True,
            ),
            "multimodal",
        ),
    ],
)
def test_unsupported_models_rejected_before_normalizing_config(model, reason):
    config = _tau_scheduler().vllm_config
    config.model_config = model
    with pytest.raises(ValueError, match=reason):
        configure_tau_batch(config)


@pytest.mark.parametrize(
    "layout", ["multiple_groups", "no_groups", "window", "block_size"]
)
def test_unsupported_kv_layout_rejected_on_scheduler_construction(layout):
    sched = _tau_scheduler()
    config = sched.kv_cache_config
    group = config.kv_cache_groups[0]
    if layout == "multiple_groups":
        config = replace(config, kv_cache_groups=[group, group])
    elif layout == "no_groups":
        config = replace(config, kv_cache_groups=[])
    elif layout == "window":
        config = replace(
            config,
            kv_cache_groups=[
                replace(
                    group,
                    kv_cache_spec=replace(group.kv_cache_spec, sliding_window=128),
                )
            ],
        )
    else:
        config = replace(
            config,
            kv_cache_groups=[
                replace(
                    group, kv_cache_spec=replace(group.kv_cache_spec, block_size=32)
                )
            ],
        )
    with pytest.raises(ValueError, match="TauScheduler requires"):
        validate_tau_kv_layout(config, 16)
    # The scheduler must enforce the same contract, not just export a helper.
    with pytest.raises(ValueError, match="TauScheduler requires"):
        TauScheduler(
            vllm_config=sched.vllm_config,
            kv_cache_config=config,
            structured_output_manager=sched.structured_output_manager,
            block_size=16,
        )


@pytest.mark.parametrize(
    "feature", ["structured", "lora", "multimodal", "pooling", "transfer"]
)
def test_unsupported_request_finishes_without_becoming_a_cached_forward(feature):
    sched = _tau_scheduler()
    req = _req("unsupported", tpot_slo_ms=10)
    req.client_index = 3
    if feature == "structured":
        req.structured_output_request = StructuredOutputRequest(
            params=StructuredOutputsParams(json_object=True)
        )
        req.status = RequestStatus.WAITING_FOR_FSM
    elif feature == "lora":
        req.lora_request = SimpleNamespace()
    elif feature == "multimodal":
        req.has_encoder_inputs = True
    elif feature == "pooling":
        req.pooling_params = SimpleNamespace()
    else:
        req.kv_transfer_params = {"remote": True}
    sched.add_request(req)
    with patch.object(sched.kv_cache_manager, "allocate_slots") as allocate:
        output = sched.schedule()
        allocate.assert_not_called()
    assert not output.num_scheduled_tokens
    assert not output.scheduled_new_reqs
    assert not output.scheduled_cached_reqs.req_ids
    assert output.finished_req_ids == {"unsupported"}
    result = sched.update_from_output(output, _sampled(output))
    assert len(result[3].outputs) == 1
    assert result[3].outputs[0].finish_reason == FinishReason.ERROR
    assert req.status == RequestStatus.FINISHED_ERROR
    assert not sched.has_requests()
    # A rejected request must not stop ordinary requests being served next.
    sched.add_request(_req("ok", tpot_slo_ms=10))
    assert sched.schedule().num_scheduled_tokens == {"ok": 8}


def test_vllm_config_construction_enforces_unsupported_feature_guard():
    from vllm.config import LoRAConfig

    config = _tau_scheduler().vllm_config
    with pytest.raises(ValueError, match="lora_config"):
        replace(config, lora_config=LoRAConfig(max_loras=1))


def test_custom_kv_overcommit_rejected_before_any_runtime_allocation():
    from vllm.v1.core.sched.tau_batch import GreedyListStrategy, TauBatchPlanner

    class IgnoreKv(GreedyListStrategy):
        def pack(self, requests, ctx):
            return super().pack(requests, replace(ctx, kv_free_blocks=None))

    sched = _tau_scheduler()
    sched.planner = TauBatchPlanner(strategy=IgnoreKv())
    requests = [
        _req(rid, prompt_len=32, max_tokens=32, tpot_slo_ms=10) for rid in ("a", "b")
    ]
    for req in requests:
        sched.add_request(req)
    with (
        patch.object(
            sched.kv_cache_manager.block_pool, "get_num_free_blocks", return_value=4
        ),
        patch.object(sched.kv_cache_manager, "allocate_slots") as allocate,
    ):
        with pytest.raises(ValueError, match="reserves 8 KV blocks, only 4"):
            sched.schedule()
        allocate.assert_not_called()
    assert all(req.status == RequestStatus.WAITING for req in requests)
    assert not sched.running and sched._list is None
