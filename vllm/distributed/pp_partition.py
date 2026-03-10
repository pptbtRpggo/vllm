from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.utils.mem_utils import GiB_bytes, format_gib
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size
from vllm.v1.core.kv_cache_utils import estimate_decoder_layer_kv_cache_bytes

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

AUTO_PP_LOG_PREFIX = ">>>AUTO_PP<<<"
AUTO_PP_MOCK_LOG_PREFIX = ">>>AUTO_PP_MOCK<<<"


@dataclass(slots=True)
class StageDeviceInfo:
    device_id: int
    total_memory: int
    multi_processor_count: int
    clock_rate: int
    major: int
    minor: int


@dataclass(slots=True)
class StageResourceBudget:
    stage_idx: int
    device_ids: list[int]
    total_memory_bytes: int
    usable_memory_bytes: int
    fixed_bytes: int
    weight_per_layer_bytes: int
    kv_cache_per_layer_bytes: int
    per_layer_bytes: int
    compute_score: float
    max_layers: int


def _get_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in STR_DTYPE_TO_TORCH_DTYPE:
        return STR_DTYPE_TO_TORCH_DTYPE[dtype]
    raise ValueError(f"Unsupported dtype for PP partition solver: {dtype!r}")


def _get_mock_stage_profile() -> tuple[int, float, float] | None:
    """Testing-only mock hook for heterogeneous PP budgeting.

    Enable with:
    VLLM_PP_MOCK_DEVICE_PROFILE="<stage_idx>:<memory_scale>:<clock_scale>"

    Example:
    VLLM_PP_MOCK_DEVICE_PROFILE="1:0.8:0.3"
    """
    mock_profile = os.getenv("VLLM_PP_MOCK_DEVICE_PROFILE")
    if not mock_profile:
        return None

    try:
        stage_idx_str, memory_scale_str, clock_scale_str = mock_profile.split(":")
        stage_idx = int(stage_idx_str)
        memory_scale = float(memory_scale_str)
        clock_scale = float(clock_scale_str)
    except ValueError as exc:
        raise ValueError(
            "VLLM_PP_MOCK_DEVICE_PROFILE must use the format "
            "'<stage_idx>:<memory_scale>:<clock_scale>', "
            f"but got {mock_profile!r}."
        ) from exc

    if stage_idx < 0:
        raise ValueError("Mock PP stage index must be non-negative.")
    if memory_scale <= 0 or clock_scale <= 0:
        raise ValueError("Mock memory/clock scales must be positive.")

    return stage_idx, memory_scale, clock_scale


def _maybe_apply_mock_stage_device_info(
    stage_idx: int,
    device_info: StageDeviceInfo,
) -> StageDeviceInfo:
    mock_profile = _get_mock_stage_profile()
    if mock_profile is None:
        return device_info

    mock_stage_idx, memory_scale, clock_scale = mock_profile
    if stage_idx != mock_stage_idx:
        return device_info

    mocked_info = StageDeviceInfo(
        device_id=device_info.device_id,
        total_memory=max(1, int(device_info.total_memory * memory_scale)),
        multi_processor_count=device_info.multi_processor_count,
        clock_rate=max(1, int(device_info.clock_rate * clock_scale)),
        major=device_info.major,
        minor=device_info.minor,
    )
    logger.info(
        "%s applying stage mock: stage=%d device=%d "
        "total_memory=%s->%s clock_rate_khz=%d->%d",
        AUTO_PP_MOCK_LOG_PREFIX,
        stage_idx,
        device_info.device_id,
        format_gib(device_info.total_memory),
        format_gib(mocked_info.total_memory),
        device_info.clock_rate,
        mocked_info.clock_rate,
    )
    return mocked_info


def _get_stage_device_infos(vllm_config: "VllmConfig") -> list[list[StageDeviceInfo]]:
    stage_device_map = vllm_config.parallel_config.get_pp_stage_device_map()
    assert stage_device_map is not None

    stage_infos: list[list[StageDeviceInfo]] = []
    for stage_idx, device_ids in enumerate(stage_device_map):
        one_stage: list[StageDeviceInfo] = []
        for device_id in device_ids:
            props = torch.cuda.get_device_properties(device_id)
            one_stage.append(
                _maybe_apply_mock_stage_device_info(
                    stage_idx,
                    StageDeviceInfo(
                        device_id=device_id,
                        total_memory=props.total_memory,
                        multi_processor_count=props.multi_processor_count,
                        clock_rate=props.clock_rate,
                        major=props.major,
                        minor=props.minor,
                    ),
                )
            )
        stage_infos.append(one_stage)

    if vllm_config.parallel_config.pp_auto_partition_log_details:
        for stage_idx, device_infos in enumerate(stage_infos):
            logger.info(
                "%s stage %d device inventory: %s",
                AUTO_PP_LOG_PREFIX,
                stage_idx,
                [
                    {
                        "device_id": device_info.device_id,
                        "total_memory": format_gib(device_info.total_memory),
                        "multi_processor_count": device_info.multi_processor_count,
                        "clock_rate_khz": device_info.clock_rate,
                        "capability": f"{device_info.major}.{device_info.minor}",
                    }
                    for device_info in device_infos
                ],
            )
    return stage_infos


def _estimate_decoder_layer_weight_bytes(vllm_config: "VllmConfig") -> int:
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    assert model_config is not None

    hidden_size = model_config.get_hidden_size()
    intermediate_size = getattr(
        model_config.hf_text_config, "intermediate_size", hidden_size * 4
    )
    total_num_attention_heads = getattr(
        model_config.hf_text_config,
        "num_attention_heads",
        model_config.get_num_attention_heads(parallel_config)
        * parallel_config.tensor_parallel_size,
    )
    total_num_kv_heads = model_config.get_total_num_kv_heads()
    head_size = model_config.get_head_size()
    weight_dtype = _get_torch_dtype(model_config.dtype)
    weight_dtype_size = get_dtype_size(weight_dtype)

    qkv_output_dim = (total_num_attention_heads + 2 * total_num_kv_heads) * head_size
    attn_weight_elems = hidden_size * qkv_output_dim + hidden_size * hidden_size
    ffn_weight_elems = 3 * hidden_size * intermediate_size
    return math.ceil(
        (attn_weight_elems + ffn_weight_elems)
        * weight_dtype_size
        / parallel_config.tensor_parallel_size
    )


def _estimate_stage_fixed_bytes(vllm_config: "VllmConfig", stage_idx: int) -> int:
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    assert model_config is not None

    hidden_size = model_config.get_hidden_size()
    vocab_size = model_config.get_vocab_size()
    weight_dtype = _get_torch_dtype(model_config.dtype)
    weight_dtype_size = get_dtype_size(weight_dtype)

    fixed_bytes = 0
    if stage_idx == 0:
        fixed_bytes += math.ceil(
            vocab_size
            * hidden_size
            * weight_dtype_size
            / parallel_config.tensor_parallel_size
        )
    if stage_idx == parallel_config.pipeline_parallel_size - 1:
        fixed_bytes += math.ceil(
            vocab_size
            * hidden_size
            * weight_dtype_size
            / parallel_config.tensor_parallel_size
        )
        fixed_bytes += 2 * hidden_size * weight_dtype_size
    return fixed_bytes


def _compute_stage_score(device_infos: list[StageDeviceInfo]) -> float:
    score = 0.0
    for device_info in device_infos:
        capability = device_info.major + device_info.minor / 10.0
        score += (
            max(1, device_info.multi_processor_count)
            * max(1, device_info.clock_rate)
            * max(1.0, capability)
        )
    return score


def _build_stage_budgets(vllm_config: "VllmConfig") -> list[StageResourceBudget]:
    stage_infos = _get_stage_device_infos(vllm_config)
    cache_config = vllm_config.cache_config
    parallel_config = vllm_config.parallel_config
    safety_margin_bytes = int(
        parallel_config.pp_auto_partition_safety_margin_gb * GiB_bytes
    )
    weight_per_layer_bytes = _estimate_decoder_layer_weight_bytes(vllm_config)
    kv_cache_per_layer_bytes = estimate_decoder_layer_kv_cache_bytes(vllm_config)
    per_layer_bytes = weight_per_layer_bytes + kv_cache_per_layer_bytes

    if parallel_config.pp_auto_partition_log_details:
        model_config = vllm_config.model_config
        assert model_config is not None
        logger.info(
            "%s static model budget inputs: model=%s total_layers=%d hidden_size=%d "
            "total_kv_heads=%d head_size=%d dtype=%s max_model_len=%d block_size=%s "
            "gpu_memory_utilization=%.3f safety_margin=%s weight_per_layer=%s "
            "kv_cache_per_layer=%s total_per_layer=%s",
            AUTO_PP_LOG_PREFIX,
            model_config.model,
            model_config.get_total_num_hidden_layers(),
            model_config.get_hidden_size(),
            model_config.get_total_num_kv_heads(),
            model_config.get_head_size(),
            model_config.dtype,
            model_config.max_model_len,
            cache_config.block_size,
            cache_config.gpu_memory_utilization,
            format_gib(safety_margin_bytes),
            format_gib(weight_per_layer_bytes),
            format_gib(kv_cache_per_layer_bytes),
            format_gib(per_layer_bytes),
        )

    budgets: list[StageResourceBudget] = []
    for stage_idx, device_infos in enumerate(stage_infos):
        min_total_memory = min(device_info.total_memory for device_info in device_infos)
        usable_memory_bytes = int(min_total_memory * cache_config.gpu_memory_utilization)
        usable_memory_bytes = max(0, usable_memory_bytes - safety_margin_bytes)
        fixed_bytes = _estimate_stage_fixed_bytes(vllm_config, stage_idx)
        max_layers = 0
        if usable_memory_bytes > fixed_bytes and per_layer_bytes > 0:
            max_layers = (usable_memory_bytes - fixed_bytes) // per_layer_bytes
        budgets.append(
            StageResourceBudget(
                stage_idx=stage_idx,
                device_ids=[device_info.device_id for device_info in device_infos],
                total_memory_bytes=min_total_memory,
                usable_memory_bytes=usable_memory_bytes,
                fixed_bytes=fixed_bytes,
                weight_per_layer_bytes=weight_per_layer_bytes,
                kv_cache_per_layer_bytes=kv_cache_per_layer_bytes,
                per_layer_bytes=per_layer_bytes,
                compute_score=_compute_stage_score(device_infos),
                max_layers=max_layers,
            )
        )
    return budgets


def allocate_heterogeneous_pp_partition(
    total_num_hidden_layers: int,
    max_layers_per_stage: list[int],
    stage_scores: list[float],
) -> list[int]:
    num_stages = len(max_layers_per_stage)
    if total_num_hidden_layers < num_stages:
        raise ValueError(
            "pipeline_parallel_size cannot exceed total_num_hidden_layers when "
            "using automatic heterogeneous PP partitioning."
        )
    if any(max_layers < 1 for max_layers in max_layers_per_stage):
        raise ValueError(
            "At least one PP stage does not have enough memory budget for even "
            "a single decoder layer."
        )
    if sum(max_layers_per_stage) < total_num_hidden_layers:
        raise ValueError(
            "The visible heterogeneous PP stages do not have enough aggregate "
            "memory budget for the requested model and max_model_len."
        )

    partition = [1 for _ in range(num_stages)]
    remaining_layers = total_num_hidden_layers - num_stages

    while remaining_layers > 0:
        candidates = [
            idx
            for idx, num_layers in enumerate(partition)
            if num_layers < max_layers_per_stage[idx]
        ]
        if not candidates:
            raise ValueError(
                "Unable to resolve an automatic heterogeneous PP partition "
                "within the computed stage budgets."
            )
        chosen_idx = min(
            candidates,
            key=lambda idx: partition[idx] / max(stage_scores[idx], 1e-6),
        )
        partition[chosen_idx] += 1
        remaining_layers -= 1

    return partition


def resolve_auto_pp_layer_partition(vllm_config: "VllmConfig") -> list[int]:
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    assert model_config is not None
    mock_profile = _get_mock_stage_profile()

    if parallel_config.pp_auto_partition_log_details:
        logger.info(
            "%s static budget calculation started: strategy=%s "
            "total_layers=%d pp_size=%d tp_size=%d",
            AUTO_PP_LOG_PREFIX,
            parallel_config.pp_partition_strategy,
            model_config.get_total_num_hidden_layers(),
            parallel_config.pipeline_parallel_size,
            parallel_config.tensor_parallel_size,
        )
        if mock_profile is not None:
            logger.info(
                "%s enabled via VLLM_PP_MOCK_DEVICE_PROFILE=%s",
                AUTO_PP_MOCK_LOG_PREFIX,
                os.getenv("VLLM_PP_MOCK_DEVICE_PROFILE"),
            )
    budgets = _build_stage_budgets(vllm_config)
    total_num_hidden_layers = model_config.get_total_num_hidden_layers()
    if parallel_config.pp_auto_partition_log_details:
        logger.info(
            "%s static budget calculation finished: stages=%d "
            "aggregate_max_layers=%d stage_max_layers=%s",
            AUTO_PP_LOG_PREFIX,
            len(budgets),
            sum(budget.max_layers for budget in budgets),
            [budget.max_layers for budget in budgets],
        )
    if parallel_config.pp_partition_strategy == "auto_memory":
        stage_scores = [max(1.0, float(budget.max_layers)) for budget in budgets]
    else:
        stage_scores = [max(1.0, budget.compute_score) for budget in budgets]

    partition = allocate_heterogeneous_pp_partition(
        total_num_hidden_layers,
        [budget.max_layers for budget in budgets],
        stage_scores,
    )

    if parallel_config.pp_auto_partition_log_details:
        logger.info(
            "%s resolved heterogeneous PP partition %s using strategy=%s.",
            AUTO_PP_LOG_PREFIX,
            partition,
            parallel_config.pp_partition_strategy,
        )
        for budget in budgets:
            logger.info(
                "%s stage %d devices=%s total_memory=%s usable=%s fixed=%s "
                "weight_per_layer=%s kv_cache_per_layer=%s total_per_layer=%s "
                "score=%.3f max_layers=%d",
                AUTO_PP_LOG_PREFIX,
                budget.stage_idx,
                budget.device_ids,
                format_gib(budget.total_memory_bytes),
                format_gib(budget.usable_memory_bytes),
                format_gib(budget.fixed_bytes),
                format_gib(budget.weight_per_layer_bytes),
                format_gib(budget.kv_cache_per_layer_bytes),
                format_gib(budget.per_layer_bytes),
                stage_scores[budget.stage_idx],
                budget.max_layers,
            )

    return partition
