# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


def _validate_supported_config(config: "VllmConfig") -> None:
    unsupported = []
    for name in ("lora_config", "kv_transfer_config", "ec_transfer_config"):
        if getattr(config, name) is not None:
            unsupported.append(name)
    model = config.model_config
    if model is not None:
        if model.runner_type != "generate":
            unsupported.append("non-generation model")
        if model.is_encoder_decoder:
            unsupported.append("encoder-decoder model")
        if model.is_multimodal_model:
            unsupported.append("multimodal model")
    parallel = config.parallel_config
    for name in (
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ):
        if getattr(parallel, name) != 1:
            unsupported.append(name)
    if unsupported:
        raise ValueError(
            "TauScheduler currently supports text generation with PP/TP only; "
            "unsupported configuration: " + ", ".join(unsupported)
        )


def validate_tau_kv_layout(config: "KVCacheConfig", block_size: int) -> None:
    """The reservation formula counts one ordinary full-attention block group."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    groups = config.kv_cache_groups
    if len(groups) != 1:
        raise ValueError("TauScheduler requires exactly one full-attention KV group")
    spec = groups[0].kv_cache_spec
    if (
        type(spec) is not FullAttentionSpec
        or spec.sliding_window is not None
        or spec.attention_chunk_size is not None
        or spec.block_size != block_size
    ):
        raise ValueError(
            "TauScheduler requires ordinary full-attention KV with the same "
            "block_size as the scheduler; this KV layout is not supported"
        )


def _unsupported_fields(config: "VllmConfig") -> list[tuple[object, str, object]]:
    sched = config.scheduler_config
    expected = [
        (sched, "enable_chunked_prefill", False),
        (sched, "long_prefill_token_threshold", 0),
        (sched, "max_num_partial_prefills", 1),
        (sched, "max_long_partial_prefills", 1),
        (sched, "async_scheduling", False),
        (config.cache_config, "enable_prefix_caching", False),
        (config, "speculative_config", None),
    ]
    return [
        (obj, name, value)
        for obj, name, value in expected
        if getattr(obj, name) != value
    ]


def configure_tau_batch(config: "VllmConfig") -> None:
    """Normalize the full-slot contract during VllmConfig initialization.

    This runs before async/PP validation, profiling, cache allocation,
    executor creation and worker serialization.
    """
    _validate_supported_config(config)
    changes = _unsupported_fields(config)
    for obj, name, value in changes:
        setattr(obj, name, value)
    if changes:
        logger.warning(
            "TauScheduler disabled unsupported features: %s",
            ", ".join(name for _, name, _ in changes),
        )
    if config.model_config is not None:
        config.scheduler_config.verify_max_model_len(config.model_config.max_model_len)


def validate_tau_batch_config(config: "VllmConfig") -> None:
    """Reject late changes instead of silently diverging from worker config."""
    _validate_supported_config(config)
    changes = _unsupported_fields(config)
    if changes:
        names = ", ".join(name for _, name, _ in changes)
        raise ValueError(
            f"TauScheduler requires configuration before worker startup: {names}. "
            "Set scheduler_cls to TauScheduler when constructing VllmConfig."
        )
