# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


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
    changes = _unsupported_fields(config)
    if changes:
        names = ", ".join(name for _, name, _ in changes)
        raise ValueError(
            f"TauScheduler requires configuration before worker startup: {names}. "
            "Set scheduler_cls to TauScheduler when constructing VllmConfig."
        )
