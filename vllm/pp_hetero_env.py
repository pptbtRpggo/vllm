# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Env-driven PP worker selection. Kept out of ``vllm.distributed`` so
``VllmConfig`` can call it without importing ``distributed/__init__.py``
(that would start OpenMP threads before mp fork).
"""

from __future__ import annotations

import os
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

PP_ASCEND_WORKER = "vllm.v1.worker.pp_ascend_worker.PPAscendWorker"
TP_ASCEND_WORKER = "vllm.v1.worker.tp_ascend_worker.TPAscendWorker"


def hetero_env_requested() -> bool:
    """True when the user asked for PP hetero emulation or stage tracing."""
    return bool(
        os.environ.get("VLLM_PP_HETERO")
        or os.environ.get("VLLM_PP_STAGE_TRACE")
        or os.environ.get("VLLM_PP_NETWORK")
    )


def maybe_override_pp_worker(parallel_config: Any) -> None:
    """Select an Ascend PP or TP experiment worker after platform setup."""
    tp_requested = bool(
        os.environ.get("VLLM_TP_COMPUTE_SCALES")
        or os.environ.get("VLLM_TP_CROSS_EXTRA_BANDWIDTH_GBPS")
    )
    if tp_requested and hetero_env_requested():
        raise ValueError("TP and PP heterogeneity settings cannot be combined")
    if not tp_requested and not hetero_env_requested():
        return
    if tp_requested and getattr(parallel_config, "worker_cls", None) == TP_ASCEND_WORKER:
        return
    # The platform has already resolved "auto". Replace only its built-in
    # worker; custom worker names/classes must not be matched by substring.
    if getattr(parallel_config, "worker_cls", None) != "vllm_ascend.worker.worker.NPUWorker":
        if tp_requested:
            raise ValueError("TP heterogeneity currently requires Ascend NPUWorker")
        return
    parallel_config.worker_cls = TP_ASCEND_WORKER if tp_requested else PP_ASCEND_WORKER
    logger.info("heterogeneity worker_cls=%s", parallel_config.worker_cls)
