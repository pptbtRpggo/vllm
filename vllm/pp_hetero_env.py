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


def hetero_env_requested() -> bool:
    """True when the user asked for PP hetero emulation or stage tracing."""
    return bool(
        os.environ.get("VLLM_PP_HETERO")
        or os.environ.get("VLLM_PP_COMPUTE_SCALE")
        or os.environ.get("VLLM_PP_COMM_SCALE")
        or os.environ.get("VLLM_PP_STAGE_TRACE")
    )


def maybe_override_pp_worker(parallel_config: Any) -> None:
    """Use ``PPAscendWorker`` when hetero/trace is on and the platform picked NPUWorker.

    CUDA ``gpu_worker.Worker`` already stretches, so it is left alone.
    Must run *after* ``Platform.check_and_update_config``.
    """
    if not hetero_env_requested():
        return
    current = getattr(parallel_config, "worker_cls", None)
    if current == PP_ASCEND_WORKER:
        return
    current_s = "" if current is None else str(current)
    is_auto = current_s in ("", "auto")
    is_npu = any(
        marker in current_s.lower()
        for marker in ("vllm_ascend", "npuworker", "npu_worker")
    )
    if not is_auto and not is_npu:
        return
    if is_auto:
        try:
            import vllm_ascend  # noqa: F401
        except ImportError:
            return
    parallel_config.worker_cls = PP_ASCEND_WORKER
    logger.info("PP hetero/trace enabled: worker_cls=%s", PP_ASCEND_WORKER)
