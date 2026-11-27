# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Agreement between the two independently configured HCCL launchers."""

import os
from pathlib import Path
from typing import Any

import regex as re


def _first_difference(expected, actual, prefix=""):
    for key in sorted(expected.keys() | actual.keys()):
        name = prefix + key
        first, second = expected.get(key), actual.get(key)
        if key not in expected or key not in actual:
            return name, first, second
        if isinstance(first, dict) and isinstance(second, dict):
            difference = _first_difference(first, second, name + ".")
            if difference is not None:
                return difference
        elif first != second:
            return name, first, second
    return None


def validate_native_reports(reports):
    """Reject mismatched effective options before any rank loads weights."""
    if len(reports) != 4 or [r.get("rank") for r in reports] != list(range(4)):
        raise ValueError("HCCL startup agreement requires reports from ranks 0–3")
    reference = reports[0]
    for rank, report in enumerate(reports):
        difference = _first_difference(reference["settings"], report["settings"])
        if difference is not None:
            field, head, worker = difference
            raise ValueError(
                f"HCCL startup mismatch at {field}: "
                f"rank 0={head!r}, rank {rank}={worker!r}"
            )
    head_checkpoint = reference.get("checkpoint")
    for rank, report in enumerate(reports):
        digest = report.get("checkpoint")
        # Direct Python users hash once on the first rank of each node.
        if (
            digest is None
            and rank in (1, 3)
            and reference["settings"]["mode"] == "direct"
        ):
            continue
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError(f"rank {rank} lacks a valid HCCL checkpoint digest")
        if digest != head_checkpoint:
            raise ValueError(f"HCCL checkpoint content differs on rank {rank}")


def check_native_startup(worker, plan):
    """Use the existing CPU control group to compare the four startup reports."""
    import torch.distributed as dist

    from vllm.distributed.hetero_pp.profiling import serving_config_observation
    from vllm.distributed.parallel_state import get_pp_group

    cfg = worker.vllm_config
    group = get_pp_group()
    if group.ranks != list(range(4)) or group.rank_in_group != worker.rank:
        raise ValueError("HCCL startup agreement requires a four-rank PP group")
    settings = serving_config_observation(worker)
    settings["model"].pop("model")
    settings["cache"].pop("num_gpu_blocks")
    mode = os.environ.get("HETERO_PP_RUN_MODE", "direct")
    if mode not in ("profile", "serve", "direct"):
        raise ValueError(f"invalid HCCL startup mode: {mode!r}")
    settings.update(
        mode=mode,
        use_v2_model_runner=cfg.use_v2_model_runner,
        seed=cfg.model_config.seed,
        stages={
            str(i): dict(
                node_ip=stage.node_ip,
                soc=stage.soc,
                layers=stage.layers,
                npu_core_fraction=stage.npu_core_fraction,
            )
            for i, stage in enumerate(plan.stages)
        },
    )
    if mode == "profile":
        settings["profile_compute_timing"] = os.environ.get(
            "HETERO_PP_PROFILE_COMPUTE_TIMING"
        )
    digest = os.environ.get("HETERO_PP_CHECKPOINT_SHA256")
    if digest is None and mode == "direct" and worker.rank in (0, 2):
        from vllm.distributed.hetero_pp.checkpoint_identity import checkpoint_identity

        digest = checkpoint_identity(Path(cfg.model_config.model))
    reports: list[dict[str, Any]] = [{} for _ in range(4)]
    dist.all_gather_object(
        reports,
        dict(rank=worker.rank, settings=settings, checkpoint=digest),
        group=group.cpu_group,
    )
    validate_native_reports(reports)
    return reports[0]["checkpoint"]
