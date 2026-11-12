# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch-free, fail-closed admission for the first heterogeneous PP backend."""

import math
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    node_ip: str
    soc: str
    layers: int


@dataclass(frozen=True)
class PipelinePlan:
    stages: tuple[Stage, ...]
    timeout_s: float = 120.0
    validation_mode: str = "mixed"

    @classmethod
    def from_dict(cls, value, total_layers):
        if set(value) - {"stages", "timeout_s", "validation_mode"}:
            raise ValueError("unknown pipeline plan fields")
        raw = value.get("stages", [])
        if len(raw) != 4:
            raise ValueError("first backend requires four stages")
        stages = []
        for item in raw:
            if set(item) != {"node_ip", "soc", "layers"}:
                raise ValueError("stage requires node_ip, soc and layers")
            if not isinstance(item["node_ip"], str) or not item["node_ip"].strip():
                raise ValueError("node_ip must be nonempty")
            if item["soc"] not in ("910B", "310P"):
                raise ValueError("only 910B and 310P are admitted")
            if type(item["layers"]) is not int or item["layers"] <= 0:
                raise ValueError("each stage must have positive integer layers")
            stages.append(Stage(**item))
        if sum(s.layers for s in stages) != total_layers:
            raise ValueError("stage layers must sum to model hidden layers")
        mode = value.get("validation_mode", "mixed")
        expected = ["910B", "910B", "310P", "310P"] if mode == "mixed" else ["910B"] * 4
        if mode not in ("mixed", "910b_only"):
            raise ValueError("unknown validation_mode")
        if not (
            stages[0].node_ip == stages[1].node_ip
            and stages[2].node_ip == stages[3].node_ip
            and stages[0].node_ip != stages[2].node_ip
            and [s.soc for s in stages] == expected
        ):
            raise ValueError(
                "two stages per node and SoC layout must match validation_mode"
            )
        timeout = value.get("timeout_s", 120.0)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or not 0 < timeout <= 1800
        ):
            raise ValueError("timeout_s must be in (0, 1800]")
        return cls(tuple(stages), float(timeout), mode)

    @property
    def layer_ranges(self):
        end = 0
        result = []
        for s in self.stages:
            result.append((end, end + s.layers))
            end += s.layers
        return tuple(result)

    @property
    def partition(self):
        return ",".join(str(s.layers) for s in self.stages)

    @property
    def nodes(self):
        return tuple(dict.fromkeys(s.node_ip for s in self.stages))


def select_worker(expected_soc, local_worker):
    supported = {
        "910B": "vllm_ascend.worker.worker.NPUWorker",
        "310P": "vllm_ascend._310p.worker_310p.NPUWorker310",
    }
    if expected_soc not in supported or local_worker != supported[expected_soc]:
        raise ValueError(
            f"local hardware worker {local_worker} does not match {expected_soc}"
        )
    return "vllm.distributed.hetero_pp.worker.Staged" + expected_soc + "Worker"


def worker_for_soc_name(name):
    """Legacy plugin compatibility, selected from the local ACL SoC name."""
    import regex as re

    if re.fullmatch(r"Ascend910B(?:[0-9]+(?:-[0-9]+)?)?", name, re.IGNORECASE):
        return "vllm_ascend.worker.worker.NPUWorker"
    if re.fullmatch(r"Ascend310P(?:[0-9]+(?:-[0-9]+)?)?", name, re.IGNORECASE):
        return "vllm_ascend._310p.worker_310p.NPUWorker310"
    raise ValueError(f"unsupported local ACL SoC: {name!r}")


def validate_runtime(cfg, size):
    p, m = cfg.parallel_config, cfg.model_config
    # The platform populates additional_config before constructing Executor.
    # Admit only the exact observed defaults, not arbitrary enabled fusions.
    native_defaults = {
        "ascend_compilation_config": {
            "fuse_norm_quant": True,
            "fuse_qknorm_rope": True,
            "fuse_allreduce_rms": False,
            "enable_npugraph_ex": False,
            "enable_static_kernel": False,
            "fuse_muls_add": True,
        },
        "ascend_fusion_config": {"fusion_ops_gmmswigluquant": True},
    }
    checks = {
        "PP size": p.pipeline_parallel_size == size,
        "TP=1": p.tensor_parallel_size == 1,
        "DP=1": p.data_parallel_size == 1,
        "PCP=1": getattr(p, "prefill_context_parallel_size", 1) == 1,
        "DCP=1": getattr(p, "decode_context_parallel_size", 1) == 1,
        "no expert parallel": not getattr(p, "enable_expert_parallel", False),
        "no DBO": not getattr(p, "enable_dbo", False),
        "Ray manages nodes, nnodes=1": p.nnodes == 1,
        "no worker extension": not p.worker_extension_cls,
        "FP16": str(m.dtype) in ("torch.float16", "float16"),
        "eager": m.enforce_eager,
        "no quantization": m.quantization is None,
        "generate": m.runner_type == "generate",
        "text only": not m.is_multimodal_model,
        "no sleep": not m.enable_sleep_mode,
        "synchronous scheduling": cfg.scheduler_config.async_scheduling is False,
        "MRv1": not cfg.use_v2_model_runner,
        "no speculation": cfg.speculative_config is None,
        "no KV connector": cfg.kv_transfer_config is None,
        "no LoRA": cfg.lora_config is None,
        "no custom Ascend optimizations": getattr(cfg, "additional_config", {})
        in ({}, native_defaults),
    }
    if failed := [name for name, ok in checks.items() if not ok]:
        raise ValueError("heterogeneous PP unsupported config: " + ", ".join(failed))


def profile_actor_environment(environment=None):
    """Forward only explicit profile settings and clear stale daemon values."""
    source = os.environ if environment is None else environment
    keys = ("VLLM_PP_STAGE_TRACE", "VLLM_PP_TRACE_SESSION", "VLLM_PP_COMPUTE_MODEL")
    if not source.get(keys[0]):
        return dict.fromkeys(keys, "")
    values = {key: source.get(key, "") for key in keys}
    if (
        not all(isinstance(v, str) for v in values.values())
        or not Path(values[keys[0]]).is_absolute()
        or not values[keys[1]]
        or values[keys[2]] != "layer-measured"
    ):
        raise ValueError(
            "profiling needs an absolute trace path, session and layer-measured mode"
        )
    return values
