# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit memory envelopes for fixed-block, fixed-order dense PP candidates.

These are measured-workload envelopes plus an explicit margin, not a proof of
arbitrary-shard safety. A recommendation still requires a trace-disabled live
deployment. The raw observations must be retained alongside the profile.
"""

from __future__ import annotations

import math
from typing import Any

import regex as re


def serving_config_from_engine(config: Any) -> dict[str, Any]:
    """Return the exact engine scope understood by PPMemoryProfile."""
    model, cache = config.model_config, config.cache_config
    scheduler = config.scheduler_config
    return {
        "model": model.model,
        "revision": model.revision,
        "dtype": str(model.dtype).removeprefix("torch."),
        "quantization": model.quantization,
        "kv_cache_dtype": cache.cache_dtype,
        "block_size": cache.block_size,
        "max_model_len": model.max_model_len,
        "max_num_seqs": scheduler.max_num_seqs,
        "max_num_batched_tokens": scheduler.max_num_batched_tokens,
        "gpu_memory_utilization": cache.gpu_memory_utilization,
    }


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _layer_id(name: str) -> int:
    if str(name).isdigit():
        return int(name)
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    if match is None:
        raise ValueError(f"cannot identify decoder layer in KV spec {name!r}")
    return int(match[1])


def memory_envelope_assumptions() -> dict[str, Any]:
    """Describe which quantities are observations and which are projections."""
    return {
        "kind": "observed_fixed_blocks_dense_layer_envelope",
        "layer_weights": "per-device maximum observed local decoder storage",
        "layer_weights_extrapolation": "same-structure dense layers on that device",
        "kv": "native page bytes times fixed block count, pool-owner reconciled",
        "reserve": "peak_reserved minus model minus KV, plus current non-Torch",
        "margin": "explicit caller-supplied bytes, not a measured allocation",
        "roles": "four original devices in original order; endpoints stay fixed",
        "guarantee_for_unprofiled_workloads": False,
        "requires_live_redeployment_validation": True,
    }


def build_fixed_block_memory_profile(
    observations: list[dict[str, Any]],
    config: Any,
    safety_margin_bytes: int = 1024**3,
):
    """Build v2 bounds without charging the observed KV pool twice.

    Args:
        observations: Four drained worker observations with native KV specs and
            actual unique KV storage-owner bytes.
        config: The live VllmConfig used to obtain the observations.
        safety_margin_bytes: Explicit additional reserve per device.

    Returns:
        A version-2 PPMemoryProfile for the fixed original device order.

    Raises:
        ValueError: Geometry, scope, pool ownership or coverage is unknown.
    """
    from vllm.distributed.pp_memory import DeviceMemory, PPMemoryProfile

    scope = serving_config_from_engine(config)
    cache, parallel = config.cache_config, config.parallel_config
    model = config.model_config
    n_layers = model.get_total_num_hidden_layers()
    _integer(safety_margin_bytes, "safety_margin_bytes")
    blocks = _integer(cache.num_gpu_blocks_override, "fixed block count", 1)
    if cache.kv_cache_memory_bytes is not None:
        raise ValueError("fixed-byte KV overrides are not supported")
    if (parallel.pipeline_parallel_size, parallel.tensor_parallel_size) != (4, 1):
        raise ValueError("automatic envelope requires fixed PP4 and TP1")
    if scope["dtype"] != "float16" or model.quantization is not None:
        raise ValueError("automatic envelope supports unquantized FP16 only")
    if model.hf_config.architectures not in (
        ["LlamaForCausalLM"],
        ["Qwen2ForCausalLM"],
    ):
        raise ValueError("automatic envelope requires supported dense architecture")
    if len(observations) != 4:
        raise ValueError("four complete worker observations are required")
    ordered = sorted(observations, key=lambda row: row["pp_rank"])
    if [(r["pp_rank"], r["tp_rank"]) for r in ordered] != [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
    ]:
        raise ValueError("observations must cover each original PP rank once")
    all_layers, partitions, devices = [], [], []
    for rank, row in enumerate(ordered):
        if row.get("includes_kv_pool") is not True:
            raise ValueError("allocator observation must explicitly include KV pool")
        if row.get("num_gpu_blocks") != blocks:
            raise ValueError("actual KV block count differs from configured policy")
        weights = {
            int(k): _integer(v, "layer storage bytes", 1)
            for k, v in row["layer_storage_bytes"].items()
        }
        if not weights:
            raise ValueError("loaded layer weights are missing")
        indices = sorted(weights)
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError("each worker must own a consecutive layer interval")
        all_layers.extend(indices)
        partitions.append(len(indices))
        specs = row.get("kv_cache_specs")
        if not isinstance(specs, dict) or not specs:
            raise ValueError("native per-layer KV specs are required")
        page_bytes = {}
        for name, spec in specs.items():
            index = _layer_id(name)
            if index in page_bytes or index not in weights:
                raise ValueError("KV specs do not match unique local decoder layers")
            if spec.get("block_size") != scope["block_size"]:
                raise ValueError("native KV block size differs from engine config")
            dtype = str(spec.get("dtype", "")).removeprefix("torch.")
            expected = (
                scope["dtype"]
                if scope["kv_cache_dtype"] == "auto"
                else scope["kv_cache_dtype"]
            )
            if dtype != expected:
                raise ValueError("native KV dtype differs from engine config")
            page_bytes[index] = _integer(
                spec.get("page_size_bytes"), "KV page bytes", 1
            )
        if set(page_bytes) != set(weights):
            raise ValueError("native KV specs do not cover every loaded layer")
        pool = _integer(row.get("kv_cache_storage_bytes"), "actual KV owner bytes", 1)
        if pool != blocks * sum(page_bytes.values()):
            raise ValueError(
                "actual KV pool does not reconcile with native page geometry"
            )
        if len(set(page_bytes.values())) != 1:
            raise ValueError("automatic envelope requires homogeneous attention pages")
        endpoints = _integer(row["non_layer_storage_bytes"], "non-layer storage")
        model_bytes = sum(weights.values()) + endpoints
        reserved = _integer(row["reserved_bytes"], "reserved bytes")
        peak = _integer(row["peak_reserved_bytes"], "peak reserved bytes")
        total = _integer(row["total_bytes"], "total bytes", 1)
        free = _integer(row["free_bytes"], "free bytes")
        if free > total or peak < reserved or peak < model_bytes + pool:
            raise ValueError("allocator observation is inconsistent")
        non_torch = max(0, total - free - reserved)
        other_peak = peak - model_bytes - pool
        budget = math.floor(total * scope["gpu_memory_utilization"])
        devices.append(
            DeviceMemory(
                pp_rank=rank,
                tp_rank=0,
                budget_bytes=budget,
                layer_weights_bytes=(max(weights.values()),) * n_layers,
                layer_kv_bytes=(blocks * next(iter(page_bytes.values())),) * n_layers,
                runtime_bytes=other_peak
                + non_torch
                + (endpoints if rank in (1, 2) else 0),
                activation_bytes=0,
                workspace_bytes=0,
                communication_bytes=0,
                graph_bytes=0,
                safety_margin_bytes=safety_margin_bytes,
                first_stage_bytes=endpoints if rank == 0 else 0,
                last_stage_bytes=endpoints if rank == 3 else 0,
            )
        )
    if all_layers != list(range(n_layers)):
        raise ValueError("worker layers must cover the whole model in rank order")
    profile = PPMemoryProfile(
        n_layers,
        4,
        1,
        scope,
        tuple(devices),
        cache_policy={
            "mode": "fixed_blocks",
            "num_gpu_blocks": blocks,
            "block_size": scope["block_size"],
            "kv_cache_dtype": scope["kv_cache_dtype"],
        },
    )
    profile.validate_engine_config(config)
    if any(row["headroom_bytes"] < 0 for row in profile.plan_usage(partitions)):
        raise ValueError("observed profile partition exceeds memory envelope")
    return profile
