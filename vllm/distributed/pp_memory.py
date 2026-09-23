# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit, per-device memory bounds for PP partition planning.

Timing traces cannot establish these bounds. Inputs must describe the target
serving workload, including block-rounded KV allocation and conservative peak
reserves valid for every candidate shard. No CUDA/HCCL memory is queried here.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


def observe_worker_memory(worker: Any) -> dict[str, Any]:
    """Observe loaded tensor storage and allocator peaks after a drained workload.

    This is evidence for capacity bounds, not an automatic safety guarantee for
    different partitions. In particular allocator peaks include the allocated KV
    pool and must not be added to a second KV estimate.
    """
    import torch

    from vllm.distributed.parallel_state import get_pp_group, get_tp_group

    device = worker.device
    backend = getattr(torch, device.type)
    backend.synchronize(device)
    model = worker.model_runner.model
    layers: dict[str, int] = {}
    endpoint = 0
    seen = set()
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        storage = tensor.untyped_storage()
        identity = (str(tensor.device), storage.data_ptr())
        if identity in seen:
            continue
        seen.add(identity)
        size = storage.nbytes()
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if match:
            index = match.group(1)
            layers[index] = layers.get(index, 0) + size
        else:
            endpoint += size
    free, total = backend.mem_get_info(device)
    return dict(
        pp_rank=get_pp_group().rank_in_group,
        tp_rank=get_tp_group().rank_in_group,
        layer_storage_bytes=layers,
        non_layer_storage_bytes=endpoint,
        allocated_bytes=backend.memory_allocated(device),
        reserved_bytes=backend.memory_reserved(device),
        peak_allocated_bytes=backend.max_memory_allocated(device),
        peak_reserved_bytes=backend.max_memory_reserved(device),
        free_bytes=free,
        total_bytes=total,
        includes_kv_pool=True,
    )


@dataclass(frozen=True)
class DeviceMemory:
    pp_rank: int
    tp_rank: int
    budget_bytes: int
    layer_weights_bytes: tuple[int, ...]
    layer_kv_bytes: tuple[int, ...]
    runtime_bytes: int
    activation_bytes: int
    workspace_bytes: int
    communication_bytes: int
    graph_bytes: int
    safety_margin_bytes: int
    first_stage_bytes: int
    last_stage_bytes: int

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            values = value if isinstance(value, tuple) else (value,)
            if any(type(v) is not int or v < 0 for v in values):
                raise ValueError(f"{key} must contain nonnegative integer bytes/ranks")
        if self.budget_bytes == 0:
            raise ValueError("budget_bytes must be positive")
        if not self.layer_weights_bytes or (
            len(self.layer_weights_bytes) != len(self.layer_kv_bytes)
        ):
            raise ValueError("weight and KV vectors must have the same nonzero length")

    def estimate(
        self, start: int, end: int, pp_size: int, *, stage_rank: int | None = None
    ) -> dict[str, int]:
        role = self.pp_rank if stage_rank is None else stage_rank
        weights = sum(self.layer_weights_bytes[start:end])
        kv = sum(self.layer_kv_bytes[start:end])
        endpoints = (self.first_stage_bytes if role == 0 else 0) + (
            self.last_stage_bytes if role == pp_size - 1 else 0
        )
        reserve = (
            self.runtime_bytes
            + self.activation_bytes
            + self.workspace_bytes
            + self.communication_bytes
            + self.graph_bytes
            + self.safety_margin_bytes
        )
        required = weights + kv + endpoints + reserve
        return {
            "pp_rank": self.pp_rank,
            "tp_rank": self.tp_rank,
            "start_layer": start,
            "end_layer": end,
            "weights_bytes": weights,
            "kv_cache_bytes": kv,
            "endpoint_bytes": endpoints,
            "reserve_bytes": reserve,
            "required_bytes": required,
            "budget_bytes": self.budget_bytes,
            "headroom_bytes": self.budget_bytes - required,
        }


@dataclass(frozen=True)
class PPMemoryProfile:
    num_layers: int
    pp_size: int
    tp_size: int
    # Exact configuration covered by the supplied memory bounds. Keep this in
    # exported plans, so offline replay cannot silently lose the workload scope.
    serving_config: dict[str, Any]
    devices: tuple[DeviceMemory, ...]

    def __post_init__(self) -> None:
        for key in ("num_layers", "pp_size", "tp_size"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        required = {
            "model",
            "revision",
            "dtype",
            "quantization",
            "kv_cache_dtype",
            "block_size",
            "max_model_len",
            "max_num_seqs",
            "max_num_batched_tokens",
            "gpu_memory_utilization",
        }
        if set(self.serving_config) != required:
            raise ValueError(f"serving_config must contain exactly {sorted(required)}")
        for key in (
            "block_size",
            "max_model_len",
            "max_num_seqs",
            "max_num_batched_tokens",
        ):
            value = self.serving_config[key]
            if type(value) is not int or value <= 0:
                raise ValueError(f"serving_config.{key} must be a positive integer")
        for key in ("model", "dtype", "kv_cache_dtype"):
            value = self.serving_config[key]
            if not isinstance(value, str) or not value:
                raise ValueError(f"serving_config.{key} must be a nonempty string")
        utilization = self.serving_config["gpu_memory_utilization"]
        if (
            type(utilization) not in (int, float)
            or not math.isfinite(utilization)
            or not 0 < utilization <= 1
        ):
            raise ValueError("gpu_memory_utilization must be finite and in (0, 1]")
        for key in ("revision", "quantization"):
            value = self.serving_config[key]
            if value is not None and not isinstance(value, str):
                raise ValueError(f"serving_config.{key} must be a string or null")
        ranks = [(d.pp_rank, d.tp_rank) for d in self.devices]
        expected = {(p, t) for p in range(self.pp_size) for t in range(self.tp_size)}
        if len(ranks) != len(expected) or set(ranks) != expected:
            raise ValueError(
                "memory profile must cover every PP/TP device exactly once"
            )
        if any(len(d.layer_weights_bytes) != self.num_layers for d in self.devices):
            raise ValueError("each device needs memory costs for all model layers")

    def validate_dimensions(self, num_layers: int, pp_size: int) -> None:
        if (self.num_layers, self.pp_size) != (num_layers, pp_size):
            raise ValueError("memory profile dimensions do not match timing rank costs")

    def fits(self, rank: int, start: int, end: int, pp_size: int) -> bool:
        return all(
            d.estimate(start, end, pp_size)["headroom_bytes"] >= 0
            for d in self.devices
            if d.pp_rank == rank
        )

    def plan_usage(self, partitions: list[int]) -> list[dict[str, int]]:
        result = []
        start = 0
        for rank, count in enumerate(partitions):
            result.extend(
                d.estimate(start, start + count, len(partitions))
                for d in sorted(self.devices, key=lambda d: (d.pp_rank, d.tp_rank))
                if d.pp_rank == rank
            )
            start += count
        return result

    def select_devices(self, order: tuple[int, ...]) -> PPMemoryProfile:
        if (
            not order
            or len(set(order)) != len(order)
            or any(d not in range(self.pp_size) for d in order)
        ):
            raise ValueError("invalid selected device order")
        return replace(
            self,
            pp_size=len(order),
            devices=tuple(
                replace(d, pp_rank=stage)
                for stage, original in enumerate(order)
                for d in self.devices
                if d.pp_rank == original
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, **asdict(self)}

    @classmethod
    def from_file(cls, path: str | Path) -> PPMemoryProfile:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if type(data["version"]) is not int or data.pop("version") != 1:
                raise ValueError("unsupported memory profile version")
            devices = []
            for item in data.pop("devices"):
                item["layer_weights_bytes"] = tuple(item["layer_weights_bytes"])
                item["layer_kv_bytes"] = tuple(item["layer_kv_bytes"])
                devices.append(DeviceMemory(**item))
            return cls(devices=tuple(devices), **data)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"invalid memory profile {path}: {exc}") from exc

    def validate_engine_config(self, config: Any) -> None:
        model = config.model_config
        cache = config.cache_config
        scheduler = config.scheduler_config
        parallel = config.parallel_config
        if (
            getattr(cache, "kv_cache_memory_bytes", None) is not None
            or getattr(cache, "num_gpu_blocks_override", None) is not None
        ):
            raise ValueError(
                "memory profile does not support fixed KV allocation overrides; "
                "remove kv_cache_memory_bytes/num_gpu_blocks_override"
            )
        actual = {
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
        mismatches = [k for k, v in actual.items() if self.serving_config[k] != v]
        if parallel.tensor_parallel_size != self.tp_size:
            mismatches.append("tensor_parallel_size")
        if parallel.pipeline_parallel_size != self.pp_size:
            mismatches.append("pipeline_parallel_size")
        if model.get_total_num_hidden_layers() != self.num_layers:
            mismatches.append("num_layers")
        if mismatches:
            raise ValueError(
                f"memory profile does not match serving config: {mismatches}"
            )


def resolve_memory_profile(
    profile: PPMemoryProfile | str | Path | None,
    dump_dir: str | Path | None,
    *,
    allow_unchecked_memory: bool = False,
) -> PPMemoryProfile | None:
    if isinstance(profile, PPMemoryProfile):
        return profile
    if profile is not None:
        return PPMemoryProfile.from_file(profile)
    if dump_dir is not None:
        path = Path(dump_dir) / "pp_memory_profile.json"
        if path.exists():
            return PPMemoryProfile.from_file(path)
    if allow_unchecked_memory:
        return None
    raise ValueError(
        "memory feasibility requires --memory-profile or pp_memory_profile.json "
        "in the trace directory; timing traces cannot establish memory budgets. "
        "Use --allow-unchecked-memory only for timing-only analysis"
    )
