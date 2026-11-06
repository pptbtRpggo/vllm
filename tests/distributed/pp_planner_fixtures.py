# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure planner fixtures; no engine, model, device or profiler imports."""

from types import SimpleNamespace

from vllm.distributed.pp_memory import DeviceMemory, PPMemoryProfile
from vllm.distributed.pp_partition import RankCost


def _scope():
    return dict(
        model="test-model",
        revision=None,
        dtype="float16",
        quantization=None,
        kv_cache_dtype="auto",
        block_size=16,
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=32,
        gpu_memory_utilization=0.9,
    )


def _device(rank, budget, weights, kv=None, tp=0, **overrides):
    values = dict(
        pp_rank=rank,
        tp_rank=tp,
        budget_bytes=budget,
        layer_weights_bytes=tuple(weights),
        layer_kv_bytes=tuple(kv if kv is not None else [0] * len(weights)),
        runtime_bytes=0,
        activation_bytes=0,
        workspace_bytes=0,
        communication_bytes=0,
        graph_bytes=0,
        safety_margin_bytes=0,
        first_stage_bytes=0,
        last_stage_bytes=0,
    )
    return DeviceMemory(**(values | overrides))


def _profile(devices, tp=1):
    return PPMemoryProfile(
        len(devices[0].layer_weights_bytes),
        len(devices) // tp,
        tp,
        _scope(),
        tuple(devices),
    )


def _costs(n=32, speeds=(1, 1), comm=100):
    return [
        RankCost(r, n // len(speeds), speed, comm if r < len(speeds) - 1 else None, 10)
        for r, speed in enumerate(speeds)
    ]


def _engine_config(memory):
    scope = memory.serving_config
    return SimpleNamespace(
        model_config=SimpleNamespace(
            **{
                k: scope[k]
                for k in ("model", "revision", "dtype", "quantization", "max_model_len")
            },
            get_total_num_hidden_layers=lambda: memory.num_layers,
        ),
        cache_config=SimpleNamespace(
            cache_dtype=scope["kv_cache_dtype"],
            block_size=scope["block_size"],
            gpu_memory_utilization=0.9,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=scope["max_num_seqs"],
            max_num_batched_tokens=scope["max_num_batched_tokens"],
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=memory.tp_size, pipeline_parallel_size=memory.pp_size
        ),
    )
