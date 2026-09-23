# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Subset/order DP for representative measured layer costs and blocking PP.

The unfinished last stage stays in the state until its successor is chosen.
Only then is its outgoing transfer known, so sender occupancy is never lost.
Source device 0 is fixed; every selected device hosts one contiguous segment.
"""

from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache

from vllm.distributed.pp_memory import resolve_memory_profile
from vllm.distributed.pp_partition import PPPartitionPlan, _compute_ms


def partition_devices(
    costs,
    links,
    *,
    objective="throughput",
    num_layers=None,
    min_pp_size=1,
    max_pp_size=None,
    memory_profile=None,
    allow_unchecked_memory=False,
    overlap_comm=False,
):
    if not costs or any(c.compute_model != "layer-measured" for c in costs):
        raise ValueError(
            "device selection requires layer-measured representative costs"
        )
    if objective not in ("latency", "throughput"):
        raise ValueError("unknown objective")
    if overlap_comm:
        raise ValueError("device selection uses vLLM blocking occupancy only")
    size = len(costs)
    n = sum(c.n_layers for c in costs) if num_layers is None else num_layers
    lo, hi = min_pp_size, min(size if max_pp_size is None else max_pp_size, size, n)
    if n < 1 or lo < 1 or lo > hi:
        raise ValueError("invalid layer count or PP size range")
    for rank, c in enumerate(costs):
        if c.pp_rank != rank:
            raise ValueError("costs must use contiguous profiling device IDs")
        values = [c.t_layer_ms, c.t_fixed_ms]
        values += [
            v
            for v in (c.t_embedding_ms, c.t_head_ms, c.t_final_norm_ms)
            if v is not None
        ]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("device costs must be finite and nonnegative")
    for (a, b), value in links.items():
        if a not in range(size) or b not in range(size) or a == b:
            raise ValueError("invalid measured link endpoints")
        if not math.isfinite(value) or value < 0:
            raise ValueError("link costs must be finite and nonnegative")
    memory = resolve_memory_profile(
        memory_profile, None, allow_unchecked_memory=allow_unchecked_memory
    )
    if memory is not None:
        memory.validate_dimensions(n, size)
        if memory.tp_size != 1:
            raise ValueError("device selection currently requires TP=1")

    @lru_cache(None)
    def compute(device, start, end):
        c = costs[device]
        if end - start < c.min_layers or (
            c.max_layers is not None and end - start > c.max_layers
        ):
            return math.inf
        if memory is not None:
            role = 0 if start == 0 else 1
            # Endpoint memory follows the candidate segment, not original rank.
            pp = role + (1 if end == n else 2)
            if any(
                d.estimate(start, end, pp, stage_rank=role)["headroom_bytes"] < 0
                for d in memory.devices
                if d.pp_rank == device
            ):
                return math.inf
        return _compute_ms(costs, device, end - start, start, n)

    def solve(limit=None):
        additive = objective == "latency" or limit is not None

        @lru_cache(None)
        def visit(mask, previous, device, start, end):
            local = compute(device, start, end)
            if not math.isfinite(local):
                return math.inf, (), ()
            incoming = 0.0 if previous == -1 else links[previous, device]
            used = mask.bit_count()
            if end == n:
                occupancy = incoming + local
                if used < lo or (limit is not None and occupancy > limit):
                    return math.inf, (), ()
                return (local if additive else occupancy), (device,), (end - start,)
            if used >= hi:
                return math.inf, (), ()
            best = (math.inf, (), ())
            for successor in range(size):
                if mask & (1 << successor) or (device, successor) not in links:
                    continue
                outgoing = links[device, successor]
                occupancy = incoming + local + outgoing
                if limit is not None and occupancy > limit:
                    continue
                for stop in range(end + 1, n + 1):
                    rest, order, parts = visit(
                        mask | (1 << successor), device, successor, end, stop
                    )
                    value = (
                        local + outgoing + rest if additive else max(occupancy, rest)
                    )
                    candidate = value, (device,) + order, (end - start,) + parts
                    if candidate < best:
                        best = candidate
            return best

        return min(visit(1, -1, 0, 0, end) for end in range(1, n + 1))

    primary, order, parts = solve()
    if not math.isfinite(primary):
        raise ValueError(
            "No feasible device selection with measured links, "
            "endpoint coverage and memory bounds"
        )
    if objective == "throughput":
        # A locally smaller max does not ensure the globally smallest sum.
        _, order, parts = solve(primary)
    selected = [
        replace(
            costs[d],
            pp_rank=r,
            t_comm_out_ms=links[d, order[r + 1]] if r + 1 < len(order) else None,
            comm_source="measured_idle_replay" if r + 1 < len(order) else "none",
        )
        for r, d in enumerate(order)
    ]
    return PPPartitionPlan(
        partitions=list(parts),
        objective=objective,
        cost_ms=primary,
        rank_costs=selected,
        device_order=order,
        memory_profile=memory.select_devices(order) if memory else None,
    )
