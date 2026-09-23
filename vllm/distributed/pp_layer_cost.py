# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Representative decoder cost per device, from direct serving measurements.

Each microbatch contributes the mean of its measured decoder layers. The
microbatch aggregation weights every observed step equally, including mixed
steps. The optional phase-balanced protocol averages prefill and decode means
equally and excludes mixed steps when workload=all.
No device/layer identity extrapolation coverage table is required: identical
layer cost on a device is an explicit approximation, not a measured fact.
"""

from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING, Any

from vllm.distributed.pp_link_profile import measured_transfer_ms

if TYPE_CHECKING:
    from vllm.distributed.pp_partition import RankCost, Workload


def _duration(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("layer-measured durations must be finite and nonnegative")
    return float(value)


def phase_groups(rows, workload, warmup_steps, aggregation="phase-balanced"):
    from vllm.distributed.pp_partition import _keep_record

    if aggregation not in ("phase-balanced", "microbatch"):
        raise ValueError(f"unknown layer aggregation: {aggregation}")
    phases = (
        ("prefill", "decode")
        if workload == "all" and aggregation == "phase-balanced"
        else (workload,)
    )
    groups = {p: [r for r in rows if _keep_record(r, p, warmup_steps)] for p in phases}
    for phase, values in groups.items():
        if not values:
            raise ValueError(f"layer-measured requires {phase} samples after warmup")
    return groups


def phase_mean(groups, getter):
    return statistics.mean(
        statistics.mean(getter(r) for r in rows) for rows in groups.values()
    )


def measured_layer_rank_costs(
    trace_sets: list[dict[int, list[dict[str, Any]]]],
    *,
    workload: Workload,
    warmup_steps: int,
    num_layers: int | None = None,
    include_communication: bool = True,
    layer_aggregation: str = "phase-balanced",
) -> list[RankCost]:
    from vllm.distributed.pp_partition import RankCost

    if not trace_sets:
        raise ValueError("layer-measured requires traces")
    ranks = sorted(trace_sets[0])
    if not ranks or ranks != list(range(len(ranks))):
        raise ValueError("PP ranks must be contiguous from 0")
    totals = set()
    for traces in trace_sets:
        if sorted(traces) != ranks:
            raise ValueError(
                "layer-measured requires the same PP ranks in every profile"
            )
        end = 0
        for rank in ranks:
            ranges = {(r["start_layer"], r["end_layer"]) for r in traces[rank]}
            if len(ranges) != 1:
                raise ValueError(
                    "use separate trace directories for different partitions"
                )
            start, stop = ranges.pop()
            if (
                type(start) is not int
                or type(stop) is not int
                or start != end
                or stop <= start
            ):
                raise ValueError("profile layer ranges must cover a contiguous model")
            end = stop
        totals.add(end)
    if len(totals) != 1 or (num_layers is not None and num_layers not in totals):
        raise ValueError("profiles must have the same total number of layers")
    costs = []
    for rank in ranks:
        groups = phase_groups(
            [
                r
                for traces in trace_sets
                for source, rs in traces.items()
                for r in rs
                if r.get("profile_device_id", source) == rank
            ],
            workload,
            warmup_steps,
            layer_aggregation,
        )
        rows = [r for group in groups.values() for r in group]
        for row in rows:
            if row.get("compute_model") != "layer-measured":
                raise ValueError("layer-measured requires direct layer traces")
            if row["pp_size"] != len(ranks):
                raise ValueError("trace PP size does not match its ranks")
            if row.get("tp_size") != 1 or row.get("tp_rank") != 0:
                raise ValueError("layer-measured currently requires TP=1")
            if row.get("compute_scale", 1.0) != 1.0:
                raise ValueError("layer-measured cannot use mock compute slowdown")
            layers = row.get("layer_compute_ms")
            if not isinstance(layers, dict) or set(layers) != {
                str(i) for i in range(row["start_layer"], row["end_layer"])
            }:
                raise ValueError("layer trace must contain every local global layer ID")
            duration = sum(_duration(v) for v in layers.values())
            residual = _duration(row.get("non_layer_compute_ms"))
            wall = _duration(row.get("compute_wall_ms"))
            if abs(duration + residual - wall) > 0.01:
                raise ValueError("layer times plus residual must match stage wall time")
            endpoints = sum(
                _duration(row[k])
                for k in ("embedding_ms", "lm_head_ms", "final_norm_ms")
                if row.get(k) is not None
            )
            overhead = _duration(row.get("runner_overhead_ms"))
            if abs(endpoints + overhead - residual) > 0.01:
                raise ValueError(
                    "endpoint times plus runner overhead must match residual"
                )
        scales = {r.get("comm_scale", 1.0) for r in rows}
        if len(scales) != 1 or any(not math.isfinite(v) or v <= 0 for v in scales):
            raise ValueError("profiles mix communication scales or have invalid scales")

        def endpoint(key, groups=groups):
            # Endpoint roles may be collected in additional reordered runs.
            measured = {
                p: [r for r in rs if r.get(key) is not None] for p, rs in groups.items()
            }
            if not any(measured.values()):
                return None
            if not all(measured.values()):
                raise ValueError(f"inconsistent phase coverage for {key}")
            return phase_mean(measured, lambda r: _duration(r[key]))

        summary = tuple(
            dict(
                phase=p,
                samples=len(rs),
                t_layer_ms=statistics.mean(
                    statistics.mean(r["layer_compute_ms"].values()) for r in rs
                ),
            )
            for p, rs in groups.items()
        )
        transfer = None
        sources = set()
        if include_communication and rank < len(ranks) - 1:
            primary_groups = phase_groups(
                trace_sets[0][rank], workload, warmup_steps, layer_aggregation
            )
            transfer = phase_mean(primary_groups, measured_transfer_ms)
            sources = {
                r["send_service_source"] for rs in primary_groups.values() for r in rs
            }
            if len(sources) != 1:
                raise ValueError("profiles mix communication measurement methods")
        initial = trace_sets[0][rank][0]
        costs.append(
            RankCost(
                pp_rank=rank,
                n_layers=initial["end_layer"] - initial["start_layer"],
                t_layer_ms=phase_mean(
                    groups, lambda r: statistics.mean(r["layer_compute_ms"].values())
                ),
                t_fixed_ms=phase_mean(groups, lambda r: r["runner_overhead_ms"]),
                t_embedding_ms=endpoint("embedding_ms"),
                t_head_ms=endpoint("lm_head_ms"),
                t_final_norm_ms=endpoint("final_norm_ms"),
                t_comm_out_ms=transfer,
                n_steps=len(rows),
                comm_source=next(iter(sources), "none"),
                compute_model="layer-measured",
                profiled_pp_size=len(ranks),
                phase_summary=summary,
                layer_aggregation=layer_aggregation,
            )
        )
    return costs
