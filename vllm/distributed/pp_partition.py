# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Choose a vLLM PP layer split from ``VLLM_PP_STAGE_TRACE`` JSONL files.

Trace files measure stage work, including fixed endpoint work. Multiple
partitions identify fixed and per-layer costs using actual microbatch shapes.
When exact shapes do not recur, a nonnegative token/context feature fit is used.
The primary trace defines the reference workload distribution for every rank.
Communication comes from paired serving timestamps (default) or measured idle
replay (explicit). No configured bandwidth or slowdown is a planner cost.
Only fixed device order and the profiled PP size are supported by this fit;
layer counts outside measured coverage are excluded. This is a mean occupancy
surrogate, not an E2E serving schedule simulation.

Two objectives, adapted from EdgeShard:

* ``latency`` — sequential token time: sum(compute) + sum(comm hops)
* ``throughput`` — blocking pipeline: min max(recv + compute + send).
  EdgeShard's ideal compute/communication overlap is an explicit opt-in.

Usage::

    python -m vllm.distributed.pp_partition ./pp_traces --objective throughput
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from vllm.distributed.pp_comm_trace import (
    attach_paired_communication,
    select_serving_communication,
)
from vllm.distributed.pp_link_profile import (
    attach_link_measurements,
)
from vllm.distributed.pp_memory import PPMemoryProfile, resolve_memory_profile

Objective = Literal["latency", "throughput"]
Workload = Literal["all", "decode", "prefill", "mixed"]
ComputeModel = Literal["provided", "shape-affine"]

_INF = math.inf


@dataclass(frozen=True)
class RankCost:
    pp_rank: int
    n_layers: int
    t_layer_ms: float
    t_comm_out_ms: float | None
    n_steps: int
    comm_source: str = "provided"
    t_fixed_ms: float = 0.0
    compute_model: ComputeModel = "provided"
    min_layers: int = 1
    max_layers: int | None = None
    profiled_pp_size: int | None = None
    shape_fits: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class PPPartitionPlan:
    partitions: list[int]
    objective: Objective
    cost_ms: float
    rank_costs: list[RankCost]
    overlap_comm: bool = False
    memory_profile: PPMemoryProfile | None = None

    @property
    def cost_model(self) -> str:
        if self.objective == "latency":
            return "sequential"
        return "ideal_overlap" if self.overlap_comm else "blocking"

    @property
    def compute_model(self) -> str:
        return self.rank_costs[0].compute_model

    @property
    def env_value(self) -> str:
        return ",".join(str(n) for n in self.partitions)

    @property
    def pp_size(self) -> int:
        return len(self.partitions)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view used by ``vllm pp-profile``."""
        return {
            "objective": self.objective,
            "cost_model": self.cost_model,
            "compute_model": self.compute_model,
            "tie_breaker": "sequential_cost"
            if self.objective == "throughput"
            else None,
            "predicted_sequential_ms": sum(
                c.t_fixed_ms + n * c.t_layer_ms
                for c, n in zip(self.rank_costs, self.partitions)
            )
            + sum(c.t_comm_out_ms or 0.0 for c in self.rank_costs[:-1]),
            "cost_kind": (
                (
                    "mean_stage_occupancy_ms"
                    if self.compute_model == "shape-affine"
                    else "steady_state_cycle_ms"
                )
                if self.objective == "throughput"
                else "sequential_latency_ms"
            ),
            "workload_aggregation": (
                "reference_shape_weighted_mean"
                if self.compute_model == "shape-affine"
                else "provided_costs"
            ),
            "memory_feasibility_checked": self.memory_profile is not None,
            "memory_feasibility_basis": (
                "supplied_memory_bounds" if self.memory_profile else "unchecked"
            ),
            "memory_profile": (
                self.memory_profile.to_dict() if self.memory_profile else None
            ),
            "memory_usage": (
                self.memory_profile.plan_usage(self.partitions)
                if self.memory_profile
                else []
            ),
            "predicted_cost_ms": self.cost_ms,
            "pp_size": self.pp_size,
            "VLLM_PP_LAYER_PARTITION": self.env_value,
            "partitions": list(self.partitions),
            "rank_costs": [
                {
                    "pp_rank": cost.pp_rank,
                    "n_layers_traced": cost.n_layers,
                    "t_layer_ms": cost.t_layer_ms,
                    "t_comm_out_ms": cost.t_comm_out_ms,
                    "n_steps": cost.n_steps,
                    "comm_source": cost.comm_source,
                    "t_fixed_ms": cost.t_fixed_ms,
                    "min_layers": cost.min_layers,
                    "max_layers": cost.max_layers,
                    "shape_fits": list(cost.shape_fits),
                }
                for cost in self.rank_costs
            ],
        }


def load_trace_records(
    dump_dir: str | Path, *, comm_source: str = "serving"
) -> dict[int, list[dict[str, Any]]]:
    """Load ``pp_stage_pp*_tp0.jsonl`` (or any tp rank if tp0 is absent)."""
    dump_dir = Path(dump_dir)
    files = sorted(dump_dir.glob("pp_stage_pp*_tp*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No pp_stage_pp*_tp*.jsonl files in {dump_dir}")

    by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_key[(int(rec["pp_rank"]), int(rec["tp_rank"]))].append(rec)

    attach_link_measurements(dump_dir, [r for rows in by_key.values() for r in rows])
    attach_paired_communication(by_key)
    if comm_source == "serving":
        select_serving_communication([r for rows in by_key.values() for r in rows])
    elif comm_source != "replay":
        raise ValueError(f"unknown communication source: {comm_source}")
    by_rank: dict[int, list[dict[str, Any]]] = {}
    ranks = sorted({rank for rank, _ in by_key})
    for rank in ranks:
        if (rank, 0) in by_key:
            by_rank[rank] = by_key[(rank, 0)]
        else:
            tp_ranks = sorted(tp for r, tp in by_key if r == rank)
            by_rank[rank] = by_key[(rank, tp_ranks[0])]
    return by_rank


def _keep_record(
    rec: dict[str, Any],
    workload: Workload,
    warmup_steps: int,
) -> bool:
    if rec.get("is_warmup", False):
        return False
    if int(rec.get("step", 0)) < warmup_steps:
        return False
    if int(rec.get("num_tokens", 0)) <= 0:
        return False
    ctx = int(rec.get("num_ctx_tokens", 0))
    gen = int(rec.get("num_generation_tokens", 0))
    if rec.get("batch_shape") is not None:
        ctx = sum(s["prompt_tokens"] for s in rec["batch_shape"])
        gen = sum(s["query_tokens"] - s["prompt_tokens"] for s in rec["batch_shape"])
    if workload == "decode":
        return gen > 0 and ctx == 0
    if workload == "prefill":
        return ctx > 0 and gen == 0
    if workload == "mixed":
        return ctx > 0 and gen > 0
    return True


def _comm_ms(costs: Sequence[RankCost], from_rank: int) -> float:
    value = costs[from_rank].t_comm_out_ms
    if value is None:
        if from_rank == len(costs) - 1:
            return 0.0
        raise ValueError(
            f"rank {from_rank} has no send_ms samples; cannot cost hop "
            f"{from_rank}->{from_rank + 1}"
        )
    return value


def _compute_ms(costs: Sequence[RankCost], rank: int, n_layers: int) -> float:
    cost = costs[rank]
    return cost.t_fixed_ms + n_layers * cost.t_layer_ms


def partition_layers(
    costs: Sequence[RankCost],
    *,
    objective: Objective = "throughput",
    num_layers: int | None = None,
    min_pp_size: int = 1,
    max_pp_size: int | None = None,
    overlap_comm: bool = False,
    memory_profile: PPMemoryProfile | None = None,
    allow_unchecked_memory: bool = False,
) -> PPPartitionPlan:
    """DP over contiguous layer counts on a fixed PP rank order.

    Rank 0 is always the source. Using ``pp_size < len(costs)`` drops trailing
    ranks; it does not reorder devices. Every candidate is checked against
    all TP devices in the supplied memory profile before entering the DP.
    Missing memory bounds require explicit timing-only opt-in.

    The blocking model assumes identical independent batches, enough in-flight
    work, and rendezvous transfers occupying both endpoints. Each rank performs
    recv -> compute -> send without overlap. Its asymptotic cycle is the maximum
    of C_i + D_(i-1) + D_i (missing endpoint hops are zero). This is not a model
    of finite-batch autoregressive feedback, shared-link contention or CPU work.
    With shape-affine costs it is only a shape-weighted mean occupancy surrogate;
    PP size is fixed and candidates outside profiled length coverage are excluded.
    """
    if not costs:
        raise ValueError("no rank costs")
    if objective not in ("latency", "throughput"):
        raise ValueError(f"unknown objective: {objective}")
    for rank, cost in enumerate(costs):
        if cost.pp_rank != rank:
            raise ValueError("rank costs must be ordered contiguously from rank 0")
        values = [cost.t_layer_ms, cost.t_fixed_ms]
        if cost.t_comm_out_ms is not None:
            values.append(cost.t_comm_out_ms)
        if any(not math.isfinite(x) or x < 0 for x in values):
            raise ValueError(f"rank {rank} costs must be finite and nonnegative")
    models = {cost.compute_model for cost in costs}
    if len(models) != 1:
        raise ValueError("cannot mix compute models across ranks")
    n_ranks = len(costs)
    n_layers = num_layers if num_layers is not None else sum(c.n_layers for c in costs)
    if n_layers < 1:
        raise ValueError("num_layers must be >= 1")
    memory_profile = resolve_memory_profile(
        memory_profile, None, allow_unchecked_memory=allow_unchecked_memory
    )
    if memory_profile is not None:
        memory_profile.validate_dimensions(n_layers, n_ranks)
    max_pp = n_ranks if max_pp_size is None else min(max_pp_size, n_ranks)
    min_pp = max(1, min_pp_size)
    if models == {"shape-affine"}:
        if overlap_comm:
            raise ValueError("shape-affine currently supports blocking occupancy only")
        if any(c.profiled_pp_size != n_ranks for c in costs):
            raise ValueError("shape-affine costs require the profiled PP size")
        # Changing PP size changes endpoint work; never reuse a middle-stage
        # intercept as the last-stage intercept.
        if min_pp > n_ranks or max_pp < n_ranks:
            raise ValueError("shape-affine cannot change the profiled PP size")
        min_pp = max_pp = n_ranks
    if min_pp > max_pp:
        raise ValueError(f"invalid pp_size range [{min_pp}, {max_pp}]")
    if n_layers < min_pp:
        raise ValueError(f"need at least one layer per rank: {n_layers=} {min_pp=}")

    best_plan = None
    best_key = (_INF, _INF)
    # Fix the final PP size before building the DP: the final rank has no
    # outgoing hop, while a prefix's last rank still sends to its successor.
    # Charging only the new rank's incoming hop loses the sender's occupancy.
    for pp_size in range(min_pp, min(max_pp, n_layers) + 1):
        hops = [0.0] + [_comm_ms(costs, r) for r in range(pp_size - 1)] + [0.0]

        def solve(
            pp_size: int, hops: Sequence[float], stage_limit: float | None = None
        ):
            dp = [[_INF] * (pp_size + 1) for _ in range(n_layers + 1)]
            prev = [[-1] * (pp_size + 1) for _ in range(n_layers + 1)]
            dp[0][0] = 0.0
            for r in range(1, pp_size + 1):
                incoming, outgoing = hops[r - 1], hops[r]
                for i in range(r, n_layers + 1):
                    for k in range(r - 1, i):
                        if not math.isfinite(dp[k][r - 1]):
                            continue
                        cost = costs[r - 1]
                        if i - k < cost.min_layers or (
                            cost.max_layers is not None and i - k > cost.max_layers
                        ):
                            continue
                        if memory_profile is not None and not memory_profile.fits(
                            r - 1, k, i, pp_size
                        ):
                            continue
                        compute = _compute_ms(costs, r - 1, i - k)
                        occupancy = (
                            max(incoming, compute)
                            if overlap_comm
                            else incoming + compute + outgoing
                        )
                        if stage_limit is not None and occupancy > stage_limit:
                            continue
                        if objective == "latency" or stage_limit is not None:
                            cand = dp[k][r - 1] + incoming + compute
                        else:
                            cand = max(dp[k][r - 1], occupancy)
                        if cand < dp[i][r]:
                            dp[i][r] = cand
                            prev[i][r] = k
            return dp[n_layers][pp_size], prev

        primary, prev = solve(pp_size, hops)
        if not math.isfinite(primary):
            continue
        secondary = primary
        if objective == "throughput":
            # A later bottleneck can hide different prefix maxima. Keeping only
            # the locally best (max, sum) pair loses globally better ties.
            # First find the optimal bottleneck, then minimize the sequential
            # cost subject to every stage staying within that exact bound.
            secondary, prev = solve(pp_size, hops, stage_limit=primary)
        key = (primary, secondary)
        if key >= best_key:
            continue
        best_key = key
        partitions = [0] * pp_size
        i = n_layers
        for r in range(pp_size, 0, -1):
            k = prev[i][r]
            partitions[r - 1] = i - k
            i = k
        if i != 0 or sum(partitions) != n_layers or any(n <= 0 for n in partitions):
            raise RuntimeError(f"invalid backtrace {partitions} for {n_layers} layers")
        best_plan = PPPartitionPlan(
            partitions=partitions,
            objective=objective,
            cost_ms=float(primary),
            rank_costs=list(costs[:pp_size]),
            overlap_comm=overlap_comm,
            memory_profile=memory_profile,
        )
    if best_plan is None:
        if models == {"shape-affine"}:
            raise ValueError(
                "No partition within both profiled shard-length coverage and "
                "memory bounds; collect wider shard profiles"
            )
        if memory_profile is not None:
            raise ValueError(
                "No memory-feasible partition within the requested PP size range; "
                "check per-device budgets, layer/KV costs and endpoint reserves"
            )
        raise ValueError("DP failed to find a finite partition")
    return best_plan


def plan_from_trace_dir(
    dump_dir: str | Path,
    *,
    objective: Objective = "throughput",
    workload: Workload = "all",
    warmup_steps: int = 5,
    num_layers: int | None = None,
    min_pp_size: int = 1,
    max_pp_size: int | None = None,
    overlap_comm: bool = False,
    memory_profile: PPMemoryProfile | str | Path | None = None,
    allow_unchecked_memory: bool = False,
    hetero: str | None = None,
    compute_model: ComputeModel = "shape-affine",
    fit_trace_dirs: Sequence[str | Path] = (),
    comm_source: str = "serving",
) -> PPPartitionPlan:
    if hetero:
        raise ValueError(
            "planner costs must come from measured traces; changed hetero scales "
            "require recollecting traces, not rescaling costs"
        )
    records = load_trace_records(dump_dir, comm_source=comm_source)
    if compute_model == "shape-affine":
        from vllm.distributed.pp_shape_cost import fit_shape_rank_costs

        trace_sets = [records] + [
            load_trace_records(p, comm_source=comm_source) for p in fit_trace_dirs
        ]
        costs = fit_shape_rank_costs(
            trace_sets,
            workload=workload,
            warmup_steps=warmup_steps,
            num_layers=num_layers,
        )
    else:
        raise ValueError(f"unknown compute_model: {compute_model}")
    memory_profile = resolve_memory_profile(
        memory_profile, dump_dir, allow_unchecked_memory=allow_unchecked_memory
    )
    if memory_profile is not None:
        trace_tp_sizes = {
            rec["tp_size"]
            for rows in records.values()
            for rec in rows
            if "tp_size" in rec
        }
        if trace_tp_sizes and trace_tp_sizes != {memory_profile.tp_size}:
            raise ValueError("memory profile TP size does not match timing traces")
    return partition_layers(
        costs,
        objective=objective,
        num_layers=num_layers,
        min_pp_size=min_pp_size,
        max_pp_size=max_pp_size,
        overlap_comm=overlap_comm,
        memory_profile=memory_profile,
        allow_unchecked_memory=allow_unchecked_memory,
    )


def format_plan(plan: PPPartitionPlan) -> str:
    lines = [
        f"objective={plan.objective}",
        f"cost_model={plan.cost_model}",
        f"compute_model={plan.compute_model}",
        f"predicted_cost_ms={plan.cost_ms:.4f}",
        f"pp_size={plan.pp_size}",
        f"VLLM_PP_LAYER_PARTITION={plan.env_value}",
        f"memory_feasibility_checked={plan.memory_profile is not None}",
        "rank t_layer_ms n_layers_selected t_comm_out_ms steps comm_source fixed_ms",
    ]
    for cost, n in zip(plan.rank_costs, plan.partitions):
        comm = "n/a" if cost.t_comm_out_ms is None else f"{cost.t_comm_out_ms:.4f}"
        lines.append(
            f"  {cost.pp_rank} {cost.t_layer_ms:.4f} {n} {comm} "
            f"{cost.n_steps} {cost.comm_source} {cost.t_fixed_ms:.4f}"
        )
    if plan.compute_model == "shape-affine":
        lines.append(
            "Shape-weighted mean resource demand; not finite-workload E2E throughput. "
            "Only profiled shard-length coverage is searched."
        )
    if plan.memory_profile is not None:
        lines.append("memory bounds (bytes): pp_rank tp_rank required budget headroom")
        for usage in plan.memory_profile.plan_usage(plan.partitions):
            lines.append(
                f"  {usage['pp_rank']} {usage['tp_rank']} "
                f"{usage['required_bytes']} {usage['budget_bytes']} "
                f"{usage['headroom_bytes']}"
            )
    return "\n".join(lines)


def add_cost_model_args(parser: Any) -> None:
    parser.add_argument(
        "--comm-source",
        choices=("replay", "serving"),
        default="serving",
        help="Idle link replay or paired same-host serving send/recv windows.",
    )
    parser.add_argument(
        "--compute-model",
        choices=("shape-affine",),
        default="shape-affine",
        help="Measured multi-shard fit with microbatch token/context features.",
    )
    parser.add_argument(
        "--fit-trace-dir",
        action="append",
        default=[],
        dest="fit_trace_dirs",
        help="Additional shard profile (repeatable); primary traces set shape weights.",
    )
    parser.add_argument(
        "--memory-profile",
        help=(
            "Target serving memory bounds JSON "
            "(default: trace-dir/pp_memory_profile.json)."
        ),
    )
    parser.add_argument(
        "--allow-unchecked-memory",
        action="store_true",
        help="Explicit timing-only analysis without memory bounds; no deployable plan.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--overlap-comm",
        dest="overlap_comm",
        action="store_true",
        help="Use ideal compute/communication overlap (does not change runtime).",
    )
    group.add_argument(
        "--no-overlap-comm",
        dest="overlap_comm",
        action="store_false",
        help="Use blocking recv+compute+send occupancy (default).",
    )
    parser.set_defaults(overlap_comm=False)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", help="Directory of pp_stage_pp*_tp*.jsonl files")
    parser.add_argument(
        "--objective",
        choices=("latency", "throughput"),
        default="throughput",
        help="latency = sequential sum; throughput = pipeline bottleneck",
    )
    parser.add_argument(
        "--workload",
        choices=("all", "decode", "prefill", "mixed"),
        default="all",
        help="Which traced steps to fit costs from",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Override hidden layer count (must match traced start/end)",
    )
    parser.add_argument(
        "--pp-size",
        type=int,
        default=None,
        help="Force exactly this many PP ranks (default: search 1..traced)",
    )
    parser.add_argument(
        "--min-pp-size",
        type=int,
        default=1,
        help="Smallest PP size to consider when --pp-size is omitted",
    )
    add_cost_model_args(parser)
    args = parser.parse_args(list(argv) if argv is not None else None)

    min_pp = args.pp_size if args.pp_size is not None else args.min_pp_size
    max_pp = args.pp_size
    plan = plan_from_trace_dir(
        args.dump_dir,
        objective=args.objective,
        workload=args.workload,
        warmup_steps=args.warmup_steps,
        num_layers=args.num_layers,
        min_pp_size=min_pp,
        max_pp_size=max_pp,
        overlap_comm=args.overlap_comm,
        memory_profile=args.memory_profile,
        allow_unchecked_memory=args.allow_unchecked_memory,
        compute_model=args.compute_model,
        fit_trace_dirs=args.fit_trace_dirs,
        comm_source=args.comm_source,
    )
    print(format_plan(plan))


if __name__ == "__main__":
    main()
