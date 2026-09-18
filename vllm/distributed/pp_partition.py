# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Choose a vLLM PP layer split from ``VLLM_PP_STAGE_TRACE`` JSONL files.

Trace files only know *stage* time (all local layers together). Llama-style
decoders are treated as homogeneous, so each rank yields one per-layer compute
cost:

    t_layer[rank] = median(compute_ms / n_layers)

Hops are independent of how many layers sit on a rank:

    t_comm[rank -> rank+1] = median(send_transfer_ms)

Legacy ``send_ms`` includes peer waiting. Using it as a link service cost
requires explicit opt-in and marks the resulting plan as an approximation.

vLLM PP is a linear pipeline with a fixed rank order (rank 0 owns embed /
layer 0). The DP therefore only chooses how many *contiguous* layers each
already-ordered rank gets, and optionally whether trailing ranks should be
dropped.

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
import statistics
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from vllm.distributed.pp_hetero import parse_hetero_spec, scale_at
from vllm.distributed.pp_memory import PPMemoryProfile, resolve_memory_profile

Objective = Literal["latency", "throughput"]
Workload = Literal["all", "decode", "prefill"]

_INF = math.inf


@dataclass(frozen=True)
class RankCost:
    pp_rank: int
    n_layers: int
    t_layer_ms: float
    t_comm_out_ms: float | None
    n_steps: int
    comm_source: str = "provided"


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
            "cost_kind": (
                "steady_state_cycle_ms"
                if self.objective == "throughput" else "sequential_latency_ms"
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
                if self.memory_profile else []
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
                }
                for cost in self.rank_costs
            ],
        }


def load_trace_records(dump_dir: str | Path) -> dict[int, list[dict[str, Any]]]:
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
    if int(rec.get("step", 0)) < warmup_steps:
        return False
    if int(rec.get("num_tokens", 0)) <= 0:
        return False
    ctx = int(rec.get("num_ctx_tokens", 0))
    gen = int(rec.get("num_generation_tokens", 0))
    if workload == "decode":
        return gen > 0 and ctx == 0
    if workload == "prefill":
        return ctx > 0
    return True


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median of empty sample")
    return float(statistics.median(values))


def fit_rank_costs(
    records_by_rank: dict[int, list[dict[str, Any]]],
    *,
    workload: Workload = "all",
    warmup_steps: int = 5,
    num_layers: int | None = None,
    allow_wall_time_comm: bool = False,
) -> list[RankCost]:
    """Turn stage traces into per-rank layer compute and outbound comm."""
    if not records_by_rank:
        raise ValueError("no trace records")

    ranks = sorted(records_by_rank)
    if ranks != list(range(len(ranks))):
        raise ValueError(f"PP ranks must be contiguous from 0, got {ranks}")

    costs: list[RankCost] = []
    inferred_layers = 0
    for rank in ranks:
        recs = [
            rec
            for rec in records_by_rank[rank]
            if _keep_record(rec, workload, warmup_steps)
        ]
        if not recs:
            raise ValueError(
                f"rank {rank} has no usable trace steps for {workload=} "
                f"after {warmup_steps=}; collect more matching steps"
            )

        starts = {rec.get("start_layer") for rec in recs}
        ends = {rec.get("end_layer") for rec in recs}
        start, end = recs[-1].get("start_layer"), recs[-1].get("end_layer")
        if start is None or end is None or start in (None,) or end in (None,):
            raise ValueError(
                f"rank {rank} traces missing start_layer/end_layer; "
                "re-run with a loaded model or pass --num-layers after "
                "annotating traces"
            )
        if len(starts) != 1 or len(ends) != 1:
            raise ValueError(
                f"rank {rank} traces mix multiple layer ranges: {starts} -> {ends}"
            )
        n_layers = int(end) - int(start)
        if n_layers <= 0:
            raise ValueError(f"rank {rank} has non-positive layer count {n_layers}")
        inferred_layers += n_layers

        t_layer_samples = []
        for rec in recs:
            compute = float(rec["compute_ms"])
            if not math.isfinite(compute) or compute < 0:
                raise ValueError(f"rank {rank} has invalid compute_ms={compute}")
            if compute > 0:
                t_layer_samples.append(compute / n_layers)
        if not t_layer_samples:
            raise ValueError(f"rank {rank} has no positive compute samples")
        t_comm_samples = []
        comm_sources = set()
        for rec in recs:
            comm_ms = rec.get("send_transfer_ms")
            source = "link_model"
            if comm_ms is None:
                comm_ms = rec.get("send_ms")
                source = "wall_time"
                if comm_ms is not None and not allow_wall_time_comm:
                    raise ValueError(
                        f"rank {rank} is missing send_transfer_ms; send_ms "
                        "includes peer waiting. Recollect with calibrated "
                        "VLLM_PP_COMM_BANDWIDTH_GBPS or explicitly use "
                        "--allow-wall-time-comm for an approximate plan"
                    )
            if comm_ms is not None:
                comm_ms = float(comm_ms)
                if not math.isfinite(comm_ms) or comm_ms < 0:
                    raise ValueError(f"rank {rank} has invalid transfer cost {comm_ms}")
                t_comm_samples.append(comm_ms)
                comm_sources.add(source)
        if len(comm_sources) > 1:
            raise ValueError(f"rank {rank} mixes modeled and wall-time communication")
        comm_source = next(iter(comm_sources), "none")
        if comm_source == "wall_time":
            warnings.warn(
                f"rank {rank}: using send_ms including peer waits; "
                "partition cost is only an approximation",
                UserWarning,
                stacklevel=2,
            )
        costs.append(
            RankCost(
                pp_rank=rank,
                n_layers=n_layers,
                t_layer_ms=_median(t_layer_samples),
                t_comm_out_ms=_median(t_comm_samples) if t_comm_samples else None,
                n_steps=len(recs),
                comm_source=comm_source,
            )
        )

    if num_layers is not None and num_layers != inferred_layers:
        raise ValueError(
            f"--num-layers={num_layers} does not match traced layers "
            f"{inferred_layers}"
        )
    return costs


def scale_rank_costs(
    costs: Sequence[RankCost],
    *,
    compute_scales: Sequence[float] = (),
    comm_scales: Sequence[float] = (),
) -> list[RankCost]:
    """Multiply fitted ``t_layer`` / ``t_comm`` to emulate slower ranks/hops.

    Used by ``--skip-run`` so a homogeneous trace can be planned as if the
    devices were heterogeneous, without another engine launch.
    """
    if not compute_scales and not comm_scales:
        return list(costs)
    scaled: list[RankCost] = []
    for cost in costs:
        t_layer = cost.t_layer_ms * scale_at(compute_scales, cost.pp_rank)
        t_comm = cost.t_comm_out_ms
        if t_comm is not None:
            t_comm = t_comm * scale_at(comm_scales, cost.pp_rank)
        scaled.append(replace(cost, t_layer_ms=t_layer, t_comm_out_ms=t_comm))
    return scaled


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
    return n_layers * costs[rank].t_layer_ms


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
    """
    if not costs:
        raise ValueError("no rank costs")
    if objective not in ("latency", "throughput"):
        raise ValueError(f"unknown objective: {objective}")
    for rank, cost in enumerate(costs):
        if cost.pp_rank != rank:
            raise ValueError("rank costs must be ordered contiguously from rank 0")
        values = [cost.t_layer_ms]
        if cost.t_comm_out_ms is not None:
            values.append(cost.t_comm_out_ms)
        if any(not math.isfinite(x) or x < 0 for x in values):
            raise ValueError(f"rank {rank} costs must be finite and nonnegative")
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
    if min_pp > max_pp:
        raise ValueError(f"invalid pp_size range [{min_pp}, {max_pp}]")
    if n_layers < min_pp:
        raise ValueError(
            f"need at least one layer per rank: {n_layers=} {min_pp=}"
        )

    best_plan = None
    # Fix the final PP size before building the DP: the final rank has no
    # outgoing hop, while a prefix's last rank still sends to its successor.
    # Charging only the new rank's incoming hop loses the sender's occupancy.
    for pp_size in range(min_pp, min(max_pp, n_layers) + 1):
        hops = [0.0] + [_comm_ms(costs, r) for r in range(pp_size - 1)] + [0.0]
        dp = [[_INF] * (pp_size + 1) for _ in range(n_layers + 1)]
        prev = [[-1] * (pp_size + 1) for _ in range(n_layers + 1)]
        dp[0][0] = 0.0
        for r in range(1, pp_size + 1):
            incoming, outgoing = hops[r - 1], hops[r]
            for i in range(r, n_layers + 1):
                for k in range(r - 1, i):
                    if not math.isfinite(dp[k][r - 1]):
                        continue
                    if memory_profile is not None and not memory_profile.fits(
                        r - 1, k, i, pp_size
                    ):
                        continue
                    compute = _compute_ms(costs, r - 1, i - k)
                    if objective == "latency":
                        # A link is counted once along the sequential path.
                        cand = dp[k][r - 1] + incoming + compute
                    elif overlap_comm:
                        cand = max(dp[k][r - 1], incoming, compute)
                    else:
                        # A link occupies two devices simultaneously, not two
                        # sequential transfers. Taking max avoids double-counting.
                        cand = max(dp[k][r - 1], incoming + compute + outgoing)
                    if cand < dp[i][r]:
                        dp[i][r] = cand
                        prev[i][r] = k
        if best_plan is not None and dp[n_layers][pp_size] >= best_plan.cost_ms:
            continue
        if not math.isfinite(dp[n_layers][pp_size]):
            continue
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
            cost_ms=float(dp[n_layers][pp_size]),
            rank_costs=list(costs[:pp_size]),
            overlap_comm=overlap_comm,
            memory_profile=memory_profile,
        )
    if best_plan is None:
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
    allow_wall_time_comm: bool = False,
    memory_profile: PPMemoryProfile | str | Path | None = None,
    allow_unchecked_memory: bool = False,
    hetero: str | None = None,
) -> PPPartitionPlan:
    records = load_trace_records(dump_dir)
    costs = fit_rank_costs(
        records,
        workload=workload,
        warmup_steps=warmup_steps,
        num_layers=num_layers,
        allow_wall_time_comm=allow_wall_time_comm,
    )
    if hetero is None:
        import vllm.envs as envs

        hetero = envs.VLLM_PP_HETERO
    compute, comm = parse_hetero_spec(hetero)
    if hetero:
        # New worker traces already contain their recorded slowdown. Interpret
        # the environment as the target configuration, not an extra multiplier.
        # Legacy traces without metadata are assumed to be unscaled.
        def relative_scales(target: Sequence[float], field: str) -> tuple[float, ...]:
            factors = []
            for rank in range(len(costs)):
                recorded = {
                    float(rec.get(field, 1.0))
                    for rec in records[rank]
                    if _keep_record(rec, workload, warmup_steps)
                }
                if len(recorded) != 1:
                    raise ValueError(f"rank {rank} traces mix multiple {field} values")
                source = recorded.pop()
                if not math.isfinite(source) or source <= 0:
                    raise ValueError(f"rank {rank} has invalid {field}={source}")
                factors.append(scale_at(target, rank) / source)
            return tuple(factors)

        compute = relative_scales(compute, "compute_scale")
        comm = relative_scales(comm, "comm_scale")
    costs = scale_rank_costs(
        costs,
        compute_scales=compute,
        comm_scales=comm,
    )
    memory_profile = resolve_memory_profile(
        memory_profile, dump_dir, allow_unchecked_memory=allow_unchecked_memory
    )
    if memory_profile is not None:
        trace_tp_sizes = {
            rec["tp_size"] for rows in records.values()
            for rec in rows if "tp_size" in rec
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
        f"predicted_cost_ms={plan.cost_ms:.4f}",
        f"pp_size={plan.pp_size}",
        f"VLLM_PP_LAYER_PARTITION={plan.env_value}",
        f"memory_feasibility_checked={plan.memory_profile is not None}",
        "rank t_layer_ms n_layers_selected t_comm_out_ms steps comm_source",
    ]
    for cost, n in zip(plan.rank_costs, plan.partitions):
        comm = "n/a" if cost.t_comm_out_ms is None else f"{cost.t_comm_out_ms:.4f}"
        lines.append(
            f"  {cost.pp_rank} {cost.t_layer_ms:.4f} {n} {comm} "
            f"{cost.n_steps} {cost.comm_source}"
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
    parser.add_argument(
        "--allow-wall-time-comm",
        action="store_true",
        help="Allow approximate legacy send_ms costs, including peer waiting.",
    )


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
        choices=("all", "decode", "prefill"),
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
        allow_wall_time_comm=args.allow_wall_time_comm,
        memory_profile=args.memory_profile,
        allow_unchecked_memory=args.allow_unchecked_memory,
    )
    print(format_plan(plan))


if __name__ == "__main__":
    main()
