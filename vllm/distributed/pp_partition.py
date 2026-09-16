# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Choose a vLLM PP layer split from ``VLLM_PP_STAGE_TRACE`` JSONL files.

Trace files only know *stage* time (all local layers together). Llama-style
decoders are treated as homogeneous, so each rank yields one per-layer compute
cost:

    t_layer[rank] = median(compute_ms / n_layers)

Hops are independent of how many layers sit on a rank:

    t_comm[rank -> rank+1] = median(send_ms)

vLLM PP is a linear pipeline with a fixed rank order (rank 0 owns embed /
layer 0). The DP therefore only chooses how many *contiguous* layers each
already-ordered rank gets, and optionally whether trailing ranks should be
dropped.

Two objectives, matching EdgeShard:

* ``latency`` — sequential token time: sum(compute) + sum(comm hops)
* ``throughput`` — pipeline bottleneck: min max(stage compute, inbound comm)

Usage::

    python -m vllm.distributed.pp_partition ./pp_traces --objective throughput
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from vllm.distributed.pp_hetero import parse_scale_list, scale_at

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


@dataclass(frozen=True)
class PPPartitionPlan:
    partitions: list[int]
    objective: Objective
    cost_ms: float
    rank_costs: list[RankCost]

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
            recs = [
                rec
                for rec in records_by_rank[rank]
                if int(rec.get("num_tokens", 0)) > 0
            ]
        if not recs:
            raise ValueError(f"rank {rank} has no usable trace steps")

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

        t_layer_samples = [
            float(rec["compute_ms"]) / n_layers for rec in recs if rec["compute_ms"] > 0
        ]
        t_comm_samples = [
            float(rec["send_ms"])
            for rec in recs
            if rec.get("send_ms") is not None and float(rec["send_ms"]) >= 0
        ]
        costs.append(
            RankCost(
                pp_rank=rank,
                n_layers=n_layers,
                t_layer_ms=_median(t_layer_samples),
                t_comm_out_ms=_median(t_comm_samples) if t_comm_samples else None,
                n_steps=len(recs),
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
    overlap_comm: bool = True,
) -> PPPartitionPlan:
    """DP over contiguous layer counts on a fixed PP rank order.

    Rank 0 is always the source. Using ``pp_size < len(costs)`` drops the
    slowest *trailing* ranks; it does not reorder devices.
    """
    if not costs:
        raise ValueError("no rank costs")
    n_ranks = len(costs)
    n_layers = num_layers if num_layers is not None else sum(c.n_layers for c in costs)
    if n_layers < 1:
        raise ValueError("num_layers must be >= 1")
    max_pp = n_ranks if max_pp_size is None else min(max_pp_size, n_ranks)
    min_pp = max(1, min_pp_size)
    if min_pp > max_pp:
        raise ValueError(f"invalid pp_size range [{min_pp}, {max_pp}]")
    if n_layers < min_pp:
        raise ValueError(
            f"need at least one layer per rank: {n_layers=} {min_pp=}"
        )

    # dp[i][r] = best cost of placing the first i layers on r ranks.
    # prev[i][r] = number of layers placed before the last rank.
    dp = [[_INF] * (max_pp + 1) for _ in range(n_layers + 1)]
    prev = [[-1] * (max_pp + 1) for _ in range(n_layers + 1)]
    dp[0][0] = 0.0

    for i in range(1, n_layers + 1):
        dp[i][1] = _compute_ms(costs, 0, i)
        prev[i][1] = 0

    for r in range(2, max_pp + 1):
        for i in range(r, n_layers + 1):
            comm = _comm_ms(costs, r - 2)
            best = _INF
            best_k = -1
            for k in range(r - 1, i):
                last_layers = i - k
                compute = _compute_ms(costs, r - 1, last_layers)
                if objective == "latency":
                    cand = dp[k][r - 1] + comm + compute
                elif overlap_comm:
                    cand = max(dp[k][r - 1], comm, compute)
                else:
                    cand = max(dp[k][r - 1], comm + compute)
                if cand < best:
                    best = cand
                    best_k = k
            dp[i][r] = best
            prev[i][r] = best_k

    best_r = min(range(min_pp, max_pp + 1), key=lambda r: dp[n_layers][r])
    if not math.isfinite(dp[n_layers][best_r]):
        raise ValueError("DP failed to find a finite partition")

    partitions = [0] * best_r
    i = n_layers
    r = best_r
    while r >= 1:
        k = prev[i][r]
        partitions[r - 1] = i - k
        i = k
        r -= 1
    if i != 0 or sum(partitions) != n_layers or any(n <= 0 for n in partitions):
        raise RuntimeError(f"invalid backtrace {partitions} for {n_layers} layers")

    return PPPartitionPlan(
        partitions=partitions,
        objective=objective,
        cost_ms=float(dp[n_layers][best_r]),
        rank_costs=list(costs[:best_r]),
    )


def plan_from_trace_dir(
    dump_dir: str | Path,
    *,
    objective: Objective = "throughput",
    workload: Workload = "all",
    warmup_steps: int = 5,
    num_layers: int | None = None,
    min_pp_size: int = 1,
    max_pp_size: int | None = None,
    overlap_comm: bool = True,
    compute_scale: str | None = None,
    comm_scale: str | None = None,
) -> PPPartitionPlan:
    records = load_trace_records(dump_dir)
    costs = fit_rank_costs(
        records,
        workload=workload,
        warmup_steps=warmup_steps,
        num_layers=num_layers,
    )
    costs = scale_rank_costs(
        costs,
        compute_scales=parse_scale_list(compute_scale),
        comm_scales=parse_scale_list(comm_scale),
    )
    return partition_layers(
        costs,
        objective=objective,
        num_layers=num_layers,
        min_pp_size=min_pp_size,
        max_pp_size=max_pp_size,
        overlap_comm=overlap_comm,
    )


def format_plan(plan: PPPartitionPlan) -> str:
    lines = [
        f"objective={plan.objective}",
        f"predicted_cost_ms={plan.cost_ms:.4f}",
        f"pp_size={plan.pp_size}",
        f"VLLM_PP_LAYER_PARTITION={plan.env_value}",
        "rank t_layer_ms n_layers_traced t_comm_out_ms steps",
    ]
    for cost, n in zip(plan.rank_costs, plan.partitions):
        comm = "n/a" if cost.t_comm_out_ms is None else f"{cost.t_comm_out_ms:.4f}"
        lines.append(
            f"  {cost.pp_rank} {cost.t_layer_ms:.4f} {n} {comm} {cost.n_steps}"
        )
    return "\n".join(lines)


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
    parser.add_argument(
        "--no-overlap-comm",
        action="store_true",
        help="Throughput DP uses compute+comm instead of max(compute, comm)",
    )
    parser.add_argument(
        "--compute-scale",
        type=str,
        default=None,
        help="Per-PP-rank compute slowdown, e.g. 1,2 (rank1 twice as slow).",
    )
    parser.add_argument(
        "--comm-scale",
        type=str,
        default=None,
        help="Per-hop comm slowdown rank i->i+1, e.g. 1,4.",
    )
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
        overlap_comm=not args.no_overlap_comm,
        compute_scale=args.compute_scale,
        comm_scale=args.comm_scale,
    )
    print(format_plan(plan))


if __name__ == "__main__":
    main()
