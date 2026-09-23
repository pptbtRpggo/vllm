# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shape-conditioned stage costs from multiple shard profiles.

This is a mean resource-demand surrogate, not a simulation of vLLM scheduling:
throughput minimizes max_rank E[recv + compute + send]. In particular this is
not E[max_rank(...)] or the completion time of a finite autoregressive workload.
Only homogeneous decoder layers and a fixed device order / PP size are modeled.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm.distributed.pp_link_profile import measured_transfer_ms

if TYPE_CHECKING:
    from vllm.distributed.pp_partition import RankCost, Workload

Shape = tuple[tuple[int, int, int], ...]

FEATURES = (
    "constant",
    "requests",
    "prompt_tokens",
    "generation_tokens",
    "attention_pairs",
)


def shape_features(shape: Shape) -> list[float]:
    """Work descriptors, not a hardware timing formula; coefficients are measured.

    q*k + q*(q+1)/2 counts causal query/key pairs, including cached context.
    Separate prompt/generation tokens allow different measured kernel costs.
    """
    return [
        1.0,
        float(len(shape)),
        float(sum(p for _, _, p in shape)),
        float(sum(q - p for q, _, p in shape)),
        float(sum(q * k + q * (q + 1) / 2 for q, k, _ in shape)),
    ]


def _nnls(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Small active-set nonnegative least squares; no optional scipy dependency."""
    scale = np.maximum(np.linalg.norm(x, axis=0), 1e-12)
    a = x / scale
    solution = np.zeros(a.shape[1])
    active = np.zeros(a.shape[1], dtype=bool)
    tolerance = 1e-10 * max(1.0, np.linalg.norm(y))
    for _ in range(1000):
        gradient = a.T @ (y - a @ solution)
        gradient[active] = -np.inf
        if np.max(gradient) <= tolerance:
            return solution / scale
        active[int(np.argmax(gradient))] = True
        for _ in range(1000):
            candidate = np.zeros_like(solution)
            candidate[active] = np.linalg.lstsq(a[:, active], y, rcond=None)[0]
            bad = active & (candidate <= 0)
            if not bad.any():
                solution = candidate
                break
            denominator = solution[bad] - candidate[bad]
            ratios = np.divide(
                solution[bad],
                denominator,
                out=np.zeros_like(denominator),
                where=denominator > 0,
            )
            solution += float(np.min(ratios)) * (candidate - solution)
            drop = active & (solution <= tolerance)
            solution[drop] = 0
            active[drop] = False
        else:
            raise ValueError("nonnegative cost fit did not converge")
    raise ValueError("nonnegative cost fit did not converge")


def _feature_fit(
    profiles: list[list[dict]], counts: Counter
) -> tuple[float, float, dict]:
    """Fit fixed(shape) + layers * marginal(shape) without exact shape matches."""
    matrices, targets = [], []
    lengths = set()
    for rows in profiles:
        n = rows[0]["end_layer"] - rows[0]["start_layer"]
        lengths.add(n)
        f = np.asarray([shape_features(batch_shape_key(r)) for r in rows])
        matrices.append(np.concatenate((f, f * n), axis=1))
        targets.append(np.asarray([r["compute_wall_ms"] for r in rows]))
    if len(lengths) < 2:
        raise ValueError("each rank needs at least two distinct shard lengths")

    def train(indices):
        # Equal weight per profile: a slower run must not get extra fit weight
        # simply because its scheduler produced more microbatches.
        x = np.concatenate([matrices[i] / math.sqrt(len(targets[i])) for i in indices])
        y = np.concatenate([targets[i] / math.sqrt(len(targets[i])) for i in indices])
        return _nnls(x, y)

    coef = train(range(len(profiles)))
    reference = np.average(
        np.asarray([shape_features(s) for s in counts]),
        axis=0,
        weights=list(counts.values()),
    )
    errors = []
    for i in range(len(profiles)):
        remaining = [j for j in range(len(profiles)) if j != i]
        other_lengths = {
            profiles[j][0]["end_layer"] - profiles[j][0]["start_layer"]
            for j in remaining
        }
        if len(other_lengths) >= 2:
            prediction = matrices[i] @ train(remaining)
            errors.append(
                dict(
                    profile=i,
                    wape=float(
                        np.abs(prediction - targets[i]).sum() / targets[i].sum()
                    ),
                )
            )
    x, y = np.concatenate(matrices), np.concatenate(targets)
    return (
        float(reference @ coef[:5]),
        float(reference @ coef[5:]),
        dict(
            method="feature_affine",
            features=list(FEATURES),
            fixed_coefficients=coef[:5].tolist(),
            layer_coefficients=coef[5:].tolist(),
            reference_mean_features=reference.tolist(),
            training_wape=float(np.abs(x @ coef - y).sum() / y.sum()),
            leave_one_profile_out=errors,
            design_rank=int(
                np.linalg.matrix_rank(x / np.maximum(np.linalg.norm(x, axis=0), 1e-12))
            ),
            parameter_count=int(x.shape[1]),
        ),
    )


def batch_shape_key(record: dict[str, Any]) -> Shape:
    shape = record.get("batch_shape")
    if not isinstance(shape, list) or not shape:
        raise ValueError("shape-affine needs complete batch_shape; recollect traces")
    entries = []
    for entry in shape:
        values = tuple(
            entry.get(k) for k in ("query_tokens", "context_tokens", "prompt_tokens")
        )
        if any(type(v) is not int for v in values):
            raise ValueError("batch_shape lengths must be integers")
        q, k, p = values
        if q <= 0 or k < 0 or not 0 <= p <= q:
            raise ValueError("invalid query/context/prompt lengths in batch_shape")
        entries.append((q, k, p))
    if (
        sum(q for q, _, _ in entries) != record["num_tokens"]
        or len(entries) != record["num_reqs"]
    ):
        raise ValueError("batch_shape does not match batch token/request counts")
    # Sequence ordering does not change the modeled workload. Keep q/k paired:
    # equal total tokens or equal summed context alone is insufficient.
    return tuple(sorted(entries))


def _nonnegative_affine(points: list[tuple[int, float]]) -> tuple[float, float]:
    """Least squares a + b*n with a,b >= 0; each shard length has equal weight."""
    xs, ys = zip(*points)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    slope = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x in xs)
    intercept = my - slope * mx
    candidates = [
        (max(0.0, my), 0.0),
        (0.0, max(0.0, sum(x * y for x, y in points) / sum(x * x for x in xs))),
    ]
    if intercept >= 0 and slope >= 0:
        candidates.append((intercept, slope))
    return min(
        candidates, key=lambda ab: sum((ab[0] + ab[1] * x - y) ** 2 for x, y in points)
    )


def fit_shape_rank_costs(
    trace_sets: list[dict[int, list[dict[str, Any]]]],
    *,
    workload: Workload,
    warmup_steps: int,
    num_layers: int | None = None,
) -> list[RankCost]:
    from vllm.distributed.pp_partition import RankCost, _keep_record

    if len(trace_sets) < 2:
        raise ValueError(
            "shape-affine requires multiple shard profiles; add --fit-trace-dir "
            "with a different partition"
        )
    ranks = sorted(trace_sets[0])
    if not ranks or ranks != list(range(len(ranks))):
        raise ValueError("PP ranks must be contiguous from 0")
    totals = set()
    selected = []
    metadata = {rank: set() for rank in ranks}
    for traces in trace_sets:
        if sorted(traces) != ranks:
            raise ValueError("shape-affine requires the same PP ranks in every profile")
        filtered = {}
        end = 0
        for rank in ranks:
            rows = [r for r in traces[rank] if _keep_record(r, workload, warmup_steps)]
            if not rows:
                raise ValueError(f"rank {rank} has no matching workload samples")
            ranges = {(r["start_layer"], r["end_layer"]) for r in rows}
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
            for rec in rows:
                if rec.get("compute_model", "shape-affine") != "shape-affine":
                    raise ValueError(
                        "shape-affine cannot consume layer-measured traces"
                    )
                if rec["pp_size"] != len(ranks):
                    raise ValueError("trace PP size does not match its ranks")
                batch_shape_key(rec)
                value = rec.get("compute_wall_ms")
                if value is None or not math.isfinite(value) or value <= 0:
                    raise ValueError(
                        "shape-affine needs positive measured compute_wall_ms"
                    )
                # Mixing slowdown or TP configurations would conflate device
                # changes with the effect of layer count. No implicit rescaling.
                configuration = tuple(
                    rec.get(k, 1) for k in ("tp_size", "compute_scale", "comm_scale")
                )
                if any(not math.isfinite(v) or v <= 0 for v in configuration):
                    raise ValueError("invalid TP size or hetero scales")
                tp_rank = rec["tp_rank"]
                if type(tp_rank) is not int or not 0 <= tp_rank < configuration[0]:
                    raise ValueError("invalid timing representative TP rank")
                metadata[rank].add((*configuration, tp_rank))
            filtered[rank] = rows
        totals.add(end)
        selected.append(filtered)
    if len(totals) != 1 or (num_layers is not None and num_layers not in totals):
        raise ValueError("profiles must have the same total number of layers")
    if any(len(values) != 1 for values in metadata.values()):
        raise ValueError("profiles mix TP representatives, TP size or hetero scales")
    if len({next(iter(values))[0] for values in metadata.values()}) != 1:
        raise ValueError("profiles mix TP size across ranks")

    # The primary directory defines one common workload distribution for all
    # ranks. Extra profiles supply costs, never silently change the weights.
    counts = Counter(batch_shape_key(r) for r in selected[0][0])
    total = sum(counts.values())
    costs = []
    for rank in ranks:
        samples = defaultdict(lambda: defaultdict(list))
        comm = defaultdict(list)
        comm_sources = set()
        n_steps = 0
        for traces in selected:
            for rec in traces[rank]:
                shape = batch_shape_key(rec)
                if shape not in counts:
                    continue
                value = rec.get("compute_wall_ms")
                if value is None or not math.isfinite(value) or value <= 0:
                    raise ValueError(
                        "shape-affine needs positive measured compute_wall_ms"
                    )
                n = rec["end_layer"] - rec["start_layer"]
                samples[shape][n].append(value)
                if rank < len(ranks) - 1:
                    comm[shape].append(measured_transfer_ms(rec))
                    comm_sources.add(rec["send_service_source"])
                n_steps += 1
        fixed = marginal = transfer_mean = 0.0
        lower, upper = 1, next(iter(totals))
        diagnostics = []
        exact_coverage = all(len(samples[shape]) >= 2 for shape in counts)
        if not exact_coverage:
            profiles = [traces[rank] for traces in selected]
            fixed, marginal, diagnostic = _feature_fit(profiles, counts)
            diagnostics.append(diagnostic)
            lengths = [
                rows[0]["end_layer"] - rows[0]["start_layer"] for rows in profiles
            ]
            lower, upper = min(lengths), max(lengths)
            n_steps = sum(map(len, profiles))
            # Use the reference serving distribution directly. Other partitions
            # can change both batch formation and communication contention.
            if rank < len(ranks) - 1:
                reference_rows = selected[0][rank]
                if Counter(batch_shape_key(r) for r in reference_rows) != counts:
                    raise ValueError(
                        "reference ranks have mismatched microbatch distributions"
                    )
                transfer_mean = statistics.mean(
                    measured_transfer_ms(r) for r in reference_rows
                )
                comm_sources = {r["send_service_source"] for r in reference_rows}
        for shape, count in sorted(counts.items()) if exact_coverage else []:
            by_length = samples[shape]
            points = sorted((n, statistics.mean(v)) for n, v in by_length.items())
            a, b = _nonnegative_affine(points)
            weight = count / total
            fixed += weight * a
            marginal += weight * b
            if rank < len(ranks) - 1:
                # Measurements naturally vary across runs. Give each observed
                # sample equal weight within a shape, then apply reference weights.
                transfer_mean += weight * statistics.mean(comm[shape])
            lower, upper = max(lower, points[0][0]), min(upper, points[-1][0])
            diagnostics.append(
                dict(
                    shape=[list(entry) for entry in shape],
                    weight=weight,
                    fixed_ms=a,
                    layer_ms=b,
                    measured_points=points,
                    max_fit_relative_error=max(
                        abs(a + b * n - y) / y for n, y in points
                    ),
                )
            )
        if lower > upper:
            raise ValueError(f"rank {rank} has no common shard-length coverage")
        if len(comm_sources) > 1:
            raise ValueError("profiles mix communication measurement methods")
        initial = selected[0][rank][0]
        costs.append(
            RankCost(
                pp_rank=rank,
                n_layers=initial["end_layer"] - initial["start_layer"],
                t_layer_ms=marginal,
                t_fixed_ms=fixed,
                t_comm_out_ms=transfer_mean if rank < len(ranks) - 1 else None,
                n_steps=n_steps,
                comm_source=next(iter(comm_sources), "none"),
                compute_model="shape-affine",
                min_layers=lower,
                max_layers=upper,
                profiled_pp_size=len(ranks),
                shape_fits=tuple(diagnostics),
            )
        )
    return costs
