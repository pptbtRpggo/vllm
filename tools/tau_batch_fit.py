# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fit shared stage latency models from completed, validated tau trace ranges.

Requires NumPy. No vLLM or accelerator imports. By default, pools PP ranks into
one prefill and one decode function per formula. Each label remains one stage's
compute duration. Holds out entire waves, including all their ranks and decode
tokens. Coefficients use milliseconds.
"""

import argparse
import json
from pathlib import Path

import numpy as np

MODELS = {
    "tau_affine": ("n*s_max", "n", "1"),
    "scls_bilinear": ("n*s_max", "n", "s_max", "1"),
    "unpadded_comparison": ("s_sum", "n", "s_max", "1"),
}
DEFAULT_MODELS = ("scls_bilinear", "unpadded_comparison")


def design(rows, model):
    n, s, total = rows[:, 0], rows[:, 1], rows[:, 2]
    if model == "tau_affine":
        return np.column_stack((n * s, n, np.ones(len(rows))))
    first = n * s if model == "scls_bilinear" else total
    return np.column_stack((first, n, s, np.ones(len(rows))))


def solve(x, y):
    scale = np.linalg.norm(x, axis=0)
    scale[scale == 0] = 1
    scaled = x / scale
    coefficients, _, rank, singular = np.linalg.lstsq(scaled, y, rcond=None)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    if condition is not None and not np.isfinite(condition):
        condition = None
    return {
        "rank": int(rank),
        "columns": x.shape[1],
        "scaled_condition": condition,
        "coefficients": (coefficients / scale).tolist() if rank == x.shape[1] else None,
        "status": "identified" if rank == x.shape[1] else "rank_deficient",
    }


def metrics(actual, predicted):
    error = predicted - actual
    absolute = np.abs(error)
    return {
        "n": len(actual),
        "mae_ms": float(absolute.mean()),
        "mean_error_ms": float(error.mean()),
        "rmse_ms": float(np.sqrt(np.mean(error**2))),
        "mape_percent": float(np.mean(absolute / actual) * 100),
        "p95_absolute_error_ms": float(np.quantile(absolute, 0.95)),
        "underprediction_fraction": float(np.mean(predicted < actual)),
        "nonpositive_prediction_count": int(np.sum(predicted <= 0)),
    }


def load_samples(trace, reports):
    """Reject failed/overlapping ranges and verify emit/compute joins again."""
    ranges = []
    for path in reports:
        report = json.loads(path.read_text())
        report = report.get("trace", report)
        if report.get("passed") is not True:
            raise ValueError(f"Unvalidated trace range: {path}")
        start, end = report["trace_start_offset"], report["trace_end_offset"]
        if not 0 <= start < end <= trace.stat().st_size:
            raise ValueError(f"Invalid byte range in {path}")
        ranges.append((start, end, str(path)))
    ranges.sort()
    if any(b[0] < a[1] for a, b in zip(ranges, ranges[1:])):
        raise ValueError("Overlapping ranges would duplicate training samples")
    samples, groups, seen = {}, {}, set()
    for start, end, path in ranges:
        emits = {}
        with trace.open("rb") as f:
            f.seek(start)
            while f.tell() < end:
                line = f.readline(end - f.tell())
                if not line.endswith(b"\n"):
                    raise ValueError(f"Truncated JSONL record: {path}")
                event = json.loads(line)
                kind = event.get("event")
                fid = event.get("fwd_id")
                if kind == "emit":
                    emits[fid] = event
                elif kind == "compute":
                    if fid not in emits:
                        raise ValueError("compute has no matching emit")
                    em = emits[fid]
                    rank, phase = event["pp_rank"], event["phase"]
                    if (
                        type(rank) is not int
                        or rank < 0
                        or phase not in ("prefill", "decode")
                    ):
                        raise ValueError("Invalid compute rank or phase")
                    key = (fid, rank)
                    if key in seen:
                        raise ValueError(
                            "Duplicate compute identity across input ranges"
                        )
                    seen.add(key)
                    for field in ("phase", "n", "s_max", "s_sum", "tokens", "seq_lens"):
                        if event.get(field) != em.get(field):
                            raise ValueError(f"Mismatched compute feature: {field}")
                    n, s, total = event["n"], event["s_max"], event["s_sum"]
                    duration = (event["end_ts_ns"] - event["start_ts_ns"]) / 1e6
                    values = (n, s, total, duration)
                    if not np.all(np.isfinite(values)) or min(values) <= 0:
                        raise ValueError("Invalid latency or feature")
                    lengths = event["seq_lens"]
                    if len(lengths) != n or max(lengths) != s or sum(lengths) != total:
                        raise ValueError("Inconsistent sequence lengths")
                    # One server run per invocation: wave IDs are unique here.
                    group = em["wave_id"]
                    groups.setdefault(group, len(groups))
                    samples.setdefault((rank, phase), []).append(
                        (n, s, total, duration, group, rank)
                    )
    if not samples:
        raise ValueError("No compute samples")
    return samples, groups, ranges


def fit_group(rows, train_groups, stage_layers=None, models=tuple(MODELS)):
    rows = np.asarray(rows, dtype=float)
    y = rows[:, 3]
    train = np.isin(rows[:, 4], list(train_groups))
    test = ~train
    results = {
        "samples": len(rows),
        "train_samples": int(train.sum()),
        "validation_samples": int(test.sum()),
        "n_values": sorted(set(rows[:, 0].astype(int).tolist())),
        "s_max_range": [int(rows[:, 1].min()), int(rows[:, 1].max())],
        "s_sum_range": [int(rows[:, 2].min()), int(rows[:, 2].max())],
        "observed_ms": {
            "mean": float(y.mean()),
            "p50": float(np.median(y)),
            "p99": float(np.quantile(y, 0.99)),
            "max": float(y.max()),
        },
        "models": {},
    }
    if rows.shape[1] > 5:
        results["pp_ranks"] = sorted(set(rows[:, 5].astype(int).tolist()))
    for name in models:
        columns = MODELS[name]
        x = design(rows, name)
        full = solve(x, y)
        entry = {
            "columns": columns,
            "all_data_fit": full,
            "validation_status": "insufficient_waves",
        }
        if train.any() and test.any():
            training = solve(x[train], y[train])
            entry["training_fit"] = training
            entry["validation_status"] = training["status"]
            if training["coefficients"] is not None:
                prediction = x[test] @ np.asarray(training["coefficients"])
                entry["validation"] = metrics(y[test], prediction)
                if rows.shape[1] > 5:
                    # Diagnose the shared coefficients; never refit by rank here.
                    ranks = rows[test, 5].astype(int)
                    entry["validation_by_stage"] = {
                        f"pp{rank}": metrics(
                            y[test][ranks == rank], prediction[ranks == rank]
                        )
                        for rank in sorted(set(ranks.tolist()))
                    }
                baseline = np.full(test.sum(), y[train].mean())
                entry["constant_baseline"] = metrics(y[test], baseline)
                entry["beats_constant_rmse"] = (
                    entry["validation"]["rmse_ms"]
                    < entry["constant_baseline"]["rmse_ms"]
                )
                within = np.all(
                    (x[test, :-1] >= x[train, :-1].min(axis=0))
                    & (x[test, :-1] <= x[train, :-1].max(axis=0)),
                    axis=1,
                )
                entry["validation_bounds_features"] = columns[:-1]
                entry["validation_outside_training_bounds"] = int((~within).sum())
        if name == "tau_affine" and stage_layers and full["coefficients"] is not None:
            a, b, c = full["coefficients"]
            entry["layer_normalized"] = {
                "alpha_attn_effective": a / stage_layers,
                "alpha_proj_times_d_squared_plus_beta": b / stage_layers,
                "gamma_effective": c / stage_layers,
                "note": (
                    "alpha_proj and beta are not separately identifiable at fixed d"
                ),
            }
        results["models"][name] = entry
    return results


def fit(
    trace,
    reports,
    stage_layers=None,
    validation_fraction=0.2,
    group_by="phase",
    models=DEFAULT_MODELS,
    validation_reports=None,
):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must be between 0 and 1")
    if stage_layers is not None and stage_layers <= 0:
        raise ValueError("stage layers must be positive")
    if group_by not in ("phase", "stage-phase"):
        raise ValueError("group_by must be phase or stage-phase")
    if not models or any(name not in MODELS for name in models):
        raise ValueError("Select at least one known latency model")
    samples, groups, ranges = load_samples(trace, reports)
    ordered = sorted(groups, key=groups.get)
    if validation_reports:
        report_paths = {p.resolve() for p in reports}
        validation_paths = {p.resolve() for p in validation_reports}
        if not validation_paths < report_paths:
            raise ValueError(
                "Validation reports must be a nonempty proper subset of reports"
            )
        _, validation_groups, _ = load_samples(trace, validation_reports)
        _, training_groups, _ = load_samples(
            trace, [p for p in reports if p.resolve() not in validation_paths]
        )
        if set(validation_groups) & set(training_groups):
            raise ValueError("A wave crosses training and validation reports")
        train_groups = [g for g in ordered if g in training_groups]
        test_groups = [g for g in ordered if g in validation_groups]
        split_method = "explicit benchmark holdout, disjoint whole waves"
    else:
        ntest = max(1, int(np.ceil(len(ordered) * validation_fraction)))
        train_groups = ordered[:-ntest]
        test_groups = ordered[-ntest:]
        split_method = "chronological whole-wave holdout"
    fit_samples = {}
    for (rank, phase), rows in sorted(samples.items()):
        key = phase if group_by == "phase" else f"pp{rank}/{phase}"
        fit_samples.setdefault(key, []).extend(rows)
    formulas = {
        "tau_affine": "a*n*s_max + b*n + c (tau-Batch section 5.1)",
        "scls_bilinear": "p1*n*s_max + p2*n + p3*s_max + p4; decode uses d1..d4",
        "unpadded_comparison": (
            "u1*s_sum + u2*n + u3*s_max + u4 (implementation comparison)"
        ),
    }
    return {
        "schema_version": 2,
        "label": "host runner invocation with NPU end sync",
        "prediction_target": "single_stage_compute_ms",
        "group_by": group_by,
        "sample_weighting": "equal weight per stage compute event",
        "units": "milliseconds",
        "trace": str(trace),
        "ranges": ranges,
        "split": {
            "method": split_method,
            "training_waves": train_groups,
            "validation_waves": test_groups,
            "validation_reports": [str(p) for p in validation_reports or []],
            "note": (
                "No adjacent-token split; duplicated source prompts "
                "may still cross waves"
            ),
        },
        "stage_layers": stage_layers,
        "formulas": {name: formulas[name] for name in models},
        "limitations": [
            "Coefficients are effective latency fits, not isolated kernel costs",
            "Do not extrapolate beyond observed n/context ranges "
            "or another model/PP layout",
            "All-data coefficients include held-out waves; "
            "validation uses training-only coefficients",
            "Passed structural trace checks do not calibrate device-event timing",
            "Phase-only fits share coefficients across ranks; check "
            "validation_by_stage for systematic rank bias. "
            "Predictions are not whole-pipeline latency",
        ]
        + (
            ["Fixed model: alpha_proj*d_model^2 and beta multiply the same feature n"]
            if "tau_affine" in models
            else []
        ),
        "groups": {
            key: fit_group(rows, train_groups, stage_layers, models)
            for key, rows in fit_samples.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        action="append",
        required=True,
        help="Completed summary.json (trace field); repeat for disjoint ranges",
    )
    parser.add_argument("--stage-layers", type=int)
    parser.add_argument(
        "--group-by",
        choices=("phase", "stage-phase"),
        default="phase",
        help="phase shares coefficients across PP ranks (default); "
        "stage-phase retains separate fits per rank",
    )
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODELS), default=DEFAULT_MODELS
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--validation-report",
        type=Path,
        action="append",
        help="Hold out this entire benchmark; must also appear in --report. "
        "Repeat to cover multiple load regimes; replaces chronological splitting.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new output file")
    result = fit(
        args.trace,
        args.report,
        args.stage_layers,
        args.validation_fraction,
        args.group_by,
        args.models,
        args.validation_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(args.output)
    for name, group in result["groups"].items():
        print(
            name,
            "samples",
            group["samples"],
            "n",
            group["n_values"],
            "s_max",
            group["s_max_range"],
        )
        for model, value in group["models"].items():
            print(" ", model, value["all_data_fit"], value.get("validation", {}))


if __name__ == "__main__":
    main()
