# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request SLO assignment and evaluation for serving benchmarks."""

import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.benchmarks.datasets import SampleRequest
    from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput


@dataclass(frozen=True)
class RequestSLO:
    profile: str
    ttft_slo_ms: float
    tpot_slo_ms: float


def load_slo_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text())
    validate_slo_config(config)
    return config


def _positive(value, context: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{context} must be a finite positive number")


def _normal_interval(spec: dict) -> tuple[NormalDist, float, float]:
    normal = NormalDist(spec["mean"], spec["std"])
    low, high = normal.cdf(spec["min"]), normal.cdf(spec["max"])
    # Reject extreme/narrow tails that cannot be sampled reliably at float
    # precision, rather than hanging in a rejection-sampling loop.
    if not math.isfinite(low) or not math.isfinite(high) or high - low < 1e-12:
        raise ValueError("Normal SLO bounds contain too little probability mass")
    return normal, low, high


def _validate_threshold(value, context: str) -> None:
    if not isinstance(value, dict):
        _positive(value, context)
        return
    distribution = value.get("distribution")
    if distribution == "uniform":
        keys = {"distribution", "min", "max"}
    elif distribution == "normal":
        keys = {"distribution", "mean", "std", "min", "max"}
    else:
        raise ValueError(f"{context}.distribution must be uniform or normal")
    if set(value) != keys:
        raise ValueError(f"{context} requires exactly {sorted(keys)}")
    for key in keys - {"distribution"}:
        _positive(value[key], f"{context}.{key}")
    if value["min"] >= value["max"]:
        raise ValueError(f"{context}.min must be less than max")
    if distribution == "normal":
        _normal_interval(value)


def validate_slo_config(config: dict) -> None:
    """Any number of groups; each threshold is a constant or bounded sampler."""
    if not isinstance(config, dict) or set(config) != {"profiles", "ratios"}:
        raise ValueError("SLO config requires exactly profiles and ratios")
    profiles, ratios = config["profiles"], config["ratios"]
    if (
        not isinstance(profiles, dict)
        or not isinstance(ratios, dict)
        or not profiles
        or profiles.keys() != ratios.keys()
    ):
        raise ValueError("SLO profiles and ratios must have the same nonempty keys")
    for name, profile in profiles.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(profile, dict)
            or set(profile) != {"ttft_slo_ms", "tpot_slo_ms"}
        ):
            raise ValueError(f"Invalid SLO profile: {name}")
        for field, value in profile.items():
            _validate_threshold(value, f"profiles.{name}.{field}")
        ratio = ratios[name]
        if type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio < 0:
            raise ValueError("SLO ratios must be finite nonnegative numbers")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("SLO ratios must sum to 1")


def _sample_threshold(spec, rng: random.Random) -> float:
    if not isinstance(spec, dict):
        return spec
    if spec["distribution"] == "uniform":
        return rng.uniform(spec["min"], spec["max"])
    normal, low, high = _normal_interval(spec)
    probability = low + (high - low) * rng.random()
    probability = min(
        math.nextafter(1.0, 0.0), max(math.nextafter(0.0, 1.0), probability)
    )
    value = normal.inv_cdf(probability)
    # Bound roundoff only: sampling uses the conditional CDF, not clipping an
    # unbounded normal draw (which would pile up samples at the endpoints).
    return min(spec["max"], max(spec["min"], value))


def assign_slos(requests: list["SampleRequest"], config: dict, seed: int) -> dict:
    """Largest-remainder quotas; local RNG leaves arrival/sampling RNG intact."""
    validate_slo_config(config)
    if not requests:
        raise ValueError("No sampled requests to assign SLOs")
    names = sorted(config["profiles"])
    total_ratio = sum(config["ratios"].values())
    quotas = {n: len(requests) * config["ratios"][n] / total_ratio for n in names}
    counts = {n: math.floor(quotas[n]) for n in names}
    remaining = len(requests) - sum(counts.values())
    for name in sorted(names, key=lambda n: (-(quotas[n] - counts[n]), n))[:remaining]:
        counts[name] += 1
    labels = [name for name in names for _ in range(counts[name])]
    random.Random(seed).shuffle(labels)
    # Separate value draws from label shuffling. Adding a distribution does not
    # change group assignments or the dataset/arrival RNG state.
    value_rng = random.Random(f"slo-values:{seed}")
    for request, name in zip(requests, labels):
        profile = config["profiles"][name]
        request.slo = RequestSLO(
            name,
            _sample_threshold(profile["ttft_slo_ms"], value_rng),
            _sample_threshold(profile["tpot_slo_ms"], value_rng),
        )
    return {
        "config": config,
        "seed": seed,
        "assignment_method": "largest_remainder_then_seeded_shuffle",
        "threshold_method": "constants_or_independent_seeded_bounded_draws",
        "counts": counts,
        "requests": [
            {
                "request_id": request.request_id,
                "source_index": request.source_index,
                "prompt_len": request.prompt_len,
                "expected_output_len": request.expected_output_len,
                **asdict(request.slo),
            }
            for request in requests
        ],
    }


def request_body(extra_body: dict | None, request: "SampleRequest") -> dict | None:
    if request.slo is None:
        return extra_body
    body = dict(extra_body or {})
    xargs = body.get("vllm_xargs") or {}
    if not isinstance(xargs, dict):
        raise ValueError("vllm_xargs must be an object")
    body["vllm_xargs"] = {
        **xargs,
        "ttft_slo_ms": request.slo.ttft_slo_ms,
        "tpot_slo_ms": request.slo.tpot_slo_ms,
    }
    return body


def evaluate_goodput(
    requests: list["SampleRequest"],
    outputs: list["RequestFuncOutput"],
    output_lens: list[int],
    defaults: dict[str, float],
    duration: float,
) -> dict | None:
    if not defaults and not any(r.slo is not None for r in requests):
        return None
    groups: dict[str, Counter] = {}
    records = []
    for request, output, length in zip(requests, outputs, output_lens):
        thresholds = dict(defaults)
        profile = "default"
        if request.slo is not None:
            profile = request.slo.profile
            thresholds.update(
                ttft=request.slo.ttft_slo_ms, tpot=request.slo.tpot_slo_ms
            )
        if not thresholds:
            continue
        observed = {
            "ttft": output.ttft * 1000,
            "tpot": (
                (output.latency - output.ttft) * 1000 / (length - 1)
                if length > 1
                else 0.0
            ),
            "e2el": output.latency * 1000,
        }
        metric_attainment = {
            key: bool(
                output.success
                and length > 0
                and math.isfinite(observed[key])
                and 0 <= observed[key] <= value
            )
            for key, value in thresholds.items()
        }
        attained = {
            "ttft": metric_attainment.get("ttft"),
            "tpot": metric_attainment.get("tpot"),
            "all": all(metric_attainment.values()),
        }
        group = groups.setdefault(profile, Counter())
        group["total_requests"] += 1
        group["completed"] += int(output.success)
        group["good_requests"] += int(attained["all"])
        for metric in ("ttft", "tpot"):
            group[f"{metric}_total_requests"] += int(attained[metric] is not None)
            group[f"{metric}_good_requests"] += int(attained[metric] is True)
        records.append(
            {
                "request_id": request.request_id,
                "source_index": request.source_index,
                "profile": profile,
                "thresholds_ms": thresholds,
                "observed_ms": {
                    k: v if output.success and math.isfinite(v) else None
                    for k, v in observed.items()
                },
                "success": output.success,
                "attained": attained,
            }
        )
    # Aggregate counts before adding rates: overall rates are request-weighted,
    # not the mean of profile rates. Missing thresholds are not failures.
    overall = Counter()
    for group in groups.values():
        overall.update(group)
    for group in [overall, *groups.values()]:
        total = group["total_requests"]
        group["attainment_rate"] = group["good_requests"] / total if total else None
        group["attainment_rates"] = {
            metric: (
                group[f"{metric}_good_requests"] / group[f"{metric}_total_requests"]
                if group[f"{metric}_total_requests"]
                else None
            )
            for metric in ("ttft", "tpot")
        }
        group["attainment_rates"]["all"] = group["attainment_rate"]
        group["request_goodput"] = group["good_requests"] / duration
    return {
        **overall,
        "by_profile": groups,
        "requests": records,
        "tpot_definition": "(latency - ttft) / (output_tokens - 1); 0 for one token",
    }
