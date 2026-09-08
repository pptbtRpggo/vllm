# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request SLO assignment and evaluation for serving benchmarks."""

import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
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
            not name
            or not isinstance(profile, dict)
            or set(profile) != {"ttft_slo_ms", "tpot_slo_ms"}
        ):
            raise ValueError(f"Invalid SLO profile: {name}")
        for value in profile.values():
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("TTFT/TPOT SLOs must be finite positive milliseconds")
        ratio = ratios[name]
        if type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio < 0:
            raise ValueError("SLO ratios must be finite nonnegative numbers")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("SLO ratios must sum to 1")
    return config


def assign_slos(requests: list["SampleRequest"], config: dict, seed: int) -> dict:
    """Largest-remainder quotas; local RNG leaves arrival/sampling RNG intact."""
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
    for request, name in zip(requests, labels):
        request.slo = RequestSLO(name, **config["profiles"][name])
    return {
        "config": config,
        "seed": seed,
        "assignment_method": "largest_remainder_then_seeded_shuffle",
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
            "tpot": (output.latency - output.ttft) * 1000 / (length - 1)
            if length > 1
            else 0.0,
            "e2el": output.latency * 1000,
        }
        attained = (
            output.success
            and length > 0
            and all(
                math.isfinite(observed[key]) and 0 <= observed[key] <= value
                for key, value in thresholds.items()
            )
        )
        group = groups.setdefault(profile, Counter())
        group["total_requests"] += 1
        group["completed"] += int(output.success)
        group["good_requests"] += int(attained)
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
    for group in groups.values():
        group["attainment_rate"] = group["good_requests"] / group["total_requests"]
        group["request_goodput"] = group["good_requests"] / duration
    total = sum(g["total_requests"] for g in groups.values())
    good = sum(g["good_requests"] for g in groups.values())
    return {
        "total_requests": total,
        "good_requests": good,
        "attainment_rate": good / total if total else 0.0,
        "request_goodput": good / duration,
        "by_profile": groups,
        "requests": records,
        "tpot_definition": "(latency - ttft) / (output_tokens - 1); 0 for one token",
    }
