# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "pp_sla", Path(__file__).resolve().parents[2] / "benchmarks/pp_sla.py"
)
sla = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sla)


def test_disjoint_stratified_sampling_preserves_joint_length_mixture():
    rows = [
        dict(
            id=i,
            input_tokens=32 if i < 800 else 1024,
            output_tokens=64 if i < 800 else 512,
        )
        for i in range(1000)
    ]
    sizes = dict(warmup=100, profile=200, evaluation=500)
    result = sla.split_samples(rows, sizes, 42)
    ids = []
    for name, sample in result.items():
        assert len(sample) == sizes[name]
        assert sum(r["input_tokens"] == 1024 for r in sample) == sizes[name] // 5
        ids.extend(r["id"] for r in sample)
    assert len(ids) == len(set(ids))
    assert sla.split_samples(rows, sizes, 42) == result


def test_no_silent_oversampling():
    with pytest.raises(ValueError, match="not enough"):
        sla.split_samples(
            [dict(input_tokens=4, output_tokens=4)], {"warmup": 1, "profile": 1}, 1
        )


def record(ttft):
    return dict(success=True, ttft_ms=ttft, e2e_ms=1000, output_tokens=10)


def test_p99_is_request_percentile_not_average_latency():
    result = sla.summarize([record(10)] * 98 + [record(600)] * 2, 10)
    assert result["p90_ttft_ms"] == 10
    assert result["p99_ttft_ms"] == 600
    assert result["sla"] == {"150": False, "500": False}


def test_failed_requests_cannot_improve_sla():
    result = sla.summarize([record(10), dict(success=False, error="timeout")], 10)
    assert result["failed"] == 1
    assert result["sla"] == {"150": False, "500": False}
    assert result["output_tokens_per_s"] == 1


def test_no_passing_baseline_is_zero_capacity():
    result = sla.capacity_summary(
        [dict(concurrency=1, summary=sla.summarize([record(900)], 1))]
    )
    assert result["500"]["max_tested_passing_concurrency"] == 0


def test_nonmonotonic_capacity_is_flagged():
    points = [
        dict(concurrency=1, summary=sla.summarize([record(600)], 1)),
        dict(concurrency=2, summary=sla.summarize([record(50)], 1)),
    ]
    result = sla.capacity_summary(points)
    assert not result["500"]["monotonic_observations"]
    assert result["500"]["max_tested_passing_concurrency"] == 2
