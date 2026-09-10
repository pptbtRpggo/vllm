# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SLO sampling/records/goodput tests that do not require an accelerator."""

import copy
import json
import math
import random
from collections import Counter
from statistics import mean
from types import SimpleNamespace

import pytest

from vllm.benchmarks.request_records import write_request_records
from vllm.benchmarks.slo import (
    assign_slos,
    evaluate_goodput,
    load_slo_config,
    request_body,
)


def requests(n):
    return [
        SimpleNamespace(
            request_id=f"r{i}",
            source_index=i,
            prompt_len=20,
            expected_output_len=3,
            slo=None,
        )
        for i in range(n)
    ]


def test_overall_and_profile_attainment_rates_include_failures():
    reqs = requests(4)
    for req, profile in zip(reqs, ("small", "large", "large", "large")):
        req.slo = SimpleNamespace(profile=profile, ttft_slo_ms=100, tpot_slo_ms=100)
    outputs = [
        SimpleNamespace(success=True, ttft=0.05, latency=0.15),
        SimpleNamespace(success=True, ttft=0.05, latency=0.45),
        SimpleNamespace(success=True, ttft=0.2, latency=0.3),
        SimpleNamespace(success=False, ttft=0, latency=0),
    ]
    report = evaluate_goodput(reqs, outputs, [3, 3, 3, 0], {}, 2)
    assert report["attainment_rates"] == {"ttft": 0.5, "tpot": 0.5, "all": 0.25}
    assert report["attainment_rate"] == 0.25
    assert report["request_goodput"] == 0.5
    assert report["by_profile"]["small"]["attainment_rates"]["all"] == 1
    assert report["by_profile"]["large"]["attainment_rates"] == {
        "ttft": 1 / 3,
        "tpot": 1 / 3,
        "all": 0,
    }
    assert sum(g["request_goodput"] for g in report["by_profile"].values()) == 0.5
    for metric in ("ttft", "tpot"):
        assert report[f"{metric}_total_requests"] == 4
        assert report[f"{metric}_good_requests"] == 2


def test_metric_rates_use_only_requests_with_that_threshold():
    reqs = requests(2)
    outputs = [SimpleNamespace(success=True, ttft=0.05, latency=0.45)] * 2
    report = evaluate_goodput(reqs, outputs, [3, 3], {"ttft": 100}, 2)
    assert report["attainment_rates"] == {"ttft": 1, "tpot": None, "all": 1}
    assert report["tpot_total_requests"] == 0
    reqs[1].slo = SimpleNamespace(profile="tight", ttft_slo_ms=100, tpot_slo_ms=100)
    report = evaluate_goodput(reqs, outputs, [3, 3], {"ttft": 100}, 2)
    assert report["attainment_rates"] == {"ttft": 1, "tpot": 0, "all": 0.5}
    assert report["ttft_total_requests"] == 2
    assert report["tpot_total_requests"] == 1
    # Without global defaults, the request without any SLO is not evaluated.
    report = evaluate_goodput(reqs, outputs, [3, 3], {}, 2)
    assert report["total_requests"] == 1
    assert report["attainment_rates"]["all"] == 0


def mixed_config():
    return {
        "profiles": {
            "fixed": {"ttft_slo_ms": 1000, "tpot_slo_ms": 50},
            "sampled": {
                "ttft_slo_ms": {
                    "distribution": "normal",
                    "mean": 2000,
                    "std": 400,
                    "min": 1600,
                    "max": 2400,
                },
                "tpot_slo_ms": {"distribution": "uniform", "min": 60, "max": 140},
            },
            "disabled": {"ttft_slo_ms": 3000, "tpot_slo_ms": 200},
        },
        "ratios": {"fixed": 0.3, "sampled": 0.7, "disabled": 0},
    }


def test_arbitrary_groups_add_remove_and_ratio_changes():
    cfg = mixed_config()
    result = assign_slos(requests(10), cfg, 7)
    assert result["counts"] == {"disabled": 0, "fixed": 3, "sampled": 7}
    cfg["profiles"]["new_group"] = {"ttft_slo_ms": 4000, "tpot_slo_ms": 250}
    cfg["ratios"] = dict(disabled=0, fixed=0.1, sampled=0.5, new_group=0.4)
    reqs = requests(10)
    assert assign_slos(reqs, cfg, 7)["counts"] == {
        "disabled": 0,
        "fixed": 1,
        "sampled": 5,
        "new_group": 4,
    }
    del cfg["profiles"]["new_group"]
    cfg["ratios"] = dict(disabled=0, fixed=0, sampled=1)
    assign_slos(reqs, cfg, 7)
    assert {r.slo.profile for r in reqs} == {"sampled"}


def test_sampling_is_reproducible_and_does_not_change_group_or_global_rng():
    cfg = mixed_config()
    original = copy.deepcopy(cfg)
    state = random.getstate()
    result = assign_slos(requests(100), cfg, 9)
    assert state == random.getstate() and cfg == original
    assert result == assign_slos(requests(100), cfg, 9)
    assert result["requests"] != assign_slos(requests(100), cfg, 10)["requests"]
    constants = copy.deepcopy(cfg)
    constants["profiles"]["sampled"] = dict(ttft_slo_ms=2000, tpot_slo_ms=100)
    old = assign_slos(requests(100), constants, 9)
    assert [r["profile"] for r in old["requests"]] == [
        r["profile"] for r in result["requests"]
    ]
    for row in result["requests"]:
        if row["profile"] == "fixed":
            assert (row["ttft_slo_ms"], row["tpot_slo_ms"]) == (1000, 50)


def test_sampled_values_have_bounds_and_are_not_clipped_normals():
    cfg = mixed_config()
    cfg["ratios"] = dict(disabled=0, fixed=0, sampled=1)
    reqs = requests(4000)
    assign_slos(reqs, cfg, 23)
    ttft = [r.slo.ttft_slo_ms for r in reqs]
    tpot = [r.slo.tpot_slo_ms for r in reqs]
    assert all(1600 < x < 2400 for x in ttft)
    assert all(60 <= x <= 140 for x in tpot)
    # A clipped N(2000,400) would put ~32% of values on the endpoints.
    assert len(set(ttft)) == len(ttft)
    assert abs(mean(ttft) - 2000) < 15
    assert abs(mean(tpot) - 100) < 2
    assert sum(x < 2000 for x in ttft) / len(ttft) == pytest.approx(0.5, abs=0.03)


@pytest.mark.parametrize(
    "spec",
    [
        {"distribution": "unknown", "min": 1, "max": 2},
        {"distribution": "uniform", "min": 1},
        {"distribution": "uniform", "min": 1, "max": 2, "typo": 3},
        {"distribution": "uniform", "min": 2, "max": 1},
        {"distribution": "uniform", "min": 1, "max": 1},
        {"distribution": "uniform", "min": 0, "max": 1},
        {"distribution": "uniform", "min": True, "max": 2},
        {"distribution": "uniform", "min": 1, "max": float("inf")},
        {"distribution": "normal", "mean": 2, "std": 0, "min": 1, "max": 3},
        {"distribution": "normal", "mean": 2, "std": -1, "min": 1, "max": 3},
        {"distribution": "normal", "mean": float("nan"), "std": 1, "min": 1, "max": 3},
        {"distribution": "normal", "mean": 1, "std": 1, "min": 1000, "max": 1001},
        {"distribution": "normal", "mean": 2, "min": 1, "max": 3},
        True,
        0,
        -1,
        float("nan"),
        [],
    ],
)
def test_invalid_sampler_rejected_before_assigning(tmp_path, spec):
    cfg = mixed_config()
    cfg["profiles"]["sampled"]["ttft_slo_ms"] = spec
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        load_slo_config(path)
    reqs = requests(5)
    with pytest.raises(ValueError):
        assign_slos(reqs, cfg, 0)
    assert all(r.slo is None for r in reqs)


def test_saved_plan_payload_and_goodput_use_each_sampled_threshold(tmp_path):
    cfg = mixed_config()
    cfg["ratios"] = dict(disabled=0, fixed=0, sampled=1)
    reqs = requests(10)
    assign_slos(reqs, cfg, 12)
    path = tmp_path / "requests.jsonl"
    write_request_records(path, reqs)
    planned = [json.loads(line) for line in path.read_text().splitlines()]
    # Select one common observed TTFT between the sampled thresholds: within
    # the same profile some requests pass and some fail their individual SLO.
    threshold = mean(r.slo.ttft_slo_ms for r in reqs)
    outputs = [
        SimpleNamespace(
            success=True,
            ttft=threshold / 1000,
            latency=threshold / 1000 + 0.02,
            error="",
        )
        for _ in reqs
    ]
    report = evaluate_goodput(reqs, outputs, [3] * len(reqs), {}, duration=2)
    expected = [threshold <= r.slo.ttft_slo_ms for r in reqs]
    assert any(expected) and not all(expected)
    assert [r["attained"]["all"] for r in report["requests"]] == expected
    assert [r["attained"]["ttft"] for r in report["requests"]] == expected
    assert all(r["attained"]["tpot"] for r in report["requests"])
    assert report["request_goodput"] == sum(expected) / 2
    shared = {"vllm_xargs": {"other": 7}}
    for req, saved in zip(reqs, planned):
        body = request_body(shared, req)
        assert body["vllm_xargs"]["ttft_slo_ms"] == saved["slo"]["ttft_slo_ms"]
        assert body["vllm_xargs"]["tpot_slo_ms"] == saved["slo"]["tpot_slo_ms"]
        assert math.isfinite(saved["slo"]["ttft_slo_ms"])
    assert shared == {"vllm_xargs": {"other": 7}}
    write_request_records(path, reqs, outputs, [3] * len(reqs), report)
    finished = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["slo"] for r in finished] == [r["slo"] for r in planned]
    assert [r["attained"]["all"] for r in finished] == expected
    assert [r["attained"] for r in finished] == [
        r["attained"] for r in report["requests"]
    ]
    assert Counter(r["slo"]["profile"] for r in finished) == {"sampled": 10}


@pytest.mark.parametrize(
    "ttft,latency,success,length,expected",
    [
        (0.05, 0.06, True, 3, {"ttft": True, "tpot": True, "all": True}),
        (0.05, 0.09, True, 3, {"ttft": True, "tpot": False, "all": False}),
        (0.12, 0.13, True, 3, {"ttft": False, "tpot": True, "all": False}),
        (0.12, 0.16, True, 3, {"ttft": False, "tpot": False, "all": False}),
        (0.05, 0.06, False, 3, {"ttft": False, "tpot": False, "all": False}),
        (0, 0, True, 0, {"ttft": False, "tpot": False, "all": False}),
        (0.05, 0.05, True, 1, {"ttft": True, "tpot": True, "all": True}),
        (0.05, float("nan"), True, 3, {"ttft": True, "tpot": False, "all": False}),
        (0.05, 0.04, True, 3, {"ttft": True, "tpot": False, "all": False}),
    ],
)
def test_attainment_components_and_saved_record(
    tmp_path, ttft, latency, success, length, expected
):
    reqs = requests(1)
    assign_slos(
        reqs,
        {
            "profiles": {"one": dict(ttft_slo_ms=100, tpot_slo_ms=10)},
            "ratios": {"one": 1},
        },
        0,
    )
    path = tmp_path / "requests.jsonl"
    write_request_records(path, reqs)
    assert json.loads(path.read_text())["attained"] == {
        "ttft": None,
        "tpot": None,
        "all": None,
    }
    outputs = [SimpleNamespace(ttft=ttft, latency=latency, success=success, error="")]
    report = evaluate_goodput(reqs, outputs, [length], {}, 2)
    assert report["requests"][0]["attained"] == expected
    assert report["good_requests"] == int(expected["all"])
    assert report["request_goodput"] == int(expected["all"]) / 2
    write_request_records(path, reqs, outputs, [length], report)
    assert json.loads(path.read_text())["attained"] == expected


def test_unconfigured_metrics_are_null_and_all_includes_e2el():
    reqs = requests(1)
    outputs = [SimpleNamespace(ttft=0.05, latency=0.06, success=True)]
    report = evaluate_goodput(reqs, outputs, [3], {"ttft": 100}, 1)
    assert report["requests"][0]["attained"] == {
        "ttft": True,
        "tpot": None,
        "all": True,
    }
    report = evaluate_goodput(
        reqs, outputs, [3], {"ttft": 100, "tpot": 10, "e2el": 55}, 1
    )
    assert report["requests"][0]["attained"] == {
        "ttft": True,
        "tpot": True,
        "all": False,
    }
    assert report["good_requests"] == 0


def test_no_slo_has_null_components_before_and_after_completion(tmp_path):
    reqs = requests(1)
    outputs = [SimpleNamespace(ttft=0.05, latency=0.06, success=True, error="")]
    path = tmp_path / "requests.jsonl"
    write_request_records(path, reqs)
    expected = {"ttft": None, "tpot": None, "all": None}
    assert json.loads(path.read_text())["attained"] == expected
    report = evaluate_goodput(reqs, outputs, [3], {}, 1)
    assert report is None
    write_request_records(path, reqs, outputs, [3], report)
    assert json.loads(path.read_text())["attained"] == expected
