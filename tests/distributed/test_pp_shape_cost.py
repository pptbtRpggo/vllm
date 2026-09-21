# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from dataclasses import replace

import pytest

from tests.distributed.pp_trace_fixtures import write_trace
from vllm.distributed.pp_partition import (
    _keep_record,
    partition_layers,
    plan_from_trace_dir,
)
from vllm.distributed.pp_shape_cost import batch_shape_key, fit_shape_rank_costs

SHAPES = [
    [dict(query_tokens=1, context_tokens=16, prompt_tokens=0)] * 2,
    [
        dict(query_tokens=1, context_tokens=1024, prompt_tokens=0),
        dict(query_tokens=1, context_tokens=0, prompt_tokens=1),
    ],
]
# Both shapes have Q=2, but their costs differ. Last-stage fixed work is
# intentionally large for shape 0: a zero-intercept layer model is incorrect.
COEFFICIENTS = [[(2, 1), (8, 4)], [(10, 3), (1, 1)]]


def trace(n0, repeats=(3, 1)):
    result = {0: [], 1: []}
    for rank, (start, end) in enumerate([(0, n0), (n0, 8)]):
        for shape_id, shape in enumerate(SHAPES):
            a, b = COEFFICIENTS[rank][shape_id]
            for _ in range(repeats[shape_id]):
                result[rank].append(
                    dict(
                        pp_rank=rank,
                        pp_size=2,
                        tp_rank=0,
                        tp_size=1,
                        step=10,
                        start_layer=start,
                        end_layer=end,
                        batch_shape=shape,
                        num_tokens=2,
                        num_reqs=2,
                        num_ctx_tokens=0,
                        num_generation_tokens=2,  # stale labels ignored
                        compute_wall_ms=a + b * (end - start),
                        compute_ms=99999,  # modeled value must not be fitted
                        send_transfer_ms=99999,
                        send_service_ms=(1, 5)[shape_id] if rank == 0 else None,
                        send_service_source="measured_idle_replay",
                        compute_scale=1,
                        comm_scale=1,
                    )
                )
    return result


def fit(traces=None):
    return fit_shape_rank_costs(
        traces or [trace(4), trace(2), trace(6)], workload="all", warmup_steps=0
    )


def test_shape_preserves_context_and_sequence_pairing():
    records = trace(4)[0]
    assert batch_shape_key(records[0]) != batch_shape_key(records[-1])
    reordered = copy.deepcopy(records[-1])
    reordered["batch_shape"].reverse()
    assert batch_shape_key(reordered) == batch_shape_key(records[-1])
    assert _keep_record(records[-1], "mixed", 0)
    assert not _keep_record(records[-1], "decode", 0)
    assert not _keep_record(records[-1], "prefill", 0)


def test_affine_fit_and_common_reference_weights():
    costs = fit([trace(4), trace(2, (1, 100)), trace(6, (100, 1))])
    assert costs[0].t_fixed_ms == pytest.approx(3.5)
    assert costs[0].t_layer_ms == pytest.approx(1.75)
    assert costs[1].t_fixed_ms == pytest.approx(7.75)
    assert costs[1].t_layer_ms == pytest.approx(2.5)
    assert costs[0].t_comm_out_ms == pytest.approx(2)
    assert costs[0].min_layers == 2 and costs[0].max_layers == 6
    assert all(d["max_fit_relative_error"] < 1e-12 for c in costs for d in c.shape_fits)


@pytest.mark.parametrize("objective", ["latency", "throughput"])
def test_dp_matches_independent_enumeration_with_fixed_costs(objective):
    costs = fit()
    plan = partition_layers(costs, objective=objective, allow_unchecked_memory=True)
    oracle = []
    for n0 in range(2, 7):
        means = []
        for rank, n in enumerate([n0, 8 - n0]):
            means.append(
                sum(
                    weight * (a + b * n)
                    for weight, (a, b) in zip([0.75, 0.25], COEFFICIENTS[rank])
                )
            )
        value = sum(means) + 2 if objective == "latency" else max(v + 2 for v in means)
        oracle.append((value, n0))
    expected, n0 = min(oracle)
    assert plan.cost_ms == pytest.approx(expected)
    assert plan.partitions == [n0, 8 - n0]
    if objective == "throughput":
        assert plan.partitions == [6, 2]
        assert plan.to_dict()["cost_kind"] == "mean_stage_occupancy_ms"


def test_interpolation_at_unseen_layer_count():
    costs = fit([trace(2), trace(6)])
    for rank in range(2):
        n = 5  # not a fit point
        prediction = costs[rank].t_fixed_ms + n * costs[rank].t_layer_ms
        expected = sum(
            w * (a + b * n) for w, (a, b) in zip([0.75, 0.25], COEFFICIENTS[rank])
        )
        assert prediction == pytest.approx(expected)


def test_measured_link_variation_is_averaged_not_rejected():
    traces = [trace(4), trace(2)]
    traces[1][0][0]["send_service_ms"] = 7
    # Shape 0: (five 1 ms samples + one 7 ms sample) / 6 = 2 ms.
    # Shape 1: 5 ms. Common reference distribution is 75% / 25%.
    assert fit(traces)[0].t_comm_out_ms == pytest.approx(2.75)


def test_compute_fit_uses_mean_of_repeated_measurements():
    traces = [trace(2), trace(6)]
    # Three samples at n=2 were all 4; their new mean is 6, median stays 4.
    traces[0][0][0]["compute_wall_ms"] += 6
    diagnostic = fit(traces)[0].shape_fits
    shape0 = next(d for d in diagnostic if d["shape"] == [[1, 16, 0]] * 2)
    assert shape0["measured_points"][0] == (2, 6)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda ts: ts.pop(), "multiple shard"),
        (lambda ts: ts.__setitem__(1, trace(4)), "distinct shard"),
        (lambda ts: ts[1][0][0].pop("batch_shape"), "batch_shape"),
        (lambda ts: ts[1][0][0].pop("compute_wall_ms"), "compute_wall_ms"),
        (lambda ts: ts[1][0][0].pop("send_service_ms"), "send_service_ms"),
        (lambda ts: ts[1][0][0].update(compute_scale=2), "scales"),
        (lambda ts: ts[1][0][0].update(num_tokens=3), "counts"),
    ],
)
def test_invalid_or_insufficient_profiles_fail(mutate, match):
    traces = [trace(4), trace(2)]
    mutate(traces)
    with pytest.raises(ValueError, match=match):
        fit(traces)


def test_pp_size_changes_and_ideal_overlap_require_new_model():
    costs = fit()
    with pytest.raises(ValueError, match="PP size"):
        partition_layers(costs, max_pp_size=1, allow_unchecked_memory=True)
    with pytest.raises(ValueError, match="blocking"):
        partition_layers(costs, overlap_comm=True, allow_unchecked_memory=True)


def test_shape_costs_still_obey_memory_and_coverage():
    from tests.distributed.test_pp_memory import _device, _profile

    costs = fit()
    memory = _profile([_device(0, 4, [1] * 8), _device(1, 8, [1] * 8)])
    plan = partition_layers(costs, memory_profile=memory)
    assert plan.partitions == [4, 4]  # Faster [6,2] cannot fit.
    with pytest.raises(ValueError, match="coverage and memory"):
        partition_layers(
            [replace(c, min_layers=5, max_layers=6) for c in costs],
            memory_profile=memory,
        )


def test_marked_warmup_is_excluded_even_with_zero_step_filter():
    rec = trace(4)[0][0] | {"is_warmup": True}
    assert not _keep_record(rec, "all", 0)


def test_cli_trace_roundtrip_and_scale_guard(tmp_path, monkeypatch):
    dirs = []
    for n0 in [4, 2, 6]:
        directory = tmp_path / str(n0)
        directory.mkdir()
        dirs.append(directory)
        for rank, rows in trace(n0).items():
            write_trace(directory / f"pp_stage_pp{rank}_tp0.jsonl", rows)
    monkeypatch.setenv("VLLM_PP_HETERO", "1,1/1")
    plan = plan_from_trace_dir(
        dirs[0],
        fit_trace_dirs=dirs[1:],
        compute_model="shape-affine",
        comm_source="replay",
        allow_unchecked_memory=True,
    )
    assert plan.partitions == [6, 2]
    assert plan.to_dict()["compute_model"] == "shape-affine"
    with pytest.raises(ValueError, match="changed hetero scales"):
        plan_from_trace_dir(
            dirs[0],
            fit_trace_dirs=dirs[1:],
            hetero="1,2/1",
            compute_model="shape-affine",
            comm_source="replay",
            allow_unchecked_memory=True,
        )

    from vllm.distributed.pp_partition import main

    main(
        [
            str(dirs[0]),
            "--fit-trace-dir",
            str(dirs[1]),
            "--fit-trace-dir",
            str(dirs[2]),
            "--comm-source",
            "replay",
            "--compute-model",
            "shape-affine",
            "--allow-unchecked-memory",
        ]
    )


def feature_profiles():
    """Different microbatches at every partition; independent known timing law."""
    import random

    rng = random.Random(902)
    profiles = []
    for split in [2, 4, 6]:
        rows = {0: [], 1: []}
        for step in range(80):
            shape = []
            for _ in range(rng.randint(1, 5)):
                q = rng.randint(1, 16)
                shape.append(
                    dict(
                        query_tokens=q,
                        context_tokens=rng.randint(0, 4096),
                        prompt_tokens=rng.choice([0, q]),
                    )
                )
            requests = len(shape)
            prompt = sum(s["prompt_tokens"] for s in shape)
            generation = sum(s["query_tokens"] - s["prompt_tokens"] for s in shape)
            attention = sum(
                s["query_tokens"] * s["context_tokens"]
                + s["query_tokens"] * (s["query_tokens"] + 1) / 2
                for s in shape
            )
            for rank, (start, end) in enumerate([(0, split), (split, 8)]):
                a = (
                    2
                    + 0.1 * requests
                    + 0.02 * prompt
                    + 0.03 * generation
                    + 0.0001 * attention
                )
                b = (
                    0.7
                    + 0.02 * requests
                    + 0.002 * prompt
                    + 0.001 * generation
                    + 0.00001 * attention
                )
                rows[rank].append(
                    dict(
                        pp_rank=rank,
                        pp_size=2,
                        tp_rank=0,
                        tp_size=1,
                        step=step,
                        start_layer=start,
                        end_layer=end,
                        batch_shape=shape,
                        num_tokens=prompt + generation,
                        num_reqs=requests,
                        compute_wall_ms=a + b * (end - start),
                        send_service_ms=1 + 0.001 * (prompt + generation)
                        if rank == 0
                        else None,
                        send_service_source="measured_serving_overlap",
                        expected_fixed=a,
                        expected_layer=b,
                    )
                )
        profiles.append(rows)
    return profiles


def test_unmatched_microbatches_fit_and_predict_reference_expectation():
    import statistics

    profiles = feature_profiles()
    costs = fit(profiles)
    for rank, cost in enumerate(costs):
        assert cost.t_fixed_ms == pytest.approx(
            statistics.mean(r["expected_fixed"] for r in profiles[0][rank]), rel=1e-6
        )
        assert cost.t_layer_ms == pytest.approx(
            statistics.mean(r["expected_layer"] for r in profiles[0][rank]), rel=1e-6
        )
        diagnostic = cost.shape_fits[0]
        assert diagnostic["method"] == "feature_affine"
        assert diagnostic["design_rank"] == 10
        assert diagnostic["training_wape"] < 1e-8
        assert len(diagnostic["leave_one_profile_out"]) == 3
        assert max(d["wape"] for d in diagnostic["leave_one_profile_out"]) < 1e-7
    assert costs[0].t_comm_out_ms == pytest.approx(
        statistics.mean(r["send_service_ms"] for r in profiles[0][0])
    )
    plan = partition_layers(costs, allow_unchecked_memory=True)
    assert sum(plan.partitions) == 8
    assert all(2 <= n <= 6 for n in plan.partitions)


def test_reference_communication_mean_keeps_outliers():
    import statistics

    profiles = feature_profiles()
    profiles[0][0][0]["send_service_ms"] = 100
    assert fit(profiles)[0].t_comm_out_ms == pytest.approx(
        statistics.mean(r["send_service_ms"] for r in profiles[0][0])
    )


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_measured_link_never_falls_back(value):
    profiles = feature_profiles()
    profiles[0][0][0]["send_service_ms"] = value
    profiles[0][0][0]["send_ms"] = 1
    with pytest.raises(ValueError, match="invalid measured"):
        fit(profiles)


def test_missing_measurement_never_uses_old_send_time():
    profiles = feature_profiles()
    profiles[0][0][0].pop("send_service_ms")
    profiles[0][0][0]["send_ms"] = 1
    profiles[0][0][0]["send_transfer_ms"] = 1
    with pytest.raises(ValueError, match="missing measured"):
        fit(profiles)


def test_insufficient_layer_variation_is_not_hidden_by_shape_features():
    profiles = feature_profiles()
    # Keep rank 0 at two layers in all profiles, with a valid contiguous model.
    for traces in profiles:
        for r in traces[0]:
            r["end_layer"] = 2
        for r in traces[1]:
            r["start_layer"] = 2
    with pytest.raises(ValueError, match="distinct shard lengths"):
        fit(profiles)


def test_empty_filtered_trace_never_substitutes_other_workload():
    with pytest.raises(ValueError, match="no matching workload"):
        fit_shape_rank_costs(feature_profiles(), workload="all", warmup_steps=1000)
