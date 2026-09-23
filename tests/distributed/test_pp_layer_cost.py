# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest

from tests.distributed.pp_trace_fixtures import write_trace
from vllm.distributed.pp_layer_cost import measured_layer_rank_costs
from vllm.distributed.pp_partition import partition_layers, plan_from_trace_dir


def layer_trace(cut=4, repeats=(1, 9)):
    result = {0: [], 1: []}
    for rank, (start, end) in enumerate(((0, cut), (cut, 8))):
        for phase, count in zip(("prefill", "decode"), repeats):
            for _ in range(count):
                factor = 3 if phase == "prefill" else 1
                layers = {str(i): factor * (rank + 1) for i in range(start, end)}
                embedding = factor * 2 if rank == 0 else None
                head, norm = (factor * 4, factor) if rank == 1 else (None, None)
                overhead = 0.5
                residual = sum(v or 0 for v in (embedding, head, norm)) + overhead
                result[rank].append(
                    dict(
                        pp_rank=rank,
                        pp_size=2,
                        tp_rank=0,
                        tp_size=1,
                        step=10,
                        start_layer=start,
                        end_layer=end,
                        num_tokens=32 if phase == "prefill" else 1,
                        num_reqs=1,
                        batch_shape=[
                            dict(
                                query_tokens=32 if phase == "prefill" else 1,
                                prompt_tokens=32 if phase == "prefill" else 0,
                                context_tokens=0 if phase == "prefill" else 32,
                            )
                        ],
                        compute_model="layer-measured",
                        layer_compute_ms=layers,
                        non_layer_compute_ms=residual,
                        runner_overhead_ms=overhead,
                        embedding_ms=embedding,
                        lm_head_ms=head,
                        final_norm_ms=norm,
                        compute_wall_ms=sum(layers.values()) + residual,
                        send_service_ms=factor if rank == 0 else None,
                        send_service_source="measured_serving_overlap",
                    )
                )
    return result


def aggregate(profiles, workload="all"):
    return measured_layer_rank_costs(
        profiles, workload=workload, warmup_steps=0, layer_aggregation="phase-balanced"
    )


def test_phase_means_are_equal_weighted_and_mixed_excluded():
    records = layer_trace()
    for rank, rows in records.items():
        mixed = dict(
            rows[0],
            batch_shape=[dict(query_tokens=2, prompt_tokens=1, context_tokens=32)],
        )
        mixed["layer_compute_ms"] = {k: 99999 for k in mixed["layer_compute_ms"]}
        rows.append(mixed)
    costs = aggregate([records])
    assert [c.t_layer_ms for c in costs] == [2, 4]
    assert [c.t_fixed_ms for c in costs] == [0.5, 0.5]
    assert costs[0].t_embedding_ms == 4
    assert costs[1].t_head_ms == 8
    assert costs[1].t_final_norm_ms == 2
    assert costs[0].t_comm_out_ms == 2
    assert costs[0].n_steps == 10
    assert [c.t_layer_ms for c in aggregate([layer_trace(repeats=(9, 1))])] == [2, 4]


@pytest.mark.parametrize("objective", ["latency", "throughput"])
def test_single_profile_can_choose_unmeasured_global_layers(objective):
    costs = aggregate([layer_trace()])
    plan = partition_layers(costs, objective=objective, allow_unchecked_memory=True)
    oracle = []
    for cut in range(1, 8):
        c0, c1 = 0.5 + 4 + 2 * cut, 0.5 + 10 + 4 * (8 - cut)
        serial = c0 + c1 + 2
        primary = serial if objective == "latency" else max(c0 + 2, c1 + 2)
        oracle.append((primary, serial, cut, [c0, c1]))
    expected = min(oracle)
    assert plan.partitions == [expected[2], 8 - expected[2]]
    assert plan.partitions != [4, 4]
    assert plan.cost_ms == expected[0]
    assert plan.to_dict()["predicted_sequential_ms"] == expected[1]
    assert plan.to_dict()["predicted_stage_compute_ms"] == expected[3]


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_phase_missing_is_not_silently_reweighted(phase):
    records = layer_trace(repeats=(0, 3) if phase == "prefill" else (3, 0))
    with pytest.raises(ValueError, match=f"requires {phase}"):
        aggregate([records])
    costs = aggregate([records], workload="decode" if phase == "prefill" else "prefill")
    assert costs[0].t_layer_ms == (1 if phase == "prefill" else 3)


@pytest.mark.parametrize(
    "key,value,error",
    [
        ("compute_model", "shape-affine", "direct layer traces"),
        ("layer_compute_ms", {"0": 1}, "every local global layer"),
        ("compute_wall_ms", 99999, "match stage wall"),
        ("tp_size", 2, "TP=1"),
        ("compute_scale", 2, "mock compute slowdown"),
        ("runner_overhead_ms", None, "finite and nonnegative"),
        ("embedding_ms", -1, "finite and nonnegative"),
    ],
)
def test_reject_invalid_traces(key, value, error):
    rows = layer_trace()
    rows[0][0][key] = value
    with pytest.raises(ValueError, match=error):
        aggregate([rows])


def test_missing_endpoint_is_unavailable_not_free():
    costs = aggregate([layer_trace()])
    costs[1] = replace(costs[1], t_head_ms=None)
    with pytest.raises(ValueError, match="endpoint coverage"):
        partition_layers(costs, allow_unchecked_memory=True)


def test_offline_planner_layer_mode(tmp_path):
    for rank, rows in layer_trace().items():
        write_trace(tmp_path / f"pp_stage_pp{rank}_tp0.jsonl", rows)
    plan = plan_from_trace_dir(
        tmp_path,
        compute_model="layer-measured",
        warmup_steps=0,
        allow_unchecked_memory=True,
    )
    assert plan.partitions == [7, 1]
    assert plan.to_dict()["workload_aggregation"] == "observed_microbatch_mean"


def test_layer_mode_preserves_memory_constraints():
    from tests.distributed.test_pp_memory import _device, _profile

    costs = aggregate([layer_trace()])
    plan = partition_layers(
        costs, memory_profile=_profile([_device(0, 2, [1] * 8), _device(1, 8, [1] * 8)])
    )
    assert plan.partitions == [2, 6]
    with pytest.raises(ValueError, match="memory bounds"):
        partition_layers(
            costs,
            memory_profile=_profile([_device(0, 1, [1] * 8), _device(1, 1, [1] * 8)]),
        )


def test_endpoint_profiles_can_follow_original_device_ids():
    primary, rotated = layer_trace(), layer_trace()
    for rank, rows in rotated.items():
        for row in rows:
            row["profile_device_id"] = 1 - rank
    costs = aggregate([primary, rotated])
    assert all(
        c.t_embedding_ms == 4 and c.t_head_ms == 8 and c.t_final_norm_ms == 2
        for c in costs
    )


def test_representative_cost_averages_different_layers():
    records = layer_trace()
    for row in records[0]:
        factor = 3 if row["batch_shape"][0]["prompt_tokens"] else 1
        row["layer_compute_ms"] = {str(i): factor * (2 * i + 1) for i in range(4)}
        row["compute_wall_ms"] = (
            sum(row["layer_compute_ms"].values()) + row["non_layer_compute_ms"]
        )
    c = aggregate([records])[0]
    assert c.t_layer_ms == 8  # mean(1,3,5,7) * mean(prefill 3, decode 1)
    assert c.phase_summary[0]["t_layer_ms"] == 12
    assert c.phase_summary[1]["t_layer_ms"] == 4


def test_microbatch_mean_includes_mixed_and_uses_observed_frequencies():
    records = layer_trace()  # one prefill, nine decode steps per rank
    for rows in records.values():
        mixed = dict(
            rows[0],
            batch_shape=[dict(query_tokens=32, prompt_tokens=31, context_tokens=32)],
        )
        rows.append(mixed)
        # Exclusion must apply to computation, endpoints and communication alike.
        rows.append(dict(mixed, is_warmup=True, compute_wall_ms=99999))
        rows.append(dict(mixed, step=0, compute_wall_ms=99999))
        rows.append(dict(mixed, num_tokens=0, compute_wall_ms=99999))
    costs = measured_layer_rank_costs(
        [records], workload="all", warmup_steps=5, layer_aggregation="microbatch"
    )
    # Two 3x steps and nine 1x steps; no token weighting or phase reweighting.
    assert [c.t_layer_ms for c in costs] == pytest.approx([15 / 11, 30 / 11])
    assert costs[0].n_steps == 11
    assert costs[0].t_comm_out_ms == pytest.approx(15 / 11)
    assert costs[0].t_embedding_ms == pytest.approx(30 / 11)
    assert costs[1].t_head_ms == pytest.approx(60 / 11)
    assert costs[1].t_final_norm_ms == pytest.approx(15 / 11)
    assert costs[1].t_fixed_ms == 0.5


@pytest.mark.parametrize("workload", ["all", "decode"])
def test_microbatch_mean_does_not_require_absent_prefill(workload):
    costs = measured_layer_rank_costs(
        [layer_trace(repeats=(0, 9))],
        workload=workload,
        warmup_steps=0,
        layer_aggregation="microbatch",
    )
    assert [c.t_layer_ms for c in costs] == [1, 2]


def test_invalid_layer_aggregation_rejected():
    with pytest.raises(ValueError, match="unknown layer aggregation"):
        measured_layer_rank_costs(
            [layer_trace()],
            workload="all",
            warmup_steps=0,
            layer_aggregation="unknown",
        )


@pytest.mark.parametrize("objective", ["latency", "throughput"])
def test_actual_mock_trace_flows_through_costs_to_dp(tmp_path, monkeypatch, objective):
    from types import SimpleNamespace

    import torch

    from tests.distributed.test_pp_layer_trace import Decoder
    from vllm.distributed.pp_stage_trace import PPStageTracer

    now = [0.0]
    monkeypatch.setattr("time.perf_counter", lambda: now[0])
    monkeypatch.setattr("time.sleep", lambda s: now.__setitem__(0, now[0] + s + 0.001))
    layers = torch.nn.ModuleList([Decoder(now, 0.001), Decoder(now, 0.003)])
    inner = SimpleNamespace(
        start_layer=0,
        end_layer=2,
        layers=layers,
        embed_tokens=Decoder(now, 0.001),
        norm=Decoder(now, 0.001),
    )
    model = SimpleNamespace(model=inner, logits_processor=Decoder(now, 0.001))

    def forward():
        now[0] += 0.004
        x = inner.embed_tokens(0)
        for layer in layers:
            x = layer(x)
        return model.logits_processor(inner.norm(x))

    tracer = PPStageTracer(
        str(tmp_path), 0, 1, torch.device("cpu"), compute_model="layer-measured"
    )
    try:
        _, timing = tracer.measure_stretched_compute(
            forward,
            lambda ms: pytest.fail("extra stage wait"),
            model_runner=SimpleNamespace(model=model),
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(enforce_eager=True)
            ),
            compute_scale=2,
        )
        tracer.record(
            num_tokens=1,
            num_reqs=1,
            num_ctx_requests=1,
            num_ctx_tokens=1,
            num_generation_requests=0,
            num_generation_tokens=0,
            batch_shape=[dict(query_tokens=1, prompt_tokens=1, context_tokens=0)],
            recv_ms=None,
            send_ms=None,
            recv_bytes=None,
            send_bytes=None,
            start_layer=0,
            end_layer=2,
            compute_scale=2,
            **timing,
        )
    finally:
        tracer.close()
    plan = plan_from_trace_dir(
        tmp_path,
        compute_model="layer-measured",
        objective=objective,
        warmup_steps=0,
        allow_unchecked_memory=True,
    )
    cost = plan.rank_costs[0]
    assert cost.t_layer_ms == pytest.approx(5)  # actual 3/7 ms, not scaled again
    assert cost.t_embedding_ms == pytest.approx(3)
    assert cost.t_head_ms == pytest.approx(3)
    assert cost.t_final_norm_ms == pytest.approx(3)
    assert cost.t_fixed_ms == pytest.approx(4)
    assert plan.partitions == [2]
    assert plan.cost_ms == pytest.approx(23)


@pytest.mark.parametrize("different", ["scale", "placement"])
def test_reject_mixed_compute_environment_for_same_device(different):
    first, second = layer_trace(), layer_trace()
    for rows in second.values():
        for row in rows:
            row["compute_delay_placement"] = "layer"
            row["compute_scale"] = 2 if different == "scale" else 1
    with pytest.raises(ValueError, match="mix compute slowdown"):
        aggregate([first, second])


def test_compute_environment_validation_uses_device_identity_not_rank():
    first, reordered = layer_trace(), layer_trace()
    for traces, order in [(first, [0, 1]), (reordered, [1, 0])]:
        for rank, rows in traces.items():
            for row in rows:
                row["profile_device_id"] = order[rank]
                row["compute_scale"] = [2, 4][order[rank]]
                row["compute_delay_placement"] = "layer"
    assert len(aggregate([first, reordered])) == 2
    reordered[0][0]["compute_scale"] = 2  # Wrong scale for device 1.
    with pytest.raises(ValueError, match="mix compute slowdown"):
        aggregate([first, reordered])
