# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import random
from dataclasses import replace

import pytest

from tests.distributed.test_pp_memory import _device, _profile
from vllm.distributed.pp_device_partition import partition_devices
from vllm.distributed.pp_partition import RankCost


def costs(speeds):
    return [
        RankCost(
            r,
            2,
            v,
            None,
            10,
            compute_model="layer-measured",
            t_embedding_ms=2,
            t_head_ms=3,
            t_final_norm_ms=1,
            t_fixed_ms=0.5,
        )
        for r, v in enumerate(speeds)
    ]


@pytest.mark.parametrize("objective", ["latency", "throughput"])
def test_subset_order_memory_and_ties_match_exhaustive_oracle(objective):
    rng = random.Random(652)
    for _ in range(20):
        n, p = 6, 4
        c = costs([rng.randint(1, 6) for _ in range(p)])
        links = {
            (a, b): rng.randint(1, 8) for a in range(p) for b in range(p) if a != b
        }
        memory = _profile(
            [
                _device(
                    r,
                    rng.randint(5, 12),
                    [1] * n,
                    first_stage_bytes=2,
                    last_stage_bytes=4,
                )
                for r in range(p)
            ]
        )
        oracle = []
        for size in range(1, p + 1):
            for tail in itertools.permutations(range(1, p), size - 1):
                order = (0,) + tail
                for cuts in itertools.combinations(range(1, n), size - 1):
                    edges = (0,) + cuts + (n,)
                    parts = tuple(b - a for a, b in zip(edges, edges[1:]))
                    if any(
                        parts[r] + (2 if r == 0 else 0) + (4 if r == size - 1 else 0)
                        > memory.devices[d].budget_bytes
                        for r, d in enumerate(order)
                    ):
                        continue
                    calc = [
                        c[d].t_layer_ms * parts[r]
                        + 0.5
                        + (2 if r == 0 else 0)
                        + (4 if r == size - 1 else 0)
                        for r, d in enumerate(order)
                    ]
                    hops = [0] + [links[a, b] for a, b in zip(order, order[1:])] + [0]
                    serial = sum(calc) + sum(hops)
                    primary = (
                        serial
                        if objective == "latency"
                        else max(calc[r] + hops[r] + hops[r + 1] for r in range(size))
                    )
                    oracle.append((primary, serial, order, parts))
        if not oracle:
            with pytest.raises(ValueError, match="No feasible"):
                partition_devices(
                    c, links, num_layers=n, objective=objective, memory_profile=memory
                )
            continue
        expected = min(oracle)
        plan = partition_devices(
            c, links, num_layers=n, objective=objective, memory_profile=memory
        )
        assert (
            plan.cost_ms,
            plan.to_dict()["predicted_sequential_ms"],
            plan.device_order,
            tuple(plan.partitions),
        ) == expected
        assert plan.memory_profile.pp_size == plan.pp_size
        for r, original in enumerate(plan.device_order):
            assert (
                plan.memory_profile.devices[r].budget_bytes
                == memory.devices[original].budget_bytes
            )
        assert all(u["headroom_bytes"] >= 0 for u in plan.to_dict()["memory_usage"])


def test_reordering_is_real_and_sender_communication_is_charged():
    c = costs([1, 1, 1])
    links = {(0, 1): 100, (0, 2): 1, (2, 1): 1, (1, 2): 100}
    plan = partition_devices(
        c, links, num_layers=6, min_pp_size=3, allow_unchecked_memory=True
    )
    assert plan.device_order == (0, 2, 1)
    payload = plan.to_dict()
    compute = payload["predicted_stage_compute_ms"]
    assert plan.cost_ms == max(compute[0] + 1, compute[1] + 2, compute[2] + 1)
    assert payload["rank_costs"][0]["t_comm_out_ms"] == 1


def test_unmeasured_endpoint_or_link_cannot_be_chosen():
    c = costs([1, 1, 1])
    c[0] = replace(c[0], t_head_ms=None)
    c[1] = replace(c[1], t_head_ms=None)
    with pytest.raises(ValueError, match="No feasible"):
        partition_devices(c, {(0, 1): 1}, num_layers=6, allow_unchecked_memory=True)
    plan = partition_devices(c, {(0, 2): 1}, num_layers=6, allow_unchecked_memory=True)
    assert plan.device_order == (0, 2)
    with pytest.raises(ValueError, match="blocking"):
        partition_devices(
            c, {(0, 2): 1}, overlap_comm=True, allow_unchecked_memory=True
        )


def test_cli_device_selection_uses_raw_topology_and_keeps_pool_memory(tmp_path):
    import json

    from tests.distributed.pp_trace_fixtures import write_trace
    from tests.distributed.test_pp_layer_cost import layer_trace
    from vllm.distributed.pp_link_profile import payload_key
    from vllm.distributed.pp_profile import profile_pp_partition

    for rank, rows in layer_trace().items():
        write_trace(tmp_path / f"pp_stage_pp{rank}_tp0.jsonl", rows)
    rows = [
        json.loads(line)
        for line in (tmp_path / "pp_stage_pp0_tp0.jsonl").read_text().splitlines()
    ]
    specs = {payload_key(r["send_tensor_spec"]) for r in rows}
    raw = [
        dict(
            trace_id=rows[0]["trace_id"],
            pp_rank=0,
            dst_rank=1,
            reference_rank=0,
            payload_key=key,
            source="measured_idle_replay",
            samples=[dict(sender_ms=2, receiver_ms=1)] * 2,
        )
        for key in specs
    ]
    topology = tmp_path / "pp_topology_pp0_to1.jsonl"
    topology.write_text("".join(json.dumps(r) + "\n" for r in raw))
    memory = _profile(
        [
            _device(r, 100, [1] * 8, first_stage_bytes=2, last_stage_bytes=4)
            for r in range(2)
        ]
    )
    path = tmp_path / "pp_memory_profile.json"
    path.write_text(json.dumps(memory.to_dict()))
    original = path.read_bytes()
    # Device selection uses its topology; adjacent clock pairing/replay may be
    # unavailable (e.g. cross-host clocks). It must not depend on those costs.
    for adjacent in tmp_path.glob("pp_link_*.jsonl"):
        adjacent.unlink()
    plan = profile_pp_partition(
        dump_dir=tmp_path,
        skip_run=True,
        compute_model="layer-measured",
        comm_source="replay",
        warmup_steps=0,
        device_selection=True,
    )
    assert plan.device_order == (0, 1)
    assert plan.partitions == [7, 1]
    assert path.read_bytes() == original
    assert (tmp_path / "pp_selected_memory_profile.json").exists()
    raw[0]["trace_id"] = "stale"
    topology.write_text("".join(json.dumps(r) + "\n" for r in raw))
    with pytest.raises(ValueError, match="stale"):
        profile_pp_partition(
            dump_dir=tmp_path,
            skip_run=True,
            compute_model="layer-measured",
            comm_source="replay",
            warmup_steps=0,
            device_selection=True,
        )
