# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest

from tests.distributed.pp_trace_fixtures import write_trace
from tests.distributed.test_pp_layer_cost import layer_trace
from vllm.distributed.pp_link_profile import (
    load_topology_measurements,
    measured_transfer_ms,
)
from vllm.distributed.pp_partition import load_trace_records


def test_unused_link_files_cannot_change_serving_cost(tmp_path):
    for rank, rows in layer_trace().items():
        write_trace(tmp_path / f"pp_stage_pp{rank}_tp0.jsonl", rows)
    for name in ("pp_link_pp0_tp0.jsonl", "pp_topology_pp0_to1.jsonl"):
        (tmp_path / name).write_text("invalid old data")
    records = load_trace_records(tmp_path)
    assert measured_transfer_ms(records[0][0]) == 3
    assert load_topology_measurements([records], "all", 0) == {(0, 1): 1.2}
    assert "send_replay_ms" not in records[0][0]


def test_link_means_include_mixed_and_drop_warmup():
    rows = [
        dict(
            step=10,
            num_tokens=2,
            num_ctx_tokens=2,
            num_generation_tokens=0,
            send_service_ms=10,
        ),
        dict(
            step=11,
            num_tokens=1,
            num_ctx_tokens=0,
            num_generation_tokens=1,
            send_service_ms=2,
        ),
        dict(
            step=12,
            num_tokens=2,
            num_ctx_tokens=1,
            num_generation_tokens=1,
            send_service_ms=6,
        ),
        dict(step=13, num_tokens=1, is_warmup=True, send_service_ms=999),
    ]
    for row in rows:
        row.update(profile_device_id=0, send_service_source="measured_serving_overlap")
    records = {0: rows, 1: [dict(profile_device_id=1)]}
    assert load_topology_measurements([records], "all", 5) == {(0, 1): 6}
    assert load_topology_measurements([records], "decode", 5) == {(0, 1): 2}


def test_reordered_runs_use_device_pairs_and_sample_counts(tmp_path):
    trace_sets = []
    for name, order, repeats in [("first", [0, 1], (1, 9)), ("second", [1, 0], (0, 2))]:
        directory = tmp_path / name
        directory.mkdir()
        for rank, rows in layer_trace(repeats=repeats).items():
            for row in rows:
                row["device_id"] = order[rank]
            write_trace(directory / f"pp_stage_pp{rank}_tp0.jsonl", rows)
        trace_sets.append(load_trace_records(directory))
    assert load_topology_measurements(trace_sets, "all", 0) == {(0, 1): 1.2, (1, 0): 1}
    assert load_topology_measurements(trace_sets[:1], "all", 0) == {(0, 1): 1.2}
    (tmp_path / "second" / "pp_device_map.json").write_text(json.dumps([0, 1]))
    with pytest.raises(ValueError, match="conflicts"):
        load_trace_records(tmp_path / "second")


@pytest.mark.parametrize("value", [None, -1, 0, float("nan"), True])
def test_bad_or_missing_measurement_cannot_be_a_link(value):
    with pytest.raises(ValueError, match="measured"):
        measured_transfer_ms(
            dict(send_service_source="measured_serving_overlap", send_service_ms=value)
        )
    with pytest.raises(ValueError, match="missing measured"):
        measured_transfer_ms(
            dict(send_service_source="measured_idle_replay", send_service_ms=1)
        )
