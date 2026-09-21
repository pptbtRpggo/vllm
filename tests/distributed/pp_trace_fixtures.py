# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic raw trace fixtures, including independently stored link samples."""

import json
from copy import deepcopy

from vllm.distributed.pp_link_profile import payload_key


def write_trace(path, records):
    links = {}
    for record in records:
        record["trace_id"] = "test-run"
        value = record.get("send_service_ms")
        if value is not None:
            # Distinct costs in a fixture represent distinct payload layouts.
            spec = [
                dict(
                    name="hidden_states",
                    shape=[int(value * 1000)],
                    dtype="uint8",
                    device_type="cpu",
                )
            ]
            record["send_tensor_spec"] = spec
            key = payload_key(spec)
            links[key] = dict(
                trace_id=record["trace_id"],
                pp_rank=record["pp_rank"],
                tp_rank=record["tp_rank"],
                payload_key=key,
                source="measured_idle_replay",
                samples=[dict(sender_ms=value, receiver_ms=value)] * 2,
            )
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    link_path = path.with_name(path.name.replace("pp_stage_", "pp_link_"))
    if links:
        link_path.write_text("".join(json.dumps(r) + "\n" for r in links.values()))


def write_fit_profile(directory, records_by_rank):
    """Second measured partition for integration tests using linear fixture costs."""
    directory.mkdir(exist_ok=True)
    records = deepcopy(records_by_rank)
    assert len(records) == 2
    cut = records[0][0]["end_layer"] - 1
    for rank, rows in records.items():
        for row in rows:
            old_n = row["end_layer"] - row["start_layer"]
            row["start_layer"] = 0 if rank == 0 else cut
            row["end_layer"] = cut if rank == 0 else row["end_layer"]
            row["compute_wall_ms"] *= (row["end_layer"] - row["start_layer"]) / old_n
        write_trace(directory / f"pp_stage_pp{rank}_tp0.jsonl", rows)
    return directory
