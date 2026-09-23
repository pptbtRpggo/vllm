# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic raw trace fixtures, including paired serving communication windows."""

import json
from copy import deepcopy


def write_trace(path, records):
    # Generate raw paired timestamps from fixture costs. Production code must
    # derive the cost afresh and must not trust send_service_ms stored in JSONL.
    steps = [r.get("step", 0) for r in records]
    for index, record in enumerate(records):
        record["trace_id"] = "test-run"
        if "send_start_ns" not in record and "recv_start_ns" not in record:
            step = (
                (min(steps) + index) if len(set(steps)) != len(steps) else steps[index]
            )
            record.update(
                step=step,
                trace_session="test-session",
                clock_domain="test-clock",
                batch_id=f"batch-{step}",
                tp_size=record.get("tp_size", 1),
                comm_delay_in_window=True,
            )
            value = record.get("send_service_ms")
            if value is not None:
                record.update(
                    send_bytes=record.get("send_bytes") or 64,
                    send_start_ns=step * 1_000_000_000
                    + record["pp_rank"] * 100_000_000,
                )
                record["send_end_ns"] = record["send_start_ns"] + int(value * 1e6)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    files = sorted(path.parent.glob("pp_stage_pp*_tp0.jsonl"))
    by_rank = {}
    for file in files:
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        if rows:
            by_rank[rows[0]["pp_rank"]] = (file, rows)
    for rank, (file, rows) in by_rank.items():
        senders = {r["step"]: r for r in by_rank.get(rank - 1, (None, []))[1]}
        for row in rows:
            sender = senders.get(row.get("step"))
            if sender and "send_start_ns" in sender and "recv_start_ns" not in row:
                row.update(
                    recv_start_ns=sender["send_start_ns"],
                    recv_end_ns=sender["send_end_ns"],
                    recv_bytes=sender["send_bytes"],
                )
        file.write_text("".join(json.dumps(r) + "\n" for r in rows))


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
