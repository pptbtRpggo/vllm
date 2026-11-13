# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise offline speed statistics through the actual single-file wrapper."""

import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples/online_serving/hetero_pp/profile_edgeshard.sh"
)


def write_trace(directory, *, defect=None):
    directory.mkdir()
    rows: list[list[dict]] = [[] for _ in range(4)]
    # One warmup, then speeds 8 / 2 / 4 Gbps; weighted speed is 4 Gbps.
    for step, (payload, duration, ctx, gen) in enumerate(
        [
            (1_000_000, 100_000, 4, 0),
            (1_000_000, 1_000_000, 4, 0),
            (500_000, 2_000_000, 0, 4),
            (500_000, 1_000_000, 2, 2),
        ]
    ):
        for rank in range(4):
            compute = 10_000 if step == 0 else 10 * step * (rank + 1)
            sample = 2 if rank == 3 else 0
            row = dict(
                pp_rank=rank,
                pp_size=4,
                tp_rank=0,
                tp_size=1,
                trace_session="test-session",
                step=step,
                batch_id=f"b{step}",
                is_warmup=step == 0,
                comm_timing="ack",
                transport="cpu_gloo",
                num_ctx_tokens=ctx,
                num_generation_tokens=gen,
                compute_wall_ms=compute,
                execute_device_ms=compute - sample,
                sample_device_ms=sample,
                # These host/legacy values must not replace device intervals.
                execute_host_ms=123456,
                compute_ms=999999,
            )
            if rank < 3:
                start = 100_000_000
                row.update(
                    send_bytes=payload,
                    send_start_ns=start,
                    send_end_ns=start + duration // 2,
                    send_ack=dict(
                        protocol=1,
                        ready_received=True,
                        done_received=True,
                        ready_start_ns=start - 2,
                        ready_end_ns=start - 1,
                        start_ns=start,
                        end_ns=start + duration,
                        identity=dict(
                            session="test-session",
                            step=step,
                            batch_id=f"b{step}",
                            src_rank=rank,
                            dst_rank=rank + 1,
                            payload_bytes=payload,
                        ),
                    ),
                )
            if rank:
                # Deliberately different clock domain from sender's timestamps.
                row.update(
                    recv_bytes=payload,
                    recv_start_ns=1000,
                    recv_end_ns=2000,
                    recv_ack=dict(
                        protocol=1,
                        ready_sent=True,
                        done_sent=True,
                        ready_start_ns=100,
                        ready_end_ns=200,
                        data_ready_ns=2000,
                        done_start_ns=2100,
                        done_end_ns=2200,
                        identity=dict(
                            session="test-session",
                            step=step,
                            batch_id=f"b{step}",
                            src_rank=rank - 1,
                            dst_rank=rank,
                            payload_bytes=payload,
                        ),
                    ),
                )
            rows[rank].append(row)
    if defect == "missing_peer":
        rows[2].pop()
    elif defect == "bad_ack":
        rows[2][-1]["recv_ack"]["done_sent"] = False
    elif defect == "duplicate":
        rows[1].append(rows[1][-1])
    elif defect == "warmup_mismatch":
        rows[1][0]["is_warmup"] = False
    elif defect == "phase_mismatch":
        rows[1][1]["num_ctx_tokens"] = 0
    elif defect == "missing_compute":
        rows[3][1].pop("compute_wall_ms")
    elif defect == "nonfinite_compute":
        rows[2][1]["compute_wall_ms"] = float("nan")
    elif defect == "negative_compute":
        rows[0][1]["compute_wall_ms"] = -1
    elif defect == "compute_conservation":
        rows[3][1]["sample_device_ms"] += 3
    raw = {}
    for rank, records in enumerate(rows):
        name = f"pp_stage_pp{rank}_tp0.jsonl"
        content = "".join(json.dumps(row) + "\n" for row in records).encode()
        (directory / name).write_bytes(content)
        raw[name] = hashlib.sha256(content).hexdigest()
    manifest = dict(
        version=1,
        complete=True,
        transport="cpu_gloo",
        comm_timing="ack",
        trace_session="test-session",
        raw_files=raw,
    )
    (directory / "profile_manifest.json").write_text(json.dumps(manifest))
    if defect == "hash":
        with (directory / "pp_stage_pp0_tp0.jsonl").open("a") as stream:
            stream.write("\n")


def run_summary(directory):
    return subprocess.run(
        ["bash", str(SCRIPT), "--stats-only", str(directory)],
        env={**os.environ, "PY": sys.executable},
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_speed_summary_excludes_warmup_and_distinguishes_weighted_mean(tmp_path):
    directory = tmp_path / "trace with spaces"
    write_trace(directory)
    result = run_summary(directory)
    assert result.returncode == 0, result.stderr
    with (directory / "communication_speed_summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12  # Three links, all/prefill/decode/mixed.
    for row in (r for r in rows if r["phase"] == "all"):
        assert int(row["samples"]) == 3
        assert float(row["mean_gbps"]) == pytest.approx(14 / 3)
        assert float(row["aggregate_gbps"]) == pytest.approx(4)
        for key, expected in {
            "p10_gbps": 2.4,
            "p50_gbps": 4,
            "p90_gbps": 7.2,
            "p95_gbps": 7.6,
            "p99_gbps": 7.92,
        }.items():
            assert float(row[key]) == pytest.approx(expected)
    with (directory / "communication_speed_samples.csv").open() as stream:
        samples = list(csv.DictReader(stream))
    assert len(samples) == 9
    assert {row["phase"] for row in samples} == {"prefill", "decode", "mixed"}
    assert "stage1->2" in result.stdout


@pytest.mark.parametrize(
    "defect",
    [
        "missing_peer",
        "bad_ack",
        "duplicate",
        "hash",
        "warmup_mismatch",
        "phase_mismatch",
    ],
)
def test_speed_summary_refuses_invalid_evidence(tmp_path, defect):
    directory = tmp_path / "trace"
    write_trace(directory, defect=defect)
    result = run_summary(directory)
    assert result.returncode != 0
    assert not (directory / "communication_speed_summary.csv").exists()


def test_speed_summary_requires_complete_manifest(tmp_path):
    directory = tmp_path / "trace"
    write_trace(directory)
    path = directory / "profile_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["complete"] = False
    path.write_text(json.dumps(manifest))
    assert run_summary(directory).returncode != 0


def test_timing_table_uses_device_compute_and_sender_ack_windows(tmp_path):
    directory = tmp_path / "trace"
    write_trace(directory)
    result = run_summary(directory)
    assert result.returncode == 0, result.stderr
    with (directory / "stage_and_link_timing_summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 28  # Four stages + three links, four phase groups.
    lookup = {(r["item"], r["phase"]): r for r in rows}
    for rank in range(4):
        row = lookup[(f"stage{rank}", "all")]
        assert row["kind"] == "compute"
        assert int(row["samples"]) == 3
        assert float(row["mean_ms"]) == pytest.approx(20 * (rank + 1))
        assert float(row["p90_ms"]) == pytest.approx(28 * (rank + 1))
        assert float(lookup[(f"stage{rank}", "decode")]["mean_ms"]) == 20 * (rank + 1)
    for rank in range(3):
        row = lookup[(f"stage{rank}->{rank + 1}", "all")]
        assert row["kind"] == "handoff"
        assert float(row["mean_ms"]) == pytest.approx(4 / 3)
        assert float(row["p50_ms"]) == 1
        assert float(row["p90_ms"]) == pytest.approx(1.8)
        assert float(row["p95_ms"]) == pytest.approx(1.9)
        assert float(row["p99_ms"]) == pytest.approx(1.98)
    with (directory / "stage_compute_samples.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 12
    for label in ("stage0", "stage1", "stage2", "stage3", "计算与交接耗时"):
        assert label in result.stdout


@pytest.mark.parametrize(
    "defect",
    [
        "missing_compute",
        "nonfinite_compute",
        "negative_compute",
        "compute_conservation",
    ],
)
def test_timing_table_rejects_missing_or_invalid_device_intervals(tmp_path, defect):
    directory = tmp_path / "trace"
    write_trace(directory, defect=defect)
    result = run_summary(directory)
    assert result.returncode != 0
    assert not (directory / "stage_and_link_timing_summary.csv").exists()


@pytest.mark.parametrize("collect_exit", [0, 42])
def test_profile_automatically_summarizes_only_after_success(tmp_path, collect_exit):
    fixture = tmp_path / "fixture"
    write_trace(fixture)
    model = tmp_path / "model"
    model.mkdir()
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    calls = tmp_path / "calls"
    interpreter = tmp_path / "fake-python"
    # Replace only the model CLI; run the actual inline statistics unchanged.
    interpreter.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f'if [[ "$1" == - ]]; then exec {shlex.quote(sys.executable)} "$@"; fi\n'
        'printf "%s\\n" "$*" >> "$CALL_LOG"\n'
        'if [[ " $* " == *" --collect-only "* ]]; then\n'
        f"  if [[ {collect_exit} != 0 ]]; then exit {collect_exit}; fi\n"
        '  while [[ "$1" != --trace-dir ]]; do shift; done\n'
        '  cp "$FIXTURE"/* "$2/"\n'
        "fi\n"
    )
    interpreter.chmod(0o700)
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env={
            **os.environ,
            "PY": str(interpreter),
            "MODEL": str(model),
            "PLAN": str(plan),
            "RAY_ADDRESS": "141.147.135.53:16379",
            "TRACE_ROOT": str(tmp_path / "shared"),
            "CALL_LOG": str(calls),
            "FIXTURE": str(fixture),
            "PROFILE_GPU_MEMORY_UTILIZATION": "0.7",
            "PROFILE_MAX_NUM_SEQS": "8",
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == collect_exit, result.stderr
    captured = calls.read_text().splitlines()
    assert "--comm-timing ack" in captured[0]
    assert "--gpu-memory-utilization 0.7" in captured[0]
    assert "--max-num-seqs 8" in captured[0]
    summaries = list((tmp_path / "shared").glob("*/communication_speed_summary.csv"))
    if collect_exit:
        assert len(captured) == 1 and not summaries
    else:
        assert len(captured) == 2 and len(summaries) == 1
        assert "--skip-run" in captured[1]
        assert "stage1->2" in result.stdout
