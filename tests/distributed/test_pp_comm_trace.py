# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest

from tests.distributed.pp_trace_fixtures import write_fit_profile
from vllm.distributed.pp_comm_trace import attach_paired_communication, paired_transfer
from vllm.distributed.pp_partition import load_trace_records, plan_from_trace_dir


def records(ss=0, se=10, rs=8, re=10):
    shared = dict(
        trace_session="session",
        step=6,
        clock_domain="same-boot-time-ns",
        batch_id="hash",
        tp_size=1,
        tp_rank=0,
        num_tokens=8,
        num_reqs=1,
        batch_shape=[dict(query_tokens=8, context_tokens=16, prompt_tokens=0)],
        compute_wall_ms=2,
        compute_ms=999,
        pp_size=2,
    )
    return (
        shared
        | dict(
            pp_rank=0,
            start_layer=0,
            end_layer=2,
            send_bytes=64,
            send_start_ns=int(ss * 1e6),
            send_end_ns=int(se * 1e6),
        ),
        shared
        | dict(
            pp_rank=1,
            start_layer=2,
            end_layer=4,
            recv_bytes=64,
            recv_start_ns=int(rs * 1e6),
            recv_end_ns=int(re * 1e6),
        ),
    )


@pytest.mark.parametrize("ss,rs", [(0, 8), (8, 0), (8, 8)])
def test_later_arrival_is_the_start(ss, rs):
    sender, receiver = records(ss=ss, rs=rs)
    status, values = paired_transfer(sender, receiver)
    assert status == "paired"
    assert values["send_overlap_ms"] == 2
    assert values["send_arrival_wait_ms"] == max(0, rs - ss)
    assert values["recv_arrival_wait_ms"] == max(0, ss - rs)


@pytest.mark.parametrize(
    "change,status",
    [
        (dict(clock_domain="different-host"), "unmatched_clock_domain"),
        (dict(clock_domain=None), "unmatched_clock_domain"),
        (dict(batch_id="different-batch"), "unmatched_batch"),
        (dict(recv_bytes=128), "unmatched_payload"),
        (dict(recv_start_ns=None), "missing_timestamps"),
        (dict(recv_end_ns=1), "nonoverlapping_windows"),
        (dict(tp_size=2), "unsupported_tp"),
    ],
)
def test_invalid_pairs_fail_closed(change, status):
    sender, receiver = records()
    assert paired_transfer(sender, receiver | change) == (status, {})


def test_mock_delay_is_not_silently_lost():
    sender, receiver = records()
    assert paired_transfer(sender | dict(comm_scale=2), receiver)[0] == (
        "mock_delay_outside_window"
    )


def test_mock_delay_inside_actual_windows_is_not_rescaled():
    sender, receiver = records(se=13, re=13)
    sender.update(comm_scale=100, comm_delay_in_window=True)
    # Both endpoints must explicitly mark the new timing boundary.
    assert paired_transfer(sender, receiver)[0] == "mock_delay_outside_window"
    receiver["comm_delay_in_window"] = True
    status, values = paired_transfer(sender, receiver)
    assert status == "paired"
    assert values["send_overlap_ms"] == 5  # 13 - max(0, 8), not a config multiple.


def test_duplicate_or_missing_peer_does_not_pair_by_position():
    sender, receiver = records()
    for peers in [[], [receiver, receiver], [receiver | dict(step=7)]]:
        attach_paired_communication({(0, 0): [sender], (1, 0): peers})
        assert sender["send_overlap_status"] == "missing_or_ambiguous_peer"
        assert "send_overlap_ms" not in sender


def test_serving_windows_reach_dp_without_replay_or_mock_config(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_PP_HETERO", "1,100/999")
    sender, receiver = records()
    for row in [sender, receiver]:
        (tmp_path / f"pp_stage_pp{row['pp_rank']}_tp0.jsonl").write_text(
            json.dumps(row) + "\n"
        )
    fit_dir = write_fit_profile(tmp_path / "fit", {0: [sender], 1: [receiver]})
    plan = plan_from_trace_dir(
        tmp_path,
        fit_trace_dirs=[fit_dir],
        comm_source="serving",
        min_pp_size=2,
        allow_unchecked_memory=True,
    )
    assert plan.cost_ms == 4
    assert plan.rank_costs[0].comm_source == "measured_serving_overlap"
    loaded = load_trace_records(tmp_path, comm_source="serving")
    assert loaded[0][0]["send_arrival_wait_ms"] == 8
    assert loaded[0][0]["send_service_ms"] == 2
