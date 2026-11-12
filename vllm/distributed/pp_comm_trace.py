# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pair serving handoffs using shared-clock windows or sender-clock ACKs.

recv_end - max(send_start, recv_start) removes the initial arrival gap only.
It still includes metadata, allocation, backend scheduling and synchronization;
it is not a device-event measurement of wire transmission or an idle-link cost.
ACK mode validates READY/DONE receipts on both endpoints, then uses only the
sender's data-start to DONE-received interval. Clock domains may differ.
"""

from collections import defaultdict
from typing import cast


def attach_paired_communication(by_key: dict[tuple[int, int], list[dict]]) -> None:
    for (rank, tp), rows in by_key.items():
        receivers = defaultdict(list)
        for record in by_key.get((rank + 1, tp), []):
            receivers[(record.get("trace_session"), record.get("step"))].append(record)
        for sender in rows:
            for field in (
                "send_overlap_ms",
                "send_overlap_sender_ms",
                "send_arrival_wait_ms",
                "recv_arrival_wait_ms",
                "send_ack_ms",
                "send_ack_ready_wait_ms",
            ):
                sender.pop(field, None)
            sender["send_overlap_status"] = "no_outgoing_transfer"
            if sender.get("send_bytes") is None:
                continue
            session = sender.get("trace_session")
            matches = receivers.get((session, sender.get("step")), [])
            if not session or len(matches) != 1:
                sender["send_overlap_status"] = "missing_or_ambiguous_peer"
                continue
            receiver = matches[0]
            status, values = paired_transfer(sender, receiver)
            sender["send_overlap_status"] = status
            sender.update(values)


def paired_transfer(sender: dict, receiver: dict) -> tuple[str, dict]:
    if sender.get("tp_size", 1) != 1 or receiver.get("tp_size", 1) != 1:
        return "unsupported_tp", {}
    mode = sender.get("comm_timing", "shared-clock")
    if mode != receiver.get("comm_timing", "shared-clock"):
        return "unmatched_communication_timing", {}
    if mode == "ack":
        return paired_ack_transfer(sender, receiver)
    if mode != "shared-clock":
        return "unknown_communication_timing", {}
    if not sender.get("clock_domain") or (
        sender["clock_domain"] != receiver.get("clock_domain")
    ):
        return "unmatched_clock_domain", {}
    if not sender.get("batch_id") or sender["batch_id"] != receiver.get("batch_id"):
        return "unmatched_batch", {}
    if sender.get("send_bytes") != receiver.get("recv_bytes"):
        return "unmatched_payload", {}
    if sender.get("comm_scale", 1) != 1 and not (
        sender.get("comm_delay_in_window") and receiver.get("comm_delay_in_window")
    ):
        return "mock_delay_outside_window", {}
    times = [
        sender.get("send_start_ns"),
        sender.get("send_end_ns"),
        receiver.get("recv_start_ns"),
        receiver.get("recv_end_ns"),
    ]
    if any(type(value) is not int or value < 0 for value in times):
        return "missing_timestamps", {}
    # The preceding guard establishes four nonnegative, exact int values.
    ss, se, rs, re = cast(tuple[int, int, int, int], tuple(times))
    begin = max(ss, rs)
    if se <= ss or re <= rs or min(se, re) <= begin:
        return "nonoverlapping_windows", {}
    return "paired", dict(
        send_overlap_ms=(re - begin) / 1e6,
        send_overlap_sender_ms=(se - begin) / 1e6,
        send_arrival_wait_ms=max(0, rs - ss) / 1e6,
        recv_arrival_wait_ms=max(0, ss - rs) / 1e6,
    )


def paired_ack_transfer(sender: dict, receiver: dict) -> tuple[str, dict]:
    """Validate both receipts, subtracting only timestamps from the same host."""
    sent, received = sender.get("send_ack"), receiver.get("recv_ack")
    if not isinstance(sent, dict) or not isinstance(received, dict):
        return "missing_ack_evidence", {}
    expected = dict(
        session=sender.get("trace_session"),
        step=sender.get("step"),
        batch_id=sender.get("batch_id"),
        src_rank=sender.get("pp_rank"),
        dst_rank=receiver.get("pp_rank"),
        payload_bytes=sender.get("send_bytes"),
    )
    if (
        not expected["session"]
        or not expected["batch_id"]
        or type(expected["step"]) is not int
        or type(expected["src_rank"]) is not int
        or expected["dst_rank"] != expected["src_rank"] + 1
        or receiver.get("trace_session") != expected["session"]
        or receiver.get("step") != expected["step"]
        or receiver.get("batch_id") != expected["batch_id"]
        or type(expected["payload_bytes"]) is not int
        or expected["payload_bytes"] < 0
        or receiver.get("recv_bytes") != expected["payload_bytes"]
        or sent.get("identity") != expected
        or received.get("identity") != expected
    ):
        return "unmatched_ack_identity", {}
    if (
        type(sent.get("protocol")) is not int
        or sent["protocol"] != 1
        or type(received.get("protocol")) is not int
        or received["protocol"] != 1
        or sent.get("ready_received") is not True
        or sent.get("done_received") is not True
        or received.get("ready_sent") is not True
        or received.get("done_sent") is not True
        or sender.get("comm_scale", 1) != 1
        or receiver.get("comm_scale", 1) != 1
    ):
        return "invalid_ack_evidence", {}
    send_times = [
        sent.get(k) for k in ("ready_start_ns", "ready_end_ns", "start_ns", "end_ns")
    ]
    recv_times = [
        received.get(k)
        for k in (
            "ready_start_ns",
            "ready_end_ns",
            "data_ready_ns",
            "done_start_ns",
            "done_end_ns",
        )
    ]
    raw = [
        sender.get("send_start_ns"),
        sender.get("send_end_ns"),
        receiver.get("recv_start_ns"),
        receiver.get("recv_end_ns"),
    ]
    if any(type(t) is not int or t < 0 for t in send_times + recv_times + raw):
        return "invalid_ack_evidence", {}
    sr, ready, start, end = cast(tuple[int, int, int, int], tuple(send_times))
    rr, rr_end, data, done, done_end = cast(
        tuple[int, int, int, int, int], tuple(recv_times)
    )
    ss, se, rs, re = cast(tuple[int, int, int, int], tuple(raw))
    if not (
        sr <= ready <= start == ss < se <= end
        and rr <= rr_end <= rs < re == data <= done <= done_end
    ):
        return "invalid_ack_evidence", {}
    return "paired_ack", dict(
        send_ack_ms=(end - start) / 1e6, send_ack_ready_wait_ms=(ready - sr) / 1e6
    )


def select_serving_communication(records: list[dict]) -> None:
    for record in records:
        record.pop("send_service_ms", None)
        record.pop("send_replay_ms", None)
        record.pop("send_service_source", None)
        if record.get("send_overlap_status") == "paired":
            record["send_service_ms"] = record["send_overlap_ms"]
            record["send_service_source"] = "measured_serving_overlap"
        elif record.get("send_overlap_status") == "paired_ack":
            record["send_service_ms"] = record["send_ack_ms"]
            record["send_service_source"] = "measured_serving_ack"
