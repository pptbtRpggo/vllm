# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pair serving communication windows on a verified shared monotonic clock.

recv_end - max(send_start, recv_start) removes the initial arrival gap only.
It still includes metadata, allocation, backend scheduling and synchronization;
it is not a device-event measurement of wire transmission or an idle-link cost.
"""

from collections import defaultdict


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
    ss, se, rs, re = times
    begin = max(ss, rs)
    if se <= ss or re <= rs or min(se, re) <= begin:
        return "nonoverlapping_windows", {}
    return "paired", dict(
        send_overlap_ms=(re - begin) / 1e6,
        send_overlap_sender_ms=(se - begin) / 1e6,
        send_arrival_wait_ms=max(0, rs - ss) / 1e6,
        recv_arrival_wait_ms=max(0, ss - rs) / 1e6,
    )


def select_serving_communication(records: list[dict]) -> None:
    for record in records:
        record.pop("send_service_ms", None)
        record.pop("send_replay_ms", None)
        record.pop("send_service_source", None)
        if record.get("send_overlap_status") == "paired":
            record["send_service_ms"] = record["send_overlap_ms"]
            record["send_service_source"] = "measured_serving_overlap"
