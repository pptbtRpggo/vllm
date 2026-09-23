# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure idle PP links using tensor layouts observed in serving traces.

Call via collective_rpc after draining requests, on the same workers that
collected the trace. No clocks are compared across hosts. Barriers, allocation
of sender buffers and compute are outside the timed region. The timed region
uses the real send_tensor_dict/recv_tensor_dict path, including its metadata
and receiver allocation. This estimates idle link service, not contention.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch


def tensor_spec(tensors: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Record layouts, never request data or tensor contents."""
    if not all(
        isinstance(t, torch.Tensor) and t.is_contiguous() for t in tensors.values()
    ):
        return None  # Unsupported payloads must not silently lose metadata.
    return [
        dict(
            name=k,
            shape=list(t.shape),
            dtype=str(t.dtype).removeprefix("torch."),
            device_type=t.device.type,
        )
        for k, t in tensors.items()
    ]


def payload_key(spec: list[dict[str, Any]]) -> str:
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def _broadcast(group: Any, value: Any, src: int) -> Any:
    objects = [value]
    torch.distributed.broadcast_object_list(
        objects, src=group.ranks[src], group=group.cpu_group
    )
    return objects[0]


def profile_worker_links(
    worker: Any, warmup: int = 3, repeats: int = 10, all_pairs: bool = False
) -> None:
    from vllm.distributed.parallel_state import get_pp_group, get_tp_group
    from vllm.distributed.pp_hetero import sync_torch_device

    warmup, repeats = int(warmup), int(repeats)
    if warmup < 1 or repeats < 2:
        raise ValueError("link measurement needs warmup >= 1 and repeats >= 2")
    if get_tp_group().world_size != 1:
        raise ValueError("measured PP link replay currently requires TP=1")
    group = get_pp_group()
    rank = group.rank_in_group
    tracer = getattr(worker, "_pp_stage_tracer", None)
    # Check every participant before entering any point-to-point operation.
    for source in range(group.world_size):
        available = _broadcast(group, tracer is not None, source)
        if not available:
            raise ValueError("enable VLLM_PP_STAGE_TRACE on every worker")
    assert tracer is not None
    hetero = worker._pp_hetero
    if all_pairs:
        for source in range(group.world_size):
            mocked = _broadcast(
                group,
                bool(
                    any(v != 1 for v in hetero.comm_scales)
                    or hetero.comm_bandwidth_gbps
                    or hetero.comm_latency_ms
                ),
                source,
            )
            if mocked:
                raise ValueError(
                    "all-pair replay requires real links; "
                    "disable rank-based mock communication"
                )
    pairs = (
        [
            (a, b)
            for a in range(group.world_size)
            for b in range(group.world_size)
            if a != b
        ]
        if all_pairs
        else [(a, a + 1) for a in range(group.world_size - 1)]
    )
    for hop, destination in pairs:
        # Homogeneous decoder architecture: use actual first-stage payloads
        # as the common activation workload for every candidate link.
        reference = 0 if all_pairs else hop
        specs = {}
        error = None
        if rank == reference:
            for record in tracer._records:
                if record.is_warmup:
                    continue
                if not record.send_tensor_spec:
                    error = "trace has unsupported or missing send_tensor_spec"
                    break
                specs[payload_key(record.send_tensor_spec)] = record.send_tensor_spec
            if not specs:
                error = error or "no non-warmup payloads to measure"
        error, specs = _broadcast(group, (error, specs), reference)
        reference_trace_id = _broadcast(group, tracer.trace_id, reference)
        if error:
            raise ValueError(error)
        rows = []
        for key, spec in sorted(specs.items()):
            tensors = {}
            allocation_error = None
            if rank == hop:
                try:
                    tensors = {
                        entry["name"]: torch.zeros(
                            entry["shape"],
                            dtype=getattr(torch, entry["dtype"]),
                            device=(
                                "cpu"
                                if entry["device_type"] == "cpu"
                                else worker.device
                            ),
                        )
                        for entry in spec
                    }
                    sync_torch_device(worker.device)
                except Exception as exc:
                    allocation_error = str(exc)
            allocation_error = _broadcast(group, allocation_error, hop)
            if allocation_error is not None:
                raise RuntimeError(
                    f"cannot allocate replay payload: {allocation_error}"
                )
            samples = []
            for iteration in range(warmup + repeats):
                sync_torch_device(worker.device)
                group.barrier()
                start = time.perf_counter()
                if rank == hop:
                    group.send_tensor_dict(tensors, dst=destination)
                elif rank == destination:
                    received = group.recv_tensor_dict(src=hop)
                sync_torch_device(worker.device)
                # Optional emulation executes on the worker, just as in serving.
                # Its *measured* elapsed time is recorded; no modeled time is
                # ever exported as a planner cost.
                elapsed = (time.perf_counter() - start) * 1000
                if rank in (hop, destination):
                    from vllm.distributed.pp_stage_trace import tensor_dict_nbytes

                    payload = tensors if rank == hop else received
                    hetero.stretch_send(
                        hop, elapsed, payload_bytes=tensor_dict_nbytes(payload)
                    )
                elapsed = (time.perf_counter() - start) * 1000
                group.barrier()
                sender_ms = _broadcast(group, elapsed, hop)
                receiver_ms = _broadcast(group, elapsed, destination)
                if iteration >= warmup:
                    samples.append(dict(sender_ms=sender_ms, receiver_ms=receiver_ms))
            if rank == hop:
                rows.append(
                    dict(
                        trace_id=reference_trace_id,
                        pp_rank=rank,
                        dst_rank=destination,
                        reference_rank=reference,
                        tp_rank=0,
                        payload_key=key,
                        source="measured_idle_replay",
                        samples=samples,
                    )
                )
            tensors.clear()
            if rank == destination:
                del received
        if rank == hop:
            filename = (
                f"pp_topology_pp{rank}_to{destination}.jsonl"
                if all_pairs
                else f"pp_link_pp{rank}_tp0.jsonl"
            )
            path = Path(tracer.path).with_name(filename)
            temporary = path.with_suffix(".tmp")
            temporary.write_text("".join(json.dumps(r) + "\n" for r in rows))
            temporary.replace(path)


def attach_link_measurements(directory: Path, records: list[dict]) -> None:
    """Join measured samples by trace ID and exact payload; reject stale data."""
    measurements = {}
    for path in directory.glob("pp_link_pp*_tp*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("source") != "measured_idle_replay":
                raise ValueError(f"unmeasured link data in {path}")
            samples = row["samples"]
            if len(samples) < 2:
                raise ValueError("link trace needs repeated measurements")
            values = []
            for sample in samples:
                endpoints = [sample["sender_ms"], sample["receiver_ms"]]
                if any(not math.isfinite(v) or v <= 0 for v in endpoints):
                    raise ValueError("invalid measured link time")
                values.append(max(endpoints))
            key = (row["trace_id"], row["pp_rank"], row["tp_rank"], row["payload_key"])
            if key in measurements:
                raise ValueError("duplicate link measurement")
            measurements[key] = sum(values) / len(values)
    for record in records:
        # These fields are derived only from the raw measurement sidecar.
        record.pop("send_service_ms", None)
        record.pop("send_service_source", None)
        spec = record.get("send_tensor_spec")
        if not spec:
            continue
        key = (
            record.get("trace_id"),
            record["pp_rank"],
            record["tp_rank"],
            payload_key(spec),
        )
        if key in measurements:
            record["send_service_ms"] = measurements[key]
            record["send_service_source"] = "measured_idle_replay"


def measured_transfer_ms(record: dict) -> float:
    value = record.get("send_service_ms")
    if (
        record.get("send_service_source")
        not in ("measured_idle_replay", "measured_serving_overlap")
        or value is None
    ):
        raise ValueError(
            "missing measured send_service_ms; "
            f"pairing status: {record.get('send_overlap_status')}; "
            "drain requests and run "
            "collective_rpc('profile_pp_links') on the tracing workers. "
            "send_transfer_ms from bandwidth/mock configuration is not a measurement"
        )
    if not math.isfinite(value) or value <= 0:
        raise ValueError("invalid measured send_service_ms")
    return value


def load_topology_measurements(directory, records, workload, warmup_steps):
    """Aggregate raw all-pair replay with the same phase weights as compute."""
    from vllm.distributed.pp_layer_cost import phase_groups, phase_mean

    groups = phase_groups(records[0], workload, warmup_steps)
    trace_ids = {r["trace_id"] for rs in groups.values() for r in rs}
    if len(trace_ids) != 1:
        raise ValueError("topology reference must belong to one trace run")
    measurements = {}
    for path in directory.glob("pp_topology_pp*_to*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if (
                row.get("source") != "measured_idle_replay"
                or row.get("trace_id") not in trace_ids
                or row.get("reference_rank") != 0
            ):
                raise ValueError("stale or unmeasured topology data")
            values = []
            for sample in row["samples"]:
                endpoints = [sample["sender_ms"], sample["receiver_ms"]]
                if any(
                    type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                    for v in endpoints
                ):
                    raise ValueError("invalid topology measurement")
                values.append(max(endpoints))
            if len(values) < 2:
                raise ValueError("topology requires repeated link measurements")
            key = row["pp_rank"], row["dst_rank"], row["payload_key"]
            if key in measurements:
                raise ValueError("duplicate topology measurement")
            measurements[key] = sum(values) / len(values)
    if not measurements:
        raise ValueError(
            "device selection needs all-pair link replay: "
            "collective_rpc('profile_pp_links', args=(3, 10, True))"
        )
    result = {}
    pairs = {(a, b) for a, b, _ in measurements}
    for a, b in pairs:

        def cost(row, a=a, b=b):
            spec = row.get("send_tensor_spec")
            key = a, b, payload_key(spec)
            if not spec or key not in measurements:
                raise ValueError("topology is missing a reference payload measurement")
            return measurements[key]

        result[a, b] = phase_mean(groups, cost)
    return result
