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


def profile_worker_links(worker: Any, warmup: int = 3, repeats: int = 10) -> None:
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
    for hop in range(group.world_size - 1):
        specs = {}
        error = None
        if rank == hop:
            for record in tracer._records:
                if record.is_warmup:
                    continue
                if not record.send_tensor_spec:
                    error = "trace has unsupported or missing send_tensor_spec"
                    break
                specs[payload_key(record.send_tensor_spec)] = record.send_tensor_spec
            if not specs:
                error = error or "no non-warmup payloads to measure"
        error, specs = _broadcast(group, (error, specs), hop)
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
                    group.send_tensor_dict(tensors, dst=hop + 1)
                elif rank == hop + 1:
                    received = group.recv_tensor_dict(src=hop)
                sync_torch_device(worker.device)
                # Optional emulation executes on the worker, just as in serving.
                # Its *measured* elapsed time is recorded; no modeled time is
                # ever exported as a planner cost.
                elapsed = (time.perf_counter() - start) * 1000
                if rank in (hop, hop + 1):
                    from vllm.distributed.pp_stage_trace import tensor_dict_nbytes

                    payload = tensors if rank == hop else received
                    hetero.stretch_send(
                        hop, elapsed, payload_bytes=tensor_dict_nbytes(payload)
                    )
                elapsed = (time.perf_counter() - start) * 1000
                group.barrier()
                sender_ms = _broadcast(group, elapsed, hop)
                receiver_ms = _broadcast(group, elapsed, hop + 1)
                if iteration >= warmup:
                    samples.append(dict(sender_ms=sender_ms, receiver_ms=receiver_ms))
            if rank == hop:
                rows.append(
                    dict(
                        trace_id=tracer.trace_id,
                        pp_rank=rank,
                        tp_rank=0,
                        payload_key=key,
                        source="measured_idle_replay",
                        samples=samples,
                    )
                )
            tensors.clear()
            if rank == hop + 1:
                del received
        if rank == hop:
            path = Path(tracer.path).with_name(f"pp_link_pp{rank}_tp0.jsonl")
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
