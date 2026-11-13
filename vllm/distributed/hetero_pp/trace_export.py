# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collect flushed profile files over the existing Ray/EngineCore RPC path."""

import hashlib
import json

import pybase64 as base64

MAX_TRACE_BYTES = 64 * 1024 * 1024


def export_profile(profiler):
    """Return bounded JSON-safe evidence after the profile writer is flushed."""
    if profiler.path.stat().st_size > MAX_TRACE_BYTES:
        raise ValueError("profile trace exceeds the 64 MiB RPC export limit")
    receipt = profiler.flush()
    data = profiler.path.read_bytes()
    if (
        len(data) > MAX_TRACE_BYTES
        or hashlib.sha256(data).hexdigest() != receipt["trace_sha256"]
    ):
        raise ValueError("profile changed after flush or exceeds export limit")
    return dict(
        rank=receipt["rank"],
        trace_session=receipt["trace_session"],
        trace_base64=base64.b64encode(data).decode("ascii"),
        manifest_base64=base64.b64encode(profiler.manifest_path.read_bytes()).decode(
            "ascii"
        ),
        manifest=receipt,
    )


def collect_profile_traces(llm, directory, receipts):
    """Fetch missing rank evidence without SSH or a shared filesystem."""
    if len(receipts) != 4 or {r.get("rank") for r in receipts} != set(range(4)):
        raise ValueError("trace collection requires four distinct rank receipts")
    for receipt in receipts:
        rank = receipt["rank"]
        path = directory / f"pp_stage_pp{rank}_tp0.jsonl"
        manifest_path = directory / f"pp_manifest_pp{rank}_tp0.json"
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != receipt.get(
                "trace_sha256"
            ):
                raise ValueError(f"existing rank {rank} raw SHA mismatch")
            if not manifest_path.exists():
                manifest_path.write_text(json.dumps(receipt, indent=2))
            continue
        replies = llm.collective_rpc("export_pp_profile", args=(rank,))
        if len(replies) != 4 or any(
            reply is not None for i, reply in enumerate(replies) if i != rank
        ):
            raise ValueError("unexpected rank profile export replies")
        reply = replies[rank]
        if (
            not isinstance(reply, dict)
            or reply.get("rank") != rank
            or reply.get("trace_session") != receipt.get("trace_session")
            or reply.get("manifest") != receipt
        ):
            raise ValueError("exported profile receipt mismatch")
        encoded = reply.get("trace_base64", "")
        if len(encoded) > (MAX_TRACE_BYTES + 2) // 3 * 4:
            raise ValueError("exported profile exceeds size limit")
        data = base64.b64decode(encoded, validate=True)
        if hashlib.sha256(data).hexdigest() != receipt.get("trace_sha256"):
            raise ValueError(f"exported rank {rank} raw SHA mismatch")
        manifest_data = base64.b64decode(reply["manifest_base64"], validate=True)
        if json.loads(manifest_data) != receipt:
            raise ValueError("exported rank manifest does not match flush receipt")
        with path.open("xb") as stream:
            stream.write(data)
        manifest_path.write_bytes(manifest_data)
