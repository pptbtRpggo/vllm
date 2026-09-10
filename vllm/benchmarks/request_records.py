# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional per-request JSONL output, outside the measured traffic interval."""

import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path


def write_request_records(path, requests, outputs=None, output_lens=None, report=None):
    """Save the plan exclusively; atomically replace it with completed results.

    An interrupted benchmark retains the plan, with unknown outcomes. No disk
    writes are added to individual HTTP requests or the timed benchmark loop.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    evaluations = {r["request_id"]: r for r in report["requests"]} if report else {}

    def records():
        for i, request in enumerate(requests):
            record = {
                "request_id": request.request_id,
                "source_index": request.source_index,
                "prompt_len": request.prompt_len,
                "expected_output_len": request.expected_output_len,
                "slo": asdict(request.slo) if request.slo else None,
                "status": "planned" if outputs is None else "completed",
                "success": None,
                "attained": {"ttft": None, "tpot": None, "all": None},
            }
            if outputs is not None:
                output, length = outputs[i], output_lens[i]
                observed = {
                    "ttft": output.ttft * 1000,
                    "tpot": (
                        (output.latency - output.ttft) * 1000 / (length - 1)
                        if length > 1
                        else 0.0
                    ),
                    "e2el": output.latency * 1000,
                }
                record.update(
                    success=output.success,
                    output_len=length,
                    error=output.error,
                    observed_ms={
                        k: v if output.success and math.isfinite(v) else None
                        for k, v in observed.items()
                    },
                )
                evaluation = evaluations.get(request.request_id)
                if evaluation:
                    record.update(
                        thresholds_ms=evaluation["thresholds_ms"],
                        attained=evaluation["attained"],
                    )
            yield record

    def write(stream):
        for record in records():
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    if outputs is None:
        with path.open("x") as stream:
            write(stream)
    else:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=path.parent, prefix=".requests_", delete=False
            ) as stream:
                temporary = Path(stream.name)
                write(stream)
            os.replace(temporary, path)
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
