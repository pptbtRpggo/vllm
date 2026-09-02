# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host-side JSONL trace for τ-Batch PP occupancy.

Each emit→done pair is one micro-batch traversing the whole pipeline.
The driver Future completes when the last PP stage finishes, so this
file cannot split per-stage widths. Overlapping emit/done bars are the
real in-flight occupancy on the machine.

Enable with ``--tau-batch-trace PATH`` or env ``TAU_BATCH_TRACE``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from vllm.logger import init_logger

logger = init_logger(__name__)

SCHEMA_VERSION = 1
ENV_TRACE_PATH = "TAU_BATCH_TRACE"


def resolve_trace_path(config_path: str | None) -> str | None:
    """Return the JSONL path from config or ``TAU_BATCH_TRACE``.

    Args:
        config_path: ``SchedulerConfig.tau_batch_trace``. Empty is unset.

    Returns:
        Absolute path, or None if tracing is off.
    """
    path = (config_path or "").strip()
    if not path:
        path = os.environ.get(ENV_TRACE_PATH, "").strip()
    if not path:
        return None
    return str(Path(path).expanduser().resolve())


class JsonlTracer:
    """Append-only JSONL writer. One event per line, flushed immediately."""

    def __init__(self, path: str) -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fp: TextIO = open(path, "a", encoding="utf-8")
        self._lock = threading.Lock()
        self._fwd_id = 0
        logger.warning("TauScheduler JSONL trace: %s", path)
        self.record("meta", schema=SCHEMA_VERSION)

    def close(self) -> None:
        with self._lock:
            if not self._fp.closed:
                self._fp.close()

    def next_fwd_id(self) -> int:
        with self._lock:
            self._fwd_id += 1
            return self._fwd_id

    def record(self, event: str, **fields: Any) -> None:
        """Write one event. ``ts_ns`` is wall clock, ``mono_ns`` is monotonic."""
        rec: dict[str, Any] = {
            "ts_ns": time.time_ns(),
            "mono_ns": time.monotonic_ns(),
            "event": event,
        }
        rec.update(fields)
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with self._lock:
            if self._fp.closed:
                return
            self._fp.write(line)
            self._fp.write("\n")
            self._fp.flush()


def load_events(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL events, skipping blank lines."""
    events: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


@dataclass(frozen=True)
class ForwardSpan:
    """One scheduled micro-batch from emit until driver Future done."""

    fwd_id: int
    wave_id: int | None
    batch_idx: int | None
    phase: str
    req_ids: tuple[str, ...]
    num_tokens: int | None
    emit_mono_ns: int
    done_mono_ns: int
    enqueue_mono_ns: int | None = None
    dequeue_mono_ns: int | None = None
    emit_ts_ns: int | None = None
    done_ts_ns: int | None = None

    @property
    def job(self) -> str:
        short = "pre" if self.phase == "prefill" else "dec"
        wave = f"w{self.wave_id}_" if self.wave_id is not None else ""
        batch = f"B{self.batch_idx}" if self.batch_idx is not None else "B?"
        return f"{wave}{batch}_{short}"

    @property
    def duration_ms(self) -> float:
        return (self.done_mono_ns - self.emit_mono_ns) / 1e6


def pair_forwards(events: Iterable[Mapping[str, Any]]) -> list[ForwardSpan]:
    """Join emit/done (and optional enqueue/dequeue) by ``fwd_id``."""
    pending: dict[int, dict[str, Any]] = {}
    spans: list[ForwardSpan] = []
    for ev in events:
        event = ev.get("event")
        fwd_id = ev.get("fwd_id")
        if fwd_id is None:
            continue
        fid = int(fwd_id)
        slot = pending.setdefault(fid, {})
        if event == "emit":
            slot["emit"] = ev
        elif event == "done":
            slot["done"] = ev
        elif event == "enqueue":
            slot["enqueue"] = ev
        elif event == "dequeue":
            slot["dequeue"] = ev
        emit = slot.get("emit")
        done = slot.get("done")
        if emit is None or done is None:
            continue
        req_ids = tuple(emit.get("req_ids") or done.get("req_ids") or ())
        spans.append(
            ForwardSpan(
                fwd_id=fid,
                wave_id=_opt_int(emit.get("wave_id", done.get("wave_id"))),
                batch_idx=_opt_int(
                    emit.get("batch_idx", done.get("batch_idx"))
                ),
                phase=str(emit.get("phase") or done.get("phase") or ""),
                req_ids=req_ids,
                num_tokens=_opt_int(emit.get("num_tokens")),
                emit_mono_ns=int(emit["mono_ns"]),
                done_mono_ns=int(done["mono_ns"]),
                enqueue_mono_ns=_opt_int(
                    (slot.get("enqueue") or {}).get("mono_ns")
                ),
                dequeue_mono_ns=_opt_int(
                    (slot.get("dequeue") or {}).get("mono_ns")
                ),
                emit_ts_ns=_opt_int(emit.get("ts_ns")),
                done_ts_ns=_opt_int(done.get("ts_ns")),
            )
        )
        del pending[fid]
    spans.sort(key=lambda s: (s.emit_mono_ns, s.fwd_id))
    return spans


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
