# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host-side JSONL trace for τ-Batch PP occupancy.

Driver events (emit/done) span the whole pipeline Future. Worker
``stage`` is the whole execute_model wall. Inside that, ``recv`` /
``compute`` / ``send`` split the PP rank: recv waits for the previous
send, compute is this stage's forward, send is enqueue-return only
(NCCL send is async). Plotting uses measured compute windows when available,
otherwise the original host stage envelopes. Cross-process alignment uses
``time.time_ns()``, not monotonic.

Enable with ``--tau-batch-trace PATH`` or env ``TAU_BATCH_TRACE``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

SCHEMA_VERSION = 5
ENV_TRACE_PATH = "TAU_BATCH_TRACE"


def _requires_file_lock(path: str) -> bool:
    """Only skip flock on recognized local Linux filesystems.

    NFS can emulate O_APPEND; retain serialization there and on unknown hosts.
    Resolve the longest mount prefix once, outside the per-record path.
    """
    local_filesystems = {
        "ext2",
        "ext3",
        "ext4",
        "xfs",
        "btrfs",
        "tmpfs",
        "overlay",
        "ramfs",
        "zfs",
        "f2fs",
    }
    target = Path(path).resolve()
    best_depth, filesystem = -1, ""
    try:
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            left, right = line.split(" - ", 1)
            mount = left.split()[4]
            for escaped, literal in (
                (r"\040", " "),
                (r"\011", "\t"),
                (r"\012", "\n"),
                (r"\134", "\\"),
            ):
                mount = mount.replace(escaped, literal)
            parent = Path(mount)
            if (target == parent or parent in target.parents) and len(
                parent.parts
            ) > best_depth:
                best_depth, filesystem = len(parent.parts), right.split()[0]
    except (OSError, ValueError, IndexError):
        return True
    return filesystem not in local_filesystems


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
    """Synchronous, single-write O_APPEND JSONL for a local filesystem.

    Each encoded record is appended with one OS write, without a userspace
    buffer. Recognized local Linux filesystems skip cross-process flock;
    network/unknown filesystems retain it because append may be emulated.
    Records are visible before record() returns; no background drain is needed
    at benchmark boundaries. This does not fsync to durable storage.

    If the path is removed later, the next write creates it again so a new
    run can start without restarting the server.
    """

    def __init__(
        self,
        path: str,
        *,
        write_meta: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = path
        self._write_meta = write_meta
        self._metadata = dict(metadata or {})
        self._fd: int | None = None
        self._lock = threading.RLock()
        self._fwd_id = 0
        self._file_lock = _requires_file_lock(path)
        if self._file_lock and write_meta:
            logger.warning(
                "Tau trace %s is on a shared or unidentified filesystem; "
                "retaining file locking. Prefer a local /tmp path to avoid "
                "high write latency.",
                path,
            )

    def close(self) -> None:
        with self._lock:
            if self._fd is not None:
                os.close(self._fd)
            self._fd = None

    def next_fwd_id(self) -> int:
        with self._lock:
            self._ensure_open()
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
        line = (json.dumps(rec, ensure_ascii=False, default=str) + "\n").encode("utf-8")
        with self._lock:
            if not self._ensure_open():
                return
            self._write_line(line)

    def _write_line(self, line: bytes) -> None:
        assert self._fd is not None
        if self._file_lock:
            import fcntl

            fcntl.flock(self._fd, fcntl.LOCK_EX)
        try:
            if os.write(self._fd, line) != len(line):
                # A suffix retry could interleave with another process's line.
                raise OSError("Short tau trace write; trace is incomplete")
        finally:
            if self._file_lock:
                fcntl.flock(self._fd, fcntl.LOCK_UN)

    def _fd_tracks_path(self, path: Path) -> bool:
        """False if the fd is closed, the path is gone, or it is a new inode.

        Deleting the JSONL while serve is up creates a new file on the next
        driver write. A worker that only checks ``path.exists()`` keeps the
        old fd and writes stages into the unlinked inode.
        """
        if self._fd is None:
            return False
        try:
            opened, current = os.fstat(self._fd), path.stat()
            return (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino)
        except OSError:
            return False

    def _ensure_open(self) -> bool:
        """Open or recreate the JSONL file. Caller holds ``_lock``."""
        path = Path(self.path)
        if self._fd_tracks_path(path):
            return True
        if self._fd is not None:
            with suppress(OSError):
                os.close(self._fd)
            self._fd = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Persistent writer; released by close() or when the inode changes.
            self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
            # Driver metadata must be present even if a worker recreated
            # the file first. Keep fwd IDs monotonic across inode changes.
            if self._write_meta:
                self._write_meta_line()
        except OSError:
            logger.exception("Failed to open JSONL trace %s", self.path)
            self.close()
            return False
        return True

    def _write_meta_line(self) -> None:
        rec = {
            "ts_ns": time.time_ns(),
            "mono_ns": time.monotonic_ns(),
            "event": "meta",
            "schema": SCHEMA_VERSION,
            "config": self._metadata,
        }
        self._write_line((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))


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
                batch_idx=_opt_int(emit.get("batch_idx", done.get("batch_idx"))),
                phase=str(emit.get("phase") or done.get("phase") or ""),
                req_ids=req_ids,
                num_tokens=_opt_int(emit.get("num_tokens", emit.get("tokens"))),
                emit_mono_ns=int(emit["mono_ns"]),
                done_mono_ns=int(done["mono_ns"]),
                enqueue_mono_ns=_opt_int((slot.get("enqueue") or {}).get("mono_ns")),
                dequeue_mono_ns=_opt_int((slot.get("dequeue") or {}).get("mono_ns")),
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


_worker_tracer: JsonlTracer | None = None
_worker_tracer_ready = False


def _sync_compute_device() -> None:
    """Wait for queued GPU/NPU kernels so stage end is compute, not dispatch."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    npu = getattr(torch, "npu", None)
    if (
        npu is not None
        and callable(getattr(npu, "is_available", None))
        and npu.is_available()
    ):
        npu.synchronize()


def _pp_rank() -> int:
    try:
        from vllm.distributed.parallel_state import get_pp_group

        return int(get_pp_group().rank_in_group)
    except Exception:
        return -1


def _ensure_worker_tracer(vllm_config: Any) -> JsonlTracer | None:
    global _worker_tracer, _worker_tracer_ready
    if not _worker_tracer_ready:
        _worker_tracer_ready = True
        cfg_path = ""
        if vllm_config is not None:
            cfg_path = (
                getattr(
                    getattr(vllm_config, "scheduler_config", None),
                    "tau_batch_trace",
                    "",
                )
                or ""
            )
        path = resolve_trace_path(cfg_path)
        if path:
            try:
                _worker_tracer = JsonlTracer(path, write_meta=False)
            except OSError:
                logger.exception("PP worker failed to open JSONL trace")
                _worker_tracer = None
    return _worker_tracer


def _record_worker_event(
    vllm_config: Any,
    event: str,
    *,
    fwd_id: int,
    start_ts_ns: int,
    req_ids: list[str],
    features: Mapping[str, Any] | None = None,
    sync_end: bool = True,
) -> None:
    tracer = _ensure_worker_tracer(vllm_config)
    if tracer is None:
        return
    if sync_end:
        _sync_compute_device()
    extra = dict(features) if features else {}
    extra.pop("req_ids", None)
    extra.pop("fwd_id", None)
    extra.pop("pp_rank", None)
    extra.pop("event", None)
    extra.pop("kind", None)
    tracer.record(
        event,
        fwd_id=fwd_id,
        pp_rank=_pp_rank(),
        start_ts_ns=start_ts_ns,
        end_ts_ns=time.time_ns(),
        req_ids=req_ids,
        **extra,
    )


def record_worker_stage(
    vllm_config: Any,
    *,
    fwd_id: int,
    start_ts_ns: int,
    req_ids: list[str],
    features: Mapping[str, Any] | None = None,
) -> None:
    """Append the host execute_model envelope without another device barrier.

    Device completion is measured by the compute hook where available.
    This envelope alone is not a device computation duration.
    """
    _record_worker_event(
        vllm_config,
        "stage",
        fwd_id=fwd_id,
        start_ts_ns=start_ts_ns,
        req_ids=req_ids,
        features=features,
        sync_end=False,
    )


def record_worker_phase(
    vllm_config: Any,
    *,
    kind: str,
    fwd_id: int,
    start_ts_ns: int,
    req_ids: list[str],
    features: Mapping[str, Any] | None = None,
    sync_end: bool = True,
) -> None:
    """Append recv / compute / send inside one stage.

    ``recv`` waits for the previous rank. ``compute`` is this rank's
    forward. ``send`` is enqueue-return; NCCL send does not wait for
    the transfer to finish (the next rank's recv does).
    """
    _record_worker_event(
        vllm_config,
        kind,
        fwd_id=fwd_id,
        start_ts_ns=start_ts_ns,
        req_ids=req_ids,
        features=features,
        sync_end=sync_end,
    )


@contextmanager
def trace_worker_phase(
    vllm_config: Any,
    scheduler_output: Any,
    kind: str,
    *,
    sync_end: bool = True,
) -> Iterator[None]:
    """Record successful calls only; no-op without ``tau_fwd_id``.

    Execution and synchronization exceptions propagate without writing a normal
    phase event, so incomplete work cannot masquerade as a latency sample.
    """
    fwd_id = getattr(scheduler_output, "tau_fwd_id", None)
    if fwd_id is None:
        yield
        return
    start_ts_ns = time.time_ns()
    yield
    record_worker_phase(
        vllm_config,
        kind=kind,
        fwd_id=int(fwd_id),
        start_ts_ns=start_ts_ns,
        req_ids=list(getattr(scheduler_output, "num_scheduled_tokens", {})),
        features=getattr(scheduler_output, "tau_task", None),
        sync_end=sync_end,
    )


@dataclass(frozen=True)
class StageCell:
    """One micro-batch on one PP stage, placed for a pipeline Gantt."""

    fwd_id: int
    pp_rank: int
    job: str
    phase: str
    start_ts_ns: int
    end_ts_ns: int
    kind: str = "stage"

    @property
    def duration_ms(self) -> float:
        return (self.end_ts_ns - self.start_ts_ns) / 1e6


def pipeline_cells(events: Iterable[Mapping[str, Any]]) -> list[StageCell]:
    """Use measured compute windows, falling back to host stage envelopes.

    Never manufacture a start timestamp from a different rank's end.
    """
    events = list(events)
    forwards = {span.fwd_id: span for span in pair_forwards(events)}
    by_fwd: dict[int, dict[int, Mapping[str, Any]]] = {}
    for ev in events:
        if ev.get("event") not in ("stage", "compute"):
            continue
        fwd_id = ev.get("fwd_id")
        rank = ev.get("pp_rank")
        if fwd_id is None or rank is None:
            continue
        ranks = by_fwd.setdefault(int(fwd_id), {})
        if ev.get("event") == "compute" or int(rank) not in ranks:
            ranks[int(rank)] = ev
    cells: list[StageCell] = []
    for fwd_id, ranks in by_fwd.items():
        span = forwards.get(fwd_id)
        job = span.job if span is not None else f"fwd{fwd_id}"
        phase = span.phase if span is not None else ""
        for rank in sorted(ranks):
            ev = ranks[rank]
            start = int(ev["start_ts_ns"])
            end = int(ev["end_ts_ns"])
            cells.append(
                StageCell(
                    fwd_id=fwd_id,
                    pp_rank=rank,
                    job=job,
                    phase=phase,
                    start_ts_ns=start,
                    end_ts_ns=end,
                    kind=str(ev["event"]),
                )
            )
    cells.sort(key=lambda c: (c.start_ts_ns, c.pp_rank, c.fwd_id))
    return cells
