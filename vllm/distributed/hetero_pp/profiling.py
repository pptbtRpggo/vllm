# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in direct layer events and completed CPU/Gloo handoff observations.

No delay kernel, mock slowdown, model forward replacement, or per-layer device
synchronization is used. Execute and sample are separate device intervals;
their sum deliberately excludes the host gap between the two RPCs. Host send
and receive windows include metadata, allocation, D2H/H2D and peer waiting.
They are effective handoff windows, never a measurement of physical wire rate.

Profiling alone waits for the local execute completion event before starting
the send window. Otherwise the send's blocking D2H copy could also include the
forward tail already measured by compute events. This readiness checkpoint can
remove native metadata/forward overlap; trace-off redeployment is required.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import torch


def monotonic_clock_domain():
    """Only equate clocks with the same boot, namespace, offsets and source."""
    try:
        parts = {
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "time_namespace": os.readlink("/proc/self/ns/time"),
            "offsets": Path("/proc/self/timens_offsets").read_text().strip(),
            "implementation": time.get_clock_info("perf_counter").implementation,
        }
    except OSError:
        return None
    return json.dumps(parts, sort_keys=True, separators=(",", ":"))


def tensor_bytes(values):
    return sum(
        t.numel() * t.element_size()
        for t in values.values()
        if isinstance(t, torch.Tensor)
    )


class _Capture:
    """One RPC owns fresh events until its background record is resolved."""

    def __init__(self, profiler, kind):
        self.profiler, self.kind = profiler, kind
        self.pairs = {}
        self.order = []
        self.handles = []

    def mark(self):
        backend = self.profiler.backend
        if backend is None:
            return time.perf_counter_ns()
        event = backend.Event(enable_timing=True)
        event.record(backend.current_stream(self.profiler.device))
        return event

    def before(self, name, _module, _args):
        if name in self.pairs:
            raise ValueError("layer profiling does not support reentrant modules")
        self.pairs[name] = [self.mark(), None]

    def after(self, name, _module, _args, _output):
        self.pairs[name][1] = self.mark()
        if isinstance(name, int):
            self.order.append(name)

    def run(self, fn):
        p = self.profiler
        try:
            for name, module in {**p.layers, **p.endpoints}.items():
                self.handles.append(
                    module.register_forward_pre_hook(partial(self.before, name))
                )
                self.handles.append(
                    module.register_forward_hook(partial(self.after, name))
                )
            self.host_start_ns = time.perf_counter_ns()
            self.start = self.mark()
            result = fn()
            self.end = self.mark()
            self.host_end_ns = time.perf_counter_ns()
            wanted = list(p.layers) if self.kind == "execute" else []
            if self.order != wanted:
                raise ValueError(
                    "profiling must observe every local decoder layer in order "
                    "during execute only"
                )
            return result
        finally:
            for handle in self.handles:
                handle.remove()

    def ready(self):
        return self.profiler.backend is None or self.end.query()

    def elapsed(self, pair):
        a, b = pair
        if b is None:
            raise ValueError("module completion event is missing")
        return (
            (b - a) / 1e6 if self.profiler.backend is None else float(a.elapsed_time(b))
        )


@dataclass
class _Step:
    metadata: dict[str, Any]
    execute: _Capture | None = None
    sample: _Capture | None = None
    recv_start_ns: int | None = None
    recv_end_ns: int | None = None
    send_start_ns: int | None = None
    send_end_ns: int | None = None
    send_submit_start_ns: int | None = None
    send_ready_wait_start_ns: int | None = None
    send_ready_wait_end_ns: int | None = None
    send_ready_wait_ms: float = 0.0
    send_readiness_checkpoint: bool = False
    recv_bytes: int | None = None
    send_bytes: int | None = None
    materialization: dict[str, dict[str, int]] = field(default_factory=dict)


class _TraceWork:
    def __init__(self, original, complete):
        self.original, self.complete = original, complete
        self.done = False

    def wait(self, *args, **kwargs):
        result = self.original.wait(*args, **kwargs)
        if not self.done:
            self.done = True
            self.complete()
        return result


class _TraceLink:
    def __init__(self, step, original):
        self.step, self.original = step, original

    def recv(self, sequence):
        self.step.recv_start_ns = time.perf_counter_ns()
        values = self.original.recv(sequence)
        self.step.recv_end_ns = time.perf_counter_ns()
        self.step.recv_bytes = tensor_bytes(values)
        return values

    def send(self, values, sequence):
        self.step.send_submit_start_ns = time.perf_counter_ns()
        self.step.send_bytes = tensor_bytes(values)
        capture = self.step.execute
        if capture is not None and capture.profiler.backend is not None:
            self.step.send_readiness_checkpoint = True
            self.step.send_ready_wait_start_ns = time.perf_counter_ns()
            # One local stream completion event, not a device-wide or
            # per-layer synchronization. Its forward work is counted in the
            # compute interval; this instrumentation wait is not added again.
            capture.end.synchronize()
            self.step.send_ready_wait_end_ns = time.perf_counter_ns()
            self.step.send_ready_wait_ms = (
                self.step.send_ready_wait_end_ns - self.step.send_ready_wait_start_ns
            ) / 1e6
        self.step.send_start_ns = time.perf_counter_ns()
        works = self.original.send(values, sequence)
        remaining = len(works)

        def complete():
            nonlocal remaining
            remaining -= 1
            if remaining == 0:
                self.step.send_end_ns = time.perf_counter_ns()

        if not works:
            self.step.send_end_ns = time.perf_counter_ns()
        return [_TraceWork(work, complete) for work in works]


class StageProfiler:
    """Finite background writer for real per-module timing observations."""

    def __init__(
        self,
        dump_dir,
        session,
        model,
        device,
        rank,
        pp_size,
        identity=None,
        timeout_s=60,
    ):
        inner = getattr(model, "model", model)
        start, end = (
            getattr(inner, "start_layer", None),
            getattr(inner, "end_layer", None),
        )
        layers = getattr(inner, "layers", None)
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or end <= start
            or not isinstance(layers, torch.nn.ModuleList)
            or len(layers) < end
        ):
            raise ValueError("profiling requires global-indexed local decoder layers")
        self.layers = {i: layers[i] for i in range(start, end)}
        if len({id(m) for m in self.layers.values()}) != len(self.layers):
            raise ValueError("profiling does not support shared decoder modules")
        candidates = {
            "embedding": getattr(inner, "embed_tokens", None),
            "final_norm": getattr(inner, "norm", None),
            "lm_head": getattr(model, "logits_processor", None),
        }
        self.endpoints = {
            k: m
            for k, m in candidates.items()
            if isinstance(m, torch.nn.Module) and type(m).__name__ != "PPMissingLayer"
        }
        self.device = torch.device(device)
        self.backend = (
            None
            if self.device.type == "cpu"
            else torch.get_device_module(self.device.type)
        )
        self.timeout_s = min(float(timeout_s), 60.0)
        self.rank, self.pp_size = rank, pp_size
        self.session = session
        self.clock_domain = monotonic_clock_domain()
        self.identity = json.loads(json.dumps(identity or {}))
        self.provenance = {
            "schema_version": 2,
            "transport": "cpu_gloo",
            "trace_session": session,
            "clock_domain": self.clock_domain,
            "pp_rank": rank,
            "pp_size": pp_size,
            "loaded_layers": [start, end],
            "identity": self.identity,
            "device_id": self.identity.get("profile_device_id", rank),
            "local_device_id": self.identity.get("device_id"),
            "timing_source": "device_events_and_host_perf_counter",
            "compute_interval_kind": "sum_of_execute_and_sample_device_intervals",
            "communication_scope": "completed_cpu_staged_handoff_including_peer_wait",
            "send_readiness_instrumentation": (
                "local_execute_completion_event_before_metadata_and_D2H"
            ),
            "readiness_wait_accounting": (
                "excluded_from_communication_and_not_added_to_planner_cost"
            ),
        }
        self.provenance_sha = hashlib.sha256(
            json.dumps(self.provenance, sort_keys=True).encode()
        ).hexdigest()
        directory = Path(dump_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"pp_stage_pp{rank}_tp0.jsonl"
        self.manifest_path = directory / f"pp_manifest_pp{rank}_tp0.json"
        self._fp = self.path.open("x", encoding="utf-8")
        self.is_warmup = False
        self.pending = None
        self.zero_steps = self.unassociated_sample_calls = 0
        self.records = self.warmup_records = self.sample_records = 0
        self._error = None
        self._closed = False
        self._queue: queue.Queue[tuple[_Step, float] | threading.Event | None] = (
            queue.Queue()
        )
        self._thread = threading.Thread(
            target=self._write_loop, name="staged-profile-writer", daemon=True
        )
        self._thread.start()

    def begin(self, scheduler, runner, sequence):
        from vllm.distributed.pp_batch_shape import (
            scheduled_batch_id,
            scheduled_batch_shape,
        )

        self._check()
        self._finalize_pending()
        if not scheduler.total_num_scheduled_tokens:
            self.zero_steps += 1
            return None
        shape = scheduled_batch_shape(scheduler, runner)
        if shape["batch_shape"] is None:
            raise ValueError(
                "profiling batch shape is unknown; no decode assumption is allowed"
            )
        metadata = {
            "step": sequence,
            "sequence": sequence,
            "ts_unix": time.time(),
            "pp_rank": self.rank,
            "pp_size": self.pp_size,
            "tp_rank": 0,
            "tp_size": 1,
            "start_layer": min(self.layers),
            "end_layer": max(self.layers) + 1,
            "num_tokens": scheduler.total_num_scheduled_tokens,
            "num_reqs": len(scheduler.num_scheduled_tokens),
            **shape,
            "batch_id": scheduled_batch_id(scheduler),
            "is_warmup": bool(self.is_warmup),
            "trace_session": self.session,
            "trace_id": self.provenance_sha,
            "clock_domain": self.clock_domain,
            "profile_device_id": self.identity.get("profile_device_id", self.rank),
            # The legacy planner defines device_id as a global profiling ID.
            # Ray's node-local NPU indices may repeat on another node.
            "device_id": self.identity.get("profile_device_id", self.rank),
            "local_device_id": self.identity.get("device_id"),
            "compute_scale": 1.0,
            "comm_scale": 1.0,
            "compute_model": "layer-measured",
            "compute_delay_placement": "none",
            "compute_delay_ms": 0.0,
            "transport": "cpu_gloo",
            "timing_source": "device_events_and_host_perf_counter",
            "compute_interval_kind": "sum_of_execute_and_sample_device_intervals",
            "profile_provenance_sha256": self.provenance_sha,
            "comm_delay_in_window": False,
        }
        self.pending = _Step(metadata)
        return self.pending

    def timed_link(self, step, original):
        return original if step is None else _TraceLink(step, original)

    def execute(self, step, fn):
        if step is None:
            return fn()
        capture = _Capture(self, "execute")
        try:
            result = capture.run(fn)
        except BaseException:
            self.pending = None
            self._error = RuntimeError("profile execute capture failed")
            self._failure(traceback.format_exc())
            raise
        step.execute = capture
        return result

    def sample(self, fn):
        self._check()
        if self.pending is None:
            self.unassociated_sample_calls += 1
            return fn()
        capture = _Capture(self, "sample")
        try:
            result = capture.run(fn)
        except BaseException:
            self.pending = None
            self._error = RuntimeError("profile sample capture failed")
            self._failure(traceback.format_exc())
            raise
        self.pending.sample = capture
        self._finalize_pending()
        return result

    def materialize(self, kind, output):
        if not hasattr(output, "get_output"):
            return output
        begin = time.perf_counter_ns()
        result = output.get_output()
        if self.pending is not None:
            self.pending.materialization[kind] = {
                "start_ns": begin,
                "end_ns": time.perf_counter_ns(),
            }
        return result

    def _finalize_pending(self):
        if self.pending is not None:
            if self.pending.execute is None:
                raise RuntimeError("profile step has no completed execute interval")
            self._queue.put((self.pending, time.monotonic() + self.timeout_s))
            self.pending = None

    def _resolve(self, step):
        row = dict(step.metadata)
        captures = [c for c in (step.execute, step.sample) if c is not None]
        durations = [c.elapsed((c.start, c.end)) for c in captures]
        modules = {}
        for capture in captures:
            for key, pair in capture.pairs.items():
                if key in modules:
                    raise ValueError("a model endpoint/layer executed in both RPCs")
                modules[key] = capture.elapsed(pair)
        wall = sum(durations)
        layers = {str(k): v for k, v in modules.items() if isinstance(k, int)}
        endpoints = {
            k + "_ms": modules.get(k) for k in ("embedding", "lm_head", "final_norm")
        }
        residual = wall - sum(layers.values())
        overhead = residual - sum(v or 0 for v in endpoints.values())
        if min(wall, *layers.values(), residual, overhead) < -0.01:
            raise ValueError("module events exceed the measured compute intervals")
        # Preserve conservation; tolerate only device event quantization noise.
        residual = max(0.0, residual)
        overhead = max(0.0, overhead)
        row.update(
            layer_compute_ms=layers,
            **endpoints,
            compute_ms=wall,
            compute_base_ms=wall,
            compute_wall_ms=wall,
            non_layer_compute_ms=residual,
            runner_overhead_ms=overhead,
            execute_device_ms=durations[0],
            sample_device_ms=durations[1] if step.sample is not None else 0.0,
            sample_called=step.sample is not None,
            output_materialization_windows=step.materialization,
        )
        row.update(
            send_submit_start_ns=step.send_submit_start_ns,
            send_ready_wait_start_ns=step.send_ready_wait_start_ns,
            send_ready_wait_end_ns=step.send_ready_wait_end_ns,
            send_ready_wait_ms=step.send_ready_wait_ms,
            send_readiness_checkpoint=step.send_readiness_checkpoint,
            readiness_wait_accounting="excluded_from_communication_and_not_added_to_planner_cost",
        )
        if step.send_submit_start_ns is not None and step.send_start_ns is None:
            raise ValueError("send readiness checkpoint did not complete")
        for kind, capture in (("execute", step.execute), ("sample", step.sample)):
            row[kind + "_host_start_ns"] = capture.host_start_ns if capture else None
            row[kind + "_host_end_ns"] = capture.host_end_ns if capture else None
            row[kind + "_host_ms"] = (
                (capture.host_end_ns - capture.host_start_ns) / 1e6 if capture else 0.0
            )
        for kind in ("recv", "send"):
            begin, end = (
                getattr(step, kind + "_start_ns"),
                getattr(step, kind + "_end_ns"),
            )
            if begin is not None and end is None:
                raise ValueError("communication interval has unfinished work")
            row[kind + "_start_ns"], row[kind + "_end_ns"] = begin, end
            row[kind + "_ms"] = (end - begin) / 1e6 if begin is not None else None
            row[kind + "_bytes"] = getattr(step, kind + "_bytes")
        return row

    def _write_row(self, row):
        self._fp.write(json.dumps(row, sort_keys=True) + "\n")
        self._fp.flush()
        self.records += 1
        self.warmup_records += int(row["is_warmup"])
        self.sample_records += int(row["sample_called"])

    def _failure(self, error):
        failure = self.manifest_path.with_name(
            f"pp_profile_failure_pp{self.rank}_tp0.json"
        )
        failure.write_text(json.dumps({**self.provenance, "error": error}, indent=2))

    def _write_loop(self):
        if self.backend is not None:
            try:
                self.backend.set_device(self.device)
            except Exception as error:
                self._error = error
                self._failure(traceback.format_exc())
        while True:
            item = self._queue.get()
            if item is None:
                return
            if isinstance(item, threading.Event):
                item.set()
                continue
            if self._error is not None:
                continue
            step, deadline = item
            try:
                while not all(
                    c.ready() for c in (step.execute, step.sample) if c is not None
                ):
                    if time.monotonic() > deadline:
                        raise TimeoutError("profile completion event did not finish")
                    time.sleep(0.001)
                self._write_row(self._resolve(step))
            except Exception as error:
                self._error = error
                self._failure(traceback.format_exc())

    def _check(self):
        if self._error is not None:
            raise RuntimeError("profile writer or capture failed") from self._error

    def flush(self):
        if self._closed:
            raise RuntimeError("profile writer is closed")
        self._check()
        self._finalize_pending()
        barrier = threading.Event()
        self._queue.put(barrier)
        if not barrier.wait(self.timeout_s):
            raise TimeoutError("profile writer flush timed out")
        self._check()
        receipt = {
            **self.provenance,
            "enabled": True,
            "record_count": self.records,
            "complete": True,
            "rank": self.rank,
            "session": self.session,
            "rows": self.records,
            "trace_path": str(self.path),
            "warmup_records": self.warmup_records,
            "measured_records": self.records - self.warmup_records,
            "sample_records": self.sample_records,
            "zero_steps": self.zero_steps,
            "unassociated_sample_calls": self.unassociated_sample_calls,
            "pending": 0,
            "errors": [],
            "trace_file": str(self.path),
            "trace_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "profile_provenance_sha256": self.provenance_sha,
        }
        self.manifest_path.write_text(json.dumps(receipt, indent=2))
        return receipt

    def close(self, raise_errors=True):
        if self._closed:
            return
        try:
            self.flush()
        except Exception:
            if raise_errors:
                raise
        finally:
            self._closed = True
            self._queue.put(None)
            self._thread.join(timeout=self.timeout_s)
            if self._thread.is_alive():
                raise TimeoutError("profile writer did not exit")
            self._fp.close()


def maybe_create_profiler(worker):
    dump_dir = os.environ.get("VLLM_PP_STAGE_TRACE")
    if not dump_dir:
        return None
    session = os.environ.get("VLLM_PP_TRACE_SESSION")
    if not session or os.environ.get("VLLM_PP_COMPUTE_MODEL") != "layer-measured":
        raise ValueError("staged profiling requires a session and layer-measured mode")
    cfg = worker.vllm_config
    if (
        not cfg.model_config.enforce_eager
        or cfg.parallel_config.tensor_parallel_size != 1
    ):
        raise ValueError("staged layer profiling requires eager TP=1")
    return StageProfiler(
        dump_dir,
        session,
        worker.model_runner.model,
        worker.device,
        worker.rank,
        worker.parallel_config.world_size,
        identity=getattr(worker, "_hetero_profile_identity", {}),
        timeout_s=worker._hetero_timeout_s,
    )


def describe_kv_cache(worker):
    """Describe actual cache tensor owners and native specs; never infer usage.

    Allocator peaks include the pool. A caller must not add this pool to an
    allocator peak again. Unknown tensor containers/specs remain explicit.
    """
    runner = worker.model_runner
    target = torch.device(worker.device).type
    tensors: list[dict[str, Any]] = []
    owners: dict[tuple[str, int], dict[str, Any]] = {}
    errors: list[dict[str, str]] = []

    def visit(value, path):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr())
            if key not in owners:
                owners[key] = {
                    "owner_id": len(owners),
                    "device": value.device.type,
                    "storage_nbytes": storage.nbytes(),
                }
            tensors.append(
                {
                    "path": path,
                    "owner_id": owners[key]["owner_id"],
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "logical_nbytes": value.numel() * value.element_size(),
                }
            )
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, path + "/" + str(key))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, path + "/" + str(index))
        elif value is not None:
            errors.append({"path": path, "unknown_type": type(value).__name__})

    for name in ("kv_caches", "kv_cache_tensors", "kv_cache_by_layer"):
        if hasattr(runner, name):
            visit(getattr(runner, name), name)
    matching = [o for o in owners.values() if o["device"] == target]
    known = bool(matching) and not errors
    cache = getattr(worker, "cache_config", None)
    native_config = getattr(runner, "kv_cache_config", None)
    result = {
        "kv_cache_storage_bytes": sum(o["storage_nbytes"] for o in matching)
        if known
        else None,
        "kv_cache_storage_owners": len(matching),
        "kv_cache_storage_owner_descriptions": list(owners.values()),
        "kv_cache_tensors": tensors,
        "kv_cache_storage_observation": "complete" if known else "unknown",
        "kv_cache_observation_errors": errors,
        "num_gpu_blocks": getattr(native_config, "num_blocks", None)
        or getattr(cache, "num_gpu_blocks", None),
        "cache_config_num_gpu_blocks": getattr(cache, "num_gpu_blocks", None),
        "runner_kv_cache_num_blocks": getattr(native_config, "num_blocks", None),
        "includes_kv_pool": True,
    }
    specs: dict[str, dict[str, Any]] = {}
    try:
        for name, spec in worker.get_kv_cache_spec().items():
            row: dict[str, Any] = {"type": type(spec).__name__}
            for key in (
                "page_size_bytes",
                "block_size",
                "storage_block_size",
                "num_kv_heads",
                "head_size",
                "dtype",
                "page_size_padded",
            ):
                value = getattr(spec, key, None)
                row[key] = (
                    value
                    if value is None or type(value) in (bool, int, float, str)
                    else str(value)
                )
            specs[name] = row
        result["kv_cache_specs"] = specs
        result["kv_cache_spec_observation"] = "complete" if specs else "unknown"
    except Exception as error:
        result["kv_cache_specs"] = {}
        result["kv_cache_spec_observation"] = "unknown"
        result["kv_cache_spec_error"] = repr(error)
    native_tensors = getattr(native_config, "kv_cache_tensors", None)
    result["native_kv_cache_tensor_config"] = (
        [
            {
                "size": getattr(t, "size", None),
                "shared_by": getattr(t, "shared_by", None),
            }
            for t in native_tensors
        ]
        if native_tensors is not None
        else None
    )
    return result


def serving_config_observation(worker):
    """Raw target configuration, not a claim of untested partition safety."""
    cfg = worker.vllm_config
    groups = {
        "model": (
            cfg.model_config,
            (
                "model",
                "revision",
                "dtype",
                "quantization",
                "max_model_len",
                "enforce_eager",
            ),
        ),
        "cache": (
            cfg.cache_config,
            (
                "cache_dtype",
                "block_size",
                "gpu_memory_utilization",
                "num_gpu_blocks",
                "num_gpu_blocks_override",
                "kv_cache_memory_bytes",
            ),
        ),
        "scheduler": (
            cfg.scheduler_config,
            ("max_num_seqs", "max_num_batched_tokens", "async_scheduling"),
        ),
        "parallel": (
            cfg.parallel_config,
            ("pipeline_parallel_size", "tensor_parallel_size", "data_parallel_size"),
        ),
    }
    result: dict[str, dict[str, Any]] = {}
    for group, (obj, names) in groups.items():
        result[group] = {}
        for name in names:
            value = getattr(obj, name, None)
            result[group][name] = (
                value
                if value is None or type(value) in (bool, int, float, str)
                else str(value)
            )
    return result
