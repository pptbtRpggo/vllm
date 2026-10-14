# SPDX-License-Identifier: Apache-2.0
"""Temporary experiment worker: isolate TP device-synchronization overhead.

This worker never injects compute or network delay. A shared mode file selects
which of the existing mock's synchronization points to enable per forward.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from vllm_ascend.worker.worker import NPUWorker

from vllm.distributed.parallel_state import get_tp_group
from vllm.distributed.pp_hetero import sync_torch_device
from vllm.distributed.pp_layer_trace import PPLayerTimer


class TPSyncProbeWorker(NPUWorker):
    def init_device(self) -> None:
        super().init_device()
        self._probe_mode_file = Path(os.environ["VLLM_TP_SYNC_PROBE_MODE_FILE"])
        self._probe_trace_dir = Path(os.environ["VLLM_TP_SYNC_PROBE_TRACE_DIR"])
        self._probe_trace_dir.mkdir(parents=True, exist_ok=True)
        rank = get_tp_group().rank_in_group
        self._probe_trace = (
            self._probe_trace_dir / f"rank{rank}_pid{os.getpid()}.jsonl"
        ).open("x", buffering=1)
        self._probe_modules = None
        self._probe_model = None
        self._probe_active = False
        self._probe_mode = "none"
        self._probe_sync = {}
        self._probe_events = {}
        self._probe_collectives = 0
        communicator = get_tp_group().device_communicator
        self._probe_originals = {}
        for op in ("all_reduce", "all_gather", "reduce_scatter"):
            original = getattr(communicator, op)
            self._probe_originals[op] = original

            def wrapped(*args, _original=original, **kwargs):
                if self._probe_active:
                    self._probe_collectives += 1
                    if self._probe_mode in (
                        "collective",
                        "both",
                        "event_collective",
                        "event_both",
                    ):
                        self._probe_do_sync("collective_before")
                result = _original(*args, **kwargs)
                if self._probe_active and self._probe_mode in (
                    "collective",
                    "both",
                    "event_collective",
                    "event_both",
                ):
                    self._probe_do_sync("collective_after")
                return result

            setattr(communicator, op, wrapped)
        self._probe_communicator = communicator

    def _probe_do_sync(self, name: str) -> None:
        start = time.perf_counter()
        if self._probe_mode.startswith("event_"):
            event = self._probe_events.get(name)
            if event is None:
                event = torch.npu.Event(enable_timing=False)
                self._probe_events[name] = event
            event.record()
            event.synchronize()
        else:
            sync_torch_device(self.device)
        elapsed = (time.perf_counter() - start) * 1000
        slot = self._probe_sync.setdefault(name, {"count": 0, "wall_ms": 0.0})
        slot["count"] += 1
        slot["wall_ms"] += elapsed

    def execute_model(self, scheduler_output):
        if scheduler_output.total_num_scheduled_tokens <= 0:
            return super().execute_model(scheduler_output)
        mode = self._probe_mode_file.read_text().strip()
        if mode not in (
            "none",
            "layer",
            "collective",
            "both",
            "event_layer",
            "event_collective",
            "event_both",
        ):
            raise ValueError(f"Invalid TP sync probe mode: {mode}")
        model = self.model_runner.model
        if model is not self._probe_model:
            timer = PPLayerTimer(model, self.device)
            self._probe_modules = tuple(
                (*timer.layers.values(), *timer.endpoints.values())
            )
            self._probe_model = model
        self._probe_mode = mode
        self._probe_sync = {}
        self._probe_collectives = 0
        handles = []

        def before(*_args):
            if mode in ("layer", "both", "event_layer", "event_both"):
                self._probe_do_sync("layer_before")

        def after(*_args):
            if mode in ("layer", "both", "event_layer", "event_both"):
                self._probe_do_sync("layer_after")

        started = time.perf_counter()
        try:
            for module in self._probe_modules:
                handles.append(module.register_forward_pre_hook(before))
                handles.append(module.register_forward_hook(after))
            self._probe_active = True
            result = super().execute_model(scheduler_output)
            if mode in ("both", "event_both"):
                self._probe_do_sync("forward_end")
            return result
        finally:
            self._probe_active = False
            for handle in handles:
                handle.remove()
            self._probe_trace.write(
                json.dumps(
                    {
                        "mode": mode,
                        "rank": get_tp_group().rank_in_group,
                        "scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
                        "forward_wall_ms": (time.perf_counter() - started) * 1000,
                        "collectives": self._probe_collectives,
                        "sync": self._probe_sync,
                    }
                )
                + "\n"
            )

    def shutdown(self) -> None:
        if trace := getattr(self, "_probe_trace", None):
            trace.close()
            self._probe_trace = None
        if communicator := getattr(self, "_probe_communicator", None):
            for op, original in self._probe_originals.items():
                setattr(communicator, op, original)
        parent = getattr(super(), "shutdown", None)
        if callable(parent):
            parent()
