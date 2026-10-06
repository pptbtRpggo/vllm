# SPDX-License-Identifier: Apache-2.0
"""Ascend eager/Graph worker with device-side TP heterogeneity simulation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from vllm_ascend.worker.worker import NPUWorker

from vllm.distributed.ascend_device_delay import AscendDeviceDelay
from vllm.distributed.async_trace import AsyncTraceWriter
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.distributed.pp_hetero import sync_torch_device
from vllm.distributed.tp_hetero import (
    TPCollectiveDelay,
    TPComputeDelay,
    TPHeteroConfig,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class TPAscendWorker(NPUWorker):
    def init_device(self) -> None:
        super().init_device()
        if get_pp_group().world_size != 1:
            raise ValueError("TP heterogeneity currently requires PP=1")
        tp_group = get_tp_group()
        self._tp_hetero = TPHeteroConfig.from_env(tp_group.world_size)
        if getattr(self.vllm_config.compilation_config, "mode", 0) not in (None, 0):
            raise ValueError(
                "TP device simulation supports eager or FULL Graph "
                "with compilation mode 0"
            )
        communicator = tp_group.device_communicator
        if communicator is None:
            raise ValueError("TP heterogeneity requires a device communicator")
        self._tp_stream_delay = AscendDeviceDelay()
        self._tp_collectives = TPCollectiveDelay(
            communicator,
            self._tp_hetero,
            tp_group.world_size,
            self._tp_stream_delay,
        )
        self._tp_collectives.install()
        self._tp_compute_delay: TPComputeDelay | None = None
        self._tp_compute_model: torch.nn.Module | None = None
        self._tp_logged = False
        trace_dir = os.getenv("VLLM_TP_MOCK_TRACE")
        self._tp_trace = None
        self._tp_trace_writer = None
        self._tp_trace_remaining = 1000
        if trace_dir:
            if not self.vllm_config.model_config.enforce_eager:
                raise ValueError(
                    "TP detailed tracing currently requires eager; "
                    "Graph serving may run without trace"
                )
            path = Path(trace_dir)
            path.mkdir(parents=True, exist_ok=True)
            self._tp_trace = (
                path / f"tp_mock_rank{tp_group.rank_in_group}_pid{os.getpid()}.jsonl"
            ).open("x", buffering=1)
            self._tp_trace_writer = AsyncTraceWriter(
                self._write_trace, lambda: torch.npu.set_device(self.device)
            )
        logger.info(
            "TP mock rank=%d compute_scale=%g cross_group_size=%d "
            "cross_extra_bandwidth_gbps=%s cross_extra_latency_ms=%g",
            tp_group.rank_in_group,
            self._tp_hetero.scale(tp_group.rank_in_group),
            self._tp_hetero.cross_group_size,
            self._tp_hetero.cross_extra_bandwidth_gbps,
            self._tp_hetero.cross_extra_latency_ms,
        )

    def shutdown(self) -> None:
        # Complete device work before draining snapshots and closing the file.
        try:
            if stream_delay := getattr(self, "_tp_stream_delay", None):
                sync_torch_device(self.device)
                stream_delay.close()
        finally:
            try:
                if writer := getattr(self, "_tp_trace_writer", None):
                    writer.close()
                    self._tp_trace_writer = None
            finally:
                if trace := getattr(self, "_tp_trace", None):
                    trace.close()
                    self._tp_trace = None
                if collectives := getattr(self, "_tp_collectives", None):
                    collectives.uninstall()
                if compute_delay := getattr(self, "_tp_compute_delay", None):
                    compute_delay.uninstall()
                parent = getattr(super(), "shutdown", None)
                if callable(parent):
                    parent()

    def _write_trace(self, record: dict) -> None:
        self._tp_trace.write(json.dumps(record) + "\n")

    def load_model(self):
        super().load_model()
        self._install_compute()

    def _install_compute(self):
        model = self.model_runner.model
        if self._tp_compute_model is not model:
            if self._tp_compute_delay is not None:
                self._tp_compute_delay.uninstall()
            self._tp_compute_delay = TPComputeDelay(
                model,
                self._tp_hetero.scale(get_tp_group().rank_in_group),
                self._tp_stream_delay,
                measure=self._tp_trace_writer is not None,
            )
            self._tp_compute_delay.install()
            self._tp_compute_model = model
            self._tp_collectives.compute_delay = self._tp_compute_delay

    def execute_model(self, scheduler_output):
        if self._tp_trace_writer is not None:
            self._tp_trace_writer.check()
        if scheduler_output.total_num_scheduled_tokens <= 0:
            return super().execute_model(scheduler_output)
        self._install_compute()
        compute = self._tp_compute_delay
        compute.reset()
        collective = self._tp_collectives
        collective.reset()
        tracing = self._tp_trace_writer is not None and self._tp_trace_remaining > 0
        collective.trace_enabled = tracing
        counts = collective.counts.copy()
        started = self._tp_stream_delay.event() if tracing else None
        output = super().execute_model(scheduler_output)
        if self.vllm_config.model_config.enforce_eager:
            compute.validate()
        if tracing:
            finished = self._tp_stream_delay.event()
            snapshots = [
                self._tp_stream_delay.snapshot(buf) for buf in compute.stats.intervals
            ]
            completed = self._tp_stream_delay.event()
            events = collective.stats.events.copy()
            metadata = {
                **getattr(self, "_tp_trace_metadata", {}),
                "pid": os.getpid(),
                "tp_rank": get_tp_group().rank_in_group,
                "scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
                "compute_scale": compute.scale,
                "collective_counts": {
                    op: n - counts[op] for op, n in collective.counts.items()
                },
                "collective_extra_requested_ms": collective.stats.extra_total_ms,
                "collective_cross_bytes": collective.stats.cross_bytes_total,
            }

            def resolve():
                values = [self._tp_stream_delay.durations(host) for host in snapshots]
                return {
                    **metadata,
                    "compute_base_ms": sum(v[0] for v in values),
                    "compute_sleep_ms": sum(v[1] for v in values),
                    "compute_extra_requested_ms": sum(v[0] for v in values)
                    * (metadata["compute_scale"] - 1),
                    "collective_observed_with_extra_ms": sum(
                        a.elapsed_time(b) for a, b in events
                    ),
                    "forward_wall_ms": started.elapsed_time(finished),
                    "timing_source": "npu_event_device_clock",
                }

            self._tp_trace_writer.submit_ready(completed, resolve)
            self._tp_trace_remaining -= 1
        return output
