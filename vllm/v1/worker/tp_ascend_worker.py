# SPDX-License-Identifier: Apache-2.0
"""Ascend eager worker for controlled TP heterogeneity experiments."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from vllm_ascend.worker.worker import NPUWorker

from vllm.distributed.async_trace import AsyncTraceWriter
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.distributed.pp_hetero import sync_torch_device
from vllm.distributed.tp_hetero import (
    TPCollectiveDelay,
    TPComputeDelay,
    TPHeteroConfig,
)
from vllm.distributed.tp_stream_delay import TPStreamDelay
from vllm.logger import init_logger

logger = init_logger(__name__)


class TPAscendWorker(NPUWorker):
    def init_device(self) -> None:
        super().init_device()
        if get_pp_group().world_size != 1:
            raise ValueError("TP heterogeneity currently requires PP=1")
        tp_group = get_tp_group()
        self._tp_hetero = TPHeteroConfig.from_env(tp_group.world_size)
        if not self.vllm_config.model_config.enforce_eager:
            raise ValueError("TP heterogeneity requires --enforce-eager")
        if getattr(self.vllm_config.compilation_config, "mode", 0) not in (None, 0):
            raise ValueError("TP heterogeneity requires compilation mode NONE (0)")
        communicator = tp_group.device_communicator
        if communicator is None:
            raise ValueError("TP heterogeneity requires a device communicator")
        self._tp_stream_delay = TPStreamDelay()
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
            path = Path(trace_dir)
            path.mkdir(parents=True, exist_ok=True)
            self._tp_trace = (
                path / f"tp_mock_rank{tp_group.rank_in_group}_pid{os.getpid()}.jsonl"
            ).open("x", buffering=1)
            self._tp_trace_writer = AsyncTraceWriter(self._write_trace)
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
        # Pending callbacks still own their statistics and may submit records.
        # Drain them before stopping the writer or closing its file.
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

    def _submit_trace(
        self, metadata: dict, compute, collective, started: float
    ) -> None:
        writer = self._tp_trace_writer

        def finish() -> None:
            writer.submit(
                {
                    **metadata,
                    "compute_base_ms": compute.input_compute_ms,
                    "compute_sleep_ms": compute.actual_delay_ms,
                    "compute_extra_requested_ms": compute.requested_delay_ms,
                    "collective_observed_with_extra_ms": collective.total_ms,
                    "collective_extra_requested_ms": collective.extra_total_ms,
                    "collective_extra_actual_ms": collective.actual_extra_total_ms,
                    "collective_cross_bytes": collective.cross_bytes_total,
                    "forward_wall_ms": (time.perf_counter() - started) * 1000,
                    "timing_source": "stream_callback",
                }
            )

        self._tp_stream_delay.enqueue(finish)

    def execute_model(self, scheduler_output):
        self._tp_stream_delay.check()
        if self._tp_trace_writer is not None:
            self._tp_trace_writer.check()
        if scheduler_output.total_num_scheduled_tokens <= 0:
            return super().execute_model(scheduler_output)
        model = self.model_runner.model
        if self._tp_compute_model is not model:
            if self._tp_compute_delay is not None:
                self._tp_compute_delay.uninstall()
            self._tp_compute_delay = TPComputeDelay(
                model,
                self._tp_hetero.scale(get_tp_group().rank_in_group),
                self._tp_stream_delay,
            )
            self._tp_compute_delay.install()
            self._tp_compute_model = model
        compute_delay = self._tp_compute_delay
        assert compute_delay is not None
        compute_delay.reset()
        collectives = self._tp_collectives
        collectives.compute_delay = compute_delay
        collectives.reset()
        compute_stats = compute_delay.stats
        collective_stats = collectives.stats
        before_counts = collectives.counts.copy()
        started = time.perf_counter()
        collectives.active = True
        try:
            output = super().execute_model(scheduler_output)
            self._tp_stream_delay.check()
            compute_delay.validate()
            if self._tp_trace_writer is not None and self._tp_trace_remaining > 0:
                self._submit_trace(
                    {
                        "pid": os.getpid(),
                        "tp_rank": get_tp_group().rank_in_group,
                        "scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
                        "compute_scale": self._tp_hetero.scale(
                            get_tp_group().rank_in_group
                        ),
                        "collective_counts": {
                            op: count - before_counts[op]
                            for op, count in collectives.counts.items()
                        },
                    },
                    compute_stats,
                    collective_stats,
                    started,
                )
                self._tp_trace_remaining -= 1
            if not self._tp_logged:
                logger.info("TP mock first forward collectives=%s", collectives.counts)
                self._tp_logged = True
            return output
        finally:
            collectives.active = False
            compute_delay.active = False
