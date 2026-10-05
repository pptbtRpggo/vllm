# SPDX-License-Identifier: Apache-2.0
"""Ascend eager worker for controlled TP heterogeneity experiments."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from vllm_ascend.worker.worker import NPUWorker

from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.distributed.pp_hetero import sync_torch_device
from vllm.distributed.pp_layer_trace import PPLayerTimer
from vllm.distributed.tp_hetero import TPCollectiveDelay, TPHeteroConfig
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
        self._tp_collectives = TPCollectiveDelay(
            communicator, self._tp_hetero, tp_group.world_size, self.device
        )
        self._tp_collectives.install()
        self._tp_layer_timer: PPLayerTimer | None = None
        self._tp_layer_model: torch.nn.Module | None = None
        self._tp_logged = False
        trace_dir = os.getenv("VLLM_TP_MOCK_TRACE")
        self._tp_trace = None
        self._tp_trace_remaining = 1000
        if trace_dir:
            path = Path(trace_dir)
            path.mkdir(parents=True, exist_ok=True)
            self._tp_trace = (
                path / f"tp_mock_rank{tp_group.rank_in_group}_pid{os.getpid()}.jsonl"
            ).open(
                "x", buffering=1
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
        if trace := getattr(self, "_tp_trace", None):
            trace.close()
            self._tp_trace = None
        if collectives := getattr(self, "_tp_collectives", None):
            collectives.uninstall()
        parent = getattr(super(), "shutdown", None)
        if callable(parent):
            parent()

    def execute_model(self, scheduler_output):
        if scheduler_output.total_num_scheduled_tokens <= 0:
            return super().execute_model(scheduler_output)
        model = self.model_runner.model
        if self._tp_layer_model is not model:
            self._tp_layer_timer = PPLayerTimer(model, self.device)
            self._tp_layer_model = model
        assert self._tp_layer_timer is not None
        collectives = self._tp_collectives
        collectives.total_ms = 0.0
        collectives.extra_total_ms = 0.0
        collectives.cross_bytes_total = 0
        before_counts = collectives.counts.copy()
        started = time.perf_counter()
        collectives.active = True
        try:
            with self._tp_layer_timer.capture(
                self._tp_hetero.scale(get_tp_group().rank_in_group),
                excluded_ms=lambda: collectives.total_ms,
            ):
                output = super().execute_model(scheduler_output)
                sync_torch_device(self.device)
            if self._tp_trace is not None:
                try:
                    self._tp_trace.write(json.dumps({
                        "pid": os.getpid(),
                        "tp_rank": get_tp_group().rank_in_group,
                        "scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
                        "compute_scale": self._tp_hetero.scale(get_tp_group().rank_in_group),
                        "compute_base_ms": self._tp_layer_timer.mock_input_compute_ms,
                        "compute_sleep_ms": self._tp_layer_timer.mock_delay_ms,
                        "collective_total_ms": collectives.total_ms,
                        "collective_extra_requested_ms": collectives.extra_total_ms,
                        "collective_cross_bytes": collectives.cross_bytes_total,
                        "collective_counts": {
                            op: count - before_counts[op]
                            for op, count in collectives.counts.items()
                        },
                        "forward_wall_ms": (time.perf_counter() - started) * 1000,
                    }) + "\n")
                    self._tp_trace_remaining -= 1
                    if self._tp_trace_remaining == 0:
                        self._tp_trace.close()
                        self._tp_trace = None
                except OSError as exc:
                    logger.warning("Disabling TP mock trace after write failure: %s", exc)
                    self._tp_trace.close()
                    self._tp_trace = None
            if not self._tp_logged:
                logger.info("TP mock first forward collectives=%s", collectives.counts)
                self._tp_logged = True
            return output
        finally:
            collectives.active = False
