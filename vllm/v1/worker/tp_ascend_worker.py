# SPDX-License-Identifier: Apache-2.0
"""Ascend eager worker for controlled TP heterogeneity experiments."""

from __future__ import annotations

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
        logger.info(
            "TP mock rank=%d compute_scale=%g cross_group_size=%d "
            "cross_extra_bandwidth_gbps=%s",
            tp_group.rank_in_group,
            self._tp_hetero.scale(tp_group.rank_in_group),
            self._tp_hetero.cross_group_size,
            self._tp_hetero.cross_extra_bandwidth_gbps,
        )

    def shutdown(self) -> None:
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
        collectives.active = True
        try:
            with self._tp_layer_timer.capture(
                self._tp_hetero.scale(get_tp_group().rank_in_group),
                excluded_ms=lambda: collectives.total_ms,
            ):
                output = super().execute_model(scheduler_output)
                sync_torch_device(self.device)
            if not self._tp_logged:
                logger.info("TP mock first forward collectives=%s", collectives.counts)
                self._tp_logged = True
            return output
        finally:
            collectives.active = False
