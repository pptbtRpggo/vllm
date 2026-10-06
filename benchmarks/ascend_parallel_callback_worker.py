# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only toggles around the real Ascend TP/PP workers.

All cases read the same control file. Native cases uninstall TP hooks and
collective wrappers, then invoke NPUWorker directly. callback_zero activates
production callback locations with a 1.000001 factor, but suppresses sleep via
the existing sleep-overhead compensation. It tests callback cost, not slower
hardware. No source-level timing or synchronization behavior is replaced.
"""

import json
import os
from pathlib import Path

from vllm_ascend.worker.worker import NPUWorker

from vllm.distributed.parallel_state import get_tp_group
from vllm.distributed.pp_hetero import PPHeteroConfig, PPNetwork
from vllm.distributed.pp_stream import PPStreamExecution
from vllm.distributed.tp_hetero import TPComputeDelay, TPHeteroConfig
from vllm.distributed.tp_stream_delay import TPStreamDelay
from vllm.v1.worker.pp_ascend_worker import PPAscendWorker
from vllm.v1.worker.tp_ascend_worker import TPAscendWorker


def control():
    return json.loads(Path(os.environ["VLLM_CALLBACK_BENCH_CONTROL"]).read_text())


class TPCallbackBenchWorker(TPAscendWorker):
    def execute_model(self, scheduler_output):
        state = control()
        mode = state["mode"]
        if mode == "native":
            self._tp_collectives.uninstall()
            if self._tp_compute_delay is not None:
                self._tp_compute_delay.uninstall()
            self._tp_compute_model = None
            return NPUWorker.execute_model(self, scheduler_output)
        if not self._tp_collectives.originals:
            self._tp_collectives.install()
        scale = 1.000001 if mode in ("callback_zero", "trace_zero") else 1.0
        size = get_tp_group().world_size
        self._tp_hetero = TPHeteroConfig((scale,) * size, size, None)
        model = self.model_runner.model
        if self._tp_compute_model is not model or self._tp_compute_delay.scale != scale:
            if self._tp_compute_delay is not None:
                self._tp_compute_delay.uninstall()
            self._tp_compute_delay = TPComputeDelay(model, scale, self._tp_stream_delay)
            self._tp_compute_delay.install()
            self._tp_compute_model = model
        if mode in ("callback_zero", "trace_zero"):
            self._tp_compute_delay.sleep_overhead_ms = 1e9
        writer = self._tp_trace_writer
        if mode != "trace_zero":
            self._tp_trace_writer = None
        try:
            return super().execute_model(scheduler_output)
        finally:
            self._tp_trace_writer = writer

    def _submit_trace(self, metadata, compute, collective, started):
        state = control()
        metadata.update(
            bench_label=state["label"],
            bench_phase=state["phase"],
            bench_concurrency=state["concurrency"],
        )
        return super()._submit_trace(metadata, compute, collective, started)


class PPCallbackBenchWorker(PPAscendWorker):
    def init_device(self):
        super().init_device()
        self._pp_stream_delay = TPStreamDelay()
        self._pp_stream_execution = PPStreamExecution(self._pp_stream_delay)

    def execute_model(self, scheduler_output):
        state = control()
        mode = state["mode"]
        if mode == "native":
            return NPUWorker.execute_model(self, scheduler_output)
        scale = 1.000001 if mode == "callback_zero" else 1.0
        self._pp_hetero = PPHeteroConfig(
            compute_scales=(scale,) * 4,
            network=PPNetwork.from_json(
                '{"bandwidth_gbps":[[null,null,null],[null,null],[null],[]]}'
            ),
        )
        if mode == "callback_zero":
            self._pp_stream_execution._sleep_overhead["compute"] = 1e9
        tracer = self._pp_stage_tracer
        if mode != "trace_zero":
            self._pp_stage_tracer = None
        else:
            tracer.is_warmup = state["phase"] == "warmup"
            tracer.trace_session = f"{state['label']}/c{state['concurrency']}"
        try:
            return super().execute_model(scheduler_output)
        finally:
            self._pp_stage_tracer = tracer
