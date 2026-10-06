# SPDX-License-Identifier: Apache-2.0
"""Serving comparisons using production device-delay workers.

Eager modes can switch in one server. Graph TP uses one fixed mode per server:
hooks must be present during capture, and capture-time factors cannot be
changed by subsequently changing Python configuration.
"""

import json
import os
from pathlib import Path

from vllm_ascend.worker.worker import NPUWorker

from vllm.distributed.pp_hetero import PPHeteroConfig, PPNetwork
from vllm.distributed.tp_hetero import TPHeteroConfig
from vllm.v1.worker.pp_ascend_worker import PPAscendWorker
from vllm.v1.worker.tp_ascend_worker import TPAscendWorker


def track_graph(worker):
    """Count real cache hits, rather than assuming --graph enabled replay."""
    model = worker.model_runner.model
    if not hasattr(model, "concrete_aclgraph_entries"):
        return
    from vllm.forward_context import get_forward_context

    original = type(model).__call__
    worker._bench_graph_replays = 0

    def call(instance, *args, **kwargs):
        if instance is model:
            context = get_forward_context()
            if context.cudagraph_runtime_mode == instance.runtime_mode:
                entry = instance.concrete_aclgraph_entries.get(context.batch_descriptor)
                if entry is not None and entry.aclgraph is not None:
                    worker._bench_graph_replays += 1
                    if worker._bench_graph_replays == 1:
                        print("DEVICE_BENCH_ACTUAL_GRAPH_REPLAY", flush=True)
        return original(instance, *args, **kwargs)

    type(model).__call__ = call


def control():
    return json.loads(Path(os.environ["VLLM_DEVICE_BENCH_CONTROL"]).read_text())


def scales(mode):
    if mode == "kernel_zero":
        # Activate the actual timing kernels; rounding produces zero requested
        # wait for ordinary intervals. This measures their launch cost.
        return (1.000000000001,) * 4
    return (1, 1, 2, 4) if mode == "hetero" else (1,) * 4


class TPDeviceBenchWorker(TPAscendWorker):
    def init_device(self):
        super().init_device()
        self._bench_mode = control()["mode"]
        self._set_mode(self._bench_mode)

    def _set_mode(self, mode):
        if self._tp_compute_delay is not None:
            self._tp_compute_delay.uninstall()
        self._tp_compute_delay = None
        self._tp_compute_model = None
        self._tp_hetero = TPHeteroConfig(
            scales(mode), 2, 25 if mode == "hetero" else None
        )
        self._tp_collectives.config = self._tp_hetero
        if mode == "native":
            self._tp_collectives.uninstall()
        elif not self._tp_collectives.originals:
            self._tp_collectives.install()
        self._bench_mode = mode

    def load_model(self):
        super().load_model()
        if self._bench_mode == "native":
            self._tp_compute_delay.uninstall()
        track_graph(self)

    def shutdown(self):
        print(
            "DEVICE_BENCH_GRAPH_REPLAYS",
            getattr(self, "_bench_graph_replays", 0),
            flush=True,
        )
        super().shutdown()

    def execute_model(self, scheduler_output):
        state = control()
        self._tp_trace_metadata = {
            "bench_label": state["label"],
            "bench_phase": state["phase"],
            "bench_concurrency": state["concurrency"],
        }
        mode = state["mode"]
        if mode != self._bench_mode:
            if not self.vllm_config.model_config.enforce_eager:
                raise RuntimeError("TP Graph benchmark requires a fixed mode")
            self._set_mode(mode)
        if mode == "native":
            return NPUWorker.execute_model(self, scheduler_output)
        writer = self._tp_trace_writer
        if mode != "trace_zero":
            self._tp_trace_writer = None
        try:
            return super().execute_model(scheduler_output)
        finally:
            self._tp_trace_writer = writer


class PPDeviceBenchWorker(PPAscendWorker):
    def load_model(self):
        super().load_model()
        track_graph(self)

    def shutdown(self):
        print(
            "DEVICE_BENCH_GRAPH_REPLAYS",
            getattr(self, "_bench_graph_replays", 0),
            flush=True,
        )
        super().shutdown()

    def execute_model(self, scheduler_output):
        state = control()
        mode = state["mode"]
        if mode == "native":
            return NPUWorker.execute_model(self, scheduler_output)
        network = (
            PPNetwork.from_json('{"bandwidth_gbps":[[null,25,25],[25,25],[null],[]]}')
            if mode == "hetero"
            else None
        )
        self._pp_hetero = PPHeteroConfig(compute_scales=scales(mode), network=network)
        tracer = self._pp_stage_tracer
        if mode != "trace_zero":
            self._pp_stage_tracer = None
        elif tracer is not None:
            tracer.is_warmup = state["phase"] == "warmup"
            tracer.trace_session = f"{state['label']}/c{state['concurrency']}"
        try:
            return super().execute_model(scheduler_output)
        finally:
            self._pp_stage_tracer = tracer
