# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in native Ascend worker for HCCL PP core limits and EdgeShard timing."""

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from vllm.distributed.hetero_pp.config import PipelinePlan, worker_for_soc_name
from vllm.distributed.hetero_pp.worker import ProfileWorkerMixin, apply_npu_core_limit

NATIVE_WORKER = __name__ + ".Native910BWorker"


@contextmanager
def _observe_method(owner, name, observe):
    original = getattr(owner, name)
    overrides = vars(owner)
    inherited = name not in overrides
    setattr(owner, name, lambda *a, **kw: observe(original, *a, **kw))
    try:
        yield
    finally:
        if inherited:
            delattr(owner, name)
        else:
            setattr(owner, name, original)


class _NativeLink:
    """Give the shared ACK timer completed native tensor operations."""

    def __init__(self, group, synchronize):
        self.channel = SimpleNamespace(group=group.cpu_group)
        self.rank = group.rank_in_group
        self.synchronize = synchronize

    @staticmethod
    def validate_payload(values):
        import torch

        if not values or not any(
            isinstance(t, torch.Tensor) and t.numel() > 0 for t in values.values()
        ):
            raise ValueError("HCCL profile expected a nonempty activation payload")
        if any(
            isinstance(t, torch.Tensor) and t.device.type != "npu"
            for t in values.values()
        ):
            raise ValueError("HCCL profile received a non-NPU tensor")

    def receive(self, timed, original, *args, **kwargs):
        def recv(_sequence):
            values, handles, postprocess = original(*args, **kwargs)
            for handle in handles:
                handle.wait()
            for fn in postprocess:
                fn()
            self.synchronize()
            self.validate_payload(values)
            return values

        self.recv = recv
        return timed.recv(timed.step.metadata["step"]), [], []

    def transmit(self, timed, original, values, *args, **kwargs):
        self.validate_payload(values)

        def send(values, _sequence):
            for handle in original(values, *args, **kwargs):
                handle.wait()
            self.synchronize()
            return []

        self.send = send
        return timed.send(values, timed.step.metadata["step"])


class NativeWorkerMixin(ProfileWorkerMixin):
    # The concrete NPUWorker base is selected lazily in __getattr__.
    _hetero_profiler = None
    _profile_sequence = 0

    def _init_worker_distributed_environment(self):
        import torch

        from vllm.utils.network_utils import get_ip

        cfg = self.vllm_config
        parallel = cfg.parallel_config
        plan = PipelinePlan.from_dict(
            json.loads(os.environ["VLLM_HETERO_PP_PLAN"]),
            cfg.model_config.get_total_num_hidden_layers(),
        )
        if (
            any(s.soc != "910B" for s in plan.stages)
            or parallel.nnodes != 2
            or parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 4
            or parallel.world_size != 4
            or parallel.data_parallel_size != 1
            or plan.partition != os.environ.get("VLLM_PP_LAYER_PARTITION")
        ):
            raise ValueError("native HCCL plan requires two 910B nodes with PP4/TP1")
        self._native_plan = plan
        stage = plan.stages[self.rank]
        self._native_node_ip = get_ip()
        self._native_device_name = torch.npu.get_device_name(self.local_rank)
        worker_for_soc_name(self._native_device_name)
        if stage.node_ip != self._native_node_ip or "910b" not in (
            self._native_device_name.lower()
        ):
            raise ValueError(f"native rank {self.rank} node/SoC does not match plan")
        if stage.npu_core_fraction != "1":
            self._hetero_npu_core_limit = apply_npu_core_limit(
                stage.soc, stage.npu_core_fraction
            )
        super()._init_worker_distributed_environment()  # type: ignore[misc]

    def start_native_pp_profile(self, directory, session, compute_timing):
        import torch
        import torch_npu

        import vllm
        from vllm.distributed.hetero_pp.profiling import StageProfiler
        from vllm.distributed.parallel_state import get_pp_group

        if self._hetero_profiler is not None:
            raise ValueError("native profiling is already active")
        cfg = self.vllm_config
        group = get_pp_group()
        backend = str(torch.distributed.get_backend(group.device_group))
        world_backend = str(torch.distributed.get_backend())
        if (
            backend != "hccl"
            or str(torch.distributed.get_backend(group.cpu_group)) != "gloo"
            or group.use_cpu_custom_send_recv
            or group.rank_in_group != self.rank
            or group.ranks != list(range(4))
            or not cfg.model_config.enforce_eager
            or cfg.scheduler_config.async_scheduling is not False
            or cfg.use_v2_model_runner
            or cfg.speculative_config is not None
            or cfg.kv_transfer_config is not None
            or getattr(self.model_runner, "broadcast_pp_output", False)
        ):
            raise ValueError("profile requires native HCCL tensor P2P and eager PP4")
        base = Path(__file__).parent
        sources = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in base.glob("*.py")
        }
        model = self.model_runner.model
        inner = getattr(model, "model", model)
        layers = (inner.start_layer, inner.end_layer)
        if layers != self._native_plan.layer_ranges[self.rank]:
            raise ValueError(f"rank {self.rank} loaded layers differ from native plan")
        self._hetero_profile_identity = {
            "rank": self.rank,
            "profile_device_id": self.rank,
            "identity": {"device_id": self.local_rank},
            "device_id": self.local_rank,
            "device_name": self._native_device_name,
            "node_ip": self._native_node_ip,
            "soc": "910B",
            "layers": layers,
            "vllm": str(vllm.__version__),
            "torch": str(torch.__version__),
            "torch_npu": str(torch_npu.__version__),
            "model_path": cfg.model_config.model,
            "model_config_sha256": hashlib.sha256(
                cfg.model_config.hf_config.to_json_string().encode()
            ).hexdigest(),
            "sources": sources,
            "world_backend": world_backend,
            "pp_tensor_backend": backend,
            "control_backend": "gloo",
        }
        core_limit = getattr(self, "_hetero_npu_core_limit", None)
        if core_limit is not None:
            self._hetero_profile_identity["npu_core_limit"] = core_limit
        self._hetero_profiler = StageProfiler(
            directory,
            session,
            model,
            self.device,
            self.rank,
            4,
            identity=self._hetero_profile_identity,
            timeout_s=self._native_plan.timeout_s,
            transport="native_hccl",
            comm_timing="ack",
            compute_timing=compute_timing,
        )
        return self._hetero_profiler.identity

    def execute_model(self, scheduler_output):
        execute = super().execute_model  # type: ignore[misc]
        profiler = self._hetero_profiler
        if profiler is None:
            return execute(scheduler_output)

        import torch

        from vllm.distributed.hetero_pp.profiling import _AckTraceLink
        from vllm.distributed.parallel_state import get_pp_group

        step = profiler.begin(
            scheduler_output, self.model_runner, self._profile_sequence
        )
        self._profile_sequence += 1
        if step is None:
            return execute(scheduler_output)

        group = get_pp_group()
        link = _NativeLink(group, lambda: torch.npu.synchronize(self.device))
        timed = _AckTraceLink(step, link, profiler.timeout_s)

        def compute(original, *args, **kwargs):
            return profiler.execute(step, lambda: original(*args, **kwargs))

        with (
            _observe_method(
                group,
                "irecv_tensor_dict",
                lambda *a, **kw: link.receive(timed, *a, **kw),
            ),
            _observe_method(
                group,
                "isend_tensor_dict",
                lambda *a, **kw: link.transmit(timed, *a, **kw),
            ),
            _observe_method(self.model_runner, "execute_model", compute),
        ):
            return execute(scheduler_output)

    def sample_tokens(self, grammar_output):
        sample = super().sample_tokens  # type: ignore[misc]
        if self._hetero_profiler is None:
            return sample(grammar_output)
        return self._hetero_profiler.sample(lambda: sample(grammar_output))

    def shutdown(self):
        try:
            if self._hetero_profiler is not None:
                self._hetero_profiler.close()
        finally:
            super().shutdown()  # type: ignore[misc]


def __getattr__(name):
    if name != "Native910BWorker":
        raise AttributeError(name)
    from vllm_ascend.worker.worker import NPUWorker

    cls = type(name, (NativeWorkerMixin, NPUWorker), {"__module__": __name__})
    globals()[name] = cls
    return cls
