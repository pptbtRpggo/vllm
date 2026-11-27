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
from vllm.logger import init_logger

NATIVE_WORKER = __name__ + ".Native910BWorker"
logger = init_logger(__name__)


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


def _native_layer_range(model, num_layers):
    for candidate in (
        getattr(model, "model", None),
        getattr(model, "transformer", None),
        model,
    ):
        start = getattr(candidate, "start_layer", None)
        end = getattr(candidate, "end_layer", None)
        if type(start) is int and type(end) is int:
            return start, end

    from torch import nn

    if not isinstance(model, nn.Module):
        return None

    from vllm.model_executor.models.utils import PPMissingLayer

    # The Transformers backend keeps the global layer indices by inserting
    # placeholders for every layer owned by another PP stage.
    ranges = set()
    for module in model.modules():
        if not isinstance(module, nn.ModuleList) or len(module) != num_layers:
            continue
        missing = [isinstance(layer, PPMissingLayer) for layer in module]
        if not any(missing):
            continue
        local = [i for i, absent in enumerate(missing) if not absent]
        if not local or local != list(range(local[0], local[-1] + 1)):
            raise ValueError("loaded PP layers must form a contiguous range")
        ranges.add((local[0], local[-1] + 1))
    return ranges.pop() if len(ranges) == 1 else None


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
        pp_size = len(plan.stages)
        if (
            any(s.soc != "910B" for s in plan.stages)
            or parallel.nnodes != len(plan.nodes)
            or parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != pp_size
            or parallel.world_size != pp_size
            or parallel.data_parallel_size != 1
            or plan.partition != os.environ.get("VLLM_PP_LAYER_PARTITION")
        ):
            raise ValueError("native HCCL requires a matching 910B PP plan and TP1")
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
        self._check_native_startup()

    def _check_native_startup(self):
        from vllm.distributed.hetero_pp.native_startup import check_native_startup

        self._native_checkpoint = check_native_startup(self, self._native_plan)
        if os.environ.get("HETERO_PP_RUN_MODE") == "profile":
            from vllm.distributed.hetero_pp.profile_memory import (
                validate_fixed_kv_capacity,
            )

            validate_fixed_kv_capacity(self.vllm_config)

    def _verified_native_layers(self, *, allow_unavailable=False):
        model = self.model_runner.model
        layers = _native_layer_range(model, self._native_plan.layer_ranges[-1][1])
        if layers is None and allow_unavailable:
            logger.warning("Cannot verify loaded PP layers for rank %d", self.rank)
            return None
        if layers is None:
            raise ValueError(f"rank {self.rank} does not expose its loaded PP layers")
        if layers != self._native_plan.layer_ranges[self.rank]:
            expected = self._native_plan.layer_ranges[self.rank]
            raise ValueError(
                f"rank {self.rank} loaded layers {layers} differ from native plan "
                f"{expected}"
            )
        return layers

    def load_model(self):
        super().load_model()  # type: ignore[misc]
        profile = os.environ.get("HETERO_PP_RUN_MODE") == "profile"
        self._verified_native_layers(allow_unavailable=not profile)

    def start_native_pp_profile(self, directory, session, compute_timing):
        import torch
        import torch_npu

        import vllm
        from vllm.distributed.hetero_pp.profiling import (
            StageProfiler,
            serving_config_observation,
        )
        from vllm.distributed.parallel_state import get_pp_group

        if self._hetero_profiler is not None:
            raise ValueError("native profiling is already active")
        cfg = self.vllm_config
        group = get_pp_group()
        pp_size = len(self._native_plan.stages)
        backend = str(torch.distributed.get_backend(group.device_group))
        world_backend = str(torch.distributed.get_backend())
        if (
            backend != "hccl"
            or str(torch.distributed.get_backend(group.cpu_group)) != "gloo"
            or group.use_cpu_custom_send_recv
            or group.rank_in_group != self.rank
            or group.ranks != list(range(pp_size))
            or not cfg.model_config.enforce_eager
            or cfg.scheduler_config.async_scheduling is not False
            or cfg.use_v2_model_runner
            or cfg.speculative_config is not None
            or cfg.kv_transfer_config is not None
            or getattr(self.model_runner, "broadcast_pp_output", False)
        ):
            raise ValueError("profile requires native HCCL tensor P2P and eager PP")
        base = Path(__file__).parent
        sources = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in base.glob("*.py")
        }
        model = self.model_runner.model
        layers = self._verified_native_layers()
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
            "checkpoint_sha256": self._native_checkpoint,
            "serving_config_observation": serving_config_observation(self),
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
            pp_size,
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

        phase = "execute"

        def handoff(operation, fn, *args, **kwargs):
            nonlocal phase
            phase = operation
            result = fn(timed, *args, **kwargs)
            phase = "execute"
            return result

        def compute(original, *args, **kwargs):
            return profiler.execute(step, lambda: original(*args, **kwargs))

        try:
            group = get_pp_group()
            link = _NativeLink(group, lambda: torch.npu.synchronize(self.device))
            timed = _AckTraceLink(step, link, profiler.timeout_s)
            with (
                _observe_method(
                    group,
                    "irecv_tensor_dict",
                    lambda *a, **kw: handoff("receive", link.receive, *a, **kw),
                ),
                _observe_method(
                    group,
                    "isend_tensor_dict",
                    lambda *a, **kw: handoff("send", link.transmit, *a, **kw),
                ),
                _observe_method(self.model_runner, "execute_model", compute),
            ):
                return execute(scheduler_output)
        except BaseException as error:
            profiler.abort_step(step, error, phase=phase)
            raise

    def sample_tokens(self, grammar_output):
        sample = super().sample_tokens  # type: ignore[misc]
        if self._hetero_profiler is None:
            return sample(grammar_output)
        return self._hetero_profiler.sample(lambda: sample(grammar_output))

    def shutdown(self):
        try:
            profiler = self._hetero_profiler
            if profiler is not None:
                profiler.close(raise_errors=not profiler.step_failure_reported)
        finally:
            super().shutdown()  # type: ignore[misc]


def __getattr__(name):
    if name != "Native910BWorker":
        raise AttributeError(name)
    from vllm_ascend.worker.worker import NPUWorker

    cls = type(name, (NativeWorkerMixin, NPUWorker), {"__module__": __name__})
    globals()[name] = cls
    return cls
