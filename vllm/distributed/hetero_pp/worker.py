# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local Ascend subclasses, created lazily in each Ray actor after SoC checks."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from vllm.config import CacheConfig, ParallelConfig, VllmConfig
    from vllm.distributed.hetero_pp.profiling import StageProfiler


class _AdjacentLink:
    def __init__(self, channel, rank):
        self.channel, self.rank = channel, rank

    def recv(self, sequence):
        return self.channel.recv(self.rank - 1, sequence)

    def send(self, values, sequence):
        return self.channel.send(values, self.rank + 1, sequence)


class StagingWorkerMixin:
    if TYPE_CHECKING:
        # Supplied by the native SoC worker selected lazily in __getattr__.
        vllm_config: VllmConfig
        cache_config: CacheConfig
        parallel_config: ParallelConfig
        model_runner: Any
        rank: int
        local_rank: int
        device: torch.device
        distributed_init_method: str
        _hetero_timeout_s: float
        _hetero_profiler: StageProfiler | None

    def _get_hetero_profiler(self):
        if hasattr(self, "_hetero_profiler"):
            return self._hetero_profiler
        import os

        if not os.environ.get("VLLM_PP_STAGE_TRACE"):
            return None
        from vllm.distributed.hetero_pp.profiling import maybe_create_profiler

        self._hetero_profiler = maybe_create_profiler(self)
        if self._hetero_profiler is not None:
            self._hetero_profiler.is_warmup = getattr(
                self, "_hetero_profile_warmup", False
            )
        return self._hetero_profiler

    def set_pp_profile_warmup(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("profile warmup flag must be a bool")
        self._hetero_profile_warmup = enabled
        profiler = self._get_hetero_profiler()
        if profiler is not None:
            profiler.is_warmup = enabled

    def flush_pp_profile(self):
        profiler = self._get_hetero_profiler()
        return (
            profiler.flush()
            if profiler is not None
            else {"enabled": False, "complete": False, "pending": 0}
        )

    def get_pp_memory_observation(self):
        import json

        from vllm.distributed.hetero_pp.profiling import (
            describe_kv_cache,
            serving_config_observation,
        )
        from vllm.distributed.pp_memory import observe_worker_memory

        with self._hetero_context():
            profiler = self._get_hetero_profiler()
            if profiler is not None:
                profiler.flush()
            result = observe_worker_memory(self)
            result.update(describe_kv_cache(self))
            result["serving_config_observation"] = serving_config_observation(self)
            result["profile_identity"] = getattr(self, "_hetero_profile_identity", {})
            # Ray accepts str subclasses such as TorchVersion, while the
            # EngineCore msgspec RPC boundary requires builtin primitives.
            # Normalize a detached copy without mutating startup identity or
            # enabling pickle/custom-object deserialization.
            return json.loads(json.dumps(result))

    def _init_worker_distributed_environment(self):
        from datetime import timedelta

        from vllm_ascend.distributed.parallel_state import init_ascend_model_parallel

        from vllm.distributed import parallel_state as ps
        from vllm.distributed.hetero_pp.groups import install_group_factory

        install_group_factory()
        ps.init_distributed_environment(
            self.parallel_config.world_size,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            "gloo",
            timeout=timedelta(seconds=self._hetero_timeout_s),
        )
        ps.ensure_model_parallel_initialized(
            1, self.parallel_config.pipeline_parallel_size, 1, 1, backend="gloo"
        )
        init_ascend_model_parallel(self.parallel_config)

    def init_device(self):
        # The lazily composed native worker, not object, supplies this method.
        super().init_device()  # type: ignore[misc]
        from vllm.distributed.hetero_pp.channel import TensorDictChannel
        from vllm.distributed.parallel_state import get_pp_group

        group = get_pp_group()
        self._hetero_channel = _AdjacentLink(
            TensorDictChannel(group.cpu_group, self.device, self._hetero_timeout_s),
            self.rank,
        )
        self._hetero_sequence = 0
        # MRv1 speculation/broadcast path uses another stage-send protocol;
        # reject rather than duplicate stage sends inside the runner.
        if getattr(self.model_runner, "broadcast_pp_output", False):
            raise ValueError("broadcast_pp_output runner path is not admitted")

    def _hetero_context(self):
        from contextlib import ExitStack

        import torch

        from vllm.config import set_current_vllm_config

        stack = ExitStack()
        stack.enter_context(set_current_vllm_config(self.vllm_config))
        stack.enter_context(torch.inference_mode())
        torch.npu.set_device(self.device)
        return stack

    def _run_staged_step(self, *args):
        from vllm.distributed.hetero_pp.step import run_stage

        return run_stage(*args)

    def execute_model(self, scheduler_output):
        from vllm.sequence import IntermediateTensors

        profiler = self._get_hetero_profiler()
        step = (
            profiler.begin(scheduler_output, self.model_runner, self._hetero_sequence)
            if profiler is not None
            else None
        )

        def execute(values):
            inputs = None if values is None else IntermediateTensors(values)
            invoke = lambda: self.model_runner.execute_model(scheduler_output, inputs)
            output = (
                profiler.execute(step, invoke) if profiler is not None else invoke()
            )
            if (
                self.rank < self.parallel_config.world_size - 1
                and scheduler_output.total_num_scheduled_tokens
            ):
                if not isinstance(output, IntermediateTensors):
                    raise TypeError(
                        "nonfinal runner did not return IntermediateTensors"
                    )
                return output.tensors
            if hasattr(output, "get_output"):
                output = (
                    profiler.materialize("execute", output)
                    if profiler is not None
                    else output.get_output()
                )
            return output

        with self._hetero_context():
            output = self._run_staged_step(
                self.rank,
                self.parallel_config.world_size,
                self._hetero_sequence,
                scheduler_output.total_num_scheduled_tokens > 0,
                (
                    profiler.timed_link(step, self._hetero_channel)
                    if profiler is not None
                    else self._hetero_channel
                ),
                execute,
            )
            self._hetero_sequence += 1
            return output

    def sample_tokens(self, grammar_output):
        with self._hetero_context():
            profiler = self._get_hetero_profiler()

            def invoke():
                output = self.model_runner.sample_tokens(grammar_output)
                if hasattr(output, "get_output"):
                    output = (
                        profiler.materialize("sample", output)
                        if profiler is not None
                        else output.get_output()
                    )
                return output

            output = profiler.sample(invoke) if profiler is not None else invoke()
            return output

    def shutdown(self):
        try:
            try:
                profiler = getattr(self, "_hetero_profiler", None)
                if profiler is not None:
                    profiler.close()
            finally:
                if hasattr(self, "_hetero_channel"):
                    self._hetero_channel.channel.close()
        finally:
            # The lazily composed native worker supplies the teardown method.
            super().shutdown()  # type: ignore[misc]


def __getattr__(name):
    if name not in ("Staged910BWorker", "Staged310PWorker"):
        raise AttributeError(name)
    if name == "Staged310PWorker":
        from vllm_ascend._310p.worker_310p import NPUWorker310 as Base
    else:
        from vllm_ascend.worker.worker import NPUWorker as Base
    # __init__ is intentionally inherited unchanged; timeout is installed on
    # the class by the actor, before construction in its private process.
    cls = type(
        name,
        (StagingWorkerMixin, Base),
        {"__module__": __name__, "_hetero_timeout_s": 120.0},
    )
    globals()[name] = cls
    return cls
