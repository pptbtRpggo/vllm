# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local Ascend subclasses, created lazily in each Ray actor after SoC checks."""


class _AdjacentLink:
    def __init__(self, channel, rank):
        self.channel, self.rank = channel, rank

    def recv(self, sequence):
        return self.channel.recv(self.rank - 1, sequence)

    def send(self, values, sequence):
        return self.channel.send(values, self.rank + 1, sequence)


class StagingWorkerMixin:
    def _init_worker_distributed_environment(self):
        from datetime import timedelta
        from vllm.distributed import parallel_state as ps
        from vllm.distributed.hetero_pp.groups import install_group_factory
        from vllm_ascend.distributed.parallel_state import init_ascend_model_parallel
        install_group_factory()
        ps.init_distributed_environment(
            self.parallel_config.world_size, self.rank,
            self.distributed_init_method, self.local_rank, 'gloo',
            timeout=timedelta(seconds=self._hetero_timeout_s))
        ps.ensure_model_parallel_initialized(1, self.parallel_config.pipeline_parallel_size,
                                             1, 1, backend='gloo')
        init_ascend_model_parallel(self.parallel_config)

    def init_device(self):
        super().init_device()
        from vllm.distributed.parallel_state import get_pp_group
        from vllm.distributed.hetero_pp.channel import TensorDictChannel
        group = get_pp_group()
        self._hetero_channel = _AdjacentLink(
            TensorDictChannel(group.cpu_group, self.device, self._hetero_timeout_s), self.rank)
        self._hetero_sequence = 0
        # MRv1 speculation/broadcast path uses another stage-send protocol;
        # reject rather than duplicate stage sends inside the runner.
        if getattr(self.model_runner, 'broadcast_pp_output', False):
            raise ValueError('broadcast_pp_output runner path is not admitted')

    def _hetero_context(self):
        import torch
        from vllm.config import set_current_vllm_config
        from contextlib import ExitStack
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
        def execute(values):
            inputs = None if values is None else IntermediateTensors(values)
            output = self.model_runner.execute_model(scheduler_output, inputs)
            if self.rank < self.parallel_config.world_size - 1 and scheduler_output.total_num_scheduled_tokens:
                if not isinstance(output, IntermediateTensors):
                    raise TypeError('nonfinal runner did not return IntermediateTensors')
                return output.tensors
            if hasattr(output, 'get_output'):
                output = output.get_output()
            return output
        with self._hetero_context():
            output = self._run_staged_step(
                self.rank, self.parallel_config.world_size, self._hetero_sequence,
                scheduler_output.total_num_scheduled_tokens > 0,
                self._hetero_channel, execute)
            self._hetero_sequence += 1
            return output

    def sample_tokens(self, grammar_output):
        with self._hetero_context():
            output = self.model_runner.sample_tokens(grammar_output)
            if hasattr(output, 'get_output'):
                output = output.get_output()
            return output

    def shutdown(self):
        try:
            if hasattr(self, '_hetero_channel'):
                self._hetero_channel.channel.close()
        finally:
            super().shutdown()


def __getattr__(name):
    if name not in ('Staged910BWorker', 'Staged310PWorker'):
        raise AttributeError(name)
    if name == 'Staged310PWorker':
        from vllm_ascend._310p.worker_310p import NPUWorker310 as Base
    else:
        from vllm_ascend.worker.worker import NPUWorker as Base
    # __init__ is intentionally inherited unchanged; timeout is installed on
    # the class by the actor, before construction in its private process.
    cls = type(name, (StagingWorkerMixin, Base), {'__module__': __name__, '_hetero_timeout_s': 120.0})
    globals()[name] = cls
    return cls
