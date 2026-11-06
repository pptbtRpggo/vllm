"""CPU-only integration of the actual core GroupCoordinator API and adapter."""
import os
from pathlib import Path
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[3]


def bootstrap():
    # Test an uninstalled source overlay against the existing vLLM runtime.
    import vllm.distributed
    path = str(ROOT / 'vllm/distributed')
    if path not in vllm.distributed.__path__: vllm.distributed.__path__.insert(0, path)
    import vllm.distributed.device_communicators
    extra = str(ROOT / 'vllm/distributed/device_communicators')
    if extra not in vllm.distributed.device_communicators.__path__:
        vllm.distributed.device_communicators.__path__.insert(0, extra)


def exchange(rank, rendezvous):
    bootstrap()
    from vllm.distributed.hetero_pp.groups import StagedGroup
    assert not (hasattr(torch, 'npu') and torch.npu.is_initialized())
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=15))
    from vllm.platforms import current_platform as real_platform
    class CpuPlatformView:
        def __getattr__(self, name):
            return getattr(real_platform, name)
        def is_cuda_alike(self): return False
        def is_tpu(self): return False
        def is_xpu(self): return False
        def is_out_of_tree(self): return False
        def is_cpu(self): return True
    cpu_platform = CpuPlatformView()
    group = None
    try:
        with patch('vllm.platforms.current_platform', cpu_platform):
            group = StagedGroup([[0, 1]], rank, 'gloo', True, group_name='pp')
        assert group.device_communicator is None
        if rank == 0:
            works = group.isend_tensor_dict({'hidden_states': torch.tensor([[3., 5.]]).half(), 'residual': None})
            for work in works: work.wait()
        else:
            values, handles, post = group.irecv_tensor_dict()
            assert not handles and not post
            assert torch.equal(values['hidden_states'], torch.tensor([[3., 5.]]).half())
        values = group.broadcast_tensor_dict({'token': torch.tensor(29)} if rank == 1 else None, src=1)
        assert values['token'].item() == 29
        dist.barrier()
        assert not (hasattr(torch, 'npu') and torch.npu.is_initialized())
    finally:
        if group is not None: group.destroy()
        dist.destroy_process_group()


class GroupTests(unittest.TestCase):
    def test_actor_rpc_uses_actual_core_dispatch_on_wrapper(self):
        bootstrap()
        from vllm.distributed.hetero_pp.actor import PipelineActor, make_wrapper
        from vllm.v1.worker.worker_base import WorkerWrapperBase
        actor = PipelineActor()
        actor.wrapper = make_wrapper(WorkerWrapperBase, NS(), 2)
        self.assertEqual(actor.wrapper.rpc_rank, 0)
        self.assertEqual(actor.wrapper.global_rank, 2)
        actor.wrapper.worker = NS(add=lambda a, b: a + b)
        self.assertEqual(actor.call('add', (2,), {'b': 5}), 7)

    def test_group_send_receive_broadcast_and_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            mp.spawn(exchange, args=(f'file://{d}/store',), nprocs=2)

    def test_executor_resolves_as_actual_core_subclass(self):
        bootstrap()
        import vllm.v1.executor
        vllm.v1.executor.__path__.insert(0, str(ROOT / 'vllm/v1/executor'))
        from vllm.v1.executor.abstract import Executor
        from vllm.v1.executor.hetero_pp_executor import HeteroPPExecutor
        self.assertTrue(issubclass(HeteroPPExecutor, Executor))
        self.assertFalse(HeteroPPExecutor.__abstractmethods__)
        self.assertTrue(HeteroPPExecutor.uses_ray)

    def test_serve_entrypoint_passes_pp_admission_as_a_class(self):
        bootstrap()
        import vllm.v1.executor
        vllm.v1.executor.__path__.insert(0, str(ROOT / 'vllm/v1/executor'))
        from vllm.distributed.hetero_pp.serve import configure_executor
        from vllm.engine.arg_utils import EngineArgs
        args = NS(distributed_executor_backend=None)
        configure_executor(args)
        self.assertTrue(args.distributed_executor_backend.supports_pp)
        engine = EngineArgs(model=str(ROOT), pipeline_parallel_size=4,
                            distributed_executor_backend=args.distributed_executor_backend)
        # The older core takes model_config here; 0.23 takes no argument.
        import inspect
        check = engine._check_feature_supported
        check(*([NS()] if inspect.signature(check).parameters else []))
        with self.assertRaises(ValueError):
            configure_executor(NS(distributed_executor_backend='mp'))


if __name__ == '__main__': unittest.main()
