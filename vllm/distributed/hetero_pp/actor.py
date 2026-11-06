# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One Ray actor per NPU. All Ascend selection happens on the actor's host."""
import copy
import hashlib
import os
from pathlib import Path


def make_wrapper(wrapper_cls, cfg, rank):
    import inspect
    kwargs = {'rpc_rank': 0, 'global_rank': rank}
    if 'vllm_config' in inspect.signature(wrapper_cls).parameters:
        kwargs['vllm_config'] = cfg
    return wrapper_cls(**kwargs)


def local_native_worker():
    try:
        from vllm_ascend.device.hardware_profile import get_current_hardware_profile
    except ModuleNotFoundError as error:
        if error.name not in ('vllm_ascend.device', 'vllm_ascend.device.hardware_profile'):
            raise
        import acl
        from vllm.distributed.hetero_pp.config import worker_for_soc_name
        return worker_for_soc_name(acl.get_soc_name())
    return get_current_hardware_profile().default_worker_cls


class PipelineActor:
    def identity(self):
        import ray
        ids = ray.get_runtime_context().get_accelerator_ids().get('NPU', [])
        if len(ids) != 1:
            raise RuntimeError(f'actor must own exactly one Ray NPU, got {ids}')
        return {'node_id': ray.get_runtime_context().get_node_id(), 'device_id': int(ids[0])}

    def initialize(self, cfg, rank, plan, rendezvous):
        import ray
        from vllm.distributed.hetero_pp.config import select_worker, validate_runtime
        stage = plan.stages[rank]
        os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
        os.environ['VLLM_PP_LAYER_PARTITION'] = plan.partition
        os.environ['VLLM_HOST_IP'] = stage.node_ip
        # Keep each node's own CANN/plugin library environment. Do not copy the
        # 910B head environment or its wheel to the 310P host.
        from vllm.plugins import load_general_plugins
        load_general_plugins()
        worker_cls = select_worker(stage.soc, local_native_worker())
        cfg = copy.deepcopy(cfg)
        cfg.parallel_config.worker_cls = worker_cls
        cfg.parallel_config.distributed_executor_backend = 'ray'
        cfg.parallel_config.rank = rank
        validate_runtime(cfg, len(plan.stages))
        from vllm.utils.import_utils import resolve_obj_by_qualname
        selected = resolve_obj_by_qualname(worker_cls)
        selected._hetero_timeout_s = plan.timeout_s
        from vllm.v1.worker.worker_base import WorkerWrapperBase
        self.wrapper = make_wrapper(WorkerWrapperBase, cfg, rank)
        kwargs = dict(vllm_config=cfg, rank=rank, local_rank=0,
                      distributed_init_method=rendezvous, is_driver_worker=True)
        self.wrapper.init_worker([kwargs])
        self.wrapper.init_device()
        self.call('load_model', (), {})
        runner = self.wrapper.worker.model_runner
        model = runner.model
        wanted = plan.layer_ranges[rank]
        actual = (getattr(model, 'start_layer', None), getattr(model, 'end_layer', None))
        # Llama/Qwen wrappers hold the partitioned transformer as .model.
        if actual == (None, None) and hasattr(model, 'model'):
            actual = (getattr(model.model, 'start_layer', None), getattr(model.model, 'end_layer', None))
        if actual != wanted:
            raise RuntimeError(f'rank {rank} loaded layer range {actual}, expected {wanted}')
        import torch
        import torch_npu
        device_name = torch.npu.get_device_name(0)
        if stage.soc.lower() not in device_name.lower().replace('_', ''):
            raise RuntimeError(f'actual device {device_name} does not match {stage.soc}')
        import vllm
        here = Path(__file__).parent
        sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in here.glob('*.py')}
        sources['cpu_staging.py'] = hashlib.sha256((here.parent / 'device_communicators/cpu_staging.py').read_bytes()).hexdigest()
        return dict(rank=rank, soc=stage.soc, node_ip=stage.node_ip, layers=actual,
                    device_name=device_name, identity=self.identity(), vllm=vllm.__version__,
                    torch=torch.__version__, torch_npu=torch_npu.__version__, ray=ray.__version__,
                    model_config_sha256=hashlib.sha256(cfg.model_config.hf_config.to_json_string().encode()).hexdigest(),
                    sources=sources, world_backend=str(torch.distributed.get_backend()))

    def call(self, method, args, kwargs):
        from vllm.v1.serial_utils import run_method
        return run_method(self.wrapper, method, args, kwargs)

    def shutdown(self):
        if hasattr(self, 'wrapper'):
            try:
                self.wrapper.shutdown()
            finally:
                from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
                destroy_model_parallel()
                destroy_distributed_environment()
