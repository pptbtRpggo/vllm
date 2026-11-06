# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit opt-in Ray executor for CPU-staged four-stage Ascend PP.

Ray only carries scheduler/configuration/control and final CPU model results.
NPU activation tensors never cross Ray serialization. The cluster must already
exist; this executor does not launch Ray daemons or alter node environments.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import threading

from vllm.v1.executor.abstract import Executor


class HeteroPPExecutor(Executor):
    uses_ray = True
    supports_pp = True

    def _init_executor(self):
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        from vllm.distributed.hetero_pp.actor import PipelineActor
        from vllm.distributed.hetero_pp.config import PipelinePlan, validate_runtime
        from vllm.utils.network_utils import get_open_port
        self.ray = ray
        self.workers = []
        self._failed = False
        self._shutdown_lock = threading.RLock()
        self._shutting_down = self._shutdown_complete = False
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='hetero-pp-rpc')
        try:
            raw = os.environ.get('VLLM_HETERO_PP_PLAN')
            if not raw:
                raise ValueError('VLLM_HETERO_PP_PLAN must contain the pipeline JSON plan')
            layers = self.model_config.hf_text_config.num_hidden_layers
            self.plan = PipelinePlan.from_dict(json.loads(raw), layers)
            self.timeout_s = self.plan.timeout_s
            os.environ['VLLM_PP_LAYER_PARTITION'] = self.plan.partition
            validate_runtime(self.vllm_config, len(self.plan.stages))
            architectures = self.model_config.hf_config.architectures
            if architectures not in (['LlamaForCausalLM'], ['Qwen2ForCausalLM']):
                raise ValueError('first backend admits LlamaForCausalLM/Qwen2ForCausalLM only')
            if not ray.is_initialized():
                ray.init(address='auto')
            nodes = {n['NodeManagerAddress']: n for n in ray.nodes() if n['Alive']}
            for ip in self.plan.nodes:
                if ip not in nodes or nodes[ip]['Resources'].get('NPU', 0) < 2:
                    raise ValueError(f'{ip} must be an alive Ray node with two NPUs')
            # rank0 hosts the TCP store; do not silently choose an IP on another
            # machine. Run the engine/head on the first plan node.
            head_node = ray.get_runtime_context().get_node_id()
            if nodes[self.plan.nodes[0]]['NodeID'] != head_node:
                raise ValueError('engine must run on the first pipeline node')
            remote = ray.remote(num_cpus=1, resources={'NPU': 1},
                                max_restarts=0, max_task_retries=0)(PipelineActor)
            for stage in self.plan.stages:
                worker = remote.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=nodes[stage.node_ip]['NodeID'], soft=False)).remote()
                self.workers.append(worker)
            identities = ray.get([w.identity.remote() for w in self.workers], timeout=self.timeout_s)
            # Actor allocation order is not device order. Sort each node's two
            # allocations by actual Ray NPU id before assigning PP ranks.
            ordered = []
            for ip in self.plan.nodes:
                owned = [(w, ident) for w, ident in zip(self.workers, identities)
                         if ident['node_id'] == nodes[ip]['NodeID']]
                if len(owned) != 2 or len({i['device_id'] for _, i in owned}) != 2:
                    raise ValueError('Ray did not reserve two distinct devices per node')
                ordered.extend(w for w, _ in sorted(owned, key=lambda item: item[1]['device_id']))
            self.workers = ordered
            rendezvous = f'tcp://{self.plan.nodes[0]}:{get_open_port()}'
            # All initialize RPCs are submitted before waiting (group creation
            # is collective). Same rule applies to every subsequent RPC.
            reports = ray.get([w.initialize.remote(self.vllm_config, rank, self.plan, rendezvous)
                               for rank, w in enumerate(self.workers)], timeout=self.timeout_s)
            for key in ('vllm', 'torch', 'torch_npu', 'ray', 'model_config_sha256', 'sources'):
                if any(r[key] != reports[0][key] for r in reports):
                    raise ValueError(f'pipeline endpoints disagree on {key}')
            if any(r['world_backend'] != 'gloo' for r in reports):
                raise ValueError('mixed-SoC WORLD must not use HCCL')
            self.startup_reports = reports
            import logging
            logging.getLogger(__name__).info('Heterogeneous PP startup reports: %s', json.dumps(reports))
        except BaseException:
            self._kill_owned()
            self._pool.shutdown(wait=False, cancel_futures=True)
            raise

    def _kill_owned(self):
        import logging
        import threading
        import traceback
        logging.getLogger(__name__).debug('Owned actor termination thread=%s caller=%s',
            threading.get_ident(), ''.join(traceback.format_stack(limit=5)))
        self._failed = True
        for worker in self.workers:
            try:
                self.ray.kill(worker, no_restart=True)
            except Exception:
                pass

    def _rpc(self, method, args, kwargs, timeout):
        if self._failed:
            raise RuntimeError('heterogeneous PP executor has failed or shut down')
        try:
            refs = [w.call.remote(method, args, kwargs) for w in self.workers]
            return self.ray.get(refs, timeout=self.timeout_s if timeout is None else timeout)
        except BaseException:
            self._kill_owned()
            raise

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None, non_block=False):
        kwargs = {} if kwargs is None else kwargs
        with self._shutdown_lock:
            self._check_accepting()
            future = self._pool.submit(self._rpc, method, args, kwargs, timeout)
        # Both forms use the same queue: direct synchronous dispatch could
        # overtake an accepted model step and reorder calls between ranks.
        return future if non_block else future.result()

    def _check_accepting(self):
        if self._shutting_down or self._shutdown_complete:
            raise RuntimeError('heterogeneous PP executor is shutting down')

    def _last_output(self, method, value):
        # Accepted pool work must still finish after shutdown closes admission.
        return self._rpc(method, (value,), {}, None)[-1]

    def execute_model(self, scheduler_output, non_block=False):
        with self._shutdown_lock:
            self._check_accepting()
            future = self._pool.submit(self._last_output, 'execute_model', scheduler_output)
        return future if non_block else future.result()

    def sample_tokens(self, grammar_output, non_block=False):
        with self._shutdown_lock:
            self._check_accepting()
            future = self._pool.submit(self._last_output, 'sample_tokens', grammar_output)
        return future if non_block else future.result()

    def check_health(self):
        self.collective_rpc('check_health')

    def shutdown(self):
        with self._shutdown_lock:
            if self._shutting_down or self._shutdown_complete:
                return
            self._shutting_down = True
            try:
                if not self._failed:
                    # generate() may return while a final bookkeeping step is
                    # still in flight. Drain accepted RPCs before group teardown.
                    self._pool.submit(lambda: None).result(timeout=min(self.timeout_s, 20))
                    if self._failed:
                        raise RuntimeError('pending PP RPC failed during shutdown')
                    refs = [w.shutdown.remote() for w in self.workers]
                    self.ray.get(refs, timeout=min(self.timeout_s, 20))
            finally:
                self._kill_owned()
                self._pool.shutdown(wait=False, cancel_futures=True)
                self._shutdown_complete = True
