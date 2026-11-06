"""Torch-free tests of topology, admission and Ray-step orchestration contracts."""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[3]


def load(name):
    path = ROOT / 'vllm/distributed/hetero_pp' / (name + '.py')
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location('hetero_test_' + name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.policy = load('config')
        self.assertIsNotNone(self.policy, 'heterogeneous PP policy is not implemented')

    def topology(self):
        return {'stages': [
            {'node_ip': '10.0.0.1', 'soc': '910B', 'layers': 18},
            {'node_ip': '10.0.0.1', 'soc': '910B', 'layers': 18},
            {'node_ip': '10.0.0.2', 'soc': '310P', 'layers': 6},
            {'node_ip': '10.0.0.2', 'soc': '310P', 'layers': 6}]}

    def test_uneven_layers_and_two_node_layout(self):
        plan = self.policy.PipelinePlan.from_dict(self.topology(), 48)
        self.assertEqual(plan.layer_ranges, ((0, 18), (18, 36), (36, 42), (42, 48)))
        self.assertEqual(plan.partition, '18,18,6,6')
        self.assertEqual(plan.nodes, ('10.0.0.1', '10.0.0.2'))

    def test_invalid_plan_rejected_before_actor_creation(self):
        for field, value in [('layers', 0), ('layers', True), ('soc', 'guess'), ('node_ip', '')]:
            d = self.topology(); d['stages'][0][field] = value
            with self.assertRaises(ValueError): self.policy.PipelinePlan.from_dict(d, 48)
        d = self.topology(); d['stages'][0]['layers'] = 17
        with self.assertRaises(ValueError): self.policy.PipelinePlan.from_dict(d, 48)

    def test_910b_validation_mode_does_not_impersonate_310p(self):
        d = self.topology()
        for stage in d['stages']: stage['soc'] = '910B'
        with self.assertRaises(ValueError): self.policy.PipelinePlan.from_dict(d, 48)
        d['validation_mode'] = '910b_only'
        plan = self.policy.PipelinePlan.from_dict(d, 48)
        self.assertEqual([s.soc for s in plan.stages], ['910B'] * 4)
        d['stages'][3]['soc'] = '310P'
        with self.assertRaises(ValueError): self.policy.PipelinePlan.from_dict(d, 48)

    def test_worker_selected_locally_and_mismatch_refused(self):
        choose = self.policy.select_worker
        self.assertTrue(choose('910B', 'vllm_ascend.worker.worker.NPUWorker').endswith('Staged910BWorker'))
        self.assertTrue(choose('310P', 'vllm_ascend._310p.worker_310p.NPUWorker310').endswith('Staged310PWorker'))
        with self.assertRaises(ValueError): choose('310P', 'vllm_ascend.worker.worker.NPUWorker')

    def test_legacy_runtime_worker_selection_requires_actual_soc(self):
        choose = self.policy.worker_for_soc_name
        self.assertEqual(choose('Ascend910B3'), 'vllm_ascend.worker.worker.NPUWorker')
        self.assertEqual(choose('Ascend310P3'), 'vllm_ascend._310p.worker_310p.NPUWorker310')
        for name in ('unknown', 'Ascend910_9391', 'Ascend910B_guess', ''):
            with self.assertRaises(ValueError): choose(name)

    def test_admission_rejects_incompatible_runtime(self):
        cfg = NS(parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=4,
            data_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1,
            enable_expert_parallel=False, enable_dbo=False, nnodes=1, worker_extension_cls=''),
            model_config=NS(dtype='torch.float16', enforce_eager=True, quantization=None,
                runner_type='generate', is_multimodal_model=False, enable_sleep_mode=False),
            scheduler_config=NS(async_scheduling=False), speculative_config=None,
            kv_transfer_config=None, lora_config=None, additional_config={}, use_v2_model_runner=False)
        self.policy.validate_runtime(cfg, 4)
        # Ascend 0.23 inserts these defaults even with no user extras. This
        # must be accepted without accepting enabled cross-rank fusion.
        cfg.additional_config = {
            'ascend_compilation_config': {
                'fuse_norm_quant': True, 'fuse_qknorm_rope': True,
                'fuse_allreduce_rms': False, 'enable_npugraph_ex': False,
                'enable_static_kernel': False, 'fuse_muls_add': True},
            'ascend_fusion_config': {'fusion_ops_gmmswigluquant': True}}
        self.policy.validate_runtime(cfg, 4)
        cfg.additional_config['ascend_compilation_config']['fuse_allreduce_rms'] = True
        with self.assertRaises(ValueError): self.policy.validate_runtime(cfg, 4)
        cfg.additional_config = {}
        for obj, attr, value in [(cfg.parallel_config, 'tensor_parallel_size', 2),
                (cfg.scheduler_config, 'async_scheduling', True),
                (cfg.model_config, 'dtype', 'torch.bfloat16'), (cfg, 'additional_config', {'enable_flashcomm1': True}), (cfg, 'use_v2_model_runner', True)]:
            old = getattr(obj, attr); setattr(obj, attr, value)
            with self.assertRaises(ValueError): self.policy.validate_runtime(cfg, 4)
            setattr(obj, attr, old)


class StepTests(unittest.TestCase):
    def setUp(self):
        self.runtime = load('step')
        self.assertIsNotNone(self.runtime, 'pipeline step contract is not implemented')

    def test_stage_waits_for_receive_and_send_before_releasing(self):
        calls = []
        class Work:
            def wait(self): calls.append('send_complete')
        class Link:
            def recv(self, sequence): calls.append(('recv_complete', sequence)); return {'hidden_states': 4}
            def send(self, data, sequence): calls.append(('send', data, sequence)); return [Work()]
        def execute(inputs): calls.append(('compute', inputs)); return {'hidden_states': inputs['hidden_states'] + 3}
        result = self.runtime.run_stage(1, 4, 7, True, Link(), execute)
        self.assertIsNone(result)
        self.assertEqual(calls, [('recv_complete', 7), ('compute', {'hidden_states': 4}),
            ('send', {'hidden_states': 7}, 7), 'send_complete'])

    def test_no_tokens_skips_communication_but_runs_runner(self):
        def fail(*args): self.fail('empty scheduler step attempted transfer')
        result = self.runtime.run_stage(2, 4, 8, False, NS(recv=fail, send=fail), lambda inputs: 'empty')
        self.assertEqual(result, 'empty')

    def test_final_stage_returns_runner_result(self):
        result = self.runtime.run_stage(3, 4, 2, True, NS(recv=lambda seq: {'x': 9}), lambda x: ('result', x['x']))
        self.assertEqual(result, ('result', 9))

    def test_submit_all_ranks_before_wait_and_check_nonfinal_errors(self):
        events = []
        def submit(rank): events.append(('submit', rank)); return rank
        def gather(refs):
            self.assertEqual(events, [('submit', 0), ('submit', 1), ('submit', 2), ('submit', 3)])
            raise RuntimeError('middle rank failed')
        with self.assertRaises(RuntimeError): self.runtime.submit_pipeline(4, submit, gather)


if __name__ == '__main__': unittest.main()
