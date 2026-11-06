"""Torch-free lifecycle tests using a stand-in Ascend dependency boundary."""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch


class WorkerTests(unittest.TestCase):
    def test_mixin_uses_staged_step_and_blocks_for_output(self):
        path = Path(__file__).resolve().parents[3] / 'vllm/distributed/hetero_pp/worker.py'
        self.assertTrue(path.exists(), 'worker adapter not implemented')
        spec = importlib.util.spec_from_file_location('worker_adapter_test', path)
        mod = importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        calls = []
        class Parent:
            def execute_model(self, scheduler_output): calls.append('wrong_worker_entry')
        class Intermediate:
            def __init__(self, values): self.tensors = values
        class Runner:
            broadcast_pp_output = False
            def execute_model(self, output, inputs):
                calls.append(inputs.tensors);return Intermediate({'hidden_states': 5})
        # Dependency injections stand in only for device/vLLM imports; the
        # worker executes its real stage, sequence and state transitions.
        class Worker(mod.StagingWorkerMixin, Parent):
            def _hetero_context(self):
                from contextlib import nullcontext
                return nullcontext()
            def _run_staged_step(self, *args):
                calls.append(('stage', args[:4]));return args[5]({'hidden_states': 2})
        w = Worker(); w.rank = 1; w.parallel_config = NS(world_size=4)
        w.model_runner = Runner();w._hetero_sequence = 0;w._hetero_channel = 'link'
        with patch.dict(sys.modules, {'vllm.sequence': NS(IntermediateTensors=Intermediate)}):
            w.execute_model(NS(total_num_scheduled_tokens=2))
        self.assertNotIn('wrong_worker_entry', calls)
        self.assertIn({'hidden_states': 2}, calls)
        self.assertEqual(w._hetero_sequence, 1)


if __name__ == '__main__': unittest.main()
