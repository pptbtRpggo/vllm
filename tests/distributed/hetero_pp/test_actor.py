"""Compatibility with the worker wrapper constructors used by both cores."""
import importlib.util
import pathlib
import unittest

p = pathlib.Path(__file__).resolve().parents[3] / 'vllm/distributed/hetero_pp/actor.py'
spec = importlib.util.spec_from_file_location('actor_contract', p)
actor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(actor)


class ActorTests(unittest.TestCase):
    def test_wrapper_constructor_does_not_treat_config_as_rpc_rank(self):
        class CurrentWrapper:
            def __init__(self, rpc_rank=0, global_rank=None):
                self.rpc_rank, self.global_rank = rpc_rank, global_rank
        class OlderWrapper:
            def __init__(self, vllm_config, rpc_rank=0, global_rank=None):
                self.config = vllm_config
                self.rpc_rank, self.global_rank = rpc_rank, global_rank
        cfg = object()
        for cls in (CurrentWrapper, OlderWrapper):
            wrapper = actor.make_wrapper(cls, cfg, 3)
            self.assertEqual((wrapper.rpc_rank, wrapper.global_rank), (0, 3))
            if cls is OlderWrapper:
                self.assertIs(wrapper.config, cfg)


if __name__ == '__main__': unittest.main()
