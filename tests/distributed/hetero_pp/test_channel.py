"""Real four-rank CPU/Gloo pipeline; use unittest directly, no NPU initialized."""
import importlib.util
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[3]


def load_channel():
    # Load source files without requiring compiled vLLM extensions.
    for name, relative in [('staging_test', 'device_communicators/cpu_staging.py'),
                           ('channel_test', 'hetero_pp/channel.py')]:
        p = ROOT / 'vllm/distributed' / relative
        if not p.exists(): return None
        spec = importlib.util.spec_from_file_location(name, p)
        mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules['channel_test']


def roundtrip(rank, rendezvous):
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=20))
    mod = load_channel()
    channel = mod.TensorDictChannel(dist.group.WORLD, 'cpu',
        transport_cls=sys.modules['staging_test'].CpuStagingTransport)
    try:
        for seq in range(5):
            if rank == 0:
                x = torch.arange(6).reshape(2, 3).half().T + seq
                data = {'hidden_states': x, 'residual': x * 2, 'optional': None}
            else: data = channel.recv(rank - 1, seq)
            data['hidden_states'] += rank
            data['residual'] += rank * 2
            if rank < 3:
                works = channel.send(data, rank + 1, seq)
                data['hidden_states'].fill_(-99)
                for work in works: work.wait()
            else:
                want = torch.arange(6).reshape(2, 3).half().T + seq + 6
                assert torch.equal(data['hidden_states'], want)
                assert torch.equal(data['residual'], want * 2)
                assert data['optional'] is None
        # Bidirectional token return on the same CPU transport.
        if rank == 3:
            for work in channel.send({'token': torch.tensor(17)}, 0, 20): work.wait()
        elif rank == 0:
            assert channel.recv(3, 20)['token'].item() == 17
        channel.close()
        dist.barrier()
    finally: dist.destroy_process_group()


class ChannelTests(unittest.TestCase):
    def test_four_stage_activations_residuals_and_return(self):
        self.assertIsNotNone(load_channel(), 'tensor dictionary channel is not implemented')
        with tempfile.TemporaryDirectory() as d:
            mp.spawn(roundtrip, args=(f'file://{d}/store',), nprocs=4)

    def test_protocol_rejects_wrong_sequence_and_unknown_dtype(self):
        mod = load_channel()
        self.assertIsNotNone(mod, 'tensor dictionary channel is not implemented')
        with self.assertRaises(ValueError): mod.validate_header({'version': 1, 'sequence': 3, 'items': []}, 4)
        with self.assertRaises(ValueError):
            mod.validate_header({'version': 1, 'sequence': 4, 'items': [('x', 'cuda_magic', [2])]}, 4)


if __name__ == '__main__': unittest.main()
