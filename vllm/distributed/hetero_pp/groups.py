# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actor-local group factory. No HCCL communicator is constructed.

The Ascend plugin replaces the core group constructor. Explicitly use the
core ancestor's constructor with Gloo and device communicators disabled;
keep the plugin's source and torch.distributed APIs untouched.
"""
from collections import defaultdict

import torch.distributed as dist
from vllm.distributed import parallel_state as ps
from vllm.distributed.hetero_pp.channel import TensorDictChannel

CoreGroup = next(base for base in ps.GroupCoordinator.__mro__
                 if base.__module__ == 'vllm.distributed.parallel_state'
                 and base.__name__ == 'GroupCoordinator')


class StagedGroup(CoreGroup):
    def __init__(self, group_ranks, local_rank, torch_distributed_backend,
                 use_device_communicator=False, use_message_queue_broadcaster=False,
                 group_name=None, **kwargs):
        if str(torch_distributed_backend).lower() != 'gloo':
            raise ValueError('heterogeneous groups must use Gloo')
        CoreGroup.__init__(self, group_ranks, local_rank, 'gloo', False,
                           False, group_name)
        self.channel = TensorDictChannel(self.cpu_group, self.device)
        self._tx = defaultdict(int)
        self._rx = defaultdict(int)

    def _peer(self, relative, default):
        index = default if relative is None else relative
        if not 0 <= index < self.world_size:
            raise ValueError('invalid group-relative peer')
        return self.ranks[index]

    def isend_tensor_dict(self, tensor_dict, dst=None, all_gather_group=None):
        if all_gather_group is not None and all_gather_group.world_size != 1:
            raise ValueError('TP all-gather is not supported by staged PP')
        peer = self._peer(dst, (self.rank_in_group + 1) % self.world_size)
        seq = self._tx[peer]
        works = self.channel.send(tensor_dict, peer, seq)
        self._tx[peer] += 1
        return works

    def send_tensor_dict(self, tensor_dict, dst=None, all_gather_group=None):
        for work in self.isend_tensor_dict(tensor_dict, dst, all_gather_group):
            work.wait(self.channel.timeout)
        return tensor_dict

    def recv_tensor_dict(self, src=None, all_gather_group=None):
        if all_gather_group is not None and all_gather_group.world_size != 1:
            raise ValueError('TP all-gather is not supported by staged PP')
        peer = self._peer(src, (self.rank_in_group - 1) % self.world_size)
        values = self.channel.recv(peer, self._rx[peer]); self._rx[peer] += 1
        return values

    def irecv_tensor_dict(self, src=None, all_gather_group=None):
        # Deliberately blocking first implementation: no consumer observes an
        # incomplete H2D copy. The native AsyncIntermediateTensors API is kept.
        return self.recv_tensor_dict(src, all_gather_group), [], []

    def broadcast_tensor_dict(self, tensor_dict=None, src=0, group=None, **kwargs):
        if group is not None:
            raise ValueError('external broadcast groups are not admitted')
        if self.world_size == 1:
            return tensor_dict
        if self.rank_in_group == src:
            for dst in range(self.world_size):
                if dst != src:
                    self.send_tensor_dict(tensor_dict, dst)
            return tensor_dict
        return self.recv_tensor_dict(src)

    def destroy(self):
        if hasattr(self, 'channel'):
            self.channel.close()
        # Do not invoke Ascend HCCL-specific teardown for these CPU groups.
        CoreGroup.destroy(self)


def install_group_factory():
    if ps._WORLD is not None:
        raise RuntimeError('staged group factory must be installed before WORLD init')
    ps.GroupCoordinator = StagedGroup
