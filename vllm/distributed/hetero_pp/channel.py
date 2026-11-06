# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordered tensor dictionaries over a trusted, explicit CPU/Gloo group.

Metadata is sent separately from CPU-staged tensor bytes. A receive returns
only after all H2D copies finish. This initial backend permits one executing
scheduler step per actor, with no speculative or asynchronous scheduling.
"""
import math
from datetime import timedelta

import torch
import torch.distributed as dist

DTYPES = {str(t): t for t in (torch.float16, torch.float32, torch.int32,
                             torch.int64, torch.bool, torch.uint8)}


def validate_header(header, sequence):
    if not isinstance(header, dict) or header.get('version') != 1 or header.get('sequence') != sequence:
        raise ValueError('PP tensor header version/sequence mismatch')
    items = header.get('items')
    if not isinstance(items, list) or len(items) > 64:
        raise ValueError('invalid PP tensor dictionary')
    keys, bytes_total = set(), 0
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise ValueError('invalid PP metadata entry')
        key, dtype, shape = item
        if not isinstance(key, str) or key in keys:
            raise ValueError('invalid or duplicate PP tensor key')
        keys.add(key)
        if dtype is None and shape is None:
            continue
        if dtype not in DTYPES or not isinstance(shape, (list, tuple)) or len(shape) > 8:
            raise ValueError('unsupported tensor dtype/shape')
        if any(type(n) is not int or n < 0 for n in shape):
            raise ValueError('invalid tensor dimension')
        bytes_total += math.prod(shape) * torch.empty((), dtype=DTYPES[dtype]).element_size()
    if bytes_total > 8 * 1024**3:
        raise ValueError('PP message exceeds 8 GiB bound')
    return items


class TensorDictChannel:
    def __init__(self, group, device, timeout_s=120, transport_cls=None):
        if transport_cls is None:
            from vllm.distributed.device_communicators.cpu_staging import CpuStagingTransport
            transport_cls = CpuStagingTransport
        self.group = group
        self.device = device
        self.transport = transport_cls(group)
        self.timeout = timedelta(seconds=timeout_s)

    def send(self, values, dst, sequence):
        if not isinstance(values, dict):
            raise TypeError('nonfinal stage must return a tensor dictionary')
        items = []
        for key, tensor in values.items():
            if tensor is None:
                items.append((key, None, None))
            elif isinstance(tensor, torch.Tensor):
                items.append((key, str(tensor.dtype), list(tensor.shape)))
            else:
                raise TypeError('PP values must be tensors or None')
        header = {'version': 1, 'sequence': sequence, 'items': items}
        validate_header(header, sequence)
        dist.send_object_list([header], dst=dst, group=self.group)
        # Each isend snapshots its input before returning. The caller must
        # wait/drain these handles, even if the model output is then reused.
        return [self.transport.isend(values[key], dst, tag=100 + i)
                for i, (key, dtype, _) in enumerate(items) if dtype is not None]

    def recv(self, src, sequence):
        header = [None]
        dist.recv_object_list(header, src=src, group=self.group)
        items = validate_header(header[0], sequence)
        result, handles = {}, []
        for i, (key, dtype, shape) in enumerate(items):
            if dtype is None:
                result[key] = None
            else:
                result[key] = torch.empty(tuple(shape), dtype=DTYPES[dtype], device=self.device)
                handles.append(self.transport.irecv(result[key], src, tag=100 + i))
        for handle in handles:
            handle.wait(self.timeout)
        return result

    def close(self):
        self.transport.close(self.timeout)
