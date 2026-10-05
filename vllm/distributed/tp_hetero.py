# SPDX-License-Identifier: Apache-2.0
"""Software-only TP heterogeneity for single-host Ascend experiments.

Compute factors apply separately to each TP rank's decoder layers.  The
cross-group network knob adds a modeled delay after each TP collective; it
does not change HCCL's physical bandwidth or promise a target link rate.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm.distributed.pp_hetero import parse_scale_list, sync_torch_device


@dataclass(frozen=True)
class TPHeteroConfig:
    compute_scales: tuple[float, ...]
    cross_group_size: int
    cross_extra_bandwidth_gbps: float | None

    @classmethod
    def from_env(cls, tp_size: int) -> TPHeteroConfig:
        scales = parse_scale_list(os.getenv("VLLM_TP_COMPUTE_SCALES"))
        raw_bandwidth = os.getenv("VLLM_TP_CROSS_EXTRA_BANDWIDTH_GBPS")
        bandwidth = float(raw_bandwidth) if raw_bandwidth else None
        raw_group_size = os.getenv("VLLM_TP_CROSS_GROUP_SIZE")
        group_size = int(raw_group_size) if raw_group_size else tp_size
        config = cls(scales, group_size, bandwidth)
        config.validate(tp_size)
        return config

    def validate(self, tp_size: int) -> None:
        if tp_size < 2:
            raise ValueError("TP heterogeneity requires TP >= 2")
        if self.compute_scales and len(self.compute_scales) != tp_size:
            raise ValueError("VLLM_TP_COMPUTE_SCALES must have one value per TP rank")
        if any(scale < 1 for scale in self.compute_scales):
            raise ValueError("TP compute scales must be >= 1; sleep cannot speed up compute")
        if self.cross_extra_bandwidth_gbps is not None:
            if (
                not math.isfinite(self.cross_extra_bandwidth_gbps)
                or self.cross_extra_bandwidth_gbps <= 0
            ):
                raise ValueError("TP cross-group bandwidth must be finite and > 0")
            if not 0 < self.cross_group_size < tp_size:
                raise ValueError("TP cross-group size must split the TP ranks")
        elif not 0 < self.cross_group_size <= tp_size:
            raise ValueError("invalid TP cross-group size")

    def scale(self, tp_rank: int) -> float:
        return self.compute_scales[tp_rank] if self.compute_scales else 1.0

    def cross_bytes(self, op: str, input_bytes: int, tp_size: int) -> int:
        """Minimum bytes crossing a two-group cut in each direction.

        This is an idealized collective lower bound, not HCCL's actual route.
        Both groups execute the same added delay to avoid rank skew.
        """
        if self.cross_extra_bandwidth_gbps is None:
            return 0
        left = self.cross_group_size
        right = tp_size - left
        if op == "all_reduce":
            return input_bytes
        if op == "all_gather":
            return max(left, right) * input_bytes
        if op == "reduce_scatter":
            return math.ceil(max(left, right) * input_bytes / tp_size)
        raise ValueError(f"unsupported TP collective: {op}")

    def extra_ms(self, op: str, input_bytes: int, tp_size: int) -> float:
        if self.cross_extra_bandwidth_gbps is None:
            return 0.0
        return (
            8 * self.cross_bytes(op, input_bytes, tp_size)
            / (self.cross_extra_bandwidth_gbps * 1_000_000)
        )


class TPCollectiveDelay:
    """Wrap the TP communicator during forward only, tracking excluded time.

    Synchronizing before/after each collective makes this eager-only path
    intentionally conservative.  It isolates collective time from the
    per-layer compute factor and lets downstream kernels see the added delay.
    """

    def __init__(
        self,
        communicator: Any,
        config: TPHeteroConfig,
        tp_size: int,
        device: torch.device,
    ) -> None:
        self.communicator = communicator
        self.config = config
        self.tp_size = tp_size
        self.device = device
        self.total_ms = 0.0
        self.active = False
        self.counts = {"all_reduce": 0, "all_gather": 0, "reduce_scatter": 0}
        self.originals: dict[str, Callable[..., Any]] = {}

    def install(self) -> None:
        for op in self.counts:
            original = getattr(self.communicator, op)
            self.originals[op] = original

            def wrapped(input_: torch.Tensor, *args: Any,
                        _op: str = op, _original: Callable[..., Any] = original,
                        **kwargs: Any) -> Any:
                if not self.active:
                    return _original(input_, *args, **kwargs)
                sync_torch_device(self.device)
                start = time.perf_counter()
                result = _original(input_, *args, **kwargs)
                sync_torch_device(self.device)
                extra_ms = self.config.extra_ms(
                    _op, input_.numel() * input_.element_size(), self.tp_size
                )
                if extra_ms:
                    time.sleep(extra_ms / 1000)
                self.total_ms += (time.perf_counter() - start) * 1000
                self.counts[_op] += 1
                return result

            setattr(self.communicator, op, wrapped)

    def uninstall(self) -> None:
        for op, original in self.originals.items():
            setattr(self.communicator, op, original)
        self.originals.clear()
