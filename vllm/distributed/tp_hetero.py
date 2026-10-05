# SPDX-License-Identifier: Apache-2.0
"""Software-only TP heterogeneity for single-host Ascend experiments.

Compute delay precedes each decoder-layer all-reduce. A cross-group delay
follows each TP collective. Neither changes NPU compute capability or HCCL's
physical bandwidth.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm.distributed.pp_hetero import parse_scale_list
from vllm.distributed.tp_stream_delay import TPStreamDelay


@dataclass(frozen=True)
class TPHeteroConfig:
    compute_scales: tuple[float, ...]
    cross_group_size: int
    cross_extra_bandwidth_gbps: float | None
    cross_extra_latency_ms: float = 0.0

    @classmethod
    def from_env(cls, tp_size: int) -> TPHeteroConfig:
        scales = parse_scale_list(os.getenv("VLLM_TP_COMPUTE_SCALES"))
        raw_bandwidth = os.getenv("VLLM_TP_CROSS_EXTRA_BANDWIDTH_GBPS")
        bandwidth = float(raw_bandwidth) if raw_bandwidth else None
        raw_latency = os.getenv("VLLM_TP_CROSS_EXTRA_LATENCY_MS")
        latency_ms = float(raw_latency) if raw_latency else 0.0
        raw_group_size = os.getenv("VLLM_TP_CROSS_GROUP_SIZE")
        group_size = int(raw_group_size) if raw_group_size else tp_size
        config = cls(scales, group_size, bandwidth, latency_ms)
        config.validate(tp_size)
        return config

    def validate(self, tp_size: int) -> None:
        if tp_size < 2:
            raise ValueError("TP heterogeneity requires TP >= 2")
        if self.compute_scales and len(self.compute_scales) != tp_size:
            raise ValueError("VLLM_TP_COMPUTE_SCALES must have one value per TP rank")
        if any(scale < 1 for scale in self.compute_scales):
            raise ValueError("TP compute scales must be >= 1; sleep cannot speed up compute")
        if not math.isfinite(self.cross_extra_latency_ms) or self.cross_extra_latency_ms < 0:
            raise ValueError("TP cross-group extra latency must be finite and >= 0")
        if self.cross_extra_bandwidth_gbps is not None:
            if (
                not math.isfinite(self.cross_extra_bandwidth_gbps)
                or self.cross_extra_bandwidth_gbps <= 0
            ):
                raise ValueError("TP cross-group bandwidth must be finite and > 0")
            if not 0 < self.cross_group_size < tp_size:
                raise ValueError("TP cross-group size must split the TP ranks")
        elif self.cross_extra_latency_ms > 0:
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
            return self.cross_extra_latency_ms
        return self.cross_extra_latency_ms + (
            8 * self.cross_bytes(op, input_bytes, tp_size)
            / (self.cross_extra_bandwidth_gbps * 1_000_000)
        )


class TPComputeDelay:
    """Stretch local decoder compute before its two TP all-reduces.

CodeLlama/Llama decoder layers have one all-reduce in attention's o_proj and
one in MLP's down_proj. Stream callbacks mark the beginning and end of each
local compute segment without synchronizing the device on the caller thread.
"""

    def __init__(self, model: Any, scale: float, stream_delay: TPStreamDelay) -> None:
        inner = getattr(model, "model", model)
        start = getattr(inner, "start_layer", None)
        end = getattr(inner, "end_layer", None)
        layers = getattr(inner, "layers", None)
        if (
            type(start) is not int
            or type(end) is not int
            or not isinstance(layers, torch.nn.ModuleList)
            or not 0 <= start < end <= len(layers)
        ):
            raise ValueError("TP compute mock requires indexed decoder layers")
        self.layers = [layers[index] for index in range(start, end)]
        if len({id(layer) for layer in self.layers}) != len(self.layers):
            raise ValueError("TP compute mock does not support shared layers")
        self.scale = scale
        self.stream_delay = stream_delay
        self.handles: list[Any] = []
        self.segment: list[float] | None = None
        self.collectives_in_layer = 0
        self.input_compute_ms = 0.0
        self.requested_delay_ms = 0.0
        self.actual_delay_ms = 0.0
        self.sleep_overhead_ms = 0.0
        self.finished_layers = 0
        self.active = False

    def install(self) -> None:
        if self.handles:
            raise RuntimeError("TP compute hooks already installed")
        if self.scale <= 1:
            return
        for layer in self.layers:
            self.handles.append(layer.register_forward_pre_hook(self._before_layer))
            self.handles.append(layer.register_forward_hook(self._after_layer))

    def uninstall(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reset(self) -> None:
        self.segment = None
        self.collectives_in_layer = 0
        self.input_compute_ms = 0.0
        self.requested_delay_ms = 0.0
        self.actual_delay_ms = 0.0
        self.finished_layers = 0
        self.active = self.scale > 1

    def _begin_segment(self) -> None:
        if self.segment is not None:
            raise RuntimeError("TP compute segment already started")
        segment = [0.0]
        self.segment = segment

        def mark_start() -> None:
            segment[0] = time.perf_counter()

        self.stream_delay.enqueue(mark_start)

    def _before_layer(self, _module: Any, _args: Any) -> None:
        if not self.active:
            return
        if self.segment is not None or self.collectives_in_layer != 0:
            raise RuntimeError("TP compute mock expects non-reentrant layers")
        self._begin_segment()

    def before_collective(self, op: str) -> None:
        if not self.active or self.segment is None:
            return
        if op != "all_reduce" or self.collectives_in_layer >= 2:
            raise RuntimeError("TP compute mock expects two layer all-reduces")
        self._finish_segment()
        self.collectives_in_layer += 1

    def _finish_segment(self) -> None:
        if self.segment is None:
            raise RuntimeError("TP compute segment was not started")
        segment = self.segment
        self.segment = None

        def stretch_segment() -> None:
            base_ms = (time.perf_counter() - segment[0]) * 1000
            requested_ms = base_ms * (self.scale - 1)
            sleep_ms = max(0.0, requested_ms - self.sleep_overhead_ms)
            started = time.perf_counter()
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000)
            actual_ms = (time.perf_counter() - started) * 1000
            if sleep_ms > 0:
                overhead_ms = max(0.0, actual_ms - sleep_ms)
                self.sleep_overhead_ms = (
                    0.75 * self.sleep_overhead_ms + 0.25 * overhead_ms
                )
            self.input_compute_ms += base_ms
            self.requested_delay_ms += requested_ms
            self.actual_delay_ms += actual_ms

        self.stream_delay.enqueue(stretch_segment)

    def after_collective(self, op: str) -> None:
        if self.active and op == "all_reduce" and self.collectives_in_layer in (1, 2):
            self._begin_segment()

    def _after_layer(self, _module: Any, _args: Any, _output: Any) -> None:
        if not self.active:
            return
        if self.segment is None or self.collectives_in_layer != 2:
            raise RuntimeError("TP compute mock expects two layer all-reduces")
        self._finish_segment()
        self.collectives_in_layer = 0
        self.finished_layers += 1

    def validate(self) -> None:
        if self.active and self.finished_layers != len(self.layers):
            raise RuntimeError("TP compute mock did not observe all decoder layers")


class TPCollectiveDelay:
    """Wrap TP collectives, optionally adding stream-ordered link delay."""

    def __init__(
        self,
        communicator: Any,
        config: TPHeteroConfig,
        tp_size: int,
        stream_delay: TPStreamDelay,
    ) -> None:
        self.communicator = communicator
        self.config = config
        self.tp_size = tp_size
        self.stream_delay = stream_delay
        self.compute_delay: TPComputeDelay | None = None
        self.total_ms = 0.0
        self.extra_total_ms = 0.0
        self.actual_extra_total_ms = 0.0
        self.sleep_overhead_ms = 0.0
        self.cross_bytes_total = 0
        self.active = False
        self.counts = {"all_reduce": 0, "all_gather": 0, "reduce_scatter": 0}
        self.originals: dict[str, Callable[..., Any]] = {}

    def install(self) -> None:
        for op in self.counts:
            original = getattr(self.communicator, op)
            self.originals[op] = original

            def wrapped(
                input_: torch.Tensor,
                *args: Any,
                _op: str = op,
                _original: Callable[..., Any] = original,
                **kwargs: Any,
            ) -> Any:
                if not self.active:
                    return _original(input_, *args, **kwargs)
                if self.compute_delay is not None:
                    self.compute_delay.before_collective(_op)
                extra_ms = self.config.extra_ms(
                    _op, input_.numel() * input_.element_size(), self.tp_size
                )
                if extra_ms:
                    started = [0.0]
                    self.stream_delay.enqueue(
                        lambda: started.__setitem__(0, time.perf_counter())
                    )
                result = _original(input_, *args, **kwargs)
                if extra_ms:

                    def stretch_collective() -> None:
                        elapsed_ms = (time.perf_counter() - started[0]) * 1000
                        sleep_ms = max(0.0, extra_ms - self.sleep_overhead_ms)
                        sleep_started = time.perf_counter()
                        if sleep_ms > 0:
                            time.sleep(sleep_ms / 1000)
                        actual_ms = (time.perf_counter() - sleep_started) * 1000
                        if sleep_ms > 0:
                            overhead_ms = max(0.0, actual_ms - sleep_ms)
                            self.sleep_overhead_ms = (
                                0.75 * self.sleep_overhead_ms + 0.25 * overhead_ms
                            )
                        self.actual_extra_total_ms += actual_ms
                        self.total_ms += elapsed_ms + actual_ms

                    self.stream_delay.enqueue(stretch_collective)
                if self.compute_delay is not None:
                    self.compute_delay.after_collective(_op)
                self.extra_total_ms += extra_ms
                self.cross_bytes_total += self.config.cross_bytes(
                    _op, input_.numel() * input_.element_size(), self.tp_size
                )
                self.counts[_op] += 1
                return result

            setattr(self.communicator, op, wrapped)

    def uninstall(self) -> None:
        for op, original in self.originals.items():
            setattr(self.communicator, op, original)
        self.originals.clear()
