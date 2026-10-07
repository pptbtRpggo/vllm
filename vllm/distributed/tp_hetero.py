# SPDX-License-Identifier: Apache-2.0
"""Software-only TP heterogeneity for single-host Ascend experiments.

Compute delay precedes each decoder-layer all-reduce. A cross-group delay
follows each TP collective. Target-total networks reuse the PP formula and
subtract a measured native collective curve. Neither changes NPU compute
capability or HCCL's physical bandwidth.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.distributed.ascend_device_delay import AscendDeviceDelay
from vllm.distributed.pp_hetero import PPNetwork, parse_scale_list


@dataclass(frozen=True)
class TPHeteroConfig:
    compute_scales: tuple[float, ...]
    cross_group_size: int
    cross_extra_bandwidth_gbps: float | None
    cross_extra_latency_ms: float = 0.0
    cross_networks: tuple[tuple[str, PPNetwork], ...] = ()

    @classmethod
    def from_env(cls, tp_size: int) -> TPHeteroConfig:
        scales = parse_scale_list(os.getenv("VLLM_TP_COMPUTE_SCALES"))
        raw_bandwidth = os.getenv("VLLM_TP_CROSS_EXTRA_BANDWIDTH_GBPS")
        bandwidth = float(raw_bandwidth) if raw_bandwidth else None
        raw_latency = os.getenv("VLLM_TP_CROSS_EXTRA_LATENCY_MS")
        latency_ms = float(raw_latency) if raw_latency else 0.0
        raw_group_size = os.getenv("VLLM_TP_CROSS_GROUP_SIZE")
        group_size = int(raw_group_size) if raw_group_size else tp_size
        networks = ()
        raw_network = os.getenv("VLLM_TP_CROSS_NETWORK")
        if raw_network:
            data = json.loads(raw_network)
            if set(data) != {
                "tp_size",
                "cross_group_size",
                "bandwidth_gbps",
                "native_collectives",
            }:
                raise ValueError(
                    "TP target network requires topology, bandwidth "
                    "and native collectives"
                )
            if data["tp_size"] != tp_size or data["cross_group_size"] != group_size:
                raise ValueError("TP native calibration topology does not match")
            native = data["native_collectives"]
            if set(native) != {"all_reduce", "all_gather", "reduce_scatter"}:
                raise ValueError(
                    "TP native calibration must cover all three collectives"
                )
            pairs = []
            for op, curve in native.items():
                if set(curve) != {"bandwidth_gbps", "latency_ms"}:
                    raise ValueError("TP native curve requires bandwidth and latency")
                network = PPNetwork.from_json(
                    json.dumps(
                        dict(
                            mode="target_total",
                            bandwidth_gbps=[[data["bandwidth_gbps"]], []],
                            latency_ms=[[0], []],
                            native_bandwidth_gbps=[[curve["bandwidth_gbps"]], []],
                            native_latency_ms=[[curve["latency_ms"]], []],
                        )
                    )
                )
                pairs.append((op, network))
            networks = tuple(pairs)
        config = cls(scales, group_size, bandwidth, latency_ms, networks)
        config.validate(tp_size)
        return config

    def validate(self, tp_size: int) -> None:
        if tp_size < 2:
            raise ValueError("TP heterogeneity requires TP >= 2")
        if self.compute_scales and len(self.compute_scales) != tp_size:
            raise ValueError("VLLM_TP_COMPUTE_SCALES must have one value per TP rank")
        if any(scale < 1 for scale in self.compute_scales):
            raise ValueError(
                "TP compute scales must be >= 1; sleep cannot speed up compute"
            )
        if (
            not math.isfinite(self.cross_extra_latency_ms)
            or self.cross_extra_latency_ms < 0
        ):
            raise ValueError("TP cross-group extra latency must be finite and >= 0")
        if self.cross_networks:
            if (
                self.cross_extra_bandwidth_gbps is not None
                or self.cross_extra_latency_ms
            ):
                raise ValueError(
                    "TP target network cannot be combined with extra network settings"
                )
            if not 0 < self.cross_group_size < tp_size:
                raise ValueError("TP cross-group size must split the TP ranks")
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
        if self.cross_extra_bandwidth_gbps is None and not self.cross_networks:
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
        if self.cross_networks:
            for name, network in self.cross_networks:
                if name == op:
                    return network.delay_ms(
                        0, 1, self.cross_bytes(op, input_bytes, tp_size)
                    )
            raise ValueError(f"missing native TP calibration for {op}")
        if self.cross_extra_bandwidth_gbps is None:
            return self.cross_extra_latency_ms
        return self.cross_extra_latency_ms + (
            8
            * self.cross_bytes(op, input_bytes, tp_size)
            / (self.cross_extra_bandwidth_gbps * 1_000_000)
        )


@dataclass
class TPComputeStats:
    input_compute_ms: float = 0.0
    requested_delay_ms: float = 0.0
    actual_delay_ms: float = 0.0
    intervals: list[Any] = field(default_factory=list)


@dataclass
class TPCollectiveStats:
    total_ms: float = 0.0
    extra_total_ms: float = 0.0
    actual_extra_total_ms: float = 0.0
    cross_bytes_total: int = 0
    events: list[tuple[Any, Any]] = field(default_factory=list)


class TPComputeDelay:
    """Measure and stretch attention/MLP compute on the device before all-reduce.

    Buffers and factors are stable during Graph capture. Graph replay reruns
    device timing, so delay scales the current execution instead of a duration
    frozen during capture. Python hooks do not run again during replay.
    """

    def __init__(
        self,
        model: Any,
        scale: float,
        stream_delay: AscendDeviceDelay,
        measure: bool = False,
    ):
        while hasattr(model, "runnable"):
            model = model.runnable
        inner = getattr(model, "model", model)
        start, end = (
            getattr(inner, "start_layer", None),
            getattr(inner, "end_layer", None),
        )
        layers = getattr(inner, "layers", None)
        if (
            type(start) is not int
            or type(end) is not int
            or not isinstance(layers, torch.nn.ModuleList)
            or not 0 <= start < end <= len(layers)
        ):
            raise ValueError("TP compute mock requires indexed decoder layers")
        self.layers = [layers[i] for i in range(start, end)]
        if len({id(layer) for layer in self.layers}) != len(self.layers):
            raise ValueError("TP compute mock does not support shared layers")
        self.scale = scale
        self.stream_delay = stream_delay
        self.handles = []
        self.buffers = (
            [stream_delay.buffer() for _ in range(2 * len(self.layers))]
            if scale > 1 or measure
            else []
        )
        self.segment = None
        self.collectives_in_layer = 0
        self.finished_layers = 0
        self.stats = TPComputeStats()
        self.active = bool(self.buffers)

    def install(self):
        if self.handles:
            raise RuntimeError("TP compute hooks already installed")
        if not self.buffers:
            return
        for layer in self.layers:
            self.handles.append(layer.register_forward_pre_hook(self._before_layer))
            self.handles.append(layer.register_forward_hook(self._after_layer))

    def uninstall(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reset(self):
        self.segment = None
        self.collectives_in_layer = 0
        self.finished_layers = 0
        self.stats = TPComputeStats()
        self.active = bool(self.buffers)

    def _begin_segment(self):
        index = self.finished_layers * 2 + self.collectives_in_layer
        self.segment = self.stream_delay.begin_interval(self.buffers[index])

    def _before_layer(self, _module, _args):
        if not self.active:
            return
        # Dummy forwards used in warmup/capture bypass worker.execute_model.
        if self.finished_layers == len(self.layers):
            self.reset()
        if self.segment is not None or self.collectives_in_layer:
            raise RuntimeError("TP compute mock expects non-reentrant layers")
        self._begin_segment()

    def before_collective(self, op):
        if not self.active or self.segment is None:
            return
        if op != "all_reduce" or self.collectives_in_layer >= 2:
            raise RuntimeError("TP compute mock expects two layer all-reduces")
        self.stream_delay.end_interval(self.segment, factor=self.scale - 1)
        self.stats.intervals.append(self.segment)
        self.segment = None
        self.collectives_in_layer += 1

    def after_collective(self, op):
        if self.active and op == "all_reduce" and self.collectives_in_layer == 1:
            self._begin_segment()

    def _after_layer(self, _module, _args, _output):
        if not self.active:
            return
        if self.segment is not None or self.collectives_in_layer != 2:
            raise RuntimeError("TP compute mock expects two layer all-reduces")
        # No third wait after down_proj's all-reduce: Llama returns its output.
        self.collectives_in_layer = 0
        self.finished_layers += 1

    def validate(self):
        if self.active and self.finished_layers != len(self.layers):
            raise RuntimeError("TP compute mock did not observe all decoder layers")


class TPCollectiveDelay:
    """Native TP collectives followed by optional device-side link delay."""

    def __init__(self, communicator, config, tp_size, stream_delay):
        self.communicator = communicator
        self.config = config
        self.tp_size = tp_size
        self.stream_delay = stream_delay
        self.compute_delay = None
        self.stats = TPCollectiveStats()
        self.active = True
        self.trace_enabled = False
        self.counts = {"all_reduce": 0, "all_gather": 0, "reduce_scatter": 0}
        self.originals = {}

    def reset(self):
        self.stats = TPCollectiveStats()

    def install(self):
        for op in self.counts:
            original = getattr(self.communicator, op)
            self.originals[op] = original

            def wrapped(input_, *args, _op=op, _original=original, **kwargs):
                if not self.active:
                    return _original(input_, *args, **kwargs)
                if self.compute_delay is not None:
                    self.compute_delay.before_collective(_op)
                extra_ms = self.config.extra_ms(
                    _op, input_.numel() * input_.element_size(), self.tp_size
                )
                begin = self.stream_delay.event() if self.trace_enabled else None
                result = _original(input_, *args, **kwargs)
                self.stream_delay.wait_ms(extra_ms)
                if begin is not None:
                    self.stats.events.append((begin, self.stream_delay.event()))
                if self.compute_delay is not None:
                    self.compute_delay.after_collective(_op)
                self.stats.extra_total_ms += extra_ms
                self.stats.cross_bytes_total += self.config.cross_bytes(
                    _op, input_.numel() * input_.element_size(), self.tp_size
                )
                self.counts[_op] += 1
                return result

            setattr(self.communicator, op, wrapped)

    def uninstall(self):
        for op, original in self.originals.items():
            setattr(self.communicator, op, original)
        self.originals.clear()
