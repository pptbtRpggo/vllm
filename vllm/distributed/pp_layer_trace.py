# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct decoder-layer timing during eager serving, without per-layer sync."""

from __future__ import annotations

import time
from contextlib import contextmanager
from functools import partial
from typing import Any

import torch


class PPLayerTimer:
    """Hook only local top-level decoder layers, not their nested modules.

    Events use the current device stream. Resolve after the stage's existing
    synchronization; synchronizing inside each hook would change execution.
    CPU timing exists for CPU execution/tests. CUDA Graph/compiled execution,
    reentrant layers and forwards that skip layers are intentionally rejected.
    """

    def __init__(self, model: Any, device: torch.device) -> None:
        inner = getattr(model, "model", model)
        start, end = (
            getattr(inner, "start_layer", None),
            getattr(inner, "end_layer", None),
        )
        layers = getattr(inner, "layers", None)
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or end <= start
            or not isinstance(layers, torch.nn.ModuleList)
            or len(layers) < end
        ):
            raise ValueError(
                "layer-measured needs model[.model].layers indexed by global "
                "decoder layer IDs, with start_layer/end_layer"
            )
        self.layers = {i: layers[i] for i in range(start, end)}
        if len({id(m) for m in self.layers.values()}) != len(self.layers):
            raise ValueError("layer-measured does not support shared decoder modules")
        # ParallelLMHead is normally used through logits_processor, which
        # calls its quantization method directly (not lm_head.forward).
        candidates = {
            "embedding": getattr(inner, "embed_tokens", None),
            "final_norm": getattr(inner, "norm", None),
            "lm_head": getattr(model, "logits_processor", None),
        }
        self.endpoints = {
            name: module
            for name, module in candidates.items()
            if isinstance(module, torch.nn.Module)
            and type(module).__name__ != "PPMissingLayer"
        }
        self.device = device
        self.backend = getattr(torch, device.type) if device.type != "cpu" else None
        if self.backend is not None and not hasattr(self.backend, "Event"):
            raise ValueError(f"layer-measured requires {device.type} timing events")
        self.events: dict[int | str, tuple[Any, Any]] = {}
        self.finished: list[int] = []
        self.device_events = (
            {
                i: (
                    self.backend.Event(enable_timing=True),
                    self.backend.Event(enable_timing=True),
                )
                for i in (*self.layers, *self.endpoints)
            }
            if self.backend is not None
            else {}
        )

    def _mark(self, index, endpoint):
        if self.backend is None:
            return time.perf_counter()
        event = self.device_events[index][endpoint]
        event.record()
        return event

    @contextmanager
    def capture(self):
        self.events.clear()
        self.finished.clear()
        handles = []

        def before(index, _module, _args):
            if index in self.events:
                raise ValueError("layer-measured expects one call per decoder layer")
            self.events[index] = (self._mark(index, 0), None)

        def after(index, _module, _args, _output):
            begin, _ = self.events[index]
            self.events[index] = (begin, self._mark(index, 1))
            if isinstance(index, int):
                self.finished.append(index)

        try:
            for index, module in {**self.layers, **self.endpoints}.items():
                handles.append(module.register_forward_pre_hook(partial(before, index)))
                handles.append(module.register_forward_hook(partial(after, index)))
            yield
            if self.finished != list(self.layers):
                raise ValueError(
                    "layer-measured did not observe every decoder layer in order; "
                    "use eager execution without graph replay or compilation"
                )
        finally:
            for handle in handles:
                handle.remove()

    def elapsed_ms(self) -> dict[str, float]:
        """Read after the caller synchronizes the stage's device work."""
        return {
            str(index): float(start.elapsed_time(end))
            if self.backend is not None
            else (end - start) * 1000
            for index, (start, end) in self.events.items()
            if isinstance(index, int)
        }

    def endpoint_elapsed_ms(self) -> dict[str, float | None]:
        """None means not executed; zero must not disguise absent measurement."""
        result = {}
        for name in ("embedding", "lm_head", "final_norm"):
            pair = self.events.get(name)
            result[name + "_ms"] = (
                (
                    float(pair[0].elapsed_time(pair[1]))
                    if self.backend is not None
                    else (pair[1] - pair[0]) * 1000
                )
                if pair is not None
                else None
            )
        return result
