# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the event probe's acceptance rules without claiming NPU coverage."""

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples/online_serving/hetero_pp/check_npu_event_timing.py"
)


def load_probe():
    assert SCRIPT.exists(), "isolated NPU event probe is not implemented"
    spec = importlib.util.spec_from_file_location("event_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rejects_reported_negative_offset_and_nonadditive_pairs():
    probe = load_probe()
    assert not probe.consistent(
        dict(total_ms=101.33099, start_ms=-9691.91, end_ms=101.3309, inner_ms=9793.24)
    )
    assert not probe.consistent(
        dict(total_ms=99.83, start_ms=93.46, end_ms=99.0, inner_ms=280.44)
    )


def test_accepts_contained_additive_interval_and_rejects_invalid_numbers():
    probe = load_probe()
    row = dict(total_ms=10.0, start_ms=2.0, end_ms=6.0, inner_ms=4.0)
    assert probe.consistent(row)
    for name in row:
        for value in (float("nan"), float("inf"), -20):
            assert not probe.consistent({**row, name: value})
