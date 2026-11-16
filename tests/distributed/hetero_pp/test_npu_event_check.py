# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the event probe's acceptance rules without claiming NPU coverage."""

import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def queued_runtime(*, invalid=False, read_error=False):
    """CPU event fixture; explicitly does not validate the NPU runtime."""
    state = SimpleNamespace(clock=0, created=0, readers=set(), queries=0)

    class Event:
        def __init__(self, **kwargs):
            state.created += 1
            self.queried = False

        def record(self, stream):
            self.value = state.clock
            state.clock += 1

        def query(self):
            state.queries += 1
            previous, self.queried = self.queried, True
            return previous

        def elapsed_time(self, other):
            state.readers.add(threading.get_ident())
            if read_error:
                raise RuntimeError("injected event read failure")
            return -100.0 if invalid else float(other.value - self.value)

    runtime = SimpleNamespace(
        npu=SimpleNamespace(Event=Event, set_device=lambda device: None),
        mm=lambda x, y: x,
    )
    return runtime, state


def test_queued_capture_reads_all_layers_on_persistent_background_thread():
    probe = load_probe()
    runtime, state = queued_runtime()
    rows = probe.queued_case(runtime, 0, object(), object(), repeats=9, depth=4)
    assert len(rows) == 9
    assert all(row["consistent"] and len(row["intervals"]) == 13 for row in rows)
    assert state.created == 9 * 28
    assert len(state.readers) == 1
    assert threading.get_ident() not in state.readers
    assert state.queries >= 18  # Each end event was initially not ready.


def test_queued_capture_preserves_invalid_intervals():
    probe = load_probe()
    runtime, _ = queued_runtime(invalid=True)
    rows = probe.queued_case(runtime, 0, object(), object(), repeats=3, depth=1)
    assert len(rows) == 3
    assert not any(row["consistent"] for row in rows)
    assert rows[0]["intervals"][0]["start_ms"] == -100.0


def test_queued_reader_exception_is_propagated_without_blocking_producer():
    probe = load_probe()
    runtime, _ = queued_runtime(read_error=True)
    with pytest.raises(RuntimeError, match="injected event read failure"):
        probe.queued_case(runtime, 0, object(), object(), repeats=9, depth=4)
