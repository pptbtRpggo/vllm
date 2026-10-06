# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def calibration():
    source = (
        Path(__file__).resolve().parents[2] / "benchmarks/pp_sla_network_calibration.py"
    )
    spec = importlib.util.spec_from_file_location("network_calibration", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trace_dir(tmp_path, large):
    pairs = [(1024, 2.167)] * 24 + large
    send, recv = [], []
    for index, (size, duration) in enumerate(pairs):
        start = index * 100_000_000
        end = start + round(duration * 1e6)
        shared = dict(
            trace_session="test",
            step=index,
            clock_domain="same",
            batch_id=str(index),
            tp_size=1,
            is_warmup=False,
        )
        send.append(
            shared | dict(send_bytes=size, send_start_ns=start, send_end_ns=end)
        )
        recv.append(
            shared | dict(recv_bytes=size, recv_start_ns=start, recv_end_ns=end)
        )
    for rank, rows in ((1, send), (2, recv)):
        (tmp_path / f"pp_stage_pp{rank}_tp0.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    return tmp_path


def test_small_scheduling_time_is_not_subtracted_from_large_cost(calibration, tmp_path):
    # Large transfers follow a known 156 Gbps curve; small serving transfers
    # have 2 ms of other overhead. Mixing the regimes inflates the old estimate.
    large = [
        (mib * 1024**2, 0.02 + 8 * mib * 1024**2 / (156 * 1e6))
        for mib in (24, 32, 48, 64)
    ]
    result = calibration.calibrate(trace_dir(tmp_path, large))
    assert result["native_bandwidth_gbps"] == pytest.approx(156, rel=0.001)
    assert result["native_latency_ms"] == pytest.approx(0.02, abs=0.001)
    assert result["small_payload_median_ms"] == pytest.approx(2.167)


def test_equal_large_sizes_cannot_identify_bandwidth(calibration, tmp_path):
    with pytest.raises(ValueError, match="varied large payload"):
        calibration.calibrate(trace_dir(tmp_path, [(32 * 1024**2, 2)] * 4))


def test_negative_fitted_intercept_is_constrained(calibration, tmp_path):
    large = [(mib * 1024**2, 0.05 * mib - 0.5) for mib in (16, 32, 64)]
    result = calibration.calibrate(trace_dir(tmp_path, large))
    assert result["native_latency_ms"] == 0
    assert result["native_bandwidth_gbps"] > 0


def test_noisy_large_costs_fail_closed(calibration, tmp_path):
    large = [(mib * 1024**2, ms) for mib, ms in zip((16, 32, 48, 64), (3, 1, 4, 3))]
    with pytest.raises(ValueError, match="affine fit"):
        calibration.calibrate(trace_dir(tmp_path, large))


@pytest.mark.parametrize("size", [float("inf"), float("nan"), 0, -1, 16.5, True])
def test_invalid_payload_is_rejected(calibration, tmp_path, size):
    large = [(mib * 1024**2, mib / 16) for mib in (16, 32, 64)]
    with pytest.raises(ValueError, match="payload"):
        calibration.calibrate(trace_dir(tmp_path, large + [(size, 3)]))
