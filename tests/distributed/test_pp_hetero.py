# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.pp_hetero import (
    PPHeteroConfig,
    parse_hetero_spec,
    parse_scale_list,
    scale_at,
    stretch_after,
)
from vllm.pp_hetero_env import PP_ASCEND_WORKER, maybe_override_pp_worker


def test_parse_scale_list_empty():
    assert parse_scale_list(None) == ()
    assert parse_scale_list("") == ()
    assert parse_scale_list("  ,  ") == ()


def test_parse_scale_list_values():
    assert parse_scale_list("1,2,1.5") == (1.0, 2.0, 1.5)


def test_parse_scale_list_rejects_non_positive():
    with pytest.raises(ValueError, match="> 0"):
        parse_scale_list("1,0")
    with pytest.raises(ValueError, match="> 0"):
        parse_scale_list("-2")


def test_scale_at_pads_with_default():
    assert scale_at((1.0, 2.0), 0) == 1.0
    assert scale_at((1.0, 2.0), 1) == 2.0
    assert scale_at((1.0, 2.0), 2) == 1.0
    assert scale_at((1.0, 2.0), -1) == 1.0


def test_stretch_after_sleeps_extra(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(
        "vllm.distributed.pp_hetero.time.sleep", lambda s: slept.append(s)
    )
    assert stretch_after(10.0, 1.0) == 10.0
    assert slept == []
    assert stretch_after(10.0, 2.5) == pytest.approx(25.0)
    assert slept == [pytest.approx(0.015)]


def test_hetero_config_stretch_uses_rank_and_hop(monkeypatch):
    monkeypatch.setattr("vllm.distributed.pp_hetero.time.sleep", lambda s: None)
    cfg = PPHeteroConfig(
        compute_scales=(1.0, 3.0),
        comm_scales=(4.0,),
        comm_bandwidth_gbps=(8.0,),
    )
    assert cfg.enabled is True
    assert cfg.stretch_compute(0, 10.0) == 10.0
    assert cfg.stretch_compute(1, 10.0) == pytest.approx(30.0)
    assert cfg.stretch_send(0, 2.0, payload_bytes=2_000_000) == pytest.approx(8.0)
    assert cfg.stretch_recv(1, 2.0, payload_bytes=2_000_000) == pytest.approx(8.0)
    assert cfg.stretch_send(1, 2.0, payload_bytes=2_000_000) == 2.0


def test_parse_hetero_spec():
    assert parse_hetero_spec(None) == ((), ())
    assert parse_hetero_spec("1,2") == ((1.0, 2.0), ())
    assert parse_hetero_spec("1,2/4") == ((1.0, 2.0), (4.0,))
    assert parse_hetero_spec("/4") == ((), (4.0,))
    assert parse_hetero_spec("1,2/") == ((1.0, 2.0), ())


def test_from_env_reads_unified_spec(monkeypatch):
    monkeypatch.setenv("VLLM_PP_HETERO", "1,2/4")
    monkeypatch.setenv("VLLM_PP_COMM_BANDWIDTH_GBPS", "8")
    monkeypatch.setenv("VLLM_PP_COMM_LATENCY_MS", "0.5")
    cfg = PPHeteroConfig.from_env()
    assert cfg.compute_scales == (1.0, 2.0)
    assert cfg.comm_scales == (4.0,)
    assert cfg.enabled is True
    assert cfg.comm_bandwidth_gbps == (8.0,)
    assert cfg.comm_latency_ms == (0.5,)


@pytest.mark.parametrize("wait_ms", [0.0, 100.0, 300.0])
@pytest.mark.parametrize("endpoint", ["send", "recv"])
def test_comm_delay_does_not_scale_peer_wait(monkeypatch, wait_ms, endpoint):
    sleeps = []
    monkeypatch.setattr("vllm.distributed.pp_hetero.time.sleep", sleeps.append)
    cfg = PPHeteroConfig(comm_scales=(4,), comm_bandwidth_gbps=(8,))
    fn = cfg.stretch_send if endpoint == "send" else cfg.stretch_recv
    rank = 0 if endpoint == "send" else 1
    # 1 MB at 8 Gbit/s takes 1 ms, regardless of peer readiness.
    elapsed = fn(rank, wait_ms + 1, payload_bytes=1_000_000)
    assert sleeps == [pytest.approx(0.003)]
    assert elapsed == pytest.approx(wait_ms + 4)
    assert cfg.transfer_ms(0, 1_000_000) == pytest.approx(4)


def test_comm_delay_tracks_payload_and_hop(monkeypatch):
    sleeps = []
    monkeypatch.setattr("vllm.distributed.pp_hetero.time.sleep", sleeps.append)
    cfg = PPHeteroConfig(
        comm_scales=(4, 2), comm_bandwidth_gbps=(8, 4), comm_latency_ms=(0.5, 0)
    )
    assert cfg.transfer_ms(0, 1_000_000) == pytest.approx(6)
    assert cfg.transfer_ms(1, 1_000_000) == pytest.approx(4)
    assert cfg.transfer_ms(1, 2_000_000) == pytest.approx(8)
    assert cfg.transfer_ms(2, 1_000_000) is None
    assert cfg.stretch_recv(2, 102, payload_bytes=1_000_000) == pytest.approx(104)
    assert sleeps == [pytest.approx(0.002)]
    sleeps.clear()
    assert cfg.stretch_send(0, 100, payload_bytes=0) == 100
    assert sleeps == []


def test_compute_metadata_uses_effective_slowdown():
    cfg = PPHeteroConfig(compute_scales=(0.5,))
    assert cfg.compute_scale(0) == 1.0
    assert cfg.stretch_compute(0, 10) == 10


def test_live_comm_scale_requires_explicit_baseline():
    with pytest.raises(ValueError, match="VLLM_PP_COMM_BANDWIDTH_GBPS"):
        PPHeteroConfig(comm_scales=(4,))
    with pytest.raises(ValueError, match="Hop 1->2"):
        PPHeteroConfig(comm_scales=(1, 4), comm_bandwidth_gbps=(8,))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1])
def test_comm_bandwidth_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="bandwidth"):
        PPHeteroConfig(comm_bandwidth_gbps=(value,))


@pytest.mark.parametrize(
    "trigger", ["VLLM_PP_HETERO", "VLLM_PP_STAGE_TRACE", "VLLM_PP_NETWORK"]
)
def test_maybe_override_swaps_npu_worker(monkeypatch, trigger):
    monkeypatch.delenv("VLLM_PP_HETERO", raising=False)
    monkeypatch.delenv("VLLM_PP_STAGE_TRACE", raising=False)
    monkeypatch.setenv(trigger, "1,2" if trigger == "VLLM_PP_HETERO" else "/tmp/trace")
    cfg = type("PC", (), {"worker_cls": "vllm_ascend.worker.worker.NPUWorker"})()
    maybe_override_pp_worker(cfg)
    assert cfg.worker_cls == PP_ASCEND_WORKER


@pytest.mark.parametrize(
    "worker_cls",
    [
        "mock_layer_worker.MockLayerWorker",
        "my_project.CustomNPUWorker",
        "vllm_ascend.custom.Worker",
        type("CustomNPUWorker", (), {}),
        PP_ASCEND_WORKER,
        "auto",
    ],
)
def test_maybe_override_preserves_custom_or_unresolved_worker(monkeypatch, worker_cls):
    monkeypatch.setenv("VLLM_PP_STAGE_TRACE", "/tmp/trace")
    cfg = type("PC", (), {"worker_cls": worker_cls})()
    maybe_override_pp_worker(cfg)
    assert cfg.worker_cls == worker_cls


def test_maybe_override_leaves_gpu_worker(monkeypatch):
    monkeypatch.setenv("VLLM_PP_HETERO", "1,2")
    cfg = type("PC", (), {"worker_cls": "vllm.v1.worker.gpu_worker.Worker"})()
    maybe_override_pp_worker(cfg)
    assert cfg.worker_cls == "vllm.v1.worker.gpu_worker.Worker"


def test_maybe_override_noop_without_env(monkeypatch):
    monkeypatch.delenv("VLLM_PP_HETERO", raising=False)
    monkeypatch.delenv("VLLM_PP_STAGE_TRACE", raising=False)
    cfg = type("PC", (), {"worker_cls": "vllm_ascend.worker.worker.NPUWorker"})()
    maybe_override_pp_worker(cfg)
    assert cfg.worker_cls == "vllm_ascend.worker.worker.NPUWorker"


def test_network_is_symmetric_and_follows_devices_after_reordering(monkeypatch):
    import json

    monkeypatch.setenv(
        "VLLM_PP_NETWORK",
        json.dumps(
            dict(
                bandwidth_gbps=[[8, 4], [0.5], []],
                latency_ms=[[1, 2], [6], []],
            )
        ),
    )
    monkeypatch.setenv("VLLM_PP_DEVICE_ORDER", "0,2,1")
    monkeypatch.setenv("VLLM_PP_HETERO", "1,2,3")
    monkeypatch.delenv("VLLM_PP_COMM_BANDWIDTH_GBPS", raising=False)
    monkeypatch.delenv("VLLM_PP_COMM_LATENCY_MS", raising=False)
    cfg = PPHeteroConfig.from_env()
    cfg.validate_pp_size(3)
    assert cfg.enabled
    assert cfg.compute_scale(1) == 3  # stage 1 is device 2
    assert cfg.transfer_ms(0, 1_000_000) == 4  # device 0->2: 2+2 ms
    assert cfg.transfer_ms(1, 1_000_000) == 22  # device 2->1: 6+16 ms
    assert cfg.network.delay_ms(1, 2, 1_000_000) == 22  # reverse shares the same entry
    delays = []
    monkeypatch.setattr("vllm.distributed.pp_hetero.time.sleep", delays.append)
    assert cfg.stretch_send(1, 300, payload_bytes=1_000_000) == 322
    assert cfg.stretch_recv(2, 100, payload_bytes=1_000_000) == 122
    assert delays == [0.022, 0.022]  # peer waiting is never multiplied
    assert cfg.stretch_send(1, 300, payload_bytes=0) == 300
    with pytest.raises(ValueError, match="PP size"):
        cfg.validate_pp_size(2)


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"bandwidth_gbps": [[8]]},
        {"bandwidth_gbps": [[8], [8]]},
        {"bandwidth_gbps": [[0], []]},
        {"bandwidth_gbps": [[float("nan")], []]},
        {"bandwidth_gbps": [[0, 8], [8, 0]]},
        {"bandwidth_gbps": [[8], []], "latency_ms": [[-1], []]},
        {"bandwidth_gbps": [[8], []], "latency_ms": [[0]]},
        {"bandwidth_gbps": [[8], []], "unknown": 1},
    ],
)
def test_network_rejects_invalid_matrices(data):
    import json

    from vllm.distributed.pp_hetero import PPNetwork

    with pytest.raises(ValueError):
        PPNetwork.from_json(json.dumps(data))


def test_network_rejects_ambiguous_legacy_settings_and_bad_device_ids():
    from vllm.distributed.pp_hetero import PPNetwork, parse_device_order

    network = PPNetwork.from_json('{"bandwidth_gbps":[[8],[]]}')
    with pytest.raises(ValueError, match="per-hop"):
        PPHeteroConfig(network=network, comm_scales=(2,), comm_bandwidth_gbps=(8,))
    with pytest.raises(ValueError, match="outside"):
        PPHeteroConfig(network=network, device_order=(0, 2))
    with pytest.raises(ValueError, match="reordered"):
        PPHeteroConfig(device_order=(1, 0), comm_bandwidth_gbps=(8,))
    with pytest.raises(ValueError, match="distinct"):
        parse_device_order("0,0")


def test_upper_triangle_four_devices_and_default_latency():
    from vllm.distributed.pp_hetero import PPNetwork

    network = PPNetwork.from_json('{"bandwidth_gbps":[[1,2,4],[8,16],[32],[]]}')
    # Check every offset, including pairs beyond adjacent device IDs.
    for (a, b), bandwidth in {
        (0, 1): 1,
        (0, 2): 2,
        (0, 3): 4,
        (1, 2): 8,
        (1, 3): 16,
        (2, 3): 32,
    }.items():
        assert network.delay_ms(a, b, 1_000_000) == 8 / bandwidth
        assert network.delay_ms(b, a, 1_000_000) == 8 / bandwidth
