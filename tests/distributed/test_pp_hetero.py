# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.pp_hetero import (
    PPHeteroConfig,
    parse_scale_list,
    scale_at,
    stretch_after,
)
from vllm.distributed.pp_partition import RankCost, scale_rank_costs


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
    cfg = PPHeteroConfig(compute_scales=(1.0, 3.0), comm_scales=(4.0,))
    assert cfg.enabled is True
    assert cfg.stretch_compute(0, 10.0) == 10.0
    assert cfg.stretch_compute(1, 10.0) == pytest.approx(30.0)
    assert cfg.stretch_send(0, 2.0) == pytest.approx(8.0)
    assert cfg.stretch_recv(1, 2.0) == pytest.approx(8.0)
    assert cfg.stretch_send(1, 2.0) == 2.0


def test_scale_rank_costs_compute_and_comm():
    costs = [
        RankCost(0, 16, t_layer_ms=1.0, t_comm_out_ms=0.5, n_steps=8),
        RankCost(1, 16, t_layer_ms=1.0, t_comm_out_ms=None, n_steps=8),
    ]
    scaled = scale_rank_costs(
        costs, compute_scales=(1.0, 2.0), comm_scales=(3.0,)
    )
    assert scaled[0].t_layer_ms == pytest.approx(1.0)
    assert scaled[0].t_comm_out_ms == pytest.approx(1.5)
    assert scaled[1].t_layer_ms == pytest.approx(2.0)
    assert scaled[1].t_comm_out_ms is None
