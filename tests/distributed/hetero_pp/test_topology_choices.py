# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit stage hardware must reach plan admission without weakening legacy modes."""

import importlib.util
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[3] / "vllm/distributed/hetero_pp/config.py"
SPEC = importlib.util.spec_from_file_location("topology_choices_test", PATH)
assert SPEC is not None and SPEC.loader is not None
POLICY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = POLICY
SPEC.loader.exec_module(POLICY)


def topology(socs):
    return {
        "validation_mode": "explicit",
        "stages": [
            {
                "node_ip": "192.0.2.10" if rank < 2 else "192.0.2.20",
                "soc": soc,
                "layers": count,
            }
            for rank, (soc, count) in enumerate(zip(socs, [6, 6, 18, 18]))
        ],
    }


@pytest.mark.parametrize(
    "socs",
    [
        ["310P", "310P", "910B", "910B"],
        ["910B"] * 4,
        ["310P"] * 4,
        ["910B", "310P", "310P", "910B"],
    ],
)
def test_explicit_plan_preserves_stage_soc_order_and_layer_ownership(socs):
    plan = POLICY.PipelinePlan.from_dict(topology(socs), 48)
    assert [s.soc for s in plan.stages] == socs
    assert plan.layer_ranges == ((0, 6), (6, 12), (12, 30), (30, 48))


@pytest.mark.parametrize(
    "change", ["unknown_soc", "single_node", "third_node", "wrong_layers"]
)
def test_explicit_plan_still_rejects_unsupported_or_inconsistent_layout(change):
    value = topology(["310P", "310P", "910B", "910B"])
    stages = value["stages"]
    if change == "unknown_soc":
        stages[0]["soc"] = "guess"
    elif change == "single_node":
        for s in stages:
            s["node_ip"] = "192.0.2.10"
    elif change == "third_node":
        stages[3]["node_ip"] = "192.0.2.30"
    else:
        stages[3]["layers"] = 17
    with pytest.raises(ValueError):
        POLICY.PipelinePlan.from_dict(value, 48)
