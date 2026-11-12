# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The serving worker selector must admit the already tested device revisions."""

import importlib.util
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[3] / "vllm/distributed/hetero_pp/config.py"
SPEC = importlib.util.spec_from_file_location("soc_worker_revision_test", PATH)
assert SPEC is not None and SPEC.loader is not None
POLICY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = POLICY
SPEC.loader.exec_module(POLICY)


@pytest.mark.parametrize(
    "name,worker",
    [
        ("Ascend910B4-1", "vllm_ascend.worker.worker.NPUWorker"),
        ("Ascend310P3-1", "vllm_ascend._310p.worker_310p.NPUWorker310"),
    ],
)
def test_numeric_revision_selects_the_native_worker(name, worker):
    assert POLICY.worker_for_soc_name(name) == worker


@pytest.mark.parametrize("name", ["Ascend910B-1", "Ascend910B4--1", "Ascend310Pfoo"])
def test_invalid_revision_is_rejected(name):
    with pytest.raises(ValueError):
        POLICY.worker_for_soc_name(name)
