# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate two-host contracts without requiring Ascend hardware."""

from copy import deepcopy

import pytest


def reports():
    common = {
        "mode": "serve",
        "stages": {str(i): {"layers": 12, "npu_core_fraction": "1"} for i in range(4)},
        "cache": {"gpu_memory_utilization": 0.8},
    }
    return [
        dict(rank=i, settings=deepcopy(common), checkpoint="a" * 64) for i in range(4)
    ]


def test_agreement_accepts_normalized_configuration():
    from vllm.distributed.hetero_pp.native_startup import validate_native_reports

    validate_native_reports(reports())


@pytest.mark.parametrize(
    "field,value,expected",
    [
        (("mode",), "profile", "mode"),
        (("stages", "3", "layers"), 6, "stages.3.layers"),
        (("stages", "3", "npu_core_fraction"), "1/4", "stages.3.npu_core_fraction"),
        (("cache", "gpu_memory_utilization"), 0.5, "cache.gpu_memory_utilization"),
    ],
)
def test_agreement_rejects_different_peer_settings(field, value, expected):
    from vllm.distributed.hetero_pp.native_startup import validate_native_reports

    actual = reports()
    for report in actual[2:]:
        target = report["settings"]
        for part in field[:-1]:
            target = target[part]
        target[field[-1]] = value
    with pytest.raises(ValueError, match=expected):
        validate_native_reports(actual)


def test_agreement_rejects_different_checkpoint_content():
    from vllm.distributed.hetero_pp.native_startup import validate_native_reports

    actual = reports()
    actual[2]["checkpoint"] = actual[3]["checkpoint"] = "b" * 64
    with pytest.raises(ValueError, match="checkpoint"):
        validate_native_reports(actual)


@pytest.mark.parametrize("local_size", [1, 2])
def test_direct_pp2_requires_a_checkpoint_digest_from_each_node(local_size):
    from vllm.distributed.hetero_pp.native_startup import validate_native_reports

    actual = reports()[:2]
    for row in actual:
        row["settings"]["mode"] = "direct"
    validate_native_reports(actual, pp_size=2, local_size=local_size)
    actual[1]["checkpoint"] = None
    if local_size == 1:
        with pytest.raises(ValueError, match="rank 1.*checkpoint"):
            validate_native_reports(actual, pp_size=2, local_size=local_size)
    else:
        validate_native_reports(actual, pp_size=2, local_size=local_size)
