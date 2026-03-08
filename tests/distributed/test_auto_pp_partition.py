# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.pp_partition import allocate_heterogeneous_pp_partition


def test_allocate_heterogeneous_pp_partition_prefers_higher_score():
    partition = allocate_heterogeneous_pp_partition(
        total_num_hidden_layers=12,
        max_layers_per_stage=[12, 12],
        stage_scores=[1.0, 3.0],
    )

    assert sum(partition) == 12
    assert partition[1] > partition[0]


def test_allocate_heterogeneous_pp_partition_respects_capacity():
    partition = allocate_heterogeneous_pp_partition(
        total_num_hidden_layers=10,
        max_layers_per_stage=[3, 10],
        stage_scores=[10.0, 1.0],
    )

    assert partition == [3, 7]


def test_allocate_heterogeneous_pp_partition_requires_enough_capacity():
    with pytest.raises(ValueError, match="enough aggregate memory budget"):
        allocate_heterogeneous_pp_partition(
            total_num_hidden_layers=10,
            max_layers_per_stage=[4, 4],
            stage_scores=[1.0, 1.0],
        )
