# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import OPTConfig


@pytest.fixture(scope="session", autouse=True)
def local_tau_test_model(tmp_path_factory):
    """Scheduler probes need architecture metadata, never model downloads."""
    from tests.v1.core.tau_batch import test_scheduler

    path = tmp_path_factory.mktemp("tau-opt-config")
    OPTConfig(
        hidden_size=768,
        ffn_dim=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=8192,
        dtype="float16",
        architectures=["OPTForCausalLM"],
    ).save_pretrained(path)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(test_scheduler, "TEST_MODEL", str(path))
        yield
