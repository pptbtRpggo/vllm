# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured-pool conservation and conservative candidate memory envelopes."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_builder():
    path = ROOT / "vllm/distributed/hetero_pp/profile_memory.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("profile_memory_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def config():
    return NS(
        model_config=NS(
            model="local-model",
            revision=None,
            dtype="torch.float16",
            quantization=None,
            get_total_num_hidden_layers=lambda: 4,
            max_model_len=8,
            hf_config=NS(architectures=["LlamaForCausalLM"]),
        ),
        cache_config=NS(
            cache_dtype="auto",
            block_size=4,
            num_gpu_blocks_override=8,
            kv_cache_memory_bytes=None,
            gpu_memory_utilization=0.5,
        ),
        scheduler_config=NS(max_num_seqs=2, max_num_batched_tokens=4),
        parallel_config=NS(pipeline_parallel_size=4, tensor_parallel_size=1),
    )


def observations():
    result = []
    for rank, endpoint in enumerate([20, 0, 0, 30]):
        result.append(
            dict(
                pp_rank=rank,
                tp_rank=0,
                layer_storage_bytes={str(rank): 100},
                non_layer_storage_bytes=endpoint,
                allocated_bytes=278 + endpoint,
                reserved_bytes=328 + endpoint,
                peak_allocated_bytes=328 + endpoint,
                peak_reserved_bytes=428 + endpoint,
                free_bytes=9472 - endpoint,
                total_bytes=10000,
                includes_kv_pool=True,
                kv_cache_storage_bytes=128,
                num_gpu_blocks=8,
                kv_cache_specs={
                    f"model.layers.{rank}.self_attn.attn": {
                        "page_size_bytes": 16,
                        "block_size": 4,
                        "dtype": "torch.float16",
                        "spec_type": "FullAttentionSpec",
                    }
                },
            )
        )
    return result


class MemoryBuilderTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_builder()
        self.assertIsNotNone(self.mod, "profile memory builder is missing")

    def test_actual_kv_pool_is_not_charged_twice(self):
        profile = self.mod.build_fixed_block_memory_profile(
            observations(), config(), safety_margin_bytes=64
        )
        self.assertEqual(profile.to_dict()["version"], 2)
        self.assertEqual(profile.devices[0].runtime_bytes, 400)
        self.assertEqual(profile.devices[0].layer_weights_bytes, (100,) * 4)
        self.assertEqual(profile.devices[0].layer_kv_bytes, (128,) * 4)
        self.assertEqual(profile.devices[0].estimate(0, 1, 4)["required_bytes"], 712)
        profile.validate_engine_config(config())

    def test_pool_or_blocks_mismatch_is_rejected(self):
        for key, value in [
            ("kv_cache_storage_bytes", 129),
            ("num_gpu_blocks", 9),
            ("includes_kv_pool", False),
        ]:
            data = observations()
            data[0][key] = value
            with self.assertRaises(ValueError):
                self.mod.build_fixed_block_memory_profile(data, config())

    def test_missing_duplicate_rank_and_layer_coverage_are_rejected(self):
        for data in [observations()[:-1], observations()[:3] + [observations()[0]]]:
            with self.assertRaises(ValueError):
                self.mod.build_fixed_block_memory_profile(data, config())
        data = observations()
        data[3]["layer_storage_bytes"] = {"9": 100}
        with self.assertRaises(ValueError):
            self.mod.build_fixed_block_memory_profile(data, config())

    def test_unknown_cache_geometry_and_fixed_bytes_are_rejected(self):
        data = observations()
        data[0]["kv_cache_specs"] = {}
        with self.assertRaises(ValueError):
            self.mod.build_fixed_block_memory_profile(data, config())
        cfg = config()
        cfg.cache_config.kv_cache_memory_bytes = 2048
        with self.assertRaises(ValueError):
            self.mod.build_fixed_block_memory_profile(observations(), cfg)

    def test_too_small_budget_rejects_even_the_profiled_layout(self):
        data = observations()
        for row in data:
            row["total_bytes"] = 1000
            row["free_bytes"] = 472
        with self.assertRaises(ValueError):
            self.mod.build_fixed_block_memory_profile(
                data, config(), safety_margin_bytes=1000
            )


if __name__ == "__main__":
    unittest.main()
