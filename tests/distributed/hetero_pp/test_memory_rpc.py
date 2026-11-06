# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Memory RPC returns msgspec-safe primitives without changing actor identity."""

import importlib.util
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING
from unittest.mock import patch

import msgspec
import torch

ROOT = Path(__file__).resolve().parents[3]


class MetadataLabel(str):
    pass


class MemoryRpcTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "vllm/distributed/hetero_pp/worker.py"
        spec = importlib.util.spec_from_file_location("memory_rpc_worker", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if TYPE_CHECKING:
            from vllm.distributed.hetero_pp.worker import StagingWorkerMixin
        else:
            StagingWorkerMixin = module.StagingWorkerMixin

        class Worker(StagingWorkerMixin):
            def _hetero_context(self):
                return nullcontext()

        self.worker = Worker()
        self.worker._hetero_profiler = None
        self.identity = {
            "torch": torch.__version__,
            "label": MetadataLabel("physical-rank-2"),
            "nested": {"optional": None, "enabled": True, "layers": (20, 27)},
        }
        self.worker._hetero_profile_identity = self.identity
        memory = ModuleType("vllm.distributed.pp_memory")
        memory.__dict__["observe_worker_memory"] = lambda _: {
            "allocated_bytes": 123,
            "includes_kv_pool": True,
            "unknown": None,
        }
        profile = ModuleType("vllm.distributed.hetero_pp.profiling")
        profile.__dict__["describe_kv_cache"] = lambda _: {"kv_cache_storage_bytes": 64}
        profile.__dict__["serving_config_observation"] = lambda _: {
            "dtype": "torch.float16",
            "quantization": None,
            "gpu_memory_utilization": 0.5,
        }
        self.addCleanup(patch.stopall)
        patch.dict(
            sys.modules,
            {
                "vllm.distributed.pp_memory": memory,
                "vllm.distributed.hetero_pp.profiling": profile,
            },
        ).start()

    def test_actual_memory_rpc_encodes_with_real_torch_version_metadata(self):
        # Real TorchVersion is a str subclass rejected by msgspec, unlike JSON.
        result = self.worker.get_pp_memory_observation()
        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded)
        self.assertEqual(decoded["profile_identity"]["torch"], str(torch.__version__))
        self.assertIs(type(result["profile_identity"]["torch"]), str)
        self.assertIs(type(result["profile_identity"]["label"]), str)
        self.assertIs(type(result["serving_config_observation"]["dtype"]), str)
        self.assertEqual(decoded["profile_identity"]["nested"]["layers"], [20, 27])
        self.assertIsNone(decoded["profile_identity"]["nested"]["optional"])
        self.assertIs(decoded["profile_identity"]["nested"]["enabled"], True)
        self.assertEqual(decoded["allocated_bytes"], 123)
        self.assertEqual(decoded["kv_cache_storage_bytes"], 64)
        self.assertIsNone(decoded["unknown"])

    def test_rpc_copy_preserves_original_identity_types_and_nested_values(self):
        result = self.worker.get_pp_memory_observation()
        self.assertIs(self.worker._hetero_profile_identity, self.identity)
        self.assertIs(self.identity["torch"], torch.__version__)
        self.assertIsInstance(self.identity["label"], MetadataLabel)
        self.assertIs(type(self.identity["nested"]["layers"]), tuple)
        self.assertIsNot(result["profile_identity"], self.identity)
        self.assertIsNot(result["profile_identity"]["nested"], self.identity["nested"])
        result["profile_identity"]["nested"]["optional"] = "changed-response"
        self.assertIsNone(self.identity["nested"]["optional"])

    def test_non_json_metadata_remains_a_failure_without_pickle_fallback(self):
        self.identity["unsupported"] = object()
        with self.assertRaises(TypeError):
            self.worker.get_pp_memory_observation()


if __name__ == "__main__":
    unittest.main()
