# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Head profiling settings are explicit actor inputs, never daemon defaults."""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class ProfileEnvironmentTests(unittest.TestCase):
    def module(self):
        path = ROOT / "vllm/distributed/hetero_pp/config.py"
        spec = importlib.util.spec_from_file_location("profile_env_policy", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        self.assertTrue(
            hasattr(mod, "profile_actor_environment"),
            "actor profiling propagation is missing",
        )
        return mod

    def test_disabled_profile_explicitly_clears_daemon_settings(self):
        settings = self.module().profile_actor_environment(
            {"VLLM_PP_TRACE_SESSION": "stale", "VLLM_PP_COMPUTE_MODEL": "stale"}
        )
        self.assertEqual(
            settings,
            {
                "VLLM_PP_STAGE_TRACE": "",
                "VLLM_PP_TRACE_SESSION": "",
                "VLLM_PP_COMPUTE_MODEL": "",
                "VLLM_PP_COMM_TIMING": "",
            },
        )

    def test_active_profile_is_forwarded_without_unrelated_environment(self):
        env = {
            "VLLM_PP_STAGE_TRACE": "/tmp/new-profile",
            "VLLM_PP_TRACE_SESSION": "unique-session",
            "VLLM_PP_COMPUTE_MODEL": "layer-measured",
            "VLLM_PP_COMM_TIMING": "ack",
            "UNRELATED": "secret-fixture",
        }
        settings = self.module().profile_actor_environment(env)
        self.assertEqual(settings, {k: v for k, v in env.items() if k != "UNRELATED"})

    def test_partial_profile_is_rejected_before_actor_creation(self):
        for changes in [
            {"VLLM_PP_TRACE_SESSION": ""},
            {"VLLM_PP_COMPUTE_MODEL": "shape-affine"},
            {"VLLM_PP_STAGE_TRACE": "relative-path"},
            {"VLLM_PP_COMM_TIMING": "assume-synced"},
        ]:
            env = {
                "VLLM_PP_STAGE_TRACE": "/tmp/new-profile",
                "VLLM_PP_TRACE_SESSION": "unique-session",
                "VLLM_PP_COMPUTE_MODEL": "layer-measured",
                **changes,
            }
            with self.assertRaises(ValueError):
                self.module().profile_actor_environment(env)


if __name__ == "__main__":
    unittest.main()
