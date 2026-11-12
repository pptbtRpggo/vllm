# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual v0.23 parser admission, not source-text assertions."""

import importlib.util
import json
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


class ProfileCLITests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "vllm/entrypoints/cli/pp_profile.py"
        self.assertTrue(path.exists(), "v0.23 pp-profile CLI is not implemented")
        spec = importlib.util.spec_from_file_location("profile_cli_under_test", path)
        assert spec is not None and spec.loader is not None
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        # Parser validation uses real EngineArgs, but needs an explicit platform
        # in a source checkout without installed vLLM build metadata.
        import vllm.platforms
        from vllm.platforms.cpu import CpuPlatform

        platform = patch.object(vllm.platforms, "_current_platform", CpuPlatform())
        platform.start()
        self.addCleanup(platform.stop)
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        parser = FlexibleArgumentParser()
        sub = parser.add_subparsers(dest="subparser")
        self.mod.PPProfileSubcommand().subparser_init(sub)
        self.parser = parser

    def test_live_parser_requires_explicit_new_backend_plan(self):
        args = self.parser.parse_args(
            ["pp-profile", "fixture-model", "--enforce-eager", "--no-async-scheduling"]
        )
        with self.assertRaisesRegex(ValueError, "hetero.plan"):
            self.mod.PPProfileSubcommand().validate(args)

    def test_ack_communication_timing_is_explicit(self):
        args = self.parser.parse_args(
            [
                "pp-profile",
                "fixture-model",
                "--hetero-plan",
                "plan.json",
                "--enforce-eager",
                "--no-async-scheduling",
                "--comm-timing",
                "ack",
            ]
        )
        self.mod.PPProfileSubcommand().validate(args)
        self.assertEqual(args.comm_timing, "ack")

    def test_defaults_are_fixed_order_layer_measured_latency(self):
        args = self.parser.parse_args(
            [
                "pp-profile",
                "--hetero-plan",
                "plan.json",
                "--model",
                "fixture-model",
                "--enforce-eager",
                "--no-async-scheduling",
            ]
        )
        self.mod.PPProfileSubcommand().validate(args)
        self.assertEqual(
            (
                args.compute_model,
                args.objective,
                args.pipeline_parallel_size,
                args.tensor_parallel_size,
            ),
            ("layer-measured", "latency", 4, 1),
        )

    def test_changed_parallelism_async_and_missing_trace_are_rejected(self):
        cases = [
            (["--tensor-parallel-size", "2"], "TP"),
            (["--pipeline-parallel-size", "2"], "PP"),
            (["--async-scheduling"], "sync"),
            (["--skip-run"], "trace.dir"),
        ]
        for extra, message in cases:
            args = self.parser.parse_args(
                [
                    "pp-profile",
                    "--hetero-plan",
                    "plan.json",
                    "--model",
                    "fixture-model",
                    "--enforce-eager",
                    "--no-async-scheduling",
                    *extra,
                ]
            )
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                self.mod.PPProfileSubcommand().validate(args)

    def test_actual_engine_args_defaults_pass_profile_admission(self):
        from vllm.distributed.pp_profile import _configure_engine
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.executor.hetero_pp_executor import HeteroPPExecutor

        parsed = self.parser.parse_args(
            [
                "pp-profile",
                "--hetero-plan",
                "plan.json",
                "--model",
                str(ROOT),
                "--enforce-eager",
                "--no-async-scheduling",
            ]
        )
        args = EngineArgs.from_cli_args(parsed)
        self.assertTrue(args.disable_cascade_attn)
        selected = _configure_engine(args)
        self.assertIs(selected.distributed_executor_backend, HeteroPPExecutor)

    def test_generated_candidate_passes_actual_serve_parser_and_executor_selection(
        self,
    ):
        from tests.distributed.test_pp_profile import (
            FixtureEngine,
            engine_args,
            memory_profile,
            topology,
        )
        from vllm.distributed.hetero_pp.serve import configure_executor
        from vllm.distributed.pp_profile import (
            format_serve_command,
            profile_pp_partition,
        )
        from vllm.entrypoints.openai.api_server import (
            FlexibleArgumentParser,
            make_arg_parser,
        )
        from vllm.v1.executor.hetero_pp_executor import HeteroPPExecutor

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "trace"
            engine = FixtureEngine(expose_executor=False)
            profile_pp_partition(
                dump_dir=directory,
                hetero_plan=topology(),
                engine_args=engine_args(),
                llm_factory=lambda: engine,
                memory_profile=memory_profile(),
                collect_only=True,
                num_prompts=1,
                input_len=2,
                output_len=2,
                num_iters=1,
                num_iters_warmup=1,
            )
            plan = profile_pp_partition(
                dump_dir=directory, hetero_plan=topology(), skip_run=True
            )
            words = shlex.split(format_serve_command(plan, directory).splitlines()[-1])
            index = words.index("-m")
            # Parse the generated actual invocation; do not just grep prose.
            self.assertEqual(words[index + 1], "vllm.distributed.hetero_pp.serve")
            namespace = make_arg_parser(FlexibleArgumentParser()).parse_args(
                words[index + 2 :]
            )
            configure_executor(namespace)
            self.assertIs(namespace.distributed_executor_backend, HeteroPPExecutor)
            self.assertFalse(namespace.async_scheduling)
            self.assertFalse(namespace.enable_prefix_caching)
            self.assertFalse(namespace.disable_cascade_attn)
            assignment = next(w for w in words if w.startswith("VLLM_HETERO_PP_PLAN="))
            selected = json.loads(assignment.split("=", 1)[1])
            self.assertEqual([s["layers"] for s in selected["stages"]], plan.partitions)
            self.assertEqual(
                [s["node_ip"] for s in selected["stages"]],
                [s["node_ip"] for s in topology()["stages"]],
            )


if __name__ == "__main__":
    unittest.main()
