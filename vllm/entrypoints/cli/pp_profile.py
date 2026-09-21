# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import typing

from vllm.distributed.pp_partition import format_plan
from vllm.distributed.pp_profile import (
    add_cli_args,
    format_serve_command,
    run_from_cli_args,
)
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG

if typing.TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser

DESCRIPTION = """Profile PP stage compute/communication, then choose
VLLM_PP_LAYER_PARTITION via dynamic programming.

Loads the model, feeds dummy prompts, writes pp_stage_pp*_tp*.jsonl traces,
fits per-rank layer costs, and chooses a contiguous split within supplied
per-device memory bounds. Timing-only analysis requires explicit opt-in.
"""


class PPProfileSubcommand(CLISubcommand):
    """The ``pp-profile`` subcommand for the vLLM CLI."""

    name = "pp-profile"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        plan = run_from_cli_args(args)
        if plan is not None:
            print(format_plan(plan))
            print()
            print(format_serve_command(plan))

    def validate(self, args: argparse.Namespace) -> None:
        if args.collect_only and args.skip_run:
            raise ValueError("--collect-only cannot be combined with --skip-run")
        if args.skip_run and not args.trace_dir:
            raise ValueError("--trace-dir is required with --skip-run")
        if not args.skip_run and getattr(args, "pipeline_parallel_size", 1) <= 1:
            raise ValueError(
                "pp-profile needs --pipeline-parallel-size > 1 "
                "(or pass --skip-run with existing traces)"
            )

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help="Profile PP stages and choose VLLM_PP_LAYER_PARTITION.",
            description=DESCRIPTION,
            usage=("vllm pp-profile [model_tag] --pipeline-parallel-size N [options]"),
        )
        parser = add_cli_args(parser)
        parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return parser


def cmd_init() -> list[CLISubcommand]:
    return [PPProfileSubcommand()]
