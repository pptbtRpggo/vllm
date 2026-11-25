# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit four-stage CPU/Gloo profiling and fixed-order EdgeShard-style DP."""

import argparse

from vllm.distributed.pp_partition import format_plan
from vllm.distributed.pp_profile import (
    add_cli_args,
    format_serve_command,
    run_from_cli_args,
    validate_profile_args,
)
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.serve.utils.api_utils import VLLM_SUBCMD_PARSER_EPILOG


class PPProfileSubcommand(CLISubcommand):
    name = "pp-profile"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        plan = run_from_cli_args(args)
        if plan is not None:
            print(format_plan(plan))
            print(format_serve_command(plan, args.trace_dir))

    def validate(self, args: argparse.Namespace) -> None:
        validate_profile_args(args)

    def subparser_init(self, subparsers):
        parser = subparsers.add_parser(
            self.name,
            help="Profile CPU/Gloo PP layers and plan a fixed four-stage split.",
            description=__doc__,
        )
        parser = add_cli_args(parser)
        parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return parser


def cmd_init() -> list[CLISubcommand]:
    return [PPProfileSubcommand()]
