# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI serving entrypoint using the PP-capable executor class API."""


def configure_executor(args):
    from vllm.v1.executor.hetero_pp_executor import HeteroPPExecutor
    if args.distributed_executor_backend is not None:
        raise ValueError('this entrypoint selects HeteroPPExecutor; omit --distributed-executor-backend')
    # PP admission reads supports_pp from the class, not from a qualified-name
    # string. ParallelConfig also uses its uses_ray attribute.
    args.distributed_executor_backend = HeteroPPExecutor


def main():
    import uvloop
    # Use the same objects as api_server's own __main__; their defining
    # utility modules moved between core releases.
    from vllm.entrypoints.openai.api_server import (
        cli_env_setup, run_server, make_arg_parser, validate_parsed_serve_args,
        FlexibleArgumentParser)
    cli_env_setup()
    args = make_arg_parser(FlexibleArgumentParser()).parse_args()
    configure_executor(args)
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))


if __name__ == '__main__':
    main()
