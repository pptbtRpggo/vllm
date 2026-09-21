# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ``vllm pp-profile`` CLI subcommand."""

import json
from pathlib import Path

import pytest

from tests.distributed.pp_trace_fixtures import write_fit_profile, write_trace
from vllm.entrypoints.cli.pp_profile import PPProfileSubcommand, cmd_init
from vllm.utils.argparse_utils import FlexibleArgumentParser


def _rec(
    *,
    pp_rank: int,
    pp_size: int,
    step: int,
    start: int,
    end: int,
    compute_ms: float,
    send_ms: float | None,
) -> dict:
    return {
        "step": step,
        "ts_unix": 0.0,
        "pp_rank": pp_rank,
        "pp_size": pp_size,
        "tp_rank": 0,
        "start_layer": start,
        "end_layer": end,
        "num_tokens": 8,
        "batch_shape": [dict(query_tokens=8, context_tokens=16, prompt_tokens=0)],
        "num_reqs": 1,
        "num_ctx_requests": 0,
        "num_ctx_tokens": 0,
        "num_generation_requests": 1,
        "num_generation_tokens": 8,
        "recv_ms": None if pp_rank == 0 else 0.5,
        "compute_ms": compute_ms,
        "compute_wall_ms": compute_ms,
        "send_ms": send_ms,
        "send_transfer_ms": 99999,
        "send_service_ms": send_ms,
        "send_service_source": "measured_idle_replay",
        "recv_bytes": None if pp_rank == 0 else 4096,
        "send_bytes": 4096 if send_ms is not None else None,
    }


def _write_two_rank_traces(dump_dir: Path) -> None:
    records = {}
    for rank, start, end, send_ms in (
        (0, 0, 16, 0.2),
        (1, 16, 32, None),
    ):
        recs = [
            _rec(
                pp_rank=rank,
                pp_size=2,
                step=i,
                start=start,
                end=end,
                compute_ms=16.0,
                send_ms=send_ms,
            )
            for i in range(10)
        ]
        records[rank] = recs
        write_trace(dump_dir / f"pp_stage_pp{rank}_tp0.jsonl", recs)
    write_fit_profile(dump_dir / "fit", records)


@pytest.fixture
def pp_profile_parser():
    parser = FlexibleArgumentParser(description="test")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    PPProfileSubcommand().subparser_init(subparsers)
    return parser


def test_subcommand_name():
    assert PPProfileSubcommand().name == "pp-profile"


def test_cmd_init_returns_subcommand():
    result = cmd_init()
    assert len(result) == 1
    assert isinstance(result[0], PPProfileSubcommand)


def test_parse_skip_run_and_profile_knobs(pp_profile_parser):
    args = pp_profile_parser.parse_args(
        [
            "pp-profile",
            "--allow-unchecked-memory",
            "--skip-run",
            "--trace-dir",
            "/tmp/traces",
            "--objective",
            "latency",
            "--workload",
            "decode",
            "--input-len",
            "128",
            "--output-len",
            "16",
            "--num-prompts",
            "4",
        ]
    )
    assert args.skip_run is True
    assert args.trace_dir == "/tmp/traces"
    assert args.objective == "latency"
    assert args.workload_kind == "decode"
    assert args.input_len == 128
    assert args.output_len == 16
    assert args.num_prompts == 4


def test_parse_model_tag_and_pp_size(pp_profile_parser):
    args = pp_profile_parser.parse_args(
        [
            "pp-profile",
            "--allow-unchecked-memory",
            "facebook/opt-125m",
            "--pipeline-parallel-size",
            "4",
        ]
    )
    assert args.model_tag == "facebook/opt-125m"
    assert args.pipeline_parallel_size == 4
    assert args.skip_run is False


def test_parse_shape_profiles_and_mixed_workload(pp_profile_parser):
    args = pp_profile_parser.parse_args(
        [
            "pp-profile",
            "--skip-run",
            "--trace-dir",
            "/tmp/reference",
            "--compute-model",
            "shape-affine",
            "--workload",
            "mixed",
            "--fit-trace-dir",
            "/tmp/a",
            "--fit-trace-dir",
            "/tmp/b",
        ]
    )
    assert args.compute_model == "shape-affine"
    assert args.fit_trace_dirs == ["/tmp/a", "/tmp/b"]
    assert args.workload_kind == "mixed"


def test_validate_skip_run_requires_trace_dir(pp_profile_parser):
    args = pp_profile_parser.parse_args(
        ["pp-profile", "--allow-unchecked-memory", "--skip-run"]
    )
    with pytest.raises(ValueError, match="trace-dir"):
        PPProfileSubcommand().validate(args)


def test_validate_live_run_requires_pp_gt_1(pp_profile_parser):
    args = pp_profile_parser.parse_args(
        ["pp-profile", "--allow-unchecked-memory", "some-model"]
    )
    with pytest.raises(ValueError, match="pipeline-parallel-size"):
        PPProfileSubcommand().validate(args)


def test_cmd_skip_run_prints_partition(
    pp_profile_parser, tmp_path, capsys, monkeypatch
):
    monkeypatch.delenv("VLLM_PP_HETERO", raising=False)
    _write_two_rank_traces(tmp_path)
    args = pp_profile_parser.parse_args(
        [
            "pp-profile",
            "--allow-unchecked-memory",
            "--skip-run",
            "--trace-dir",
            str(tmp_path),
            "--fit-trace-dir",
            str(tmp_path / "fit"),
            "--comm-source",
            "replay",
            "--warmup-steps",
            "5",
            "--min-pp-size",
            "2",
            "--max-pp-size",
            "2",
        ]
    )
    PPProfileSubcommand().validate(args)
    PPProfileSubcommand.cmd(args)
    out = capsys.readouterr().out
    assert "VLLM_PP_LAYER_PARTITION=16,16" in out
    payload = json.loads(
        (tmp_path / "pp_partition_plan.json").read_text(encoding="utf-8")
    )
    assert payload["VLLM_PP_LAYER_PARTITION"] == "16,16"


def test_cmd_skip_run_hetero_env_prints_env(
    pp_profile_parser, tmp_path, capsys, monkeypatch
):
    monkeypatch.setenv("VLLM_PP_HETERO", "1,2")
    _write_two_rank_traces(tmp_path)
    args = pp_profile_parser.parse_args(
        [
            "pp-profile",
            "--allow-unchecked-memory",
            "--skip-run",
            "--trace-dir",
            str(tmp_path),
            "--fit-trace-dir",
            str(tmp_path / "fit"),
            "--comm-source",
            "replay",
            "--warmup-steps",
            "5",
            "--min-pp-size",
            "2",
            "--max-pp-size",
            "2",
        ]
    )
    PPProfileSubcommand().validate(args)
    PPProfileSubcommand.cmd(args)
    out = capsys.readouterr().out
    assert "memory feasibility was not checked" in out
    payload = json.loads(
        (tmp_path / "pp_partition_plan.json").read_text(encoding="utf-8")
    )
    assert "VLLM_PP_HETERO" not in payload
    parts = [int(x) for x in payload["VLLM_PP_LAYER_PARTITION"].split(",")]
    assert sum(parts) == 32
    assert parts == [16, 16]


def test_cli_main_registers_pp_profile():
    import inspect

    import vllm.entrypoints.cli.main as main_mod

    source = inspect.getsource(main_mod.main)
    assert "pp_profile" in source


@pytest.mark.parametrize(
    "flag,model,cost",
    [
        (None, "blocking", 16.2),
        ("--no-overlap-comm", "blocking", 16.2),
        ("--overlap-comm", "ideal_overlap", 16.0),
    ],
)
def test_cost_model_flags_reach_planner(pp_profile_parser, tmp_path, flag, model, cost):
    _write_two_rank_traces(tmp_path)
    argv = [
        "pp-profile",
        "--allow-unchecked-memory",
        "--skip-run",
        "--trace-dir",
        str(tmp_path),
        "--fit-trace-dir",
        str(tmp_path / "fit"),
        "--comm-source",
        "replay",
        "--min-pp-size",
        "2",
        "--max-pp-size",
        "2",
    ]
    if flag:
        argv.append(flag)
    args = pp_profile_parser.parse_args(argv)
    if flag == "--overlap-comm":
        with pytest.raises(ValueError, match="blocking"):
            PPProfileSubcommand.cmd(args)
        return
    PPProfileSubcommand.cmd(args)
    payload = json.loads((tmp_path / "pp_partition_plan.json").read_text())
    assert payload["cost_model"] == model
    assert payload["predicted_cost_ms"] == pytest.approx(cost)


def test_legacy_trace_cli_requires_explicit_approximation(pp_profile_parser, tmp_path):
    _write_two_rank_traces(tmp_path)
    for path in tmp_path.glob("pp_link_*.jsonl"):
        path.unlink()
    argv = [
        "pp-profile",
        "--allow-unchecked-memory",
        "--skip-run",
        "--trace-dir",
        str(tmp_path),
        "--fit-trace-dir",
        str(tmp_path / "fit"),
        "--comm-source",
        "replay",
        "--min-pp-size",
        "2",
        "--max-pp-size",
        "2",
    ]
    with pytest.raises(ValueError, match="missing measured"):
        PPProfileSubcommand.cmd(pp_profile_parser.parse_args(argv))
    argv.append("--allow-wall-time-comm")
    with pytest.raises(SystemExit):
        pp_profile_parser.parse_args(argv)
