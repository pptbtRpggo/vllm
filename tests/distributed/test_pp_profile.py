# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.distributed.pp_profile import (
    build_profile_workload,
    prepare_trace_dir,
    profile_pp_partition,
    require_complete_traces,
    run_traced_generate,
    shutdown_llm,
)


def _write_rank_jsonl(path: Path, recs: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(rec) + "\n" for rec in recs),
        encoding="utf-8",
    )


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
        "num_reqs": 1,
        "num_ctx_requests": 0,
        "num_ctx_tokens": 0,
        "num_generation_requests": 1,
        "num_generation_tokens": 8,
        "recv_ms": None if pp_rank == 0 else 0.5,
        "compute_ms": compute_ms,
        "send_ms": send_ms,
        "recv_bytes": None if pp_rank == 0 else 4096,
        "send_bytes": 4096 if send_ms is not None else None,
    }


def _two_rank_recs(*, n_steps: int = 10) -> dict[int, list[dict]]:
    recs0 = [
        _rec(
            pp_rank=0,
            pp_size=2,
            step=i,
            start=0,
            end=16,
            compute_ms=16.0,
            send_ms=0.2,
        )
        for i in range(n_steps)
    ]
    recs1 = [
        _rec(
            pp_rank=1,
            pp_size=2,
            step=i,
            start=16,
            end=32,
            compute_ms=16.0,
            send_ms=None,
        )
        for i in range(n_steps)
    ]
    return {0: recs0, 1: recs1}


def _write_two_rank_traces(dump_dir: Path, recs: dict[int, list[dict]] | None = None):
    recs = recs or _two_rank_recs()
    for rank, rows in recs.items():
        _write_rank_jsonl(dump_dir / f"pp_stage_pp{rank}_tp0.jsonl", rows)
    return recs


class FakeLLM:
    """Stand-in for vllm.LLM that records generate() and can emit traces."""

    def __init__(
        self,
        dump_dir: str | Path,
        traces_by_rank: dict[int, list[dict]],
        *,
        write_traces: bool = True,
    ) -> None:
        self.dump_dir = Path(dump_dir)
        self.traces_by_rank = traces_by_rank
        self.write_traces = write_traces
        self.generate_calls: list[dict] = []
        self.closed = False
        self.llm_engine = SimpleNamespace(
            engine_core=SimpleNamespace(shutdown=self._shutdown)
        )

    def generate(self, prompts, sampling_params, use_tqdm=True):
        self.generate_calls.append(
            {
                "n_prompts": len(prompts),
                "input_len": len(prompts[0]["prompt_token_ids"]),
                "max_tokens": sampling_params.max_tokens,
                "ignore_eos": sampling_params.ignore_eos,
                "use_tqdm": use_tqdm,
                "prompt_token_ids": [p["prompt_token_ids"] for p in prompts],
            }
        )
        if self.write_traces:
            _write_two_rank_traces(self.dump_dir, self.traces_by_rank)
        return []

    def _shutdown(self) -> None:
        self.closed = True


def test_build_profile_workload_fixed_lengths():
    workload = build_profile_workload(
        num_prompts=4,
        input_len=8,
        output_len=16,
        seed=0,
        vocab_size=32,
    )
    assert len(workload.prompts) == 4
    assert all(len(p["prompt_token_ids"]) == 8 for p in workload.prompts)
    assert all(1 <= tok < 32 for p in workload.prompts for tok in p["prompt_token_ids"])
    assert workload.max_tokens == 16
    assert workload.ignore_eos is True


def test_build_profile_workload_is_deterministic():
    kwargs = dict(
        num_prompts=3,
        input_len=4,
        output_len=2,
        seed=7,
        vocab_size=16,
    )
    first = build_profile_workload(**kwargs)
    second = build_profile_workload(**kwargs)
    assert first.prompts == second.prompts


def test_build_profile_workload_rejects_invalid_sizes():
    with pytest.raises(ValueError, match="num_prompts"):
        build_profile_workload(num_prompts=0, input_len=8, output_len=8)
    with pytest.raises(ValueError, match="input_len"):
        build_profile_workload(num_prompts=1, input_len=0, output_len=8)
    with pytest.raises(ValueError, match="output_len"):
        build_profile_workload(num_prompts=1, input_len=8, output_len=0)
    with pytest.raises(ValueError, match="vocab_size"):
        build_profile_workload(num_prompts=1, input_len=8, output_len=8, vocab_size=1)


def test_prepare_trace_dir_creates_and_exports_env(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_PP_STAGE_TRACE", raising=False)
    dump_dir = prepare_trace_dir(tmp_path / "traces")
    assert dump_dir.is_dir()
    assert dump_dir == (tmp_path / "traces").resolve()
    assert os.environ["VLLM_PP_STAGE_TRACE"] == str(dump_dir)


def test_prepare_trace_dir_reuses_env_when_unspecified(tmp_path, monkeypatch):
    existing = tmp_path / "from_env"
    existing.mkdir()
    monkeypatch.setenv("VLLM_PP_STAGE_TRACE", str(existing))
    dump_dir = prepare_trace_dir(None)
    assert dump_dir == existing.resolve()


def test_require_complete_traces_accepts_contiguous_ranks(tmp_path):
    _write_two_rank_traces(tmp_path)
    records = require_complete_traces(tmp_path, pp_size=2)
    assert sorted(records) == [0, 1]
    assert len(records[0]) == 10


def test_require_complete_traces_missing_rank(tmp_path):
    recs = _two_rank_recs()
    _write_rank_jsonl(tmp_path / "pp_stage_pp0_tp0.jsonl", recs[0])
    with pytest.raises(FileNotFoundError, match="expected traces"):
        require_complete_traces(tmp_path, pp_size=2)


def test_require_complete_traces_empty_file(tmp_path):
    (tmp_path / "pp_stage_pp0_tp0.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "pp_stage_pp1_tp0.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty traces"):
        require_complete_traces(tmp_path, pp_size=2)


def test_run_traced_generate_feeds_prompts_and_repeats():
    workload = build_profile_workload(
        num_prompts=2,
        input_len=5,
        output_len=7,
        seed=1,
        vocab_size=32,
        num_iters=2,
        num_iters_warmup=1,
    )
    llm = FakeLLM(".", {}, write_traces=False)
    run_traced_generate(llm, workload)
    assert len(llm.generate_calls) == 3
    call = llm.generate_calls[0]
    assert call["n_prompts"] == 2
    assert call["input_len"] == 5
    assert call["max_tokens"] == 7
    assert call["ignore_eos"] is True
    assert call["use_tqdm"] is False
    assert call["prompt_token_ids"] == [p["prompt_token_ids"] for p in workload.prompts]


def test_shutdown_llm_calls_engine_core():
    llm = FakeLLM(".", {}, write_traces=False)
    shutdown_llm(llm)
    assert llm.closed is True
    shutdown_llm(SimpleNamespace())


def test_write_profile_result_json(tmp_path):
    _write_two_rank_traces(tmp_path)
    plan = profile_pp_partition(
        dump_dir=tmp_path,
        skip_run=True,
        warmup_steps=5,
        min_pp_size=2,
        max_pp_size=2,
        output_json=tmp_path / "custom_plan.json",
    )
    path = tmp_path / "custom_plan.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["VLLM_PP_LAYER_PARTITION"] == "16,16"
    assert payload["pp_size"] == 2
    assert payload["partitions"] == [16, 16]
    assert plan.env_value == "16,16"


def test_profile_skip_run_requires_trace_dir():
    with pytest.raises(ValueError, match="trace-dir"):
        profile_pp_partition(skip_run=True)


def test_profile_rejects_pp1_live_run():
    with pytest.raises(ValueError, match="pipeline_parallel_size"):
        profile_pp_partition(
            dump_dir="unused",
            engine_args=SimpleNamespace(pipeline_parallel_size=1),
            llm_factory=lambda: None,
        )


def test_profile_live_run_sets_env_feeds_and_plans(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_PP_STAGE_TRACE", raising=False)
    seen: dict[str, str | None] = {}
    traces = _two_rank_recs()
    holder: dict[str, FakeLLM] = {}

    def factory() -> FakeLLM:
        seen["env"] = os.environ.get("VLLM_PP_STAGE_TRACE")
        llm = FakeLLM(seen["env"] or tmp_path, traces)
        holder["llm"] = llm
        return llm

    plan = profile_pp_partition(
        dump_dir=tmp_path,
        llm_factory=factory,
        engine_args=SimpleNamespace(pipeline_parallel_size=2),
        num_prompts=2,
        input_len=4,
        output_len=3,
        num_iters=1,
        num_iters_warmup=1,
        warmup_steps=5,
        min_pp_size=2,
        max_pp_size=2,
        vocab_size=32,
        seed=3,
    )
    assert Path(seen["env"] or "") == tmp_path.resolve()
    llm = holder["llm"]
    assert llm.closed is True
    assert len(llm.generate_calls) == 2
    assert llm.generate_calls[0]["input_len"] == 4
    assert llm.generate_calls[0]["max_tokens"] == 3
    assert plan.env_value == "16,16"
    payload = json.loads(
        (tmp_path / "pp_partition_plan.json").read_text(encoding="utf-8")
    )
    assert payload["VLLM_PP_LAYER_PARTITION"] == "16,16"


def test_profile_fails_if_engine_writes_no_traces(tmp_path):
    llm = FakeLLM(tmp_path, {}, write_traces=False)
    with pytest.raises(FileNotFoundError):
        profile_pp_partition(
            dump_dir=tmp_path,
            llm_factory=lambda: llm,
            engine_args=SimpleNamespace(pipeline_parallel_size=2),
            num_prompts=1,
            input_len=2,
            output_len=2,
            num_iters=1,
            num_iters_warmup=0,
            vocab_size=32,
        )
    assert llm.closed is True


def test_profile_shutdown_runs_when_generate_raises(tmp_path):
    class BoomLLM(FakeLLM):
        def generate(self, *args, **kwargs):
            raise RuntimeError("generate failed")

    llm = BoomLLM(tmp_path, {}, write_traces=False)
    with pytest.raises(RuntimeError, match="generate failed"):
        profile_pp_partition(
            dump_dir=tmp_path,
            llm_factory=lambda: llm,
            engine_args=SimpleNamespace(pipeline_parallel_size=2),
            num_prompts=1,
            input_len=2,
            output_len=2,
            num_iters=1,
            num_iters_warmup=0,
            vocab_size=32,
        )
    assert llm.closed is True


def test_profile_requires_engine_when_not_skipping():
    with pytest.raises(ValueError, match="engine_args or llm_factory"):
        profile_pp_partition(dump_dir="unused", skip_run=False)
