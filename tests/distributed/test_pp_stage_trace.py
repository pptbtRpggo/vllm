# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.pp_stage_trace import (
    PPStageTracer,
    layer_range_from_runner,
    tensor_dict_nbytes,
)


def test_tensor_dict_nbytes_counts_only_tensors():
    hidden = torch.zeros(4, 8, dtype=torch.float16)
    residual = torch.ones(4, 8, dtype=torch.float16)
    nbytes = tensor_dict_nbytes(
        {
            "hidden_states": hidden,
            "residual": residual,
            "meta": "skip-me",
        }
    )
    assert nbytes == hidden.numel() * 2 + residual.numel() * 2
    assert tensor_dict_nbytes(None) == 0
    assert tensor_dict_nbytes({}) == 0


def test_layer_range_from_runner_reads_nested_model():
    runner = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(start_layer=8, end_layer=20))
    )
    assert layer_range_from_runner(runner) == (8, 20)
    assert layer_range_from_runner(SimpleNamespace()) == (None, None)


def test_wire_bytes_match_tp_slicing_and_sp_override():
    tensors = {
        "hidden_states": torch.empty(8, dtype=torch.float32),
        "residual": torch.empty(8, dtype=torch.float32),
        "odd": torch.empty(3, dtype=torch.float32),
        "meta": "not a tensor",
    }
    assert tensor_dict_nbytes(tensors) == 76
    assert tensor_dict_nbytes(tensors, all_gather_size=2) == 44
    assert tensor_dict_nbytes(
        tensors, all_gather_size=2, all_gather_tensors={"residual": False}
    ) == 60


def test_pp_stage_tracer_writes_jsonl(tmp_path):
    tracer = PPStageTracer(
        dump_dir=str(tmp_path),
        pp_rank=1,
        pp_size=2,
        device=torch.device("cpu"),
        tp_rank=0,
        use_cuda=False,
    )

    def _work() -> int:
        total = 0
        for i in range(1000):
            total += i
        return total

    result, compute_ms = tracer.measure_compute(_work)
    assert result == sum(range(1000))
    assert compute_ms >= 0.0

    _, recv_ms = tracer.measure_comm(lambda: None)
    _, send_ms = tracer.measure_comm(lambda: None)

    rec = tracer.record(
        num_tokens=32,
        num_reqs=2,
        num_ctx_requests=1,
        num_ctx_tokens=24,
        num_generation_requests=1,
        num_generation_tokens=8,
        compute_ms=compute_ms,
        recv_ms=recv_ms,
        send_ms=send_ms,
        recv_bytes=4096,
        send_bytes=4096,
        start_layer=16,
        end_layer=32,
        send_transfer_ms=4.0,
        compute_scale=2.0,
        comm_scale=4.0,
    )
    tracer.close()

    path = tmp_path / "pp_stage_pp1_tp0.jsonl"
    assert path.is_file()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["step"] == 0
    assert payload["pp_rank"] == 1
    assert payload["pp_size"] == 2
    assert payload["num_tokens"] == 32
    assert payload["num_ctx_tokens"] == 24
    assert payload["num_generation_tokens"] == 8
    assert payload["start_layer"] == 16
    assert payload["end_layer"] == 32
    assert payload["recv_bytes"] == 4096
    assert payload["send_bytes"] == 4096
    assert payload["compute_ms"] == pytest.approx(rec.compute_ms)
    assert "recv_ms" in payload and "send_ms" in payload
    assert payload["send_transfer_ms"] == 4.0
    assert payload["compute_scale"] == 2.0
    assert payload["comm_scale"] == 4.0
