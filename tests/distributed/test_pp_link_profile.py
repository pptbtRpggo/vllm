# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch

from tests.distributed.pp_trace_fixtures import write_trace
from vllm.distributed.pp_link_profile import (
    attach_link_measurements,
    measured_transfer_ms,
    profile_worker_links,
    tensor_spec,
)


def test_replay_join_requires_run_and_layout_match(tmp_path):
    record = dict(pp_rank=0, tp_rank=0, send_service_ms=2.0)
    write_trace(tmp_path / "pp_stage_pp0_tp0.jsonl", [record])
    stale = record | dict(trace_id="another-run")
    wrong_layout = record | dict(send_tensor_spec=[dict(name="different")])
    attach_link_measurements(tmp_path, [record, stale, wrong_layout])
    assert measured_transfer_ms(record) == 2
    for invalid in [stale, wrong_layout, dict(send_transfer_ms=2, send_ms=2)]:
        with pytest.raises(ValueError, match="missing measured"):
            measured_transfer_ms(invalid)


def test_raw_endpoint_measurements_are_averaged(tmp_path):
    record = dict(pp_rank=0, tp_rank=0, send_service_ms=2.0)
    write_trace(tmp_path / "pp_stage_pp0_tp0.jsonl", [record])
    path = tmp_path / "pp_link_pp0_tp0.jsonl"
    row = json.loads(path.read_text())
    row["samples"] = [
        dict(sender_ms=1, receiver_ms=3),
        dict(sender_ms=7, receiver_ms=2),
    ]
    path.write_text(json.dumps(row) + "\n")
    attach_link_measurements(tmp_path, [record])
    assert measured_transfer_ms(record) == 5
    row["samples"][0]["sender_ms"] = float("nan")
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="invalid measured"):
        attach_link_measurements(tmp_path, [record])


def test_payload_spec_preserves_tensor_count_shape_dtype():
    tensors = dict(
        hidden_states=torch.zeros(8, 16, dtype=torch.float16),
        residual=torch.zeros(8, 16, dtype=torch.float32),
    )
    spec = tensor_spec(tensors)
    assert len(spec) == 2
    assert [s["dtype"] for s in spec] == ["float16", "float32"]
    assert spec[0]["shape"] == [8, 16]
    assert tensor_spec(dict(hidden_states=tensors["hidden_states"].t())) is None
    assert tensor_spec(dict(extra="unsupported")) is None


def _gloo_replay(rank, init_file, directory):
    import vllm.distributed.parallel_state as parallel
    from vllm.distributed.pp_hetero import PPHeteroConfig
    from vllm.distributed.pp_stage_trace import PPStageTracer

    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )

    # Use production tensor-dict transport methods with the real Gloo group.
    class Group:
        world_size = 2
        ranks = [0, 1]
        rank_in_group = rank
        cpu_group = torch.distributed.group.WORLD
        device_group = cpu_group
        use_cpu_custom_send_recv = False
        send_tensor_dict = parallel.GroupCoordinator.send_tensor_dict
        recv_tensor_dict = parallel.GroupCoordinator.recv_tensor_dict
        send_object = parallel.GroupCoordinator.send_object
        recv_object = parallel.GroupCoordinator.recv_object
        barrier = parallel.GroupCoordinator.barrier

    parallel.get_pp_group = lambda: Group()
    parallel.get_tp_group = lambda: SimpleNamespace(world_size=1)
    tracer = PPStageTracer(directory, rank, 2, torch.device("cpu"), use_cuda=False)
    tracer._records = [
        SimpleNamespace(
            is_warmup=False,
            send_tensor_spec=tensor_spec(
                dict(hidden_states=torch.zeros(4, 16), residual=torch.zeros(4, 16))
            ),
        ),
        SimpleNamespace(is_warmup=True, send_tensor_spec=None),
    ]
    worker = SimpleNamespace(
        _pp_stage_tracer=tracer, _pp_hetero=PPHeteroConfig(), device=torch.device("cpu")
    )
    try:
        profile_worker_links(worker, warmup=1, repeats=3)
    finally:
        tracer._records = []
        tracer.close()
        torch.distributed.destroy_process_group()


def test_real_two_process_transfer_without_bandwidth_config(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_PP_HETERO", raising=False)
    monkeypatch.delenv("VLLM_PP_COMM_BANDWIDTH_GBPS", raising=False)
    torch.multiprocessing.spawn(
        _gloo_replay,
        args=(str(tmp_path / "init"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    rows = (tmp_path / "pp_link_pp0_tp0.jsonl").read_text().splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["source"] == "measured_idle_replay"
    assert len(row["samples"]) == 3
    assert all(s["sender_ms"] > 0 and s["receiver_ms"] > 0 for s in row["samples"])
    assert not (tmp_path / "pp_link_pp1_tp0.jsonl").exists()
