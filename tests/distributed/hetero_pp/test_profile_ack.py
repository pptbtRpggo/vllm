# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CPU/Gloo acknowledgements; no NPU or Ray runtime is initialized."""

import json
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.distributed.hetero_pp.test_channel import load_channel
from tests.distributed.hetero_pp.test_profiling import (
    TinyModel,
    load_profiling,
    schedule,
)


class Adjacent:
    def __init__(self, channel, rank):
        self.channel, self.rank = channel, rank

    def recv(self, sequence):
        return self.channel.recv(self.rank - 1, sequence)

    def send(self, values, sequence):
        return self.channel.send(values, self.rank + 1, sequence)


def exchange(rank, rendezvous, directory, scenario):
    channel_module = load_channel()
    import sys

    native_transport = sys.modules["staging_test"].CpuStagingTransport
    profiler = load_profiling()
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=8),
    )

    # The class is loaded directly from source to avoid importing NPU plugins.
    class DelayedCopy(native_transport):  # type: ignore[valid-type,misc]
        def _post(self, post, finish, buffers):
            # Delay actual destination materialization, not just recv() return.
            def delayed():
                if scenario == "delayed_copy" and rank == 2 and len(buffers) == 4:
                    time.sleep(0.05)
                finish()

            return super()._post(post, delayed, buffers)

    channel = channel_module.TensorDictChannel(
        dist.group.WORLD, "cpu", transport_cls=DelayedCopy
    )
    rows = []
    live_profiler = None
    if scenario == "profile":
        model = TinyModel()
        live_profiler = profiler.StageProfiler(
            Path(directory) / f"profile-{rank}",
            "ack-session",
            model,
            "cpu",
            rank,
            4,
            comm_timing="ack",
            timeout_s=3,
        )
        runner = NS(
            requests={}, vllm_config=NS(scheduler_config=NS(async_scheduling=False))
        )
    try:
        if scenario == "missing_done" and rank == 2:
            original_send = profiler.AckControl.send

            def drop_done(control, kind, peer, identity):
                if kind != "DONE":
                    original_send(control, kind, peer, identity)

            profiler.AckControl.send = drop_done
        for sequence in range(2):
            step = profiler._Step(
                dict(
                    trace_session="ack-session",
                    step=sequence,
                    batch_id=f"batch-{sequence}",
                    pp_rank=rank,
                )
            )
            if scenario == "wrong_batch" and rank == 2:
                step.metadata["batch_id"] = "wrong"
            link = profiler._AckTraceLink(step, Adjacent(channel, rank), 2)
            if live_profiler:
                step = live_profiler.begin(schedule(), runner, sequence)
                link = live_profiler.timed_link(step, Adjacent(channel, rank))
            if scenario == "late_peer" and rank == 2:
                time.sleep(0.15)
            if rank:
                data = link.recv(sequence)
            else:
                x = torch.arange(12).half().reshape(3, 4).T
                data = {
                    "hidden_states": x,
                    "residual": x * 2,
                    "empty": torch.empty(0),
                    "optional": None,
                }
            data["hidden_states"] += rank
            data["residual"] += rank * 2
            if live_profiler:
                live_profiler.execute(step, lambda: model(torch.ones(1, 2)))
            if rank < 3:
                assert link.send(data, sequence) == []
            else:
                expected = torch.arange(12).half().reshape(3, 4).T + 6
                assert torch.equal(data["hidden_states"], expected)
                assert torch.equal(data["residual"], expected * 2)
                assert data["empty"].numel() == 0 and data["optional"] is None
            rows.append(dict(send_ack=step.send_ack, recv_ack=step.recv_ack))
        channel.close()
        if live_profiler:
            receipt = live_profiler.flush()
            assert receipt["comm_timing"] == "ack"
            assert receipt["record_count"] == 2
            saved = [
                json.loads(s)
                for s in Path(receipt["trace_path"]).read_text().splitlines()
            ]
            if rank < 3:
                assert all(row["send_ack"]["done_received"] for row in saved)
            if rank:
                assert all(row["recv_ack"]["done_sent"] for row in saved)
            live_profiler.close()
        dist.barrier()
    except Exception as exc:
        (Path(directory) / f"error{rank}.json").write_text(json.dumps(str(exc)))
        if scenario not in ("wrong_batch", "missing_done"):
            raise
    finally:
        (Path(directory) / f"rank{rank}.json").write_text(json.dumps(rows))
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "scenario",
    ["normal", "late_peer", "delayed_copy", "wrong_batch", "missing_done", "profile"],
)
def test_four_rank_ack_uses_real_data_path_and_rejects_wrong_identity(scenario):
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            exchange, args=(f"file://{directory}/store", directory, scenario), nprocs=4
        )
        if scenario in ("wrong_batch", "missing_done"):
            errors = list(Path(directory).glob("error*.json"))
            assert errors
            expected = (
                "acknowledgement mismatch"
                if scenario == "wrong_batch"
                else "DONE acknowledgement"
            )
            assert any(expected in p.read_text() for p in errors)
            return
        rows = [
            json.loads((Path(directory) / f"rank{rank}.json").read_text())
            for rank in range(4)
        ]
        for rank in range(3):
            for step in range(2):
                sent = rows[rank][step]["send_ack"]
                received = rows[rank + 1][step]["recv_ack"]
                assert sent["identity"] == received["identity"]
                assert sent["identity"]["payload_bytes"] == 48
                assert sent["done_received"] and received["done_sent"]
                assert sent["end_ns"] >= sent["start_ns"]
                assert sent["start_ns"] >= sent["ready_end_ns"]
                assert received["done_start_ns"] >= received["data_ready_ns"]
        hop = rows[1][0]["send_ack"]
        if scenario == "late_peer":
            assert hop["ready_end_ns"] - hop["ready_start_ns"] >= 80_000_000
        if scenario == "delayed_copy":
            assert hop["end_ns"] - hop["start_ns"] >= 35_000_000


def test_control_timeout_is_explicit_and_bounded(monkeypatch):
    module = load_profiling()

    class Pending:
        def wait(self, timeout):
            assert timeout == timedelta(seconds=0.01)
            raise RuntimeError("timed out")

    monkeypatch.setattr(dist, "irecv", lambda *a, **k: Pending())
    control = module.AckControl(NS(group=object()), 0, 0.01)
    with pytest.raises(RuntimeError, match="READY.*timed out"):
        control.receive(
            "READY",
            1,
            dict(
                session="s",
                step=0,
                batch_id="b",
                src_rank=0,
                dst_rank=1,
                payload_bytes=0,
            ),
        )


def test_ack_control_buffers_stay_on_cpu_under_another_default_device(monkeypatch):
    module = load_profiling()
    identity = dict(
        session="s", step=0, batch_id="b", src_rank=0, dst_rank=1, payload_bytes=0
    )
    expected = module.AckControl.frame("READY", identity)

    def receive(frame, **kwargs):
        assert frame.device.type == "cpu"
        frame.copy_(expected)
        return NS(wait=lambda timeout: True)

    monkeypatch.setattr(dist, "irecv", receive)
    with torch.device("meta"):
        assert module.AckControl.frame("READY", identity).device.type == "cpu"
        module.AckControl(NS(group=object()), 0, 1).receive("READY", 1, identity)
