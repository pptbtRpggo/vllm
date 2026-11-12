# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ACK provenance from the profiling controller through offline EdgeShard DP."""

import json
import os

import pytest

from tests.distributed.test_pp_profile import (
    FixtureEngine,
    engine_args,
    load_profile,
    memory_profile,
    topology,
)


class AckFixtureEngine(FixtureEngine):
    def __init__(self, invalid=False):
        super().__init__()
        self.invalid = invalid

    def collective_rpc(self, method, args=(), **kwargs):
        if method == "flush_pp_profile":
            for rank, rows in enumerate(self.rows):
                for row in rows:
                    base = (rank + 1) * 10**9 + row["step"] * 10**6
                    row.update(comm_timing="ack", clock_domain=f"host-{rank // 2}")

                    def identity(src, row=row):
                        return dict(
                            session=row["trace_session"],
                            step=row["step"],
                            batch_id=row["batch_id"],
                            src_rank=src,
                            dst_rank=src + 1,
                            payload_bytes=8,
                        )

                    if rank:
                        row.update(
                            recv_start_ns=base + 1,
                            recv_end_ns=base + 101,
                            recv_ack=dict(
                                protocol=1,
                                identity=identity(rank - 1),
                                ready_sent=True,
                                done_sent=True,
                                ready_start_ns=base,
                                ready_end_ns=base + 1,
                                data_ready_ns=base + 101,
                                done_start_ns=base + 102,
                                done_end_ns=base + 110,
                            ),
                        )
                    if rank < 3:
                        row.update(
                            send_start_ns=base + 200,
                            send_end_ns=base + 300,
                            send_ack=dict(
                                protocol=1,
                                identity=identity(rank),
                                ready_received=True,
                                done_received=not self.invalid,
                                ready_start_ns=base + 120,
                                ready_end_ns=base + 150,
                                start_ns=base + 200,
                                end_ns=base + 400,
                            ),
                        )
        result = super().collective_rpc(method, args, **kwargs)
        if method == "flush_pp_profile":
            for receipt in result:
                receipt["comm_timing"] = "ack"
        return result


def collect(module, directory, engine, **extra):
    return module.profile_pp_partition(
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
        comm_timing="ack",
        **extra,
    )


def test_ack_controller_provenance_offline_dp_and_environment_restore(
    tmp_path, monkeypatch
):
    module = load_profile()
    monkeypatch.setenv("VLLM_PP_COMM_TIMING", "stale")
    engine = AckFixtureEngine()
    collect(module, tmp_path, engine)
    assert engine.closed
    assert os.environ["VLLM_PP_COMM_TIMING"] == "stale"
    manifest = json.loads((tmp_path / "profile_manifest.json").read_text())
    assert manifest["comm_timing"] == "ack"
    plan = module.profile_pp_partition(
        dump_dir=tmp_path, hetero_plan=topology(), skip_run=True
    )
    assert all(r.comm_source == "measured_serving_ack" for r in plan.rank_costs[:-1])
    with pytest.raises(ValueError, match="communication timing"):
        module.profile_pp_partition(
            dump_dir=tmp_path,
            hetero_plan=topology(),
            skip_run=True,
            comm_timing="shared-clock",
        )


@pytest.mark.parametrize(
    "engine", [AckFixtureEngine, lambda: AckFixtureEngine(True), FixtureEngine]
)
def test_ack_cannot_release_incomplete_or_wrong_mode_traces(tmp_path, engine):
    # A valid ACK fixture is the positive control; others must preserve raw but
    # cannot publish a complete collection manifest.
    module = load_profile()
    instance = engine()
    if type(instance) is AckFixtureEngine and not instance.invalid:
        collect(module, tmp_path, instance)
        assert (tmp_path / "profile_manifest.json").is_file()
    else:
        with pytest.raises(ValueError, match="measured|communication timing"):
            collect(module, tmp_path, instance)
        assert not (tmp_path / "profile_manifest.json").exists()
    assert instance.closed
