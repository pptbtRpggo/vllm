# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collect real files from separate node directories through the RPC boundary."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = (
    Path(__file__).resolve().parents[3] / "vllm/distributed/hetero_pp/trace_export.py"
)


def load_export():
    assert SOURCE.exists(), "remote profile collection is not implemented"
    spec = importlib.util.spec_from_file_location("trace_export_test", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("corrupt", [False, True])
def test_master_collects_peer_raw_and_manifests_without_shared_storage(
    tmp_path, corrupt
):
    app = load_export()
    master, peer = tmp_path / "master", tmp_path / "peer"
    master.mkdir()
    peer.mkdir()
    receipts, profilers = [], []
    for rank in range(4):
        directory = master if rank < 2 else peer
        path = directory / f"pp_stage_pp{rank}_tp0.jsonl"
        content = (
            json.dumps({"pp_rank": rank, "trace_session": "run-1"}) + "\n"
        ).encode()
        path.write_bytes(content)
        receipt = dict(
            rank=rank,
            trace_session="run-1",
            complete=True,
            trace_sha256=hashlib.sha256(content).hexdigest(),
        )
        manifest = directory / f"pp_manifest_pp{rank}_tp0.json"
        manifest.write_text(json.dumps(receipt))
        profilers.append(
            SimpleNamespace(
                path=path, manifest_path=manifest, flush=lambda r=receipt: r
            )
        )
        receipts.append(receipt)

    class LLM:
        def collective_rpc(self, method, args=()):
            assert method == "export_pp_profile"
            target = args[0]
            results = [
                app.export_profile(p) if rank == target else None
                for rank, p in enumerate(profilers)
            ]
            if corrupt and target == 2:
                reply = results[2]
                assert isinstance(reply, dict)
                reply["trace_base64"] = "Y29ycnVwdA=="
            return results

    if corrupt:
        with pytest.raises(ValueError, match="SHA"):
            app.collect_profile_traces(LLM(), master, receipts)
        assert not (master / "pp_stage_pp2_tp0.jsonl").exists()
    else:
        app.collect_profile_traces(LLM(), master, receipts)
        for rank in (2, 3):
            for name in (
                f"pp_stage_pp{rank}_tp0.jsonl",
                f"pp_manifest_pp{rank}_tp0.json",
            ):
                assert (master / name).read_bytes() == (peer / name).read_bytes()
