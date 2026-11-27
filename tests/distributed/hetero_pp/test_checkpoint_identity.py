# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _write_checkpoint(path: Path, filename: str = "model.safetensors") -> None:
    path.mkdir()
    (path / "config.json").write_text('{"model_type":"llama"}')
    (path / filename).write_bytes(b"checkpoint bytes")


@pytest.mark.parametrize("filename", ["model.safetensors", "pytorch_model.bin", "x.pt"])
def test_checkpoint_identity_follows_contents_across_local_paths(tmp_path, filename):
    from vllm.distributed.hetero_pp.checkpoint_identity import checkpoint_identity

    head, peer = tmp_path / "head", tmp_path / "peer"
    _write_checkpoint(head, filename)
    shutil.copytree(head, peer)
    first = checkpoint_identity(head)
    assert len(first) == 64
    assert checkpoint_identity(peer) == first

    # Same size, different contents.
    (peer / filename).write_bytes(b"checkpoint BYTES")
    assert checkpoint_identity(peer) != first
    (peer / filename).write_bytes(b"checkpoint bytes")
    (peer / "config.json").write_text('{"model_type":"qwen2"}')
    assert checkpoint_identity(peer) != first


def test_checkpoint_identity_follows_safetensors_index_and_format_priority(tmp_path):
    from vllm.distributed.hetero_pp.checkpoint_identity import checkpoint_identity

    model = tmp_path / "model"
    _write_checkpoint(model)
    index = model / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"model.weight": "model.safetensors"}}))
    first = checkpoint_identity(model)
    (model / "ignored.safetensors").write_bytes(b"excluded by index")
    (model / "pytorch_model.bin").write_bytes(b"lower priority format")
    assert checkpoint_identity(model) == first

    index.write_text(json.dumps({"weight_map": {"other.weight": "model.safetensors"}}))
    assert checkpoint_identity(model) != first


def test_checkpoint_identity_detects_incomplete_indexed_weights(tmp_path):
    from vllm.distributed.hetero_pp.checkpoint_identity import checkpoint_identity

    model = tmp_path / "model"
    _write_checkpoint(model)
    index = model / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": "missing.safetensors"}}))
    with pytest.raises(ValueError, match="missing.safetensors"):
        checkpoint_identity(model)


def test_checkpoint_identity_detects_incomplete_numbered_weights(tmp_path):
    from vllm.distributed.hetero_pp.checkpoint_identity import checkpoint_identity

    model = tmp_path / "model"
    _write_checkpoint(model, "pytorch_model-00001-of-00002.bin")
    with pytest.raises(ValueError, match="incomplete"):
        checkpoint_identity(model)


def test_checkpoint_identity_ignores_training_state_in_selected_format(tmp_path):
    from vllm.distributed.hetero_pp.checkpoint_identity import checkpoint_identity

    model = tmp_path / "model"
    _write_checkpoint(model, "pytorch_model.bin")
    first = checkpoint_identity(model)
    (model / "training_args.bin").write_bytes(b"training artifacts")
    assert checkpoint_identity(model) == first


def test_checkpoint_identity_rejects_no_inference_weights(tmp_path):
    from vllm.distributed.hetero_pp.checkpoint_identity import checkpoint_identity

    model = tmp_path / "model"
    _write_checkpoint(model, "optimizer.pt")
    with pytest.raises(ValueError, match="no model weights"):
        checkpoint_identity(model)


def test_checkpoint_identity_follows_consolidated_auto_priority(tmp_path):
    from vllm.distributed.hetero_pp.checkpoint_identity import checkpoint_identity

    model = tmp_path / "model"
    _write_checkpoint(model, "consolidated.safetensors")
    first = checkpoint_identity(model)
    (model / "model.safetensors").write_bytes(b"standard HF checkpoint")
    assert checkpoint_identity(model) == first
    (model / "consolidated.safetensors").write_bytes(b"updated consolidated weights")
    assert checkpoint_identity(model) != first


def test_checkpoint_identity_binds_consolidated_params(tmp_path):
    from vllm.distributed.hetero_pp.checkpoint_identity import checkpoint_identity

    model = tmp_path / "model"
    _write_checkpoint(model, "consolidated.safetensors")
    params = model / "params.json"
    params.write_text('{"dim": 4096}')
    first = checkpoint_identity(model)
    params.write_text('{"dim": 8192}')
    assert checkpoint_identity(model) != first


def test_checkpoint_identity_binds_declared_hf_config_versions(tmp_path):
    from vllm.distributed.hetero_pp.checkpoint_identity import checkpoint_identity

    model = tmp_path / "model"
    _write_checkpoint(model)
    alternate = model / "config.4.0.0.json"
    (model / "config.json").write_text(
        json.dumps({"model_type": "llama", "configuration_files": [alternate.name]})
    )
    alternate.write_text('{"model_type":"llama", "hidden_size":4096}')
    first = checkpoint_identity(model)
    alternate.write_text('{"model_type":"llama", "hidden_size":8192}')
    assert checkpoint_identity(model) != first
    alternate.unlink()
    with pytest.raises(ValueError, match=r"config\.4\.0\.0\.json"):
        checkpoint_identity(model)


def test_checkpoint_identity_ignores_nonselected_config_files(tmp_path):
    from vllm.distributed.hetero_pp.checkpoint_identity import checkpoint_identity

    model = tmp_path / "model"
    _write_checkpoint(model)
    first = checkpoint_identity(model)
    (model / "params.json").write_text('{"dim": 4096}')
    (model / "config.4.0.0.json").write_text('{"model_type":"qwen2"}')
    assert checkpoint_identity(model) == first


def test_checkpoint_identity_cli_prints_only_digest(tmp_path):
    from vllm.distributed.hetero_pp import checkpoint_identity

    model = tmp_path / "model"
    _write_checkpoint(model)
    result = subprocess.run(
        [sys.executable, "-S", checkpoint_identity.__file__, str(model)],
        capture_output=True,
        check=True,
        text=True,
        timeout=10,
    )
    assert result.stdout == checkpoint_identity.checkpoint_identity(model) + "\n"
    assert result.stderr == ""
