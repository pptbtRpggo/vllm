# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compute a content identity for the local dense-model launcher."""

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path

# Keep these file patterns and training exclusions in step with DefaultModelLoader.
_HF_FORMATS = (
    ("*.safetensors", "model.safetensors.index.json"),
    ("*.bin", "pytorch_model.bin.index.json"),
    ("*.pt", None),
)
_TRAINING_FILES = (
    "training_args.bin",
    "optimizer.bin",
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
)


def _indexed_weights(index: Path, data: bytes) -> set[str]:
    try:
        weight_map = json.loads(data)["weight_map"]
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError(f"Invalid checkpoint index {index.name}") from error
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid weight_map in checkpoint index {index.name}")
    names = set()
    for name in weight_map.values():
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError(f"Invalid shard filename in checkpoint index {index.name}")
        names.add(name)
    return names


def _validate_shards(roots: dict[str, Path], *, indexed: bool) -> None:
    groups: dict[tuple[str, int], set[int]] = {}
    numbered = 0
    for path in roots.values():
        fields = path.stem.rsplit("-", 3)
        if len(fields) != 4:
            continue
        stem, index_text, separator, total_text = fields
        if (
            stem
            and separator == "of"
            and index_text.isdecimal()
            and total_text.isdecimal()
        ):
            groups.setdefault((stem, int(total_text)), set()).add(int(index_text))
            numbered += 1
    if not indexed and len(roots) > 1 and (numbered != len(roots) or len(groups) != 1):
        raise ValueError("Ambiguous checkpoint files without a shard index")
    for (stem, total), numbers in groups.items():
        if len(numbers) != total or min(numbers) != 1 or max(numbers) != total:
            raise ValueError(f"Numbered checkpoint shards are incomplete for {stem}")


def _weight_files(
    model_dir: Path, *, consolidated: bool
) -> tuple[list[Path], tuple[str, bytes] | None]:
    formats: tuple[tuple[str, str | None], ...] = _HF_FORMATS
    if consolidated:
        # Auto format selects consolidated weights even if HF weights coexist.
        formats = (
            ("consolidated*.safetensors", "consolidated.safetensors.index.json"),
        )

    for pattern, index_name in formats:
        paths = glob.glob(os.path.join(glob.escape(str(model_dir)), pattern))
        if not paths:
            continue
        safetensors = pattern.endswith(".safetensors")
        weights = {
            path.name: path
            for path in map(Path, paths)
            if safetensors or not path.name.endswith(_TRAINING_FILES)
        }
        index = None
        if index_name is not None and (model_dir / index_name).is_file():
            index_path = model_dir / index_name
            index_data = index_path.read_bytes()
            referenced = _indexed_weights(index_path, index_data)
            if missing := referenced - weights.keys():
                names = ", ".join(sorted(missing))
                raise ValueError(f"Checkpoint is missing shards: {names}")
            if safetensors:
                weights = {name: weights[name] for name in referenced}
            elif extra := weights.keys() - referenced:
                # The binary loader would load these additional files as well.
                names = ", ".join(sorted(extra))
                raise ValueError(f"Ambiguous extra checkpoint files: {names}")
            index = (index_name, index_data)
        if not weights:
            break
        _validate_shards(weights, indexed=index is not None)
        return [weights[name] for name in sorted(weights)], index

    raise ValueError(f"Checkpoint contains no model weights in {model_dir}")


def _file_digest(path: Path) -> bytes:
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"Checkpoint file is missing or empty: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def _configuration_digests(model_dir: Path, *, consolidated: bool) -> dict[str, bytes]:
    config_data = (model_dir / "config.json").read_bytes()
    try:
        config = json.loads(config_data)
    except ValueError as error:
        raise ValueError("Invalid checkpoint config.json") from error
    if not isinstance(config, dict):
        raise ValueError("Invalid checkpoint config.json")

    alternatives = config.get("configuration_files", [])
    if not isinstance(alternatives, list) or any(
        not isinstance(name, str) for name in alternatives
    ):
        raise ValueError("Invalid configuration_files in checkpoint config.json")
    # Transformers chooses a config.<version>.json based on its installed version.
    # Require and hash every selectable file without importing Transformers.
    names = {
        name
        for name in alternatives
        if name.startswith("config.")
        and name.endswith(".json")
        and name != "config.json"
    }
    if any(Path(name).name != name for name in names):
        raise ValueError("Unsupported versioned checkpoint config path")
    if consolidated and (model_dir / "params.json").is_file():
        # The Mistral parser can use both params.json and the HF configuration.
        names.add("params.json")
    digests = {name: _file_digest(model_dir / name) for name in sorted(names)}
    digests["config.json"] = hashlib.sha256(config_data).digest()
    return digests


def checkpoint_identity(model_dir: Path) -> str:
    """Hash model configuration and selected weights without depending on local paths.

    Args:
        model_dir: Local checkpoint directory used by the default dense loader.

    Returns:
        SHA-256 identity of configuration, active index and selected shards.

    Raises:
        ValueError: The checkpoint is missing files or has an ambiguous layout.
        OSError: The checkpoint could not be read.
    """
    consolidated = any(
        path.is_file() for path in model_dir.rglob("consolidated*.safetensors")
    )
    configs = _configuration_digests(model_dir, consolidated=consolidated)
    weights, index = _weight_files(model_dir, consolidated=consolidated)
    digest = hashlib.sha256(b"vllm-local-checkpoint-v2\x00")
    for name, config_digest in configs.items():
        digest.update(name.encode() + b"\x00" + config_digest)
    if index is not None:
        name, data = index
        digest.update(name.encode() + b"\x00" + hashlib.sha256(data).digest())
    for path in weights:
        digest.update(path.name.encode() + b"\x00" + _file_digest(path))
    return digest.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    try:
        print(checkpoint_identity(parser.parse_args().model_dir))
    except (OSError, ValueError) as error:
        parser.exit(status=1, message=f"{error}\n")
