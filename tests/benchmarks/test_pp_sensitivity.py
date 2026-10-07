# SPDX-License-Identifier: Apache-2.0
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_sweep_holds_requests_outputs_and_other_controls_constant(
    tmp_path, monkeypatch
):
    directory = Path(__file__).resolve().parents[2] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "pp_sensitivity", directory / "pp_sensitivity.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data, reference = tmp_path / "source", tmp_path / "reference"
    data.mkdir()
    samples = []
    for start, count, split in (
        (0, 32, "warmup"),
        (100, 64, "profile"),
        (1000, 300, "evaluation"),
    ):
        rows = []
        for i in range(start, start + count):
            tokens = [i] + [1] * 1023
            rows.append(
                dict(
                    id=i,
                    prompt=tokens,
                    input_tokens=1024,
                    output_tokens=80,
                    prompt_sha256=hashlib.sha256(
                        json.dumps(tokens).encode()
                    ).hexdigest(),
                )
            )
        (data / f"{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        if split == "evaluation":
            samples = rows[64:160]
    module.save(
        reference / "protocol.json",
        dict(model="/models/CodeLlama-34b/V1/model", scales=[1, 1, 2, 4]),
    )
    folder = reference / "p99_100g_c8"
    module.save(
        folder / "settings.json", dict(serving=dict(max_num_batched_tokens=2048))
    )
    module.save(folder / "slo.json", dict(experiment_slo_ms=800))
    module.save(folder / "complete.json", {})
    for c in (8, 12):
        records = [
            dict(r, requested_output_tokens=80, index=i, success=True)
            for i, r in enumerate(samples)
        ]
        for strategy in ("uniform", "latency_dp"):
            module.save(
                folder / strategy / f"round1_c{c}.json",
                dict(concurrency=c, summary=dict(failed=0), records=records),
            )
    args = SimpleNamespace(
        source_data=data, reference_root=reference, output=tmp_path / "result"
    )
    matrix = module.prepare(args)
    assert len(matrix["cases"]) == 14
    cases = {r["name"]: r for r in matrix["cases"]}
    for axis, varying in (
        ("bandwidth", "bandwidth"),
        ("batch_token_budget", "budget"),
        ("concurrency", "concurrency"),
        ("input_tokens", "data"),
    ):
        controls = [
            {k: v for k, v in cases[name].items() if k not in ("name", varying)}
            for name in matrix["axes"][axis]
        ]
        assert all(r == controls[0] for r in controls)
    ids = None
    for length in (128, 256, 512, 1024):
        parent = args.output / f"data/input{length}"
        rows = module.read_samples(parent / "evaluation.jsonl")
        assert len(rows) == 128
        assert all(
            r["input_tokens"] == length
            and len(r["prompt"]) == length
            and r["output_tokens"] == 64
            for r in rows
        )
        current = [r["id"] for r in rows]
        assert ids is None or ids == current
        ids = current
        hashes = [
            r["prompt_sha256"]
            for split in ("warmup", "profile", "evaluation")
            for r in module.read_samples(parent / f"{split}.jsonl")
        ]
        assert len(hashes) == len(set(hashes))
    assert module.prepare(args) == matrix
    frozen = {
        str(p.relative_to(args.output)): p.read_bytes()
        for p in args.output.rglob("*")
        if p.is_file()
    }
    profile_path = data / "profile.jsonl"
    original = profile_path.read_text()
    modified = [json.loads(line) for line in original.splitlines()]
    modified[0]["output_tokens"] = 81
    profile_path.write_text("".join(json.dumps(r) + "\n" for r in modified))
    with pytest.raises(ValueError, match="matrix changed"):
        module.prepare(args)
    assert all(
        (args.output / name).read_bytes() == value for name, value in frozen.items()
    )
    profile_path.write_text(original)
    module.save(folder / "slo.json", dict(experiment_slo_ms=900))
    with pytest.raises(ValueError, match="matrix changed"):
        module.prepare(args)
    assert all(
        (args.output / name).read_bytes() == value for name, value in frozen.items()
    )
    module.save(folder / "slo.json", dict(experiment_slo_ms=800))
    path = folder / "settings.json"
    module.save(path, dict(serving=dict(max_num_batched_tokens=1024)))
    with pytest.raises(ValueError, match="reference"):
        module.prepare(args)
