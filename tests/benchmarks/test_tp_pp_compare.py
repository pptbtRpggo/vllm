# SPDX-License-Identifier: Apache-2.0
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def comparison(monkeypatch):
    folder = Path(__file__).resolve().parents[2] / "benchmarks"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(
        "tp_pp_compare", folder / "tp_pp_compare.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sample(index):
    tokens = [index, 10]
    return dict(
        id=index,
        prompt=tokens,
        input_tokens=2,
        output_tokens=3,
        prompt_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
    )


def reference(tmp_path):
    root, data = tmp_path / "pp", tmp_path / "data"
    a, b = sample(11), sample(32)
    data.mkdir()
    for name, values in (
        ("evaluation", [a, b]),
        ("warmup", [sample(i) for i in range(50, 66)]),
    ):
        (data / f"{name}.jsonl").write_text(
            "".join(json.dumps(v) + "\n" for v in values)
        )
    spec = dict(name="case", base=1, raised=2, bandwidth=25)
    write(
        root / "protocol.json",
        dict(model="/model", pp_size=2, scales=[2, 4], scenarios=[spec]),
    )
    write(root / "case/complete.json", {})
    write(root / "case/slo.json", dict(experiment_slo_ms=500, project_slo_ms=150))
    serving = dict(
        model="/model",
        revision=None,
        dtype="float16",
        quantization=None,
        kv_cache_dtype="auto",
        block_size=128,
        max_model_len=4096,
        max_num_seqs=256,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.85,
    )
    write(root / "case/settings.json", dict(formal_ids=[11, 32], serving=serving))
    records = [
        dict(r, requested_output_tokens=3, success=True, index=i)
        for i, r in enumerate((a, b))
    ]
    for strategy in ("uniform", "latency_dp"):
        for c in (1, 2):
            write(
                root / f"case/{strategy}/round1_c{c}.json",
                dict(concurrency=c, summary=dict(failed=0), records=records),
            )
    for c in (1, 2):
        rows = [
            dict(
                sample(i),
                requested_output_tokens=16,
                output_tokens=16,
                success=True,
                index=n,
            )
            for n, i in enumerate(range(50, 66))
        ]
        write(
            root / f"case/uniform/warmup_round1_c{c}_1.json",
            dict(concurrency=c, summary=dict(failed=0), records=rows),
        )
    write(data / "manifest.json", {})
    return root, data


def test_matched_workload_uses_saved_pp_order_lengths_and_frozen_slo(
    comparison, tmp_path
):
    root, data = reference(tmp_path)
    _, _, cases = comparison.prepare_cases(root, data)
    assert [r["id"] for r in cases[0]["samples"]] == [11, 32]
    assert cases[0]["slo"]["experiment_slo_ms"] == 500
    path = root / "case/settings.json"
    settings = json.loads(path.read_text())
    settings["formal_ids"] = [32, 11]
    write(path, settings)
    with pytest.raises(ValueError, match="signature"):
        comparison.prepare_cases(root, data)


def test_matched_workload_rejects_modified_tokens_or_serving_settings(
    comparison, tmp_path
):
    root, data = reference(tmp_path)
    path = root / "case/settings.json"
    settings = json.loads(path.read_text())
    settings["serving"]["max_model_len"] = 8192
    write(path, settings)
    with pytest.raises(ValueError, match="serving parameters"):
        comparison.prepare_cases(root, data)
    settings["serving"]["max_model_len"] = 4096
    write(path, settings)
    row = sample(11)
    row["prompt"][0] = 99
    (data / "evaluation.jsonl").write_text(
        json.dumps(row) + "\n" + json.dumps(sample(32)) + "\n"
    )
    with pytest.raises(ValueError, match="stored hash"):
        comparison.prepare_cases(root, data)


def test_resume_retries_failed_point_and_skips_successful_points(
    comparison, tmp_path, monkeypatch
):
    from argparse import Namespace
    from contextlib import contextmanager

    root, data = reference(tmp_path)
    calibration = tmp_path / "native.json"
    write(calibration, dict(tp_size=2, cross_group_size=1, native_collectives={}))
    library = tmp_path / "delay.so"
    library.write_bytes(b"test-library")
    monkeypatch.setenv("VLLM_ASCEND_DELAY_LIBRARY", str(library))
    args = Namespace(
        reference_root=root,
        data=data,
        model="/model",
        output=tmp_path / "tp",
        native_calibration=calibration,
        port=18791,
        validate_only=False,
    )
    starts = []
    calls = []
    fail_first = True

    @contextmanager
    def fake_server(*_args):
        starts.append(1)
        yield "http://unused", tmp_path / "server"

    async def fake_load(samples, _url, _model, concurrency, output_override=None):
        nonlocal fail_first
        failed = output_override is None and fail_first
        if output_override is None:
            calls.append(concurrency)
            fail_first = False
        records = [
            dict(
                r,
                index=i,
                success=not failed,
                requested_output_tokens=output_override or r["output_tokens"],
                output_tokens=output_override or r["output_tokens"],
            )
            for i, r in enumerate(samples)
        ]
        return dict(
            concurrency=concurrency, records=records, summary=dict(failed=int(failed))
        )

    monkeypatch.setattr(comparison, "server", fake_server)
    monkeypatch.setattr(comparison, "run_load", fake_load)
    with pytest.raises(ValueError, match="load failed"):
        comparison.run(args)
    comparison.run(args)
    assert calls == [1, 1, 2]
    assert len(list((args.output / "case/failures").glob("*.json"))) == 1
    assert (args.output / "complete.json").exists()
    comparison.run(args)
    assert len(starts) == 2
    assert calls == [1, 1, 2]
    library.write_bytes(b"changed-library")
    with pytest.raises(ValueError, match="protocol changed"):
        comparison.run(args)
