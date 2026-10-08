# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def driver(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "length_compare", directory / "pp_length_compare.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_reference_request_matching_and_disjoint_profile(driver):
    root = Path(__file__).resolve().parents[2] / "output/pp_length_slo_20261007"
    if not (root / "metrics.csv").exists():
        pytest.skip("optional saved serving experiment is absent")
    cases = driver.prepare(root)
    assert len(cases) == 206
    assert sum(len(c["samples"]) for c in cases) == 34112
    for case in cases:
        assert all(
            len(r["prompt"]) == case["input_tokens"]
            and r["output_tokens"] == case["output_tokens"]
            for r in case["samples"] + case["warmup"] + case["profile"]
        )
        formal_ids = {
            r["id"] for c in cases if c["model"] == case["model"] for r in c["samples"]
        }
        assert not formal_ids.intersection(r["id"] for r in case["profile"])
        assert len(case["profile"]) >= 2 * case["concurrency"]
        # BOS-only prompts can be identical while source IDs remain disjoint.
        assert not {r["id"] for r in case["warmup"]}.intersection(formal_ids)


def test_graph_is_preserved_for_formal_serving_only(driver):
    case = dict(model_path="/model", size=4)
    for method, pp, tp in (("pp", "4", "1"), ("tp", "1", "4")):
        formal = driver.serving_command(case, method, 18000)
        assert formal[formal.index("--pipeline-parallel-size") + 1] == pp
        assert formal[formal.index("--tensor-parallel-size") + 1] == tp
        assert "--enforce-eager" not in formal
        assert (
            json.loads(formal[formal.index("--compilation-config") + 1]) == driver.GRAPH
        )
        assert "--enforce-eager" in driver.serving_command(
            case, method, 18000, profile=True
        )


def test_resume_rejects_wrong_output_and_request_identity(driver):
    samples = [dict(id="a", prompt_sha256="h", input_tokens=8, output_tokens=32)]
    result = dict(
        concurrency=1,
        summary=dict(failed=0),
        records=[
            dict(
                id="a",
                prompt_sha256="h",
                input_tokens=8,
                requested_output_tokens=32,
                output_tokens=32,
                index=0,
                success=True,
            )
        ],
    )
    driver.successful(result, samples, 1)
    result["records"][0]["output_tokens"] = 31
    with pytest.raises(ValueError, match="incomplete"):
        driver.successful(result, samples, 1)
    result["records"][0].update(output_tokens=32, prompt_sha256="changed")
    with pytest.raises(ValueError, match="signature"):
        driver.successful(result, samples, 1)


def test_profile_windows_exclude_warmup_and_reject_misaligned_batches(driver):
    rows = [
        dict(step=i, ts_unix=float(i), is_warmup=(i == 2), batch_id=str(i))
        for i in range(5)
    ]
    raw = {0: rows, 1: list(rows)}
    selected = driver.select_window(raw, 1.5, 3.5, 2)
    assert [r["step"] for r in selected[0]] == [3]
    raw[1] = [dict(r, batch_id="wrong") if r["step"] == 3 else r for r in rows]
    with pytest.raises(ValueError, match="matching"):
        driver.select_window(raw, 1.5, 3.5, 2)
