# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline model identification, range isolation and campaign bookkeeping."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fitter = load("tau_batch_fit")
sys.modules["tau_batch_fit"] = fitter
campaign = load("tau_batch_campaign")


def trace_fixture(tmp_path):
    path = tmp_path / "trace.jsonl"
    # Deliberately exclude an invalid earlier run using the byte range.
    path.write_text('{"event":"compute","fwd_id":-1}\n')
    start = path.stat().st_size
    fid = 0
    with path.open("a") as f:
        for wave in range(10):
            for n, s in ((1, 10), (2, 10), (1, 20), (3, 40), (4, 30)):
                for phase in ("prefill", "decode"):
                    fid += 1
                    features = dict(
                        n=n,
                        s_max=s,
                        s_sum=n * s,
                        seq_lens=[s] * n,
                        phase=phase,
                        tokens=n * s if phase == "prefill" else n,
                    )
                    f.write(
                        json.dumps(
                            dict(event="emit", fwd_id=fid, wave_id=wave, **features)
                        )
                        + "\n"
                    )
                    for rank in (0, 1):
                        # Exactly representable, per-stage/per-phase coefficients.
                        ms = (1 + rank) * (0.01 * n * s + 0.2 * n + 1)
                        if phase == "prefill":
                            ms += 3
                        f.write(
                            json.dumps(
                                dict(
                                    event="compute",
                                    fwd_id=fid,
                                    pp_rank=rank,
                                    start_ts_ns=0,
                                    end_ts_ns=round(ms * 1e6),
                                    **features,
                                )
                            )
                            + "\n"
                        )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            dict(
                passed=True,
                completed=100,
                trace_start_offset=start,
                trace_end_offset=path.stat().st_size,
            )
        )
    )
    return path, report


def test_recover_coefficients_with_whole_wave_validation(tmp_path):
    trace, report = trace_fixture(tmp_path)
    result = fitter.fit(
        trace,
        [report],
        stage_layers=16,
        group_by="stage-phase",
        models=("tau_affine",),
    )
    assert result["split"]["training_waves"] == list(range(8))
    assert result["split"]["validation_waves"] == [8, 9]
    for name, group in result["groups"].items():
        rank = int(name[2])
        expected = np.array([0.01, 0.2, 1]) * (rank + 1)
        if name.endswith("prefill"):
            expected[2] += 3
        model = group["models"]["tau_affine"]
        np.testing.assert_allclose(model["all_data_fit"]["coefficients"], expected)
        assert model["validation"]["rmse_ms"] < 1e-10
        assert group["train_samples"] == 40
        assert group["validation_samples"] == 10
        assert model["layer_normalized"]["gamma_effective"] == pytest.approx(
            expected[2] / 16
        )


def test_shared_phase_coefficients_and_stage_bias(tmp_path):
    trace, report = trace_fixture(tmp_path)
    result = fitter.fit(trace, [report])
    assert result["schema_version"] == 2
    assert result["group_by"] == "phase"
    assert result["prediction_target"] == "single_stage_compute_ms"
    assert set(result["groups"]) == {"prefill", "decode"}
    assert result["split"]["training_waves"] == list(range(8))
    assert result["split"]["validation_waves"] == [8, 9]
    for phase, group in result["groups"].items():
        assert group["pp_ranks"] == [0, 1]
        # All ranks of a wave stay on the same side of the validation split.
        assert group["samples"] == 100
        assert group["train_samples"] == 80
        assert group["validation_samples"] == 20
        assert set(group["models"]) == {"scls_bilinear", "unpadded_comparison"}
        intercept = 4.5 if phase == "prefill" else 1.5
        for model in group["models"].values():
            # Joint least squares fits the mean stage cost, not a sum or max.
            expected = [0.015, 0.3, 0, intercept]
            for fit_key in ("all_data_fit", "training_fit"):
                np.testing.assert_allclose(
                    model[fit_key]["coefficients"], expected, atol=1e-10
                )
            diagnostics = model["validation_by_stage"]
            assert set(diagnostics) == {"pp0", "pp1"}
            assert diagnostics["pp0"]["n"] == diagnostics["pp1"]["n"] == 10
            # Stage 1 is slower in this fixture. Shared parameters expose bias,
            # instead of hiding it behind separately fitted stage coefficients.
            assert diagnostics["pp0"]["mean_error_ms"] > 0
            assert diagnostics["pp1"]["mean_error_ms"] < 0
            assert diagnostics["pp1"]["underprediction_fraction"] == 1
            assert model["validation"]["rmse_ms"] > 0


def test_cli_defaults_to_shared_phase_models(tmp_path):
    trace, report = trace_fixture(tmp_path)
    output = tmp_path / "parameters.json"
    result = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "tau_batch_fit.py"),
            "--trace",
            str(trace),
            "--report",
            str(report),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    fitted = json.loads(output.read_text())
    assert set(fitted["groups"]) == {"prefill", "decode"}
    assert set(fitted["formulas"]) == {"scls_bilinear", "unpadded_comparison"}
    assert "pp0/prefill" not in result.stdout


def test_constant_batch_cannot_identify_all_parameters():
    rows = [
        (4, s, 4 * s, 0.04 * s + 2, wave) for wave in range(4) for s in range(10, 20)
    ]
    result = fitter.fit_group(rows, [0, 1])
    for model in result["models"].values():
        assert model["all_data_fit"]["status"] == "rank_deficient"
        assert model["all_data_fit"]["coefficients"] is None
        assert "validation" not in model


def test_scls_four_coefficients_and_nonpadded_design():
    rows = np.array(
        [(n, s, n * s - n + 1, 1, 0) for n in (1, 2, 4) for s in (10, 20, 40)],
        dtype=float,
    )
    x = fitter.design(rows, "scls_bilinear")
    expected = [0.01, 0.2, 0.03, 1]
    np.testing.assert_allclose(fitter.solve(x, x @ expected)["coefficients"], expected)
    unpadded = fitter.design(rows, "unpadded_comparison")
    np.testing.assert_array_equal(unpadded[:, 0], rows[:, 2])
    assert np.any(unpadded[:, 0] != x[:, 0])


def test_validation_bounds_include_formula_specific_token_features():
    rows = [
        (1, 10, 10, 10, 0),
        (2, 10, 14, 12, 1),
        (1, 20, 20, 13, 2),
        (2, 20, 25, 16, 3),
        (2, 20, 40, 20, 4),
    ]
    result = fitter.fit_group(rows, [0, 1, 2, 3])
    assert result["s_sum_range"] == [10, 40]
    # n and s_max remain in range, but the unpadded model's total exceeds all
    # training totals. Checking only n/s_max would incorrectly report zero.
    assert (
        result["models"]["unpadded_comparison"]["validation_outside_training_bounds"]
        == 1
    )
    assert result["models"]["scls_bilinear"]["validation_outside_training_bounds"] == 0


def test_reject_overlapping_or_failed_ranges(tmp_path):
    trace, report = trace_fixture(tmp_path)
    with pytest.raises(ValueError, match="Overlapping"):
        fitter.fit(trace, [report, report])
    data = json.loads(report.read_text())
    data["passed"] = False
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unvalidated"):
        fitter.fit(trace, [report])


def test_archive_rebases_offsets_and_preserves_samples(tmp_path):
    trace, report = trace_fixture(tmp_path)
    original_size = trace.stat().st_size
    # Simulate later data still being appended to the live trace.
    with trace.open("a") as f:
        f.write('{"event":"compute","fwd_id":-2}\n')
    archived, local_report = campaign.archive(trace, report, tmp_path / "copy", 100)
    local = json.loads(local_report.read_text())["trace"]
    assert local["trace_start_offset"] == 0
    assert local["trace_end_offset"] == archived.stat().st_size < original_size
    assert (
        fitter.fit(archived, [local_report])["groups"]
        == fitter.fit(trace, [report])["groups"]
    )
    with pytest.raises(ValueError, match="Unsuccessful"):
        campaign.archive(trace, report, tmp_path / "bad", 101)


def test_index_is_unique_and_matches_shuffled_valid_source_rows():
    data = [
        {"conversations": [{"value": str(n)}, {"value": "answer"}]}
        for n in (3, 4, 20, 1024, 1025)
    ]
    data.append({"conversations": []})
    tokenizer = lambda texts, **kw: SimpleNamespace(
        input_ids=[list(range(min(int(text), kw["max_length"]))) for text in texts]
    )
    ids, lengths = campaign.prepare_index(data, tokenizer, 0)
    assert set(ids) == {1, 2, 3}
    assert len(set(ids)) == len(ids)
    assert dict(zip(ids, lengths)) == {1: 4, 2: 20, 3: 1024}


def test_index_provenance_rejects_changed_source(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("[]")
    index = tmp_path / "index"
    index.mkdir()
    (index / "summary.json").write_text('{"source_sha256":"wrong"}')
    args = SimpleNamespace(dataset=source, seed=0, index_dir=index)
    with pytest.raises(ValueError, match="provenance"):
        campaign.load_index(args, {"model": "model"}, [])


@pytest.mark.parametrize("fail_second", [False, True])
def test_campaign_covers_source_once_and_stops_on_failure(
    tmp_path, monkeypatch, fail_second
):
    run = tmp_path / "run"
    run.mkdir()
    trace = run / "trace.jsonl"
    trace.write_text("")
    campaign.save(run / "server_meta.json", {"model": "test", "trace": str(trace)})
    source = tmp_path / "data.json"
    campaign.save(source, [{"id": i} for i in range(5)])
    ids = [4, 1, 3, 0, 2]
    index = tmp_path / "index"
    index.mkdir()
    campaign.save(index / "indices.json", ids)
    args = SimpleNamespace(
        run_dir=run,
        dataset=source,
        index_dir=index,
        output_dir=tmp_path / "job",
        seed=0,
        output_len=256,
        concurrency=32,
        shard_size=2,
        stage_layers=16,
        adopt_report=None,
        adopt_count=0,
        await_exit=None,
        server_pid=None,
        reserve_gib=0,
    )
    monkeypatch.setattr(campaign, "load_index", lambda *args: (ids, {}))
    monkeypatch.setattr(campaign, "ensure_idle", lambda *args: None)
    batches = []
    monkeypatch.setenv("IGNORE_EOS", "0")

    def benchmark(command, **kwargs):
        assert kwargs["env"]["IGNORE_EOS"] == "1"
        dataset = Path(command[command.index("--dataset") + 1])
        selected = json.loads(dataset.read_text())
        batches.append([x["id"] for x in selected])
        if fail_second and len(batches) == 2:
            raise RuntimeError("injected benchmark failure")
        target = Path(command[command.index("--result-dir") + 1])
        target.mkdir()
        start = trace.stat().st_size
        with trace.open("a") as f:
            f.write('{"example":"record"}\n')
        campaign.save(
            target / "summary.json",
            {
                "passed": True,
                "completed": len(selected),
                "trace_start_offset": start,
                "trace_end_offset": trace.stat().st_size,
            },
        )

    monkeypatch.setattr(campaign.subprocess, "run", benchmark)
    if fail_second:
        with pytest.raises(RuntimeError, match="injected"):
            campaign.campaign(args)
    else:
        campaign.campaign(args)
    state = json.loads((args.output_dir / "status.json").read_text())
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert "bench_tau.sh" in manifest["tools"]
    assert "serve_tau.sh" in manifest["tools"]
    if fail_second:
        assert state["status"] == "failed"
        assert state["completed"] == 2
        assert len(batches) == 2  # No retry and no third shard.
    else:
        assert state["status"] == "complete"
        assert state["completed"] == 5
        assert [i for batch in batches for i in batch] == ids
        assert not list(args.output_dir.rglob("parameters*.json"))


def test_fit_accepts_summary_trace_range(tmp_path):
    trace, report = trace_fixture(tmp_path)
    expected = fitter.fit(trace, [report])
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"trace": json.loads(report.read_text())}))
    assert fitter.fit(trace, [summary])["groups"] == expected["groups"]


@pytest.mark.parametrize("cross_wave", [False, True])
def test_explicit_benchmark_holdout_preserves_wave_isolation(tmp_path, cross_wave):
    trace, original = trace_fixture(tmp_path)
    metadata = json.loads(original.read_text())
    with trace.open("rb") as f:
        f.seek(metadata["trace_start_offset"])
        while True:
            offset = f.tell()
            event = json.loads(f.readline())
            if event["event"] == "emit" and event["wave_id"] == 2:
                if cross_wave:
                    # Cut after one complete forward, but inside the same wave.
                    f.readline()
                    f.readline()
                    offset = f.tell()
                break
    held_out, training = tmp_path / "held_out.json", tmp_path / "training.json"
    held_out.write_text(json.dumps({**metadata, "trace_end_offset": offset}))
    training.write_text(json.dumps({**metadata, "trace_start_offset": offset}))
    if cross_wave:
        with pytest.raises(ValueError, match="wave crosses"):
            fitter.fit(trace, [held_out, training], validation_reports=[held_out])
        return
    result = fitter.fit(trace, [held_out, training], validation_reports=[held_out])
    assert result["split"]["validation_waves"] == [0, 1]
    assert result["split"]["training_waves"] == list(range(2, 10))
    for group in result["groups"].values():
        assert group["validation_samples"] == 20
        assert group["train_samples"] == 80
    with pytest.raises(ValueError, match="proper subset"):
        fitter.fit(trace, [training], validation_reports=[held_out])
    with pytest.raises(ValueError, match="proper subset"):
        fitter.fit(trace, [training], validation_reports=[training])
