# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic request contracts, not a claim about real model generation."""

import importlib.util
import io
import json
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "examples/online_serving/hetero_pp/check_eos.py"
)


@pytest.fixture
def diagnostic():
    spec = importlib.util.spec_from_file_location("eos_diagnostic_test", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_forced_eos_controls_change_only_stop_policy(diagnostic):
    cases = diagnostic.make_cases("hetero-model", 2, 64)
    stop, ignore = cases[:2]
    assert stop[2]["allowed_token_ids"] == ignore[2]["allowed_token_ids"] == [2]
    assert stop[2]["ignore_eos"] is False
    assert ignore[2]["ignore_eos"] is True
    assert stop[2]["max_tokens"] == ignore[2]["max_tokens"] == 4
    assert cases[-1][1] == "/v1/chat/completions"
    assert cases[-1][2]["messages"][0]["role"] == "user"


def test_length_is_not_classified_as_an_eos_stop(diagnostic):
    response = {
        "choices": [
            {
                "text": "OK OK OK",
                "token_ids": [7, 7, 7],
                "finish_reason": "length",
                "stop_reason": None,
            }
        ],
        "usage": {"completion_tokens": 3},
    }
    summary = diagnostic.summarize("natural", response, 2)
    assert summary["finish_reason"] == "length"
    assert summary["eos_in_returned_ids"] is False
    assert summary["completion_tokens"] == 3
    assert len(summary["text_preview"]) <= 80


def test_missing_ids_are_unknown_not_proof_of_absent_eos(diagnostic):
    summary = diagnostic.summarize(
        "natural",
        {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]},
        2,
    )
    assert summary["eos_in_returned_ids"] is None


def test_saves_short_summaries_and_full_http_evidence(
    diagnostic, monkeypatch, tmp_path
):
    calls = []

    def respond(request, timeout):
        body = json.loads(request.data)
        calls.append((request.full_url, body, timeout))
        if len(calls) == 4:
            raise urllib.error.HTTPError(
                request.full_url,
                400,
                "no template",
                Message(),
                io.BytesIO(b'{"error":"no chat template"}'),
            )
        forced = "allowed_token_ids" in body
        stop = forced and not body["ignore_eos"]
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "text": "OK" * 100,
                            "token_ids": [2] if forced else [7],
                            "finish_reason": "stop" if stop else "length",
                        }
                    ],
                    "usage": {"completion_tokens": 1 if stop else body["max_tokens"]},
                }
            ).encode()
        )

    monkeypatch.setattr(diagnostic.urllib.request, "urlopen", respond)
    assert (
        diagnostic.run_cases(
            diagnostic.make_cases("hetero-model", 2, 64),
            "http://localhost:8000",
            tmp_path,
            2,
            5,
        )
        == 1
    )
    assert len(calls) == 4  # Errors are not retried.
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["forced_stop_policy_observed"] is True
    assert report["model_correctness_validated"] is False
    assert len(report["summaries"][0]["text_preview"]) == 80
    assert "OK" * 100 in (tmp_path / "forced_eos_stop.json").read_text()
    assert "no chat template" in (tmp_path / "natural_chat.json").read_text()
