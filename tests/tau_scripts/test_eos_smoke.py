# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ensure the real-service smoke checker rejects misleading trigger evidence."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "eos_smoke", Path(__file__).resolve().parents[2] / "tools/tau_batch_eos_smoke.py"
)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def evidence(tokens, reason):
    response = dict(
        id="cmpl-probe",
        choices=[dict(token_ids=tokens[:], finish_reason=reason)],
        usage=dict(completion_tokens=len(tokens)),
    )
    events = []
    for i in range(len(tokens)):
        phase = "prefill" if i == 0 else "decode"
        events.append(
            dict(event="emit", req_ids=["cmpl-probe-0"], fwd_id=i, phase=phase)
        )
        events.extend(dict(event="compute", fwd_id=i, pp_rank=r) for r in (0, 1))
        events.append(dict(event="done", req_ids=["cmpl-probe-0"], fwd_id=i))
    events.append(dict(event="eos", phase=phase, finished_ids=["cmpl-probe-0"]))
    return response, events


@pytest.mark.parametrize("name,overrides,tokens,reason", smoke.cases(2, 1))
def test_accept_matching_evidence(name, overrides, tokens, reason):
    response, events = evidence(tokens, reason)
    assert smoke.validate(response, events, tokens, reason, 2)["passed"]


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_token",
        "length_instead_of_eos",
        "missing_rank",
        "missing_callback",
        "duplicate_callback",
        "early_callback",
        "wrong_request",
        "missing_done",
    ],
)
def test_reject_incomplete_or_wrong_evidence(fault):
    response, events = evidence([1, 2], "stop")
    if fault == "wrong_token":
        response["choices"][0]["token_ids"] = [1, 3]
    elif fault == "length_instead_of_eos":
        response["choices"][0]["finish_reason"] = "length"
    elif fault == "missing_rank":
        events = [e for e in events if e.get("pp_rank") != 1]
    elif fault == "missing_callback":
        events.pop()
    elif fault == "duplicate_callback":
        events.append(events[-1].copy())
    elif fault == "early_callback":
        events.insert(4, events.pop())
    elif fault == "wrong_request":
        events[-1]["finished_ids"] = ["unrelated"]
    elif fault == "missing_done":
        events = [e for e in events if e["event"] != "done"]
    assert not smoke.validate(response, events, [1, 2], "stop", 2)["passed"]
