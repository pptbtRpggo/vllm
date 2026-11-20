# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import io
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples/online_serving/hetero_pp/verify_overlap.py"
)


@pytest.fixture
def verifier():
    spec = importlib.util.spec_from_file_location("verify_overlap_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trace(rank, starts, *, task_type="AI_CORE"):
    result = []
    for batch, kernel in enumerate(starts):
        launch = 100 + batch * 100
        result.extend(
            [
                dict(
                    ph="X",
                    name=f"hetero_pp.forward.{rank}.{batch}",
                    pid=1,
                    tid=2,
                    ts=str(launch),
                    dur=5,
                ),
                dict(
                    ph="s", name="torch_to_npu", id=batch, pid=1, tid=2, ts=launch + 1
                ),
                dict(ph="f", name="torch_to_npu", id=batch, pid=3, tid=4, ts=kernel),
                dict(
                    ph="X",
                    name="MatMul",
                    pid=3,
                    tid=4,
                    ts=str(kernel),
                    dur=20,
                    args={"Task Type": task_type},
                ),
            ]
        )
    return result


def save_trace(root, rank, events):
    path = root / f"stage{rank}" / "trace_view.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(events))


@pytest.mark.parametrize(
    ("stage1", "status"),
    [([220, 320], "passed"), ([170, 320], "not_observed")],
)
def test_requires_overlapping_device_compute_for_different_batches(
    verifier, tmp_path, stage1, status
):
    for rank, starts in enumerate(([150, 230], stage1)):
        save_trace(tmp_path, rank, trace(rank, starts))
    report = verifier.check_traces(tmp_path, 90_000, 350_000)
    assert report["status"] == status
    if status == "passed":
        assert report["witness"]["overlap_us"] == 10
        assert report["witness"]["stage0_batch"] == 1
        assert report["witness"]["stage1_batch"] == 0


def test_cpu_and_copy_events_cannot_pass_verification(verifier, tmp_path):
    for rank in (0, 1):
        save_trace(tmp_path, rank, trace(rank, [150, 230], task_type="MEMCPY"))
    assert verifier.check_traces(tmp_path, 90_000, 350_000)["status"] == "inconclusive"


@pytest.mark.parametrize("changed_ids", [False, True])
def test_verification_captures_after_warmup_and_checks_responses(
    verifier, tmp_path, monkeypatch, changed_ids
):
    calls = []

    class Client:
        def open(self, request, timeout):
            path = request.full_url.split(":8000/")[1]
            calls.append(path)
            if path == "stop_profile":
                root = tmp_path / "overlap_traces"
                save_trace(root, 0, trace(0, [150, 230]))
                save_trace(root, 1, trace(1, [220, 320]))
            if path != "v1/completions":
                return io.BytesIO()
            body = json.loads(request.data)
            assert [len(prompt) for prompt in body["prompt"]] == [384] * 4
            token = 3 if len(calls) == 3 and changed_ids else 2
            choices = [dict(index=i, token_ids=[token] * 8) for i in range(4)]
            return io.BytesIO(
                json.dumps(
                    dict(choices=choices, usage={"completion_tokens": 32})
                ).encode()
            )

    monkeypatch.setattr(verifier.urllib.request, "build_opener", lambda *_: Client())
    monkeypatch.setattr(verifier.time, "time_ns", iter((90_000, 350_000)).__next__)
    if changed_ids:
        with pytest.raises(ValueError, match="greedy output token IDs changed"):
            verifier.verify(8000, tmp_path)
    else:
        report = verifier.verify(8000, tmp_path)
        assert report["status"] == "passed" and report["token_ids_matched"]
    assert calls == [
        "v1/completions",
        "start_profile",
        "v1/completions",
        "stop_profile",
    ]
