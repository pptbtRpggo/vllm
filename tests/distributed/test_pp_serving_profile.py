# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json

import pytest

from vllm.distributed import pp_serving_profile as profile


def test_sample_preserves_arrivals_and_request_parameters():
    rows = [
        dict(at_s=i * 0.2, request=dict(prompt=str(i), max_tokens=i + 1))
        for i in range(20)
    ]
    sampled = profile.sample_request_window(rows, 5, 123)
    assert sampled == profile.sample_request_window(rows, 5, 123)
    start = int(sampled[0]["request"]["prompt"])
    assert [r["request"] for r in sampled] == [
        r["request"] for r in rows[start : start + 5]
    ]
    assert [r["at_s"] for r in sampled] == pytest.approx([i * 0.2 for i in range(5)])
    assert rows[-1]["at_s"] == pytest.approx(3.8)


@pytest.mark.parametrize("size", [0, -1, 3])
def test_sample_rejects_invalid_size(size):
    with pytest.raises(ValueError, match="sample size"):
        profile.sample_request_window([dict(at_s=0, request={})], size, 0)


def test_real_requests_warmup_drain_then_link_measurement(monkeypatch):
    calls = []

    def post(base, path, body):
        calls.append((path, body))
        return dict(usage=dict(prompt_tokens=4, completion_tokens=2))

    monkeypatch.setattr(profile, "post", post)
    rows = [
        dict(
            at_s=0,
            request=dict(
                prompt=[1, 2, 3, 4], max_tokens=2, temperature=0, ignore_eos=False
            ),
        )
    ]
    result = asyncio.run(
        profile.collect(rows, "http://local", "model", 1, 1, measure_links=True)
    )
    assert [body.get("method", "request") for _, body in calls] == [
        "set_pp_profile_warmup",
        "request",
        "set_pp_profile_warmup",
        "request",
        "profile_pp_links",
    ]
    assert calls[0][1]["args"] == [True]
    assert calls[2][1]["args"] == [False]
    assert calls[3][1]["prompt"] == [1, 2, 3, 4]
    assert calls[3][1]["ignore_eos"] is False
    assert result["requests"][0]["usage"]["completion_tokens"] == 2


def test_failed_workload_does_not_produce_link_profile(monkeypatch):
    methods = []

    def post(base, path, body):
        if path == "/collective_rpc":
            methods.append(body["method"])
            return {}
        raise RuntimeError("failed request")

    monkeypatch.setattr(profile, "post", post)
    with pytest.raises(RuntimeError, match="failed request"):
        asyncio.run(
            profile.collect(
                [dict(at_s=0, request=dict(prompt="hello", max_tokens=2))],
                "http://local",
                "model",
                1,
                1,
            )
        )
    assert "profile_pp_links" not in methods


@pytest.mark.parametrize("offsets", [[1, 0], [-1], [float("nan")]])
def test_invalid_arrival_times_rejected(tmp_path, offsets):
    path = tmp_path / "requests.jsonl"
    path.write_text(
        "".join(
            json.dumps(dict(at_s=t, request=dict(prompt="hello", max_tokens=2))) + "\n"
            for t in offsets
        )
    )
    with pytest.raises(ValueError, match="at_s"):
        profile.load_requests(path)


def test_serving_only_collection_does_not_replay_links(monkeypatch):
    calls = []

    def post(base, path, body):
        calls.append(body.get("method", "request"))
        return dict(usage=dict(prompt_tokens=2, completion_tokens=1))

    monkeypatch.setattr(profile, "post", post)
    asyncio.run(
        profile.collect(
            [dict(at_s=0, request=dict(prompt="hi", max_tokens=1))],
            "http://local",
            "model",
            1,
            1,
            measure_links=False,
        )
    )
    assert calls == [
        "set_pp_profile_warmup",
        "request",
        "set_pp_profile_warmup",
        "request",
    ]
