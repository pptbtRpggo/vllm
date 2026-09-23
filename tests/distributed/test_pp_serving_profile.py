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
    warmup, sampled, selection = profile.split_request_samples(rows, 5, 2, 123)
    assert (warmup, sampled, selection) == profile.split_request_samples(
        rows, 5, 2, 123
    )
    start = int(sampled[0]["request"]["prompt"])
    assert [r["request"] for r in sampled] == [
        r["request"] for r in rows[start : start + 5]
    ]
    assert [r["at_s"] for r in sampled] == pytest.approx([i * 0.2 for i in range(5)])
    assert rows[-1]["at_s"] == pytest.approx(3.8)


@pytest.mark.parametrize("size", [0, -1, 3])
def test_sample_rejects_invalid_size(size):
    with pytest.raises(ValueError, match="sample size"):
        profile.split_request_samples([dict(at_s=0, request={})] * 2, size, 1, 0)


def test_real_requests_warmup_then_serving_measurement(monkeypatch):
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
    warmup = [dict(at_s=0, request=dict(prompt=[5, 6, 7, 8], max_tokens=2))]
    result = asyncio.run(
        profile.collect(rows, "http://local", "model", 1, 1, warmup_rows=warmup)
    )
    assert [body.get("method", "request") for _, body in calls] == [
        "set_pp_profile_warmup",
        "request",
        "set_pp_profile_warmup",
        "request",
    ]
    assert calls[1][1]["prompt"] == [5, 6, 7, 8]
    assert calls[0][1]["args"] == [True]
    assert calls[2][1]["args"] == [False]
    assert calls[3][1]["prompt"] == [1, 2, 3, 4]
    assert calls[3][1]["ignore_eos"] is False
    assert result["requests"][0]["usage"]["completion_tokens"] == 2


def test_failed_warmup_prevents_measured_requests(monkeypatch):
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
                warmup_rows=[dict(at_s=0, request=dict(prompt="warmup", max_tokens=2))],
            )
        )
    assert methods == ["set_pp_profile_warmup"]


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


def test_collection_switches_warmup_flag_before_measurement(monkeypatch):
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
            warmup_rows=[dict(at_s=0, request=dict(prompt="warmup", max_tokens=1))],
        )
    )
    assert calls == [
        "set_pp_profile_warmup",
        "request",
        "set_pp_profile_warmup",
        "request",
    ]


@pytest.mark.parametrize("prompt", ["same prompt", [1, 2, 3]])
def test_rejects_overlapping_warmup_before_sending_anything(monkeypatch, prompt):
    def no_post(*args):
        pytest.fail("overlap must be caught before contacting the server")

    monkeypatch.setattr(profile, "post", no_post)
    with pytest.raises(ValueError, match="warmup prompts must differ"):
        asyncio.run(
            profile.collect(
                [dict(at_s=0, request=dict(prompt=prompt, max_tokens=2))],
                "http://local",
                "model",
                1,
                1,
                # Different sampling parameters must not disguise an identical prompt.
                warmup_rows=[dict(at_s=0, request=dict(prompt=prompt, max_tokens=9))],
            )
        )


def test_cli_splits_one_dataset_and_records_sample_indices(tmp_path, monkeypatch):
    source, output = tmp_path / "requests.jsonl", tmp_path / "result.json"
    rows = [
        dict(at_s=0, request=dict(prompt=f"prompt-{i}", max_tokens=2))
        for i in range(10)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(
        "sys.argv",
        [
            "pp_serving_profile",
            str(source),
            "--model",
            "model",
            "--concurrency",
            "1",
            "--warmup-rounds",
            "2",
            "--sample-size",
            "3",
            "--output",
            str(output),
        ],
    )
    calls = []

    def post(base, path, body):
        calls.append(body.get("prompt", body.get("method")))
        return dict(usage=dict(prompt_tokens=1, completion_tokens=2))

    monkeypatch.setattr(profile, "post", post)
    profile.main()
    result = json.loads(output.read_text())
    indices = result["sampling"]["warmup_indices"]
    start, stop = result["sampling"]["measured_index_range"]
    assert calls == [
        "set_pp_profile_warmup",
        f"prompt-{indices[0]}",
        f"prompt-{indices[0]}",
        "set_pp_profile_warmup",
        *[f"prompt-{i}" for i in range(start, stop)],
    ]
    assert result["warmup_requests"] == 1
    assert result["sampling"]["source"] == str(source)
    assert len(result["requests"]) == 3
    assert indices[0] not in range(start, stop)


def test_default_split_reserves_warmup_without_extra_cli_parameters():
    rows = [dict(at_s=i * 0.1, request=dict(prompt=f"prompt-{i}")) for i in range(20)]
    warmup, measured, selection = profile.split_request_samples(rows, None, 4, 3)
    assert len(warmup) == 2
    assert len(measured) == 18
    assert not set(selection["warmup_indices"]) & set(
        range(*selection["measured_index_range"])
    )
    assert warmup[0]["at_s"] == measured[0]["at_s"] == 0


def test_split_matches_exhaustive_feasibility_with_repeated_prompts():
    import random

    rng = random.Random(987)
    for n in range(2, 18):
        rows = [
            dict(at_s=i * 0.1, request=dict(prompt=str(rng.randrange(6))))
            for i in range(n)
        ]
        for size in range(1, n):
            needed = min(3, max(1, n // 10), n - size)
            valid = []
            for start in range(n - size + 1):
                keys = {r["request"]["prompt"] for r in rows[start : start + size]}
                if sum(r["request"]["prompt"] not in keys for r in rows) >= needed:
                    valid.append(start)
            if not valid:
                with pytest.raises(ValueError, match="no disjoint warmup"):
                    profile.split_request_samples(rows, size, 3, 7)
                continue
            warmup, measured, selection = profile.split_request_samples(
                rows, size, 3, 7
            )
            assert selection["measured_index_range"][0] in valid
            assert len(warmup) == needed and len(measured) == size
            assert {r["request"]["prompt"] for r in warmup}.isdisjoint(
                {r["request"]["prompt"] for r in measured}
            )


@pytest.mark.parametrize("rows", [[], [dict(at_s=0, request=dict(prompt="only"))]])
def test_too_small_dataset_rejected(rows):
    with pytest.raises(ValueError, match="at least two"):
        profile.split_request_samples(rows, None, 1, 0)
