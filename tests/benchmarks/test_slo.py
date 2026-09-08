# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import random
from collections import Counter
from types import SimpleNamespace

import pytest
from aiohttp import web

from vllm.benchmarks import serve
from vllm.benchmarks.datasets import SampleRequest, ShareGPTDataset
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput
from vllm.benchmarks.request_records import write_request_records
from vllm.benchmarks.slo import (
    RequestSLO,
    assign_slos,
    evaluate_goodput,
    load_slo_config,
    request_body,
)
from vllm.entrypoints.openai.protocol import CompletionRequest
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.core.sched.tau_batch.scheduler import snapshot_from_request
from vllm.v1.request import Request


def config():
    return {
        "profiles": {
            "tight": {"ttft_slo_ms": 100, "tpot_slo_ms": 10},
            "loose": {"ttft_slo_ms": 300, "tpot_slo_ms": 100},
        },
        "ratios": {"tight": 0.5, "loose": 0.5},
    }


def requests(n):
    return [
        SampleRequest("one two three four", 4, 3, request_id=f"r{i}", source_index=i)
        for i in range(n)
    ]


def test_quotas_seed_and_global_rng_are_independent():
    reqs = requests(7)
    random.seed(19)
    state = random.getstate()
    first = assign_slos(reqs, config(), 5)
    assert random.getstate() == state
    assert first["counts"] == {"loose": 4, "tight": 3}
    assert Counter(r.slo.profile for r in reqs) == first["counts"]
    assert first == assign_slos(requests(7), config(), 5)
    assert first["requests"] != assign_slos(requests(7), config(), 6)["requests"]
    assert [r["source_index"] for r in first["requests"]] == list(range(7))


@pytest.mark.parametrize("fault", ["nan", "negative", "bool", "sum", "keys", "missing"])
def test_invalid_config_rejected(tmp_path, fault):
    value = config()
    if fault in ("nan", "negative", "bool"):
        value["profiles"]["tight"]["ttft_slo_ms"] = {
            "nan": float("nan"),
            "negative": -1,
            "bool": True,
        }[fault]
    elif fault == "sum":
        value["ratios"]["tight"] = 0.9
    elif fault == "keys":
        value["ratios"]["unknown"] = value["ratios"].pop("tight")
    else:
        del value["profiles"]["tight"]["tpot_slo_ms"]
    path = tmp_path / "slo.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_slo_config(path)


def test_sharegpt_source_indices_survive_filtering_shuffle_and_sampling(tmp_path):
    data = [{"conversations": []}]
    data.extend(
        {
            "conversations": [
                {"value": " ".join([str(i)] * i)},
                {"value": "answer"},
            ]
        }
        for i in (3, 4, 8, 10)
    )
    path = tmp_path / "sharegpt.json"
    path.write_text(json.dumps(data))
    sampler = ShareGPTDataset(dataset_path=str(path), random_seed=3)
    tokenizer = lambda text: SimpleNamespace(input_ids=text.split())
    reqs = sampler.sample(tokenizer, 10, output_len=3, no_oversample=True)
    assert {r.source_index for r in reqs} == {2, 3, 4}
    for req in reqs:
        assert req.prompt == data[req.source_index]["conversations"][0]["value"]


def test_body_merge_and_server_snapshot_use_same_thresholds():
    req = requests(1)[0]
    req.slo = RequestSLO("tight", 100, 10)
    shared = {"temperature": 0, "vllm_xargs": {"other": 7, "ttft_slo_ms": 999}}
    body = request_body(shared, req)
    assert shared["vllm_xargs"]["ttft_slo_ms"] == 999
    assert body["vllm_xargs"] == {"other": 7, "ttft_slo_ms": 100, "tpot_slo_ms": 10}
    api = CompletionRequest(model="test", prompt=req.prompt, max_tokens=3, **body)
    params = api.to_sampling_params(3, None)
    internal = Request("r0", [1, 2, 3, 4], params, None, 2)
    snapshot = snapshot_from_request(internal)
    assert (snapshot.ttft_slo_ms, snapshot.tpot_slo_ms) == (100, 10)


def test_per_request_goodput_keeps_failed_request_alignment():
    reqs = requests(4)
    for req, name in zip(reqs, ("tight", "loose", "tight", "loose")):
        req.slo = RequestSLO(name, **config()["profiles"][name])
    outputs = [
        RequestFuncOutput(success=False),
        RequestFuncOutput(success=True, ttft=0.2, latency=0.3, output_tokens=3),
        RequestFuncOutput(success=True, ttft=0.2, latency=0.3, output_tokens=3),
        RequestFuncOutput(success=True, ttft=0.25, latency=0.25, output_tokens=1),
    ]
    metrics, lengths = serve.calculate_metrics(reqs, outputs, 2, None, [], {})
    report = metrics.slo_report
    assert lengths == [0, 3, 3, 1]
    assert [r["attained"] for r in report["requests"]] == [False, True, False, True]
    assert report["by_profile"]["tight"]["total_requests"] == 2
    assert report["by_profile"]["loose"]["attainment_rate"] == 1
    assert report["attainment_rate"] == 0.5
    assert metrics.request_goodput == 1.0


def test_global_goodput_remains_available_and_e2el_still_applies():
    reqs = requests(1)
    outputs = [RequestFuncOutput(success=True, ttft=0.1, latency=0.3, output_tokens=3)]
    assert evaluate_goodput(reqs, outputs, [3], {}, 1) is None
    report = evaluate_goodput(reqs, outputs, [3], {"ttft": 100, "tpot": 100}, 1)
    assert report["good_requests"] == 1
    reqs[0].slo = RequestSLO("loose", 300, 100)
    report = evaluate_goodput(reqs, outputs, [3], {"ttft": 1, "e2el": 200}, 1)
    assert report["requests"][0]["thresholds_ms"]["ttft"] == 300
    assert report["good_requests"] == 0


@pytest.mark.parametrize("compact", [False, True])
def test_actual_http_requests_carry_distinct_slos(tmp_path, compact):
    async def run():
        reqs = requests(4)
        assign_slos(reqs, config(), 7)
        captured = {}

        async def handler(http):
            captured[http.headers["x-request-id"]] = await http.json()
            chunks = [
                {"choices": [{"text": "ok"}]},
                {"choices": [], "usage": {"completion_tokens": 3}},
            ]
            body = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
            return web.Response(
                text=body + "data: [DONE]\n\n", content_type="text/event-stream"
            )

        app = web.Application()
        app.router.add_post("/v1/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        try:
            result = await serve.benchmark(
                task_type=serve.TaskType.GENERATION,
                endpoint_type="vllm",
                api_url=url + "/v1/completions",
                base_url=url,
                model_id="test",
                model_name=None,
                tokenizer=None,
                input_requests=reqs,
                logprobs=None,
                request_rate=float("inf"),
                burstiness=1,
                disable_tqdm=True,
                num_warmups=0,
                profile=False,
                selected_percentile_metrics=[],
                selected_percentiles=[],
                ignore_eos=True,
                goodput_config_dict={},
                max_concurrency=2,
                lora_modules=None,
                extra_headers=None,
                extra_body={"vllm_xargs": {"other": 3}},
                ready_check_timeout_sec=0,
                request_output=str(tmp_path / "requests.jsonl") if compact else None,
            )
            assert result["completed"] == 4
            assert result["request_goodput"] is not None
            if compact:
                assert "requests" not in result["slo_evaluation"]
                records = [
                    json.loads(line)
                    for line in (tmp_path / "requests.jsonl").read_text().splitlines()
                ]
                assert len(records) == 4
                assert all(r["success"] and r["status"] == "completed" for r in records)
                assert not any("generated_text" in r for r in records)
                assert (
                    sum(r["attained"] for r in records)
                    == result["slo_evaluation"]["good_requests"]
                )
            else:
                assert len(result["slo_evaluation"]["requests"]) == 4
            for req in reqs:
                body = captured[req.request_id]
                assert body["vllm_xargs"] == {
                    "other": 3,
                    "ttft_slo_ms": req.slo.ttft_slo_ms,
                    "tpot_slo_ms": req.slo.tpot_slo_ms,
                }
        finally:
            await runner.cleanup()

    asyncio.run(run())


@pytest.mark.parametrize("compact", [False, True])
def test_assignment_is_saved_before_traffic(tmp_path, monkeypatch, compact):
    path = tmp_path / "slo.json"
    path.write_text(json.dumps(config()))
    parser = FlexibleArgumentParser()
    serve.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--model",
            "test",
            "--backend",
            "vllm",
            "--dataset-name",
            "sharegpt",
            "--dataset-path",
            str(tmp_path / "dataset.json"),
            "--num-prompts",
            "7",
            "--slo-config",
            str(path),
            "--slo-seed",
            "5",
            "--result-dir",
            str(tmp_path),
            "--result-filename",
            "result.json",
        ]
    )
    if compact:
        args.request_output = str(tmp_path / "requests.jsonl")
    monkeypatch.setattr(serve, "get_tokenizer", lambda *a, **kw: None)
    monkeypatch.setattr(serve, "get_samples", lambda *a: requests(7))
    monkeypatch.setattr(serve, "freeze_gc_heap", lambda: None)

    async def interrupted(**kwargs):
        if compact:
            records = [
                json.loads(line)
                for line in (tmp_path / "requests.jsonl").read_text().splitlines()
            ]
            assert Counter(r["slo"]["profile"] for r in records) == {
                "loose": 4,
                "tight": 3,
            }
            assert all(
                r["status"] == "planned" and r["success"] is None for r in records
            )
            assert not (tmp_path / "result.slo_assignment.json").exists()
        else:
            saved = json.loads((tmp_path / "result.slo_assignment.json").read_text())
            assert saved["counts"] == {"loose": 4, "tight": 3}
        assert all(r.slo is not None for r in kwargs["input_requests"])
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(serve, "benchmark", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        asyncio.run(serve.main_async(args))


def test_request_records_without_slo_and_atomic_failure(tmp_path):
    path = tmp_path / "requests.jsonl"
    reqs = requests(2)
    write_request_records(path, reqs)
    planned = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_request_records(path, reqs)
    outputs = [
        RequestFuncOutput(success=True, ttft=0.1, latency=0.1, output_tokens=1),
        RequestFuncOutput(success=False, error="connection lost"),
    ]
    # A serialization/indexing failure leaves the previously saved plan intact.
    with pytest.raises(IndexError):
        write_request_records(path, reqs, outputs, [1])
    assert path.read_bytes() == planned
    assert not list(tmp_path.glob(".requests_*"))
    write_request_records(path, reqs, outputs, [1, 0])
    first, second = [json.loads(line) for line in path.read_text().splitlines()]
    assert first["slo"] is None and first["attained"] is None
    assert first["observed_ms"]["tpot"] == 0
    assert not second["success"]
    assert second["error"] == "connection lost"
    assert all(v is None for v in second["observed_ms"].values())
