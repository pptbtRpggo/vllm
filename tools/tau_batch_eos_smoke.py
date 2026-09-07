# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Send three small EOS probes to an existing traced TauScheduler service.

Uses only the Python stdlib. Does not start/restart services or install packages.
"""

import argparse
import json
import time
import urllib.request
import uuid
from pathlib import Path


def request_json(url, payload=None, request_id=None):
    headers = {"Content-Type": "application/json"}
    if request_id:
        headers["X-Request-Id"] = request_id
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def require_idle(base_url):
    with urllib.request.urlopen(base_url + "/metrics", timeout=10) as response:
        lines = response.read().decode().splitlines()
    for name in ("num_requests_running", "num_requests_waiting"):
        values = [
            float(line.split()[-1])
            for line in lines
            if line.startswith("vllm:" + name + "{")
        ]
        if not values or any(value != 0 for value in values):
            raise RuntimeError(f"Service must be idle: {name}={values}")


def cases(eos_id, other_id):
    return [
        ("prefill_eos", dict(allowed_token_ids=[eos_id]), [eos_id], "stop"),
        (
            "decode_eos",
            dict(
                allowed_token_ids=[other_id, eos_id],
                logit_bias={str(eos_id): 100, str(other_id): -100},
                min_tokens=1,
            ),
            [other_id, eos_id],
            "stop",
        ),
        (
            "ignore_eos",
            dict(allowed_token_ids=[eos_id], ignore_eos=True, max_tokens=3),
            [eos_id] * 3,
            "length",
        ),
    ]


def validate(response, events, expected_tokens, reason, pp_size):
    choice = response["choices"][0]
    req_id = response["id"] + "-0"
    emitted = [
        e for e in events if e.get("event") == "emit" and req_id in e.get("req_ids", [])
    ]
    finished = [
        e
        for e in events
        if e.get("event") == "eos" and req_id in e.get("finished_ids", [])
    ]
    done = [
        e for e in events if e.get("event") == "done" and req_id in e.get("req_ids", [])
    ]
    expected_phases = ["prefill"] + ["decode"] * (len(expected_tokens) - 1)
    checks = {
        "sampled_token_ids": choice.get("token_ids") == expected_tokens,
        "finish_reason": choice["finish_reason"] == reason,
        "completion_tokens": (
            response["usage"]["completion_tokens"] == len(expected_tokens)
        ),
        "forward_phases": [e["phase"] for e in emitted] == expected_phases,
        "all_forwards_finished": (
            [e["fwd_id"] for e in done] == [e["fwd_id"] for e in emitted]
        ),
        "one_completion_event": len(finished) == 1,
        "completion_phase": (
            len(finished) == 1 and finished[0]["phase"] == expected_phases[-1]
        ),
        "completion_after_last_forward": (
            bool(done)
            and len(finished) == 1
            and events.index(finished[0]) > events.index(done[-1])
        ),
        "all_pp_ranks_computed": bool(emitted)
        and all(
            {
                e["pp_rank"]
                for e in events
                if e.get("event") == "compute" and e.get("fwd_id") == emit["fwd_id"]
            }
            == set(range(pp_size))
            for emit in emitted
        ),
    }
    return dict(
        passed=all(checks.values()),
        checks=checks,
        request_id=req_id,
        token_ids=choice.get("token_ids"),
        finish_reason=choice["finish_reason"],
        forward_phases=[e["phase"] for e in emitted],
        eos_events=finished,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--eos-token-id", type=int)
    parser.add_argument("--other-token-id", type=int)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "run.json").read_text())
    config = json.loads((Path(manifest["model"]) / "config.json").read_text())
    eos_id = args.eos_token_id
    if eos_id is None:
        eos_id = config.get("eos_token_id")
    other_id = args.other_token_id
    if other_id is None:
        other_id = config.get("bos_token_id")
    if not isinstance(eos_id, int) or not isinstance(other_id, int):
        parser.error("Supply explicit --eos-token-id and --other-token-id")
    if eos_id == other_id:
        parser.error("The two token IDs must differ")
    trace = Path(manifest["trace"])
    pp_size = int(manifest["settings"]["PP"])
    base_url = args.base_url.rstrip("/")
    require_idle(base_url)
    served_models = request_json(base_url + "/v1/models")["data"]
    if not any(m["id"] == manifest["model"] for m in served_models):
        raise RuntimeError("Service model does not match run.json")
    output = (
        args.run_dir
        / "bench"
        / ("eos_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    )
    output.mkdir(parents=True)
    report = dict(model=manifest["model"], pp_size=pp_size, cases=[])
    print(f"Results: {output}", flush=True)
    try:
        for name, overrides, tokens, reason in cases(eos_id, other_id):
            require_idle(base_url)
            payload = dict(
                model=manifest["model"],
                prompt="Hello",
                temperature=0,
                max_tokens=8,
                min_tokens=0,
                ignore_eos=False,
                stop_token_ids=[],
                return_token_ids=True,
            )
            payload.update(overrides)
            (output / f"{name}_request.json").write_text(json.dumps(payload, indent=2))
            before = trace.stat()
            response = request_json(
                base_url + "/v1/completions",
                payload,
                "tau-eos-" + uuid.uuid4().hex,
            )
            (output / f"{name}_response.json").write_text(
                json.dumps(response, indent=2)
            )
            # The API response can arrive just before the scheduler writes eos.
            deadline = time.monotonic() + 10
            while True:
                after = trace.stat()
                if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                    raise RuntimeError("Trace file replaced during test")
                if after.st_size < before.st_size:
                    raise RuntimeError("Trace file truncated during test")
                with trace.open("rb") as stream:
                    stream.seek(before.st_size)
                    data = stream.read(after.st_size - before.st_size)
                complete = data[: data.rfind(b"\n") + 1]
                events = [json.loads(line) for line in complete.splitlines()]
                if (
                    any(
                        e.get("event") == "eos"
                        and response["id"] + "-0" in e.get("finished_ids", [])
                        for e in events
                    )
                    or time.monotonic() >= deadline
                ):
                    break
                time.sleep(0.05)
            (output / f"{name}_trace.jsonl").write_bytes(complete)
            result = validate(response, events, tokens, reason, pp_size)
            result.update(
                name=name,
                trace_start_offset=before.st_size,
                trace_end_offset=before.st_size + len(complete),
            )
            report["cases"].append(result)
            print(f"{name}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
            if not result["passed"]:
                raise RuntimeError(f"Probe failed: {result['checks']}")
        require_idle(base_url)
        report["passed"] = True
    except Exception as exc:
        report.update(passed=False, error=str(exc))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
