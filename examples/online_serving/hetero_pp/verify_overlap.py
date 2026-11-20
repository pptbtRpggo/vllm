# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify same-host PP overlap from Ascend's torch-to-NPU trace flows."""

import argparse
import json
import time
import urllib.request
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import regex as re

MARKER = re.compile(r"hetero_pp\.forward\.([01])\.(\d+)")
COMPUTE_TYPES = {"AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV"}
PROMPT_TOKENS = 384
OUTPUT_TOKENS = 8


def trace_batches(events, start, end):
    """Map kernels to batch IDs using CANN's flows and annotated host forwards."""
    markers, kernels = defaultdict(list), defaultdict(list)
    starts, finishes = defaultdict(list), defaultdict(list)
    ranks = set()
    for event in events:
        phase = event.get("ph")
        name = event.get("name", "")
        if phase == "X":
            marker = MARKER.fullmatch(name)
            task_type = (event.get("args") or {}).get("Task Type")
            if marker is None and task_type not in COMPUTE_TYPES:
                continue
            thread = (event["pid"], event["tid"])
            ts = Decimal(str(event["ts"]))
            stop = ts + Decimal(str(event["dur"]))
            if not start <= ts < stop <= end:
                continue
            if marker:
                rank, batch = map(int, marker.groups())
                ranks.add(rank)
                markers[thread].append((ts, stop, batch))
            else:
                kernels[(*thread, ts)].append((ts, stop, name))
        elif name == "torch_to_npu" and phase in ("s", "f"):
            target = starts if phase == "s" else finishes
            target[event["id"]].append(
                (event["pid"], event["tid"], Decimal(str(event["ts"])))
            )

    if not ranks:
        return None, {}
    if len(ranks) != 1:
        raise ValueError("one worker trace contains multiple stage IDs")
    batches = defaultdict(set)
    for flow in starts.keys() & finishes.keys():
        if len(starts[flow]) != 1 or len(finishes[flow]) != 1:
            continue
        pid, tid, launch = starts[flow][0]
        device_key = finishes[flow][0]
        batch_ids = [b for lo, hi, b in markers[(pid, tid)] if lo <= launch < hi]
        if len(batch_ids) == 1 and len(kernels[device_key]) == 1:
            kernel = kernels[device_key][0]
            if launch <= kernel[0]:
                batches[batch_ids[0]].add(kernel)
    return ranks.pop(), {batch: sorted(values) for batch, values in batches.items()}


def intersect_kernels(earlier, later):
    """Return a pair of actual intersecting kernels, excluding gaps between them."""
    i = j = 0
    while i < len(earlier) and j < len(later):
        first, second = earlier[i], later[j]
        overlap = min(first[1], second[1]) - max(first[0], second[0])
        if overlap > 0:
            return first, second, overlap
        if first[1] <= second[1]:
            i += 1
        else:
            j += 1
    return None


def check_traces(root, start_ns, end_ns):
    stages, paths = {}, {}
    start, end = Decimal(start_ns) / 1000, Decimal(end_ns) / 1000
    for path in sorted(root.rglob("trace_view.json")):
        events = json.loads(path.read_text(), parse_float=Decimal)
        if isinstance(events, dict):
            events = events["traceEvents"]
        rank, batches = trace_batches(events, start, end)
        if rank is None:
            continue
        if rank in stages:
            raise ValueError(f"multiple worker traces for stage {rank}")
        stages[rank], paths[rank] = batches, str(path.relative_to(root))

    common = sorted(stages.get(0, {}).keys() & stages.get(1, {}).keys())
    report = dict(
        status="inconclusive",
        stages=[0, 1],
        clock_scope="same_host",
        traces=paths,
        common_batches=common,
        capture_wall_time_ns=[start_ns, end_ns],
    )
    if len(common) < 2:
        return dict(report, reason="fewer than two batches with kernels on both stages")
    for batch in common:
        # Stage 1's matching forward consumes Stage 0 output after its D2H copy.
        # A violation makes these device timestamps unsuitable for comparison.
        if max(k[1] for k in stages[0][batch]) > stages[1][batch][0][0]:
            reason = f"same-batch stage dependency violated for {batch}"
            return dict(report, reason=reason)
    for index, older in enumerate(common[:-1]):
        for newer in common[index + 1 :]:
            matched = intersect_kernels(stages[1][older], stages[0][newer])
            if matched is not None:
                left, right, duration = matched
                return dict(
                    report,
                    status="passed",
                    witness=dict(
                        stage0_batch=newer,
                        stage1_batch=older,
                        stage0_kernel=[str(right[0]), str(right[1]), right[2]],
                        stage1_kernel=[str(left[0]), str(left[1]), left[2]],
                        overlap_us=float(duration),
                    ),
                )
    return dict(report, status="not_observed")


def verify(port, output):
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def post(path, body=None):
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/{path}",
            data=json.dumps(body).encode() if body is not None else b"",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with client.open(request, timeout=120) as response:
            return response.read()

    def complete(name):
        prompts = [[token] + [42] * (PROMPT_TOKENS - 1) for token in range(41, 45)]
        body = dict(
            model="hetero-model",
            prompt=prompts,
            temperature=0,
            max_tokens=OUTPUT_TOKENS,
            ignore_eos=True,
            return_token_ids=True,
        )
        data = post("v1/completions", body)
        (output / f"overlap_{name}_response.json").write_bytes(data)
        result = json.loads(data)
        choices = {choice["index"]: choice["token_ids"] for choice in result["choices"]}
        if (
            sorted(choices) != list(range(4))
            or len(result["choices"]) != 4
            or any(len(ids or []) != OUTPUT_TOKENS for ids in choices.values())
            or result["usage"]["completion_tokens"] != 4 * OUTPUT_TOKENS
        ):
            raise ValueError(f"{name} responses are missing complete token IDs")
        return choices

    warmup = complete("warmup")
    post("start_profile")
    try:
        start = time.time_ns()
        measured = complete("measured")
        end = time.time_ns()
    finally:
        post("stop_profile")
    if measured != warmup:
        raise ValueError("greedy output token IDs changed during capture")
    return dict(
        check_traces(output / "overlap_traces", start, end), token_ids_matched=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = verify(args.port, args.output)
    except Exception as error:
        report = dict(status="inconclusive", reason=f"{type(error).__name__}: {error}")
    path = args.output / "overlap_verification.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PP NPU overlap: {report['status']}; {path}", flush=True)
    return {"passed": 0, "not_observed": 1, "inconclusive": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
