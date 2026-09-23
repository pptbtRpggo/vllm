# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay sampled completion requests into a private profiling server.

Each JSONL row is {"at_s": 0.0, "request": {"prompt": ..., "max_tokens": ...}}.
Arrival offsets are relative to the start of each replay. Request fields are
passed through, except model and stream. The client records actual dispatch
times so a concurrency limit or client saturation is visible in the output.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen


def load_requests(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("request sample is empty")
    previous = -1.0
    for row in rows:
        offset = row["at_s"]
        request = row["request"]
        if not math.isfinite(offset) or offset < 0 or offset < previous:
            raise ValueError("at_s must be finite, nonnegative and nondecreasing")
        if not request.get("prompt") or request.get("max_tokens", 0) < 1:
            raise ValueError("each request needs prompt and positive max_tokens")
        if request.get("n", 1) != 1 or request.get("best_of", 1) != 1:
            raise ValueError("profiling replay currently requires n=best_of=1")
        previous = offset
    return rows


def post(base_url: str, path: str, body: dict) -> dict:
    request = Request(
        base_url.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=3600) as response:
        raw = response.read()
    return json.loads(raw) if raw else {}


async def replay(
    rows: list[dict], base_url: str, model: str, concurrency: int
) -> list[dict]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    semaphore = asyncio.Semaphore(concurrency)
    start = time.perf_counter()

    async def send(index: int, row: dict) -> dict:
        await asyncio.sleep(max(0, start + row["at_s"] - time.perf_counter()))
        async with semaphore:
            dispatched = time.perf_counter() - start
            response = await asyncio.to_thread(
                post,
                base_url,
                "/v1/completions",
                row["request"] | dict(model=model, stream=False),
            )
            if "error" in response or "usage" not in response:
                raise RuntimeError(f"request {index} failed: {response}")
            return dict(
                index=index,
                requested_at_s=row["at_s"],
                dispatched_at_s=dispatched,
                completed_at_s=time.perf_counter() - start,
                usage=response["usage"],
            )

    # Drain successful requests even if another request fails.
    results = await asyncio.gather(
        *(send(i, row) for i, row in enumerate(rows)), return_exceptions=True
    )
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results


async def collect(
    rows: list[dict],
    base_url: str,
    model: str,
    concurrency: int,
    warmup_rounds: int,
    *,
    warmup_rows: list[dict],
) -> dict:
    if warmup_rounds < 1:
        raise ValueError("at least one warmup round is required")
    if not rows or not warmup_rows:
        raise ValueError("measured and warmup request samples must both be nonempty")
    measured_prompts = {json.dumps(row["request"]["prompt"]) for row in rows}
    if any(
        json.dumps(row["request"]["prompt"]) in measured_prompts for row in warmup_rows
    ):
        raise ValueError("warmup prompts must differ from measured prompts")

    async def rpc(method: str, args: list) -> None:
        response = await asyncio.to_thread(
            post, base_url, "/collective_rpc", dict(method=method, args=args)
        )
        if "error" in response:
            raise RuntimeError(response)

    await rpc("set_pp_profile_warmup", [True])
    for _ in range(warmup_rounds):
        await replay(warmup_rows, base_url, model, concurrency)
    await rpc("set_pp_profile_warmup", [False])
    results = await replay(rows, base_url, model, concurrency)
    return dict(
        model=model,
        concurrency=concurrency,
        warmup_rounds=warmup_rounds,
        warmup_requests=len(warmup_rows),
        requests=results,
    )


def split_request_samples(
    rows: list[dict], size: int | None, concurrency: int, seed: int
) -> tuple[list[dict], list[dict], dict]:
    """Randomly split one dataset into disjoint warmup and measured samples.

    Keep the measured window contiguous to preserve its arrival rate. Warmup
    is drawn from outside that window, excluding identical prompts even when
    they occur at different dataset indices. Select a feasible window with
    reservoir sampling and a sliding counter, without quadratic rescanning.
    """
    if len(rows) < 2 or concurrency < 1:
        raise ValueError("need at least two requests and positive concurrency")
    warmup_size = min(concurrency, max(1, len(rows) // 10))
    if size is None:
        size = len(rows) - warmup_size
    if not 0 < size < len(rows):
        raise ValueError("sample size must leave requests available for warmup")
    warmup_size = min(warmup_size, len(rows) - size)
    keys = [json.dumps(row["request"]["prompt"]) for row in rows]
    totals = Counter(keys)
    window = Counter(keys[:size])
    available = len(rows) - sum(totals[key] for key in window)
    rng = random.Random(seed)
    selected_start = None
    candidates = 0
    for start in range(len(rows) - size + 1):
        if available >= warmup_size:
            candidates += 1
            if rng.randrange(candidates) == 0:
                selected_start = start
        if start + size == len(rows):
            break
        old, new = keys[start], keys[start + size]
        window[old] -= 1
        if window[old] == 0:
            available += totals[old]
        if window[new] == 0:
            available -= totals[new]
        window[new] += 1
    if selected_start is None:
        raise ValueError(
            "no disjoint warmup prompts available; reduce --sample-size "
            "or use a dataset with more distinct prompts"
        )
    stop = selected_start + size
    measured_keys = set(keys[selected_start:stop])
    warmup_indices = sorted(
        rng.sample(
            [i for i, key in enumerate(keys) if key not in measured_keys], warmup_size
        )
    )

    def rebase(indices):
        origin = rows[indices[0]]["at_s"]
        return [rows[i] | {"at_s": rows[i]["at_s"] - origin} for i in indices]

    return (
        rebase(warmup_indices),
        rebase(range(selected_start, stop)),
        dict(
            measured_index_range=[selected_start, stop], warmup_indices=warmup_indices
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Measured window size; default reserves an automatic warmup sample.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    warmup_rows, rows, selection = split_request_samples(
        load_requests(args.requests), args.sample_size, args.concurrency, args.seed
    )
    # urlopen runs in a thread; allocate enough threads to avoid an implicit
    # default thread-pool cap changing the intended request arrival pattern.
    from concurrent.futures import ThreadPoolExecutor

    async def run() -> dict:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            loop.set_default_executor(executor)
            return await collect(
                rows,
                args.base_url,
                args.model,
                args.concurrency,
                args.warmup_rounds,
                warmup_rows=warmup_rows,
            )

    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    result = asyncio.run(run())
    result["sampling"] = dict(
        source=str(args.requests),
        size=len(rows),
        seed=args.seed,
        contiguous_window=True,
        **selection,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
