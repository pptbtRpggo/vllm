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

    # Drain successful requests even if another request fails; do not run link
    # profiling after a failed workload or while requests are still in flight.
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
    measure_links: bool = False,
) -> dict:
    if warmup_rounds < 1:
        raise ValueError("at least one warmup round is required")

    async def rpc(method: str, args: list) -> None:
        response = await asyncio.to_thread(
            post, base_url, "/collective_rpc", dict(method=method, args=args)
        )
        if "error" in response:
            raise RuntimeError(response)

    await rpc("set_pp_profile_warmup", [True])
    for _ in range(warmup_rounds):
        await replay(rows, base_url, model, concurrency)
    await rpc("set_pp_profile_warmup", [False])
    results = await replay(rows, base_url, model, concurrency)
    if measure_links:
        await rpc("profile_pp_links", [])
    return dict(
        model=model,
        concurrency=concurrency,
        warmup_rounds=warmup_rounds,
        requests=results,
    )


def sample_request_window(rows: list[dict], size: int, seed: int) -> list[dict]:
    """Sample a contiguous arrival window, preserving spacing and request fields.

    Independent random request selection would silently dilute arrival rate.
    Do not mutate the source rows, since they may also be used for validation.
    """
    if not 0 < size <= len(rows):
        raise ValueError("sample size must be between 1 and the number of requests")
    start = random.Random(seed).randrange(len(rows) - size + 1)
    selected = rows[start : start + size]
    origin = selected[0]["at_s"]
    return [row | {"at_s": row["at_s"] - origin} for row in selected]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Sample a contiguous window of service requests.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--measure-links",
        action="store_true",
        help="Also collect idle link measurements for --comm-source replay.",
    )
    args = parser.parse_args()
    rows = load_requests(args.requests)
    if args.sample_size is not None:
        rows = sample_request_window(rows, args.sample_size, args.seed)
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
                measure_links=args.measure_links,
            )

    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    result = asyncio.run(run())
    result["sampling"] = dict(
        source=str(args.requests),
        size=len(rows),
        seed=args.seed,
        contiguous_window=args.sample_size is not None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
