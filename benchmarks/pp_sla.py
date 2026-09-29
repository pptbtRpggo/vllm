# SPDX-License-Identifier: Apache-2.0
"""Prepare disjoint ShareGPT workloads and measure closed-loop PP serving SLA.

This script deliberately does not import vLLM: it can prepare data and drive an
already running OpenAI-compatible server without initializing a device runtime.
Run with --help. Per-request output lengths come from reference answers, not a
single fixed output length. Performance measurements are not quality scores.
"""

import argparse
import asyncio
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def quantile(values, percentile):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * percentile / 100
    lo = int(pos)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)


def length_stats(rows):
    return {
        key: {
            "mean": statistics.mean(r[key] for r in rows),
            **{f"p{p}": quantile([r[key] for r in rows], p) for p in (50, 90, 99)},
            "max": max(r[key] for r in rows),
        }
        for key in ("input_tokens", "output_tokens")
    }


def split_samples(rows, sizes, seed):
    """Disjoint sampling, stratified by joint input/output log2 length bins."""
    if sum(sizes.values()) > len(rows) or any(n < 1 for n in sizes.values()):
        raise ValueError("not enough distinct eligible prompts for requested splits")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[
            (row["input_tokens"].bit_length(), row["output_tokens"].bit_length())
        ].append(row)
    for group in groups.values():
        rng.shuffle(group)
    result = {}
    for name, size in sizes.items():
        total = sum(map(len, groups.values()))
        exact = {k: size * len(v) / total for k, v in groups.items()}
        quotas = {k: math.floor(v) for k, v in exact.items()}
        for key in sorted(
            groups, key=lambda k: (exact[k] - quotas[k], k), reverse=True
        )[: size - sum(quotas.values())]:
            quotas[key] += 1
        selected = []
        for key, count in quotas.items():
            selected.extend(groups[key][:count])
            del groups[key][:count]
        rng.shuffle(selected)
        result[name] = selected
    return result


def prepare(args):
    from transformers import AutoTokenizer

    out = Path(args.output)
    if (out / "manifest.json").exists():
        raise FileExistsError("refusing to overwrite an existing sample manifest")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    raw_path = Path(args.dataset)
    raw = json.loads(raw_path.read_text())
    eligible, seen, rejected = [], set(), Counter()
    for start in range(0, len(raw), 256):
        pairs = []
        for offset, entry in enumerate(raw[start : start + 256]):
            turns = entry.get("conversations", [])
            if (
                len(turns) < 2
                or turns[0].get("from") not in ("human", "user")
                or turns[1].get("from") not in ("gpt", "assistant")
            ):
                rejected["invalid_first_pair"] += 1
                continue
            prompt, answer = turns[0].get("value"), turns[1].get("value")
            if not isinstance(prompt, str) or not isinstance(answer, str):
                rejected["nontext"] += 1
                continue
            pairs.append((start + offset, prompt, answer))
        if not pairs:
            continue
        prompts = tokenizer([p[1] for p in pairs], add_special_tokens=True)["input_ids"]
        answers = tokenizer([p[2] for p in pairs], add_special_tokens=False)[
            "input_ids"
        ]
        for (index, _, _), ids, answer in zip(pairs, prompts, answers):
            a, b = len(ids), len(answer)
            if not (
                4 <= a <= args.max_input
                and 4 <= b <= args.max_output
                and a + b <= args.max_total
            ):
                rejected["length"] += 1
                continue
            digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            if digest in seen:
                rejected["duplicate_prompt"] += 1
                continue
            seen.add(digest)
            eligible.append(
                dict(
                    id=index,
                    prompt_sha256=digest,
                    prompt=ids,
                    input_tokens=a,
                    output_tokens=b,
                )
            )
        if start % 10240 == 0:
            print(
                f"TOKENIZED {min(start + 256, len(raw))}/{len(raw)} "
                f"eligible={len(eligible)}",
                flush=True,
            )
    sizes = dict(warmup=args.warmup, profile=args.profile, evaluation=args.evaluation)
    splits = split_samples(eligible, sizes, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        model=args.model,
        dataset=str(raw_path),
        seed=args.seed,
        input_max=args.max_input,
        output_max=args.max_output,
        total_max=args.max_total,
        raw_records=len(raw),
        eligible=len(eligible),
        rejected=dict(rejected),
        eligible_lengths=length_stats(eligible),
        splits={},
    )
    for name, rows in splits.items():
        path = out / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        manifest["splits"][name] = dict(
            count=len(rows),
            lengths=length_stats(rows),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    save(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def read_samples(path):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    for row in rows:
        if len(row["prompt"]) != row["input_tokens"] or row["output_tokens"] < 1:
            raise ValueError("invalid request token counts")
    return rows


def summarize(records, elapsed_s, thresholds=(150, 500)):
    good = [r for r in records if r["success"]]
    failed = len(records) - len(good)
    ttfts = [r["ttft_ms"] for r in good]
    percentiles = {f"p{p}_ttft_ms": quantile(ttfts, p) for p in (50, 90, 99)}
    return dict(
        requests=len(records),
        successful=len(good),
        failed=failed,
        elapsed_s=elapsed_s,
        **percentiles,
        output_tokens=sum(r["output_tokens"] for r in good),
        output_tokens_per_s=sum(r["output_tokens"] for r in good) / elapsed_s,
        mean_e2e_ms=statistics.mean(r["e2e_ms"] for r in good) if good else None,
        sla={
            str(t): bool(not failed and ttfts and percentiles["p99_ttft_ms"] <= t)
            for t in thresholds
        },
    )


async def request_one(session, base_url, model, row, user_id, output_override=None):
    started = time.perf_counter()
    target = row["output_tokens"] if output_override is None else output_override
    result = dict(
        id=row["id"],
        user_id=user_id,
        access_point=0,
        prompt_sha256=row["prompt_sha256"],
        input_tokens=row["input_tokens"],
        requested_output_tokens=target,
        success=False,
    )
    tokens, first, usage, done = [], None, None, False
    try:
        body = dict(
            model=model,
            prompt=row["prompt"],
            max_tokens=target,
            ignore_eos=True,
            temperature=0,
            stream=True,
            return_token_ids=True,
            stream_options={"include_usage": True},
        )
        async with session.post(
            base_url.rstrip("/") + "/v1/completions", json=body
        ) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"HTTP {response.status}: {(await response.text())[:500]}"
                )
            async for raw in response.content:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                if line[6:] == "[DONE]":
                    done = True
                    break
                event = json.loads(line[6:])
                if "error" in event:
                    raise RuntimeError(str(event["error"]))
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    ids = choice.get("token_ids") or []
                    if first is None and (ids or choice.get("text")):
                        first = time.perf_counter()
                    tokens.extend(ids)
        if not done or first is None or usage is None:
            raise RuntimeError("incomplete stream, first token, or usage")
        if (
            usage["prompt_tokens"] != row["input_tokens"]
            or usage["completion_tokens"] != target
            or len(tokens) != target
        ):
            raise RuntimeError(f"token count mismatch: {usage}, ids={len(tokens)}")
        result.update(
            success=True,
            output_tokens=len(tokens),
            ttft_ms=(first - started) * 1000,
            token_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        )
    except Exception as error:
        result["error"] = str(error)
    result["e2e_ms"] = (time.perf_counter() - started) * 1000
    return result


async def run_load(rows, base_url, model, concurrency, output_override=None):
    """Closed loop: C users, at most one outstanding request per user.

    Users immediately submit another request after completion until the fixed list
    is consumed. Record actual users and all failures. Fill/drain remain explicit;
    large formal samples, not the small pilot, are required for tail SLA claims.
    """
    import aiohttp

    if concurrency < 1 or len(rows) < concurrency:
        raise ValueError("need at least one request per concurrent user")
    cursor, records = 0, []
    started = time.perf_counter()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=1800),
        connector=aiohttp.TCPConnector(limit=concurrency),
    ) as session:

        async def worker(user_id):
            nonlocal cursor
            while cursor < len(rows):
                index = cursor
                cursor += 1
                result = await request_one(
                    session, base_url, model, rows[index], user_id, output_override
                )
                result.update(index=index, completed_at_s=time.perf_counter() - started)
                records.append(result)

        await asyncio.gather(*(worker(i) for i in range(concurrency)))
    return dict(
        summary=summarize(records, time.perf_counter() - started),
        concurrency=concurrency,
        output_override=output_override,
        records=sorted(records, key=lambda r: r["index"]),
    )


def capacity_summary(points, thresholds=(150, 500)):
    """No passing point means zero observed capacity, not a fictitious baseline."""
    result = {}
    for threshold in thresholds:
        passes = sorted(
            p["concurrency"] for p in points if p["summary"]["sla"][str(threshold)]
        )
        failures = sorted(
            p["concurrency"] for p in points if not p["summary"]["sla"][str(threshold)]
        )
        result[str(threshold)] = dict(
            max_tested_passing_concurrency=max(passes, default=0),
            tested_passing=passes,
            tested_failing=failures,
            monotonic_observations=not passes
            or not failures
            or max(passes) < min(failures),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--max-input", type=int, default=2048)
    p.add_argument("--max-output", type=int, default=1024)
    p.add_argument("--max-total", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=512)
    p.add_argument("--profile", type=int, default=1024)
    p.add_argument("--evaluation", type=int, default=8192)
    p = sub.add_parser("bench")
    p.add_argument("--samples", required=True)
    p.add_argument("--warmup-samples", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:18762")
    p.add_argument("--model", default="pp-sla-34b")
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--requests", type=int, default=2048)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        rows = read_samples(args.samples)
        warm = read_samples(args.warmup_samples)
        if {x["prompt_sha256"] for x in warm} & {x["prompt_sha256"] for x in rows}:
            raise ValueError("warmup overlaps measured prompts")
        if len(rows) < args.requests or len(warm) < args.concurrency:
            raise ValueError("insufficient samples")
        warmed = asyncio.run(
            run_load(
                warm[: max(args.concurrency, 16)],
                args.base_url,
                args.model,
                args.concurrency,
            )
        )
        if warmed["summary"]["failed"]:
            raise RuntimeError("warmup failed")
        result = asyncio.run(
            run_load(rows[: args.requests], args.base_url, args.model, args.concurrency)
        )
        result.update(
            warmup=warmed,
            sample_file=args.samples,
            sample_scope="pilot" if args.requests < 1000 else "formal_candidate",
        )
        save(args.output, result)
        print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
