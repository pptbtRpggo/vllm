# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Save four small EOS diagnostic requests and print only short summaries."""

import argparse
import datetime
import json
import os
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def make_cases(model, eos, max_tokens):
    """Separate forced stop-policy controls from natural raw/chat generation."""
    base = dict(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        min_tokens=0,
        ignore_eos=False,
        return_token_ids=True,
    )
    raw = dict(base, prompt="Reply with only the word OK.")
    return [
        (
            "forced_eos_stop",
            "/v1/completions",
            dict(raw, allowed_token_ids=[eos], max_tokens=4, skip_special_tokens=False),
        ),
        (
            "forced_eos_ignore",
            "/v1/completions",
            dict(
                raw,
                allowed_token_ids=[eos],
                max_tokens=4,
                ignore_eos=True,
                skip_special_tokens=False,
            ),
        ),
        ("natural_completion", "/v1/completions", raw),
        (
            "natural_chat",
            "/v1/chat/completions",
            dict(base, messages=[{"role": "user", "content": raw["prompt"]}]),
        ),
    ]


def summarize(name, response, eos):
    """Missing returned IDs do not prove that EOS was absent."""
    choice = response["choices"][0]
    ids = choice.get("token_ids")
    text = choice.get("text") or (choice.get("message") or {}).get("content") or ""
    return dict(
        case=name,
        finish_reason=choice.get("finish_reason"),
        stop_reason=choice.get("stop_reason"),
        completion_tokens=response.get("usage", {}).get("completion_tokens"),
        eos_in_returned_ids=(eos in ids if ids is not None else None),
        text_preview=text[:80],
    )


def read_eos(model_dir, explicit=None):
    """Read local configuration only; do not import model code or weights."""
    configs = {}
    for filename in ("config.json", "generation_config.json"):
        path = model_dir / filename
        if path.is_file():
            configs[filename] = json.loads(path.read_text())
    if explicit is not None:
        return explicit, "--eos-token-id", configs
    for name, config in configs.items():
        value = config.get("eos_token_id")
        candidates = value if isinstance(value, list) else [value]
        for candidate in candidates:
            if type(candidate) is int and candidate >= 0:
                return candidate, name, configs
    raise ValueError("本地配置没有 EOS ID；请用 --model-dir 或 --eos-token-id 指定")


def run_cases(cases, url, output, eos, timeout):
    """Send once per case; preserve HTTP errors and full replies for diagnosis."""
    summaries = []
    failed = False
    for name, endpoint, body in cases:
        artifact = dict(endpoint=endpoint, request=body)
        try:
            request = urllib.request.Request(
                url.rstrip("/") + endpoint,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                artifact["response_text"] = response.read().decode()
            result = json.loads(artifact["response_text"])
            artifact["response"] = result
            summary = summarize(name, result, eos)
        except Exception as exc:
            failed = True
            artifact["error"] = str(exc)
            if isinstance(exc, urllib.error.HTTPError):
                artifact["response_text"] = exc.read().decode(errors="replace")
            summary = dict(case=name, error=str(exc))
        (output / f"{name}.json").write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n"
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    stop, ignore = summaries[:2]
    policy = (
        stop.get("finish_reason") == "stop" and ignore.get("finish_reason") == "length"
    )
    report = dict(
        summaries=summaries,
        forced_stop_policy_observed=policy,
        model_correctness_validated=False,
        note="强制 EOS 只检查停止路径；自然生成 length 不证明 ignore_eos 失效。",
    )
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return 1 if failed or not policy else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/model"))
    parser.add_argument("--model", default="hetero-model")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--eos-token-id", type=int)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_tokens <= 0 or not 0 < args.timeout <= 600:
        parser.error("max-tokens 必须为正数，timeout 范围 (0,600]")
    if args.eos_token_id is not None and args.eos_token_id < 0:
        parser.error("EOS ID 不能为负数")
    eos, source, configs = read_eos(args.model_dir, args.eos_token_id)
    root = Path("/work/hetero_logs")
    if not root.is_dir():
        root = Path.home() / "vllm_hetero_pp_logs"
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or root / f"eos_check_{stamp}_{uuid.uuid4().hex[:8]}"
    os.umask(0o022)
    output.mkdir(parents=True, exist_ok=False)
    (output / "configuration.json").write_text(
        json.dumps(
            dict(eos_token_id=eos, eos_source=source, configs=configs),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(f"诊断目录：{output}；EOS={eos}（{source}）；每项只请求一次。", flush=True)
    return run_cases(
        make_cases(args.model, eos, args.max_tokens),
        args.url,
        output,
        eos,
        args.timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
