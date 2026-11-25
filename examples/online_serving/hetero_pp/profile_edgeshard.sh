#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 在 head 的匹配服务环境内运行；当前推理服务已退出。
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
# 修改下面四项，或运行前 export 同名变量。
MODEL="${MODEL:-}"                    # 容器内模型目录，两边相同路径可用
PLAN="${PLAN:-}"                      # 当前已跑通的 PP4 topology JSON
RAY_ADDRESS="${RAY_ADDRESS:-}"         # 例如 141.147.135.53:16379
TRACE_ROOT="${TRACE_ROOT:-}"           # 本地结果根目录；自动回收远端 rank 记录
PROFILE_COMPUTE_TIMING="${PROFILE_COMPUTE_TIMING:-event}" # event / host-sync
PROFILE_TRANSPORT="${PROFILE_TRANSPORT:-gloo}"
PY="${PY:-${VIRTUAL_ENV:-$REPO/.venv}/bin/python}"

check_timing() {
  "$PY" - "$1" <<'PY'
import json
import math
import sys
from pathlib import Path

directory = Path(sys.argv[1])
paths = sorted(directory.glob("pp_stage_pp*_tp0.jsonl"))
if not directory.is_dir() or not paths:
    raise SystemExit(f"没有找到 pp_stage_pp*_tp0.jsonl：{directory}")
checked = invalid = 0
for path in paths:
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("is_warmup", False):
                continue
            shape = row.get("batch_shape")
            if shape is not None:
                ctx = sum(s["prompt_tokens"] for s in shape)
                gen = sum(s["query_tokens"] - s["prompt_tokens"] for s in shape)
            else:
                ctx, gen = row.get("num_ctx_tokens", 0), row.get("num_generation_tokens", 0)
            if ctx <= 0 or gen != 0:
                continue
            checked += 1
            fields = {k: row.get(k) for k in (
                "compute_wall_ms", "non_layer_compute_ms", "runner_overhead_ms")}
            fields.update({k: row[k] for k in (
                "embedding_ms", "lm_head_ms", "final_norm_ms") if row.get(k) is not None})
            layers = row.get("layer_compute_ms")
            if isinstance(layers, dict) and layers:
                fields.update({f"layer_compute_ms[{k}]": v for k, v in layers.items()})
            else:
                fields["layer_compute_ms"] = layers
            for name, value in fields.items():
                if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                    continue
                invalid += 1
                if invalid <= 20:
                    print(f"{path.name}:{line_number} pp_rank={row.get('pp_rank')} "
                          f"step={row.get('step')} field={name} value={value!r} "
                          f"type={type(value).__name__}")
print(f"checked_prefill_rows={checked} invalid_fields={invalid}")
print("只读数值检查，不修改原数据、不加载模型；不代替完整 profile 验收。")
if not checked:
    raise SystemExit("没有非预热的 prefill 记录，请核对本轮结果目录。")
raise SystemExit(1 if invalid else 0)
PY
}

profile_statistics() {
  "$PY" - "$1" "$REPO" <<'PY'
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

directory = Path(sys.argv[1]).resolve()
manifest = json.loads((directory / "profile_manifest.json").read_text())
compute_timing = manifest.get("compute_timing", "event")
transport = manifest.get("transport")
if compute_timing not in ("event", "host-sync"):
    raise ValueError("未知 compute timing")
if (manifest.get("complete") is not True
        or manifest.get("version") != 1
        or transport not in ("cpu_gloo", "native_hccl")
        or manifest.get("comm_timing") != "ack"):
    raise ValueError("速率统计需要完整的 PP ACK profile manifest")

# 只加载无 Torch 依赖的配对校验器；离线统计不初始化 NPU/Ray。
source = Path(sys.argv[2]) / "vllm/distributed/pp_comm_trace.py"
spec = importlib.util.spec_from_file_location("pp_comm_trace", source)
pairing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pairing)
by_rank = []
for rank in range(4):
    name = f"pp_stage_pp{rank}_tp0.jsonl"
    content = (directory / name).read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest.get("raw_files", {}).get(name):
        raise ValueError(f"原始文件 SHA 不匹配：{name}")
    indexed = {}
    for line in content.decode().splitlines():
        row = json.loads(line)
        if (row.get("pp_rank") != rank or row.get("tp_rank") != 0
                or row.get("pp_size") != 4 or row.get("tp_size") != 1
                or row.get("trace_session") != manifest["trace_session"]
                or row.get("transport") != transport
                or row.get("comm_timing") != "ack"
                or row.get("compute_timing", "event") != compute_timing
                or type(row.get("is_warmup")) is not bool
                or type(row.get("step")) is not int):
            raise ValueError(f"rank{rank} 原始记录身份或模式错误")
        key = row["step"]
        if key in indexed:
            raise ValueError(f"rank{rank} 重复 step：{key}")
        indexed[key] = row
    by_rank.append(indexed)
if any(set(rows) != set(by_rank[0]) for rows in by_rank[1:]):
    raise ValueError("四个 stage 的 step 不完整匹配")

samples, handoff_samples = [], []
warmup_count = zero_count = 0
for rank in range(3):
    for step, sender in sorted(by_rank[rank].items()):
        receiver = by_rank[rank + 1][step]
        status, timing = pairing.paired_transfer(sender, receiver)
        if status != "paired_ack":
            raise ValueError(f"stage{rank}->{rank + 1} step{step}: {status}")
        if any(sender.get(key) != receiver.get(key) for key in (
                "is_warmup", "num_ctx_tokens", "num_generation_tokens")):
            raise ValueError("两端的预热标记或 batch 阶段不匹配")
        if sender["is_warmup"]:
            warmup_count += 1
            continue
        ctx, gen = sender.get("num_ctx_tokens"), sender.get("num_generation_tokens")
        if any(type(n) is not int or n < 0 for n in (ctx, gen)) or ctx + gen == 0:
            raise ValueError("缺少有效的 prefill/decode token 计数")
        phase = "mixed" if ctx and gen else "prefill" if ctx else "decode"
        if phase != "prefill":
            continue
        payload, elapsed = sender["send_bytes"], timing["send_ack_ms"]
        handoff_samples.append(dict(
            item=f"stage{rank}->{rank + 1}", kind="handoff", phase=phase,
            step=step, elapsed_ms=elapsed,
        ))
        if payload == 0:
            zero_count += 1
            continue
        samples.append(dict(
            link=f"stage{rank}->{rank + 1}", phase=phase, step=step,
            payload_bytes=payload, handoff_ms=elapsed,
            speed_gbps=payload * 8 / (elapsed * 1e6),
        ))


def percentile(values, q):
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


compute_samples = []
for rank, rows in enumerate(by_rank):
    for step, record in sorted(rows.items()):
        if record["is_warmup"]:
            continue
        fields = ("compute_wall_ms", "execute_compute_ms", "sample_compute_ms")
        if compute_timing == "event" and "execute_compute_ms" not in record:
            fields = ("compute_wall_ms", "execute_device_ms", "sample_device_ms")
        values = [record.get(key) for key in fields]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
               for v in values):
            raise ValueError(f"stage{rank} step{step} 缺少有效的计算时长")
        if compute_timing == "host-sync" and (
                record.get("timing_source") != "synchronized_host_perf_counter"
                or record.get("execute_device_ms") is not None
                or record.get("sample_device_ms") is not None):
            raise ValueError("host-sync 不得标注为设备 Event 耗时")
        elapsed, execute, sample = values
        if not math.isclose(elapsed, execute + sample, rel_tol=1e-6, abs_tol=0.01):
            raise ValueError(f"stage{rank} step{step} 计算区间不守恒")
        ctx, gen = record["num_ctx_tokens"], record["num_generation_tokens"]
        phase = "mixed" if ctx and gen else "prefill" if ctx else "decode"
        if phase != "prefill":
            continue
        compute_samples.append(dict(
            item=f"stage{rank}", kind="compute", phase=phase, step=step,
            elapsed_ms=elapsed, compute_timing=compute_timing,
            execute_compute_ms=execute, sample_compute_ms=sample,
            execute_device_ms=record.get("execute_device_ms"),
            sample_device_ms=record.get("sample_device_ms"),
        ))

timing_summary = []
items = [f"stage{rank}" for rank in range(4)] + [
    f"stage{rank}->{rank + 1}" for rank in range(3)]
for phase in ("prefill",):
    for item in items:
        group = [s for s in compute_samples + handoff_samples
                 if s["item"] == item and s["phase"] == phase]
        if not group:
            continue
        durations = sorted(s["elapsed_ms"] for s in group)
        row = dict(item=item, kind=group[0]["kind"], phase=phase, samples=len(group),
                   total_ms=sum(durations), mean_ms=statistics.mean(durations),
                   min_ms=durations[0])
        row.update({f"p{q}_ms": percentile(durations, q / 100)
                    for q in (50, 90, 95, 99)})
        row["max_ms"] = durations[-1]
        timing_summary.append(row)

summary = []
for rank in range(3):
    link = f"stage{rank}->{rank + 1}"
    link_samples = [s for s in samples if s["link"] == link]
    if not link_samples:
        raise ValueError(f"{link} 没有非预热、正 payload 的 prefill 交接样本")
    for phase in ("prefill",):
        group = [s for s in link_samples if s["phase"] == phase]
        if not group:
            continue
        speeds = sorted(s["speed_gbps"] for s in group)
        payloads = [s["payload_bytes"] for s in group]
        total_ms = sum(s["handoff_ms"] for s in group)
        row = dict(link=link, phase=phase, samples=len(group),
                   payload_min_bytes=min(payloads), payload_max_bytes=max(payloads),
                   total_bytes=sum(payloads), total_handoff_ms=total_ms,
                   mean_gbps=statistics.mean(speeds),
                   aggregate_gbps=sum(payloads) * 8 / (total_ms * 1e6),
                   min_gbps=speeds[0])
        row.update({f"p{q}_gbps": percentile(speeds, q / 100)
                    for q in (10, 50, 90, 95, 99)})
        row["max_gbps"] = speeds[-1]
        summary.append(row)

for name, rows in (("communication_speed_samples.csv", samples),
                   ("communication_speed_summary.csv", summary),
                   ("stage_compute_samples.csv", compute_samples),
                   ("stage_and_link_timing_summary.csv", timing_summary)):
    with (directory / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

print("\nPrefill 计算与交接耗时（ms；每个 step/链路交接一个样本，预热排除）")
print(f"{'Item':<12} {'Type':<8} {'Phase':<8} {'N':>5} "
      f"{'Mean':>9} {'P50':>9} {'P90':>9} {'P95':>9} {'P99':>9}")
print("-" * 90)
for row in timing_summary:
    numbers = " ".join(f"{row[key + '_ms']:9.4f}" for key in (
        "mean", "p50", "p90", "p95", "p99"))
    print(f"{row['item']:<12} {row['kind']:<8} {row['phase']:<8} "
          f"{row['samples']:5d} {numbers}")
print("-" * 90)
print(f"compute_timing={compute_timing}")
if compute_timing == "host-sync":
    print("compute=execute+sample 的同步主机区间和，含 CPU 提交/同步成本；每层边界同步。")
else:
    print("compute=execute+sample 的设备事件区间和，不含两个 RPC 之间的空档。")
path = "HCCL 张量与 Gloo 控制/元数据" if transport == "native_hccl" else "D2H/Gloo/H2D"
print(f"handoff=发送端 ACK 完成窗口，含 {path} 和 DONE；含零 payload 交接。")
print("计算和通信的分位数不能直接相加得到请求延迟；本表不是 HTTP/排队延迟。")
print(f"耗时表：{directory / 'stage_and_link_timing_summary.csv'}")
print(f"逐步计算：{directory / 'stage_compute_samples.csv'}")

print(f"\n有效交接速率（Gbps）：含 {path} 和 DONE，按应用层 payload 计算。")
print(f"排除预热交接 {warmup_count} 条、零 payload 交接 {zero_count} 条。")
print(f"{'链路':<12} {'阶段':<8} {'样本':>5} {'均值':>9} {'汇总速率':>9} "
      f"{'P10':>9} {'P50':>9} {'P90':>9} {'P95':>9} {'P99':>9}")
for row in summary:
    numbers = " ".join(f"{row[key + '_gbps']:9.4f}" for key in (
        "mean", "aggregate", "p10", "p50", "p90", "p95", "p99"))
    print(f"{row['link']:<12} {row['phase']:<8} {row['samples']:5d} {numbers}")
print("均值=逐交接速率算术平均；汇总速率=总字节/各交接耗时之和，不是墙钟吞吐。")
print("分位数按速率排序后线性插值；P10 表示慢侧，P90/P99 表示快侧。")
print("仅统计纯 prefill step；decode 和 mixed 不参与本表或 TTFT 切分推荐。")
print("采集轮次少时分位数仅为描述性统计，不代表稳定链路性能。")
print(f"统计表：{directory / 'communication_speed_summary.csv'}")
print(f"逐交接：{directory / 'communication_speed_samples.csv'}")
PY
}

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
用法：在 head 容器中编辑本文件顶部配置后运行：
  bash examples/online_serving/hetero_pp/profile_edgeshard.sh
只统计已有完整 ACK 采集（不占 NPU、不重跑模型）：
  bash examples/online_serving/hetero_pp/profile_edgeshard.sh --stats-only /本地目录/某轮结果
只读检查原始耗时字段（不需要完整 manifest，不加载模型）：
  bash examples/online_serving/hetero_pp/profile_edgeshard.sh --check-timing /本地目录/某轮结果

也可用环境变量传入 MODEL、PLAN、RAY_ADDRESS、TRACE_ROOT、PY。
310P 保守计时：PROFILE_COMPUTE_TIMING=host-sync bash 本脚本
四个 stage 统一使用 NPU 同步后的 CPU 单调时钟；不读取 profiling Event 时间戳。
默认 event；host-sync 包含主机提交/同步成本，结果不可与旧 Event trace 混合。
PY 是当前能运行异构 vLLM 的 Python 路径；默认当前虚拟环境或项目 .venv。
Gloo 需要事先连接 Ray；HCCL 使用两端统一入口 --transport hccl --profile。
TRACE_ROOT 为本地结果根目录；采集结束后自动通过 worker RPC 回收 peer 记录。

流程：4 条合成输入，1 轮预热 + 2 轮采集，每请求只输出首个 token，不做后续 decode。
只统计 prefill 的四 stage/三链路成本并做 latency DP，排除预热，保存汇总与逐步 CSV。
使用 ACK 测通信，默认固定 64 个 KV blocks；这是小规模流程验证，不是 GSM8K 压测。
统一 --profile 入口会传入上下文、token budget、max seq、显存比例及 KV block 配置。
不启动/停止 Ray，不安装依赖，不自动切换服务到推荐配置。
EOF
  exit 0
fi
if [[ "${1:-}" == --check-timing && $# == 2 ]]; then
  if [[ ! -x "$PY" ]]; then PY=$(command -v python || true); fi
  [[ -x "$PY" ]] || { echo '请用 PY 指定已有 Python 路径。' >&2; exit 2; }
  check_timing "$2"
  exit 0
fi
if [[ "${1:-}" == --stats-only && $# == 2 ]]; then
  profile_statistics "$2"
  exit 0
fi
if (($#)); then echo '未知参数；使用 --help 查看说明。' >&2; exit 2; fi

[[ -d "$MODEL" ]] || { echo '请设置 MODEL 为可读模型目录。' >&2; exit 2; }
[[ -r "$PLAN" && -f "$PLAN" ]] || { echo '请设置 PLAN 为当前 PP4 topology JSON。' >&2; exit 2; }
engine_transport=(--pp-transport "$PROFILE_TRANSPORT")
case "$PROFILE_TRANSPORT" in
  gloo)
    [[ -n "$RAY_ADDRESS" ]] || { echo '请设置 RAY_ADDRESS=head_IP:端口。' >&2; exit 2; }
    export RAY_ADDRESS
    ;;
  hccl)
    [[ -n "${PROFILE_MASTER_ADDR:-}" && -n "${PROFILE_MASTER_PORT:-}" ]] ||
      { echo 'HCCL profiling 请用两端统一启动入口。' >&2; exit 2; }
    engine_transport+=(--distributed-executor-backend mp --nnodes 2 --node-rank 0
      --master-addr "$PROFILE_MASTER_ADDR" --master-port "$PROFILE_MASTER_PORT")
    ;;
  *) echo 'PROFILE_TRANSPORT 只能是 gloo 或 hccl。' >&2; exit 2 ;;
esac
[[ "$TRACE_ROOT" == /* ]] || { echo '请设置 TRACE_ROOT 为本地结果目录的绝对路径。' >&2; exit 2; }
[[ -x "$PY" && -f "$PY" ]] || { echo '请设置 PY 为匹配服务环境的 Python 绝对路径。' >&2; exit 2; }
[[ "$PROFILE_COMPUTE_TIMING" == event || "$PROFILE_COMPUTE_TIMING" == host-sync ]] || {
  echo 'PROFILE_COMPUTE_TIMING 只能是 event 或 host-sync。' >&2; exit 2;
}

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V2_MODEL_RUNNER=0
# 由 pp-profile 分配本轮 session 和 trace 配置，避免继承旧采集环境。
unset VLLM_PP_STAGE_TRACE VLLM_PP_TRACE_SESSION VLLM_PP_COMPUTE_MODEL VLLM_PP_COMPUTE_TIMING

mkdir -p -- "$TRACE_ROOT"
TRACE=$(mktemp -d "$TRACE_ROOT/edgeshard_$(date +%Y%m%d_%H%M%S)_XXXXXX")
echo "本轮结果：$TRACE"
echo "日志：$TRACE.log"
echo "计算计时：${PROFILE_COMPUTE_TIMING}（四个 stage 一致）"
echo '1/3 采集 prefill 至首 token 的成本（每请求输出 1 token；请勿同时运行其他占卡任务）'
# 日志放在 trace 目录外，保证 collector 启动前目录为空。
"$PY" -m vllm.entrypoints.cli.main pp-profile "$MODEL" \
  --hetero-plan "$PLAN" --trace-dir "$TRACE" --collect-only \
  "${engine_transport[@]}" \
  --compute-model layer-measured --comm-timing ack \
  --compute-timing "$PROFILE_COMPUTE_TIMING" \
  --input-len 64 --output-len 1 --num-prompts 4 \
  --num-iters-warmup 1 --num-iters 2 --seed 7 \
  --dtype float16 --enforce-eager --no-async-scheduling \
  --max-model-len "${PROFILE_MAX_MODEL_LEN:-512}" \
  --max-num-batched-tokens "${PROFILE_MAX_NUM_BATCHED_TOKENS:-256}" \
  --max-num-seqs "${PROFILE_MAX_NUM_SEQS:-4}" \
  --block-size "${PROFILE_BLOCK_SIZE:-128}" \
  --num-gpu-blocks-override "${PROFILE_NUM_GPU_BLOCKS:-64}" \
  --gpu-memory-utilization "${PROFILE_GPU_MEMORY_UTILIZATION:-0.5}" \
  --enable-chunked-prefill --no-enable-prefix-caching 2>&1 | tee "$TRACE.log"

echo '2/3 打印 prefill 的 stage 计算、链路交接耗时及有效速率表（不占 NPU）'
profile_statistics "$TRACE" 2>&1 | tee -a "$TRACE.log"

echo '3/3 仅以 prefill latency 做离线 DP 推荐（不占 NPU）'
"$PY" -m vllm.entrypoints.cli.main pp-profile \
  --hetero-plan "$PLAN" --trace-dir "$TRACE" --pp-transport "$PROFILE_TRANSPORT" \
  --skip-run --objective latency --workload prefill \
  --output-json "$TRACE/latency_plan.json" 2>&1 | tee -a "$TRACE.log"

echo "采集与规划完成：$TRACE"
echo "推荐配置：$TRACE/recommended_hetero_plan.json"
echo '推荐方案仍需重新加载、验证输出并测性能；不是已验证的性能最优方案。'
