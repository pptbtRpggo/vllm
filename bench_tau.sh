#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# ==================== 可修改参数 ====================
# 用法：bash bench_tau.sh --mode smoke（自动读取最近启动的服务配置）
# 修改这里的默认值，或临时覆盖：REQUEST_RATE=5 CONCURRENCY=16 bash bench_tau.sh ...
RUN_DIR="${RUN_DIR:-}"                 # 留空使用 trace_runs/latest；也可指定服务运行目录
MODE="${MODE:-smoke}"                 # smoke：小测试；collect：先预热、再正式采集
# 两种模式的配置直接在下面修改；命令行参数和已有环境变量仍可覆盖默认值。
# 此函数在解析 --mode 后调用，确保选择的模式正确生效。
apply_mode_defaults() {
    case "$MODE" in
        smoke) # 小测试：不额外预热
            NUM_PROMPTS="${NUM_PROMPTS:-8}"    # 总共发送 8 个请求
            OUTPUT_LEN="${OUTPUT_LEN:-16}"    # 每个请求生成 16 个 token
            CONCURRENCY="${CONCURRENCY:-8}"   # 最多 8 个未完成请求
            ;;
        collect) # 正式采集：先预热，再发送下面指定的正式请求
            NUM_PROMPTS="${NUM_PROMPTS:-1000}" # 正式请求总数
            OUTPUT_LEN="${OUTPUT_LEN:-256}"   # 每个请求生成的 token 数
            CONCURRENCY="${CONCURRENCY:-32}"  # 最多同时未完成的请求数
            ;;
        *) echo 'ERROR: MODE 必须是 smoke 或 collect' >&2; exit 2 ;;
    esac
}

# 两种模式共用的参数
REQUEST_RATE="${REQUEST_RATE:-inf}"   # 目标请求数/秒；inf 表示尽快发送，仍受并发上限限制
BURSTINESS="${BURSTINESS:-1}"          # 有限速率时：1 随机指数间隔；<1 更突发；inf 等间隔
IGNORE_EOS="${IGNORE_EOS:-1}"          # 1 忽略 EOS 以采固定输出长度；0 允许自然 EOS 结束
SEED="${SEED:-0}"                     # 数据采样和到达间隔的随机种子
DATASET="${DATASET:-$ROOT/datasets/sharegpt.json}" # ShareGPT JSON 文件
DOWNLOAD="${DOWNLOAD:-0}"             # 1：文件缺失时下载；0：使用已有文件
BASE_URL="${BASE_URL:-}"               # 留空从 run.json 读取，例如 http://127.0.0.1:8000
RESULT_DIR="${RESULT_DIR:-}"           # 留空自动创建本次 bench 结果目录，不覆盖已有目录
WARMUP_REQUESTS="${WARMUP_REQUESTS:-32}" # collect 模式额外预热请求数，不计入正式样本
READY_TIMEOUT="${READY_TIMEOUT:-300}" # 最多等待服务就绪的秒数
PYTHON="${PYTHON:-python}"             # 仅用于配置/trace 辅助程序；流量由 vllm bench 发送
# ====================================================

usage() {
    cat <<'HELP'
usage: bash bench_tau.sh [服务运行目录] [选项]

省略目录时读取本仓库 trace_runs/latest；多服务时可手动指定目录。
--mode smoke|collect       小测试或正式采集
--num-prompts N            正式请求总数
--output-len N             每个请求的目标输出 token 数
--request-rate R           目标请求数/秒，默认 inf
--concurrency N            最多同时未完成的请求数，smoke 默认 8、collect 默认 32
--burstiness B             到达间隔：1 随机，inf 等间隔
--dataset 路径 --download  数据集；缺失时允许下载
--seed N --base-url URL --result-dir 新目录 --ready-timeout 秒数
--dry-run                 只打印命令，不下载数据或发送请求
全部参数及中文说明见本脚本顶部。
HELP
}

dry_run=0
run_arg=""
while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) dry_run=1; shift ;;
        --download) DOWNLOAD=1; shift ;;
        --mode|--num-prompts|--output-len|--concurrency|--request-rate|--burstiness|--seed|--dataset|--base-url|--result-dir|--ready-timeout)
            if (($# < 2)) || [[ -z "$2" || "$2" = --* ]]; then
                echo "ERROR: $1 需要参数值" >&2; exit 2
            fi
            case "$1" in
                --mode) MODE="$2" ;; --num-prompts) NUM_PROMPTS="$2" ;;
                --output-len) OUTPUT_LEN="$2" ;; --concurrency) CONCURRENCY="$2" ;;
                --request-rate) REQUEST_RATE="$2" ;; --burstiness) BURSTINESS="$2" ;;
                --seed) SEED="$2" ;; --dataset) DATASET="$2" ;;
                --base-url) BASE_URL="$2" ;; --result-dir) RESULT_DIR="$2" ;;
                --ready-timeout) READY_TIMEOUT="$2" ;;
            esac
            shift 2 ;;
        -*) echo "ERROR: 未知选项 $1" >&2; exit 2 ;;
        *)
            if [[ -n "$run_arg" ]]; then echo 'ERROR: 只能指定一个运行目录' >&2; exit 2; fi
            run_arg="$1"; shift ;;
    esac
done
RUN_DIR="${run_arg:-${RUN_DIR:-$ROOT/trace_runs/latest}}"
if [[ ! -f "$RUN_DIR/run.json" ]]; then
    echo "ERROR: 找不到 $RUN_DIR/run.json；请先运行 serve_tau.sh，或手动指定服务运行目录。" >&2
    exit 2
fi
# 固定本次使用的实际目录，避免另一个服务更新 latest 后影响正在执行的 bench。
RUN_DIR="$(cd "$RUN_DIR" && pwd -P)"
apply_mode_defaults
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MODE NUM_PROMPTS OUTPUT_LEN CONCURRENCY REQUEST_RATE BURSTINESS IGNORE_EOS
export SEED DATASET DOWNLOAD BASE_URL RESULT_DIR WARMUP_REQUESTS READY_TIMEOUT
if ((dry_run == 0)); then
    command -v vllm >/dev/null || { echo 'ERROR: 当前环境找不到 vllm 命令' >&2; exit 127; }
fi

config_file="$(mktemp "${TMPDIR:-/tmp}/tau_bench.XXXXXX")"
bench_pid=""
trap 'rm -f "$config_file"' EXIT
stop_bench() {
    trap '' INT TERM
    if [[ -n "$bench_pid" ]]; then
        kill -TERM "$bench_pid" 2>/dev/null || true
        wait "$bench_pid" 2>/dev/null || true
    fi
    exit 130
}
trap stop_bench INT TERM

# 仅读服务配置、准备结果目录；临时文件由辅助程序使用 shell 安全转义生成。
prepare_args=("$RUN_DIR" --config-file "$config_file")
if ((dry_run)); then prepare_args+=(--dry-run); fi
"$PYTHON" "$ROOT/tools/tau_batch_run.py" prepare-bench "${prepare_args[@]}"
source "$config_file"

build_command() {
    # 读 ShareGPT prompt，通过 HTTP 流式请求 /v1/completions。
    # 这里不组 microbatch；组批由服务端 TauScheduler 完成。
    BENCH_CMD=(
        vllm bench serve
        --backend vllm --base-url "$BASE_URL" --endpoint /v1/completions
        --model "$MODEL" --tokenizer "$MODEL"
        --dataset-name sharegpt --dataset-path "$DATASET"
        --num-prompts "$1" --sharegpt-output-len "$OUTPUT_LEN"
        --request-rate "$REQUEST_RATE" --max-concurrency "$CONCURRENCY"
        --burstiness "$BURSTINESS" --seed "$SEED"
        --ready-check-timeout-sec 0 --num-warmups 0
        --no-oversample --trust-remote-code --save-result --save-detailed
        --result-dir "$RESULT_DIR" --result-filename "$2.json"
    )
    if [[ "$IGNORE_EOS" == 1 ]]; then BENCH_CMD+=(--ignore-eos); fi
}

run_benchmark() {
    local count="$1" name="$2" code=0
    build_command "$count" "$name"
    if ((dry_run)); then printf '%q ' "${BENCH_CMD[@]}"; printf '\n'; return; fi
    # 执行前记录 trace 起点；执行成功后检查请求成功数和各 stage 的 compute 记录。
    "$PYTHON" "$ROOT/tools/tau_batch_run.py" begin-bench \
        "$RUN_DIR" "$RESULT_DIR" "$name" "$count" -- "${BENCH_CMD[@]}"
    "${BENCH_CMD[@]}" > >(trap '' INT TERM; exec tee "$RESULT_DIR/$name.log") 2>&1 &
    bench_pid=$!
    wait "$bench_pid" || code=$?
    bench_pid=""
    if ((code != 0)); then echo "ERROR: benchmark 退出码 ${code}；查看 $RESULT_DIR/$name.log" >&2; return "$code"; fi
    "$PYTHON" "$ROOT/tools/tau_batch_run.py" end-bench "$RUN_DIR" "$RESULT_DIR" "$name"
}

if [[ "$MODE" == collect ]]; then run_benchmark "$WARMUP_REQUESTS" warmup; fi
run_benchmark "$NUM_PROMPTS" result
if ((dry_run == 0)); then echo "采集及覆盖检查完成：$RESULT_DIR"; fi
