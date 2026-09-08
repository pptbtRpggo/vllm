#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# 所有实验参数（含分组 SLO）见 configs/bench.yaml。
PYTHON="${PYTHON:-python}"
CONFIG="$ROOT/configs/bench.yaml"
overrides=()

usage() {
    cat <<'HELP'
usage: bash bench_tau.sh [服务运行目录] [--config 配置文件] [选项]

修改 configs/bench.yaml 后直接运行本脚本；默认服务为 output/latest。
--config 配置文件         使用另一份完整 bench 配置（包括内嵌 SLO）。
--mode smoke|collect       小测试或正式采集
--num-prompts N            正式请求总数
--output-len N             每个请求的目标输出 token 数
--request-rate R           目标请求数/秒，默认 inf
--concurrency N            最多同时未完成的请求数，smoke 默认 8、collect 默认 32
--burstiness B             到达间隔：1 随机，inf 等间隔
--slo-config 路径          SLO 档位/比例配置；自动按请求各自阈值统计 goodput
--slo-seed N               分组种子，默认使用 --seed
--dataset 路径 --download  数据集；缺失时允许下载
--seed N --base-url URL --result-dir 新目录 --ready-timeout 秒数
--dry-run                 只打印命令，不下载数据或发送请求
参数与中文说明见 configs/bench.yaml；优先级：命令行 > 环境变量 > 配置文件。
HELP
}

dry_run=0
run_arg=""
while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) dry_run=1; shift ;;
        --download) overrides+=(--set DOWNLOAD 1); shift ;;
        --config|--mode|--num-prompts|--output-len|--concurrency|--request-rate|--burstiness|--seed|--slo-config|--slo-seed|--dataset|--base-url|--result-dir|--ready-timeout)
            if (($# < 2)) || [[ -z "$2" || "$2" = --* ]]; then
                echo "ERROR: $1 需要参数值" >&2; exit 2
            fi
            case "$1" in
                --config) CONFIG="$2" ;;
                --mode) overrides+=(--set MODE "$2") ;; --num-prompts) overrides+=(--set NUM_PROMPTS "$2") ;;
                --output-len) overrides+=(--set OUTPUT_LEN "$2") ;; --concurrency) overrides+=(--set CONCURRENCY "$2") ;;
                --request-rate) overrides+=(--set REQUEST_RATE "$2") ;; --burstiness) overrides+=(--set BURSTINESS "$2") ;;
                --seed) overrides+=(--set SEED "$2") ;; --dataset) overrides+=(--set DATASET "$2") ;;
                --slo-config) overrides+=(--set SLO_CONFIG "$2") ;; --slo-seed) overrides+=(--set SLO_SEED "$2") ;;
                --base-url) overrides+=(--set BASE_URL "$2") ;; --result-dir) overrides+=(--set RESULT_DIR "$2") ;;
                --ready-timeout) overrides+=(--set READY_TIMEOUT "$2") ;;
            esac
            shift 2 ;;
        -*) echo "ERROR: 未知选项 $1" >&2; exit 2 ;;
        *)
            if [[ -n "$run_arg" ]]; then echo 'ERROR: 只能指定一个运行目录' >&2; exit 2; fi
            run_arg="$1"; shift ;;
    esac
done
if [[ -n "$run_arg" ]]; then overrides+=(--set RUN_DIR "$run_arg"); fi
config_values="$("$PYTHON" "$ROOT/tools/tau_config.py" bench "$CONFIG" ${overrides[@]+"${overrides[@]}"})"
eval "$config_values"
RUN_DIR="${RUN_DIR:-$ROOT/output/latest}"
if [[ ! -f "$RUN_DIR/server_meta.json" ]]; then
    echo "ERROR: 找不到 $RUN_DIR/server_meta.json；请先运行 serve_tau.sh，或手动指定服务运行目录。" >&2
    exit 2
fi
# 固定本次使用的实际目录，避免另一个服务更新 latest 后影响正在执行的 bench。
RUN_DIR="$(cd "$RUN_DIR" && pwd -P)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MODE NUM_PROMPTS OUTPUT_LEN CONCURRENCY REQUEST_RATE BURSTINESS IGNORE_EOS
export SEED DATASET DOWNLOAD BASE_URL RESULT_DIR WARMUP_REQUESTS READY_TIMEOUT
export SLO_CONFIG SLO_SEED SLO_INLINE
if ((dry_run == 0)); then
    command -v vllm >/dev/null || { echo 'ERROR: 当前环境找不到 vllm 命令' >&2; exit 127; }
fi

config_file="$(mktemp "${TMPDIR:-/tmp}/tau_bench.XXXXXX")"
BENCH_WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tau_bench_work.XXXXXX")"
export BENCH_WORK_DIR
bench_pid=""
tee_pid=""
prepared=0
cleanup() {
    local code=$?
    # Close the log pipe and join tee before archiving/removing its file.
    if [[ -n "$tee_pid" ]]; then
        exec 1>&3 2>&4
        wait "$tee_pid" || true
    fi
    if ((dry_run == 0)) && [[ -s "$config_file" ]]; then
        source "$config_file"
        prepared=1
    fi
    if ((code != 0 && prepared)); then
        { cat "$BENCH_WORK_DIR/bench.log"; echo "benchmark exit code: $code"; } > "$RESULT_DIR/error.log"
    fi
    rm -f "$config_file"
    rm -rf "$BENCH_WORK_DIR"
}
trap cleanup EXIT
stop_bench() {
    trap '' INT TERM
    if [[ -n "$bench_pid" ]]; then
        kill -TERM "$bench_pid" 2>/dev/null || true
        wait "$bench_pid" 2>/dev/null || true
    fi
    exit 130
}
trap stop_bench INT TERM

if ((dry_run == 0)); then
    exec 3>&1 4>&2
    exec > >(trap '' INT TERM; exec tee "$BENCH_WORK_DIR/bench.log") 2>&1
    tee_pid=$!
fi

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
        --no-oversample --trust-remote-code --save-result
        --result-dir "$BENCH_WORK_DIR" --result-filename "$2.json"
    )
    if [[ "$2" == result ]]; then
        BENCH_CMD+=(--request-output "$RESULT_DIR/requests.jsonl")
    else
        BENCH_CMD+=(--request-output "$BENCH_WORK_DIR/warmup_requests.jsonl")
    fi
    if [[ "$IGNORE_EOS" == 1 ]]; then BENCH_CMD+=(--ignore-eos); fi
    if [[ -n "$SLO_CONFIG" ]]; then
        BENCH_CMD+=(--slo-config "$SLO_CONFIG" --slo-seed "$SLO_SEED")
    fi
}

run_benchmark() {
    local count="$1" name="$2" code=0
    build_command "$count" "$name"
    if ((dry_run)); then printf '%q ' "${BENCH_CMD[@]}"; printf '\n'; return; fi
    # 执行前记录 trace 起点；执行成功后检查请求成功数和各 stage 的 compute 记录。
    "$PYTHON" "$ROOT/tools/tau_batch_run.py" begin-bench \
        "$RUN_DIR" "$RESULT_DIR" "$name" "$count" -- "${BENCH_CMD[@]}"
    "${BENCH_CMD[@]}" &
    bench_pid=$!
    wait "$bench_pid" || code=$?
    bench_pid=""
    if ((code != 0)); then echo "ERROR: benchmark 退出码 ${code}；查看 $RESULT_DIR/error.log" >&2; return "$code"; fi
    "$PYTHON" "$ROOT/tools/tau_batch_run.py" end-bench "$RUN_DIR" "$RESULT_DIR" "$name"
}

if [[ "$MODE" == collect ]]; then run_benchmark "$WARMUP_REQUESTS" warmup; fi
run_benchmark "$NUM_PROMPTS" result
if ((dry_run == 0)); then echo "压测完成：$RESULT_DIR"; fi
