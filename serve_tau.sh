#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
# 所有实验参数见 configs/serve.yaml；PYTHON 只选择配置工具的解释器。
PYTHON="${PYTHON:-python}"
CONFIG="$ROOT/configs/serve.yaml"
overrides=()

usage() {
    cat <<'HELP'
usage: bash serve_tau.sh [模型目录] [--config 配置文件] [--run-dir 新目录] [--dry-run]

修改 configs/serve.yaml 后直接运行本脚本；模型可在配置中填写。
--config   使用另一份完整服务配置；优先级：命令行 > 环境变量 > 配置文件。
--dry-run  只查看最终参数，不创建文件、不启动服务。
--no-trace 关闭 trace 和采集额外的设备同步；仍可运行 bench、统计 SLO。
--trace    开启 trace（默认）；建议写到本机 /tmp，避免共享文件系统。
启动时自动将本次目录记录到 output/latest，bench 无需再填写目录。
HELP
}

model_arg=""
dry_run=0
while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) dry_run=1; shift ;;
        --no-trace) overrides+=(--set TRACE_ENABLED 0); shift ;;
        --trace) overrides+=(--set TRACE_ENABLED 1); shift ;;
        --config|--run-dir)
            if (($# < 2)) || [[ -z "$2" ]]; then
                echo "ERROR: $1 需要参数值" >&2; exit 2
            fi
            if [[ "$1" == --config ]]; then CONFIG="$2"; else overrides+=(--set RUN_DIR "$2"); fi
            shift 2 ;;
        -*) echo "ERROR: 未知选项 $1" >&2; exit 2 ;;
        *)
            if [[ -n "$model_arg" ]]; then
                echo 'ERROR: 只能指定一个模型目录' >&2; exit 2
            fi
            model_arg="$1"; shift ;;
    esac
done
if [[ -n "$model_arg" ]]; then overrides+=(--set MODEL "$model_arg"); fi
config_values="$("$PYTHON" "$ROOT/tools/tau_config.py" serve "$CONFIG" ${overrides[@]+"${overrides[@]}"})"
eval "$config_values"
if [[ -z "$MODEL" ]]; then echo "ERROR: 请在 $CONFIG 填写 MODEL，或传入模型目录。" >&2; exit 2; fi
if [[ -d "$MODEL" ]]; then MODEL="$(cd "$MODEL" && pwd -P)"; fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_USE_V1=1 PYTHONUNBUFFERED=1
run_id="$(date +%Y%m%d_%H%M%S)_$$_${RANDOM}"
RUN_DIR="${RUN_DIR:-$ROOT/output/$run_id}"
TRACE="${TRACE:-${TMPDIR:-/tmp}/tau_${run_id}.jsonl}"
case "$TRACE_ENABLED" in
    1) ;;
    0) TRACE=""; unset TAU_BATCH_TRACE ;;
    *) echo 'ERROR: TRACE_ENABLED 必须为 0 或 1' >&2; exit 2 ;;
esac
[[ "$RUN_DIR" = /* ]] || RUN_DIR="$PWD/$RUN_DIR"
if [[ -n "$TRACE" && "$TRACE" != /* ]]; then TRACE="$PWD/$TRACE"; fi

# 实际的 vLLM 启动命令。修改服务配置，即可改变对应的 CLI 参数。
# worker-cls 接入 Ascend compute 记录；scheduler-cls 选择本项目调度器。
# trust-remote-code 允许加载模型目录提供的自定义代码。
VLLM_CMD=(
    vllm serve "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --pipeline-parallel-size "$PP"
    --tensor-parallel-size "$TP"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_MEM"
    --tau-batch-max-microbatches "$MAX_MICROBATCHES"
    --tau-batch-min-waiting "$MIN_WAITING"
    --scheduler-cls vllm.v1.core.sched.tau_batch.TauScheduler
    --trust-remote-code
)
if [[ "$TRACE_ENABLED" == 1 ]]; then
    VLLM_CMD+=(--worker-cls vllm.v1.worker.tau_ascend_worker.TauAscendWorker
        --tau-batch-trace "$TRACE")
fi

if ((dry_run == 0)); then
    command -v vllm >/dev/null || { echo 'ERROR: 当前环境找不到 vllm 命令' >&2; exit 127; }
fi
# 此辅助命令只保存 server_meta.json、环境版本等记录，供 bench 读取，不启动或监护服务。
metadata_args=(--model "$MODEL" --run-dir "$RUN_DIR" --trace "$TRACE")
if ((dry_run)); then metadata_args+=(--dry-run); fi
"$PYTHON" "$ROOT/tools/tau_batch_run.py" prepare-serve \
    "${metadata_args[@]}" -- "${VLLM_CMD[@]}"
if ((dry_run)); then exit 0; fi

# exec 后当前 PID 就是 vLLM；Ctrl+C / kill -TERM 直接交给 vLLM 处理。
# tee 只复制日志，忽略终端停止信号，读到服务关闭输出后自行退出。
exec "${VLLM_CMD[@]}" > >(trap '' INT TERM; exec tee "$RUN_DIR/server.log") 2>&1
