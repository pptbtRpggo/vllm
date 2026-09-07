#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

# ==================== 可修改参数 ====================
# ${变量:-默认值}：命令前设置的环境变量优先；也可以直接修改这里的默认值。
# 用法：bash serve_tau.sh /模型目录
# 示例：MAX_NUM_SEQS=8 bash serve_tau.sh /模型目录
MODEL="${MODEL:-}"                         # 模型目录；命令行传入的路径优先
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}" # NPU 编号
export PP="${PP:-2}"                       # 流水线 stage 数；本例每张 NPU 一个 stage
export TP="${TP:-1}"                       # 每个 stage 的张量并行数；当前 trace 采集要求 1
export HOST="${HOST:-127.0.0.1}"            # 监听地址；127.0.0.1 仅本机，0.0.0.0 所有网卡
export PORT="${PORT:-8000}"                # HTTP 服务端口
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}" # 单请求输入 + 输出的总 token 上限
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"    # 单次 forward / 单个 microbatch 的请求数上限
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}" # 单次 forward token 预算
export MAX_MICROBATCHES="${MAX_MICROBATCHES:-0}" # 一次组建的 list 最多 microbatch 数；0 不限
export MIN_WAITING="${MIN_WAITING:-0}"      # 组建新 list 的等待请求数门槛；0 不设门槛
export GPU_MEM="${GPU_MEM:-0.90}"           # 每张设备的显存预算比例；Ascend 沿用 vLLM 参数名
RUN_DIR="${RUN_DIR:-${RUN:-}}"              # 配置/日志/结果目录；留空自动生成，必须是新目录
TRACE="${TRACE:-}"                         # 原始 JSONL 路径；留空生成本地临时文件
PYTHON="${PYTHON:-python}"                  # 仅用于保存配置 JSON；服务使用当前环境的 vllm
# ====================================================

usage() {
    cat <<'HELP'
usage: bash serve_tau.sh [模型目录] [--run-dir 新目录] [--dry-run]

默认：NPU=0,1，PP=2，TP=1，每个 microbatch 最多 4 个请求。
可调参数、默认值与中文说明都在本脚本顶部。
--dry-run  只查看最终参数，不创建文件、不启动服务。
启动时自动将本次目录记录到 trace_runs/latest，bench 无需再填写目录。
HELP
}

model_arg=""
dry_run=0
while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) dry_run=1; shift ;;
        --run-dir)
            if (($# < 2)) || [[ -z "$2" ]]; then
                echo 'ERROR: --run-dir 需要一个新目录' >&2; exit 2
            fi
            RUN_DIR="$2"; shift 2 ;;
        -*) echo "ERROR: 未知选项 $1" >&2; exit 2 ;;
        *)
            if [[ -n "$model_arg" ]]; then
                echo 'ERROR: 只能指定一个模型目录' >&2; exit 2
            fi
            model_arg="$1"; shift ;;
    esac
done
MODEL="${model_arg:-$MODEL}"
if [[ -z "$MODEL" ]]; then usage >&2; exit 2; fi
if [[ -d "$MODEL" ]]; then MODEL="$(cd "$MODEL" && pwd -P)"; fi

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_USE_V1=1 PYTHONUNBUFFERED=1
run_id="$(date +%Y%m%d_%H%M%S)_$$_${RANDOM}"
RUN_DIR="${RUN_DIR:-$ROOT/trace_runs/$run_id}"
TRACE="${TRACE:-${TMPDIR:-/tmp}/tau_${run_id}.jsonl}"
[[ "$RUN_DIR" = /* ]] || RUN_DIR="$PWD/$RUN_DIR"
[[ "$TRACE" = /* ]] || TRACE="$PWD/$TRACE"

# 实际的 vLLM 启动命令。修改顶部变量，即可改变对应的 CLI 参数。
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
    --tau-batch-max-reqs-per-microbatch "$MAX_NUM_SEQS"
    --tau-batch-max-microbatches "$MAX_MICROBATCHES"
    --tau-batch-min-waiting "$MIN_WAITING"
    --worker-cls vllm.v1.worker.tau_ascend_worker.TauAscendWorker
    --scheduler-cls vllm.v1.core.sched.tau_batch.TauScheduler
    --tau-batch-trace "$TRACE"
    --trust-remote-code
)

if ((dry_run == 0)); then
    command -v vllm >/dev/null || { echo 'ERROR: 当前环境找不到 vllm 命令' >&2; exit 127; }
fi
# 此辅助命令只保存 run.json、环境版本等记录，供 bench 读取，不启动或监护服务。
metadata_args=(--model "$MODEL" --run-dir "$RUN_DIR" --trace "$TRACE")
if ((dry_run)); then metadata_args+=(--dry-run); fi
"$PYTHON" "$ROOT/tools/tau_batch_run.py" prepare-serve \
    "${metadata_args[@]}" -- "${VLLM_CMD[@]}"
if ((dry_run)); then exit 0; fi

# exec 后当前 PID 就是 vLLM；Ctrl+C / kill -TERM 直接交给 vLLM 处理。
# tee 只复制日志，忽略终端停止信号，读到服务关闭输出后自行退出。
exec "${VLLM_CMD[@]}" > >(trap '' INT TERM; exec tee "$RUN_DIR/server.log") 2>&1
