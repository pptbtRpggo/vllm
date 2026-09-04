#!/usr/bin/env bash
set -euo pipefail

# TauScheduler serve. Run from any cwd; the script cds into this repo so
# ``import vllm`` hits the inner package, not a sibling directory named vllm.
#
#   MODEL=~/models/CodeLlama-7b-Instruct/V1/model/ ./serve_tau.sh
#   ./serve_tau.sh /path/or/hf-id
#
# Packing (take then split; overflow deferred, no padding):
#   MAX_NUM_SEQS          take cap               default 32
#   MAX_REQS_PER_MB       n cap per task         default 4
#   MAX_MICROBATCHES      P; 0 = ceil(take/n)    default 0 → 8
#   MIN_WAITING           plan threshold         default = MAX_NUM_SEQS
#
# Trace is created on the first write. Delete the JSONL to start a new run
# without restarting serve.

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT}"

MODEL="${MODEL:-${1:-}}"
if [[ -z "${MODEL}" ]]; then
  echo "用法: MODEL=<纯文本模型路径或HF id> $0" >&2
  echo "   或: $0 <纯文本模型路径或HF id>" >&2
  exit 1
fi

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
export VLLM_USE_V1=1

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
PP="${PP:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_REQS_PER_MB="${MAX_REQS_PER_MB:-4}"
MAX_MICROBATCHES="${MAX_MICROBATCHES:-0}"
MIN_WAITING="${MIN_WAITING:-${MAX_NUM_SEQS}}"
GPU_MEM="${GPU_MEM:-0.90}"
TRACE="${TRACE:-${ROOT}/tau_batch_trace.jsonl}"

echo "vllm: $(python -c 'import vllm,inspect; print(vllm.__version__, inspect.getfile(vllm))')"
echo "TauScheduler: $(python -c 'from vllm.v1.core.sched.tau_batch import TauScheduler; print(TauScheduler)')"
echo "MODEL=${MODEL}  devices=${ASCEND_RT_VISIBLE_DEVICES}  TP=${TP} PP=${PP}  port=${PORT}"
echo "take=${MAX_NUM_SEQS}  per_mb=${MAX_REQS_PER_MB}  P=${MAX_MICROBATCHES}  min_waiting=${MIN_WAITING}"
echo "trace=${TRACE}  (created on first write; rm it to start a new run)"

exec vllm serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --pipeline-parallel-size "${PP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEM}" \
  --scheduler-cls vllm.v1.core.sched.tau_batch.TauScheduler \
  --tau-batch-min-waiting "${MIN_WAITING}" \
  --tau-batch-max-reqs-per-microbatch "${MAX_REQS_PER_MB}" \
  --tau-batch-max-microbatches "${MAX_MICROBATCHES}" \
  --tau-batch-trace "${TRACE}" \
  --trust-remote-code
