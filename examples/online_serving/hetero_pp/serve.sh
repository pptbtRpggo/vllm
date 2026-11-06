#!/usr/bin/env bash
# Experimental first bring-up only. Start the two-node Ray cluster separately.
set -euo pipefail
: "${MODEL:?Set MODEL to a matching local checkpoint path on both nodes}"
: "${PLAN:?Set PLAN to your edited pipeline JSON file}"
[[ -f "$PLAN" ]] || { echo 'Plan file missing' >&2; exit 1; }
export VLLM_HETERO_PP_PLAN="$(cat "$PLAN")"
export VLLM_USE_V2_MODEL_RUNNER=0
# Existing Ray daemons must use each node's own matching Ascend runtime.
# Do not pass --nnodes: this executor uses Ray node affinity, not mp launching.
python -m vllm.distributed.hetero_pp.serve --model "$MODEL" \
  --pipeline-parallel-size 4 --tensor-parallel-size 1 \
  --dtype float16 --enforce-eager --no-async-scheduling \
  --max-model-len 512 --max-num-batched-tokens 256 --max-num-seqs 4 \
  --enable-chunked-prefill --no-enable-prefix-caching \
  --host 127.0.0.1 --port 8000
