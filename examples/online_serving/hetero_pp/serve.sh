#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Start the two-node Ray cluster separately; optional vLLM flags follow this script.
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
  --host 127.0.0.1 --port 8000 "$@"
