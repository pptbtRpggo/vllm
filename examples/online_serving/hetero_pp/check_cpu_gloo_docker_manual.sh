#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Editable host-side template. Run on both machines, with the same MASTER_ADDR.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

# 910B default. For 310P use v0.23.0.post1-310p-openeuler or an approved image.
export IMAGE="${IMAGE:-quay.io/ascend/vllm-ascend:v0.23.0.post1-openeuler}"
export CONTAINER_NAME="${CONTAINER_NAME:-hete-npu-comm-test}"
export MASTER_ADDR="${MASTER_ADDR:-}" # Required: the same real host IPv4 on both nodes.
export MASTER_PORT="${MASTER_PORT:-29501}"
export NPU_DEVICES="${NPU_DEVICES:-0,1}" # Must match the device mappings below.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HOST_PROJECT_DIR="${HOST_PROJECT_DIR:-$(cd -- "$script_dir/../../.." && pwd)}"

if [[ -z "$MASTER_ADDR" ]]; then
  echo '请填写 MASTER_ADDR，例如 export MASTER_ADDR=192.168.1.10' >&2
  exit 2
fi

# Adjust these device mappings for the local host/card model.
DEVICE_ARGS=(
  --device /dev/davinci0
  --device /dev/davinci1
  --device /dev/davinci_manager
  --device /dev/devmm_svm
  --device /dev/hisi_hdc
)
# Keep the project mount. Adjust driver paths to the administrator configuration.
# Do not mount the whole host /usr/local/Ascend over the image CANN installation.
MOUNT_ARGS=(
  -v "$HOST_PROJECT_DIR:/work/vllm"
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro
  -v /usr/local/dcmi:/usr/local/dcmi:ro
)

# Do not force-remove an existing named container. Docker reports name conflicts.
# These variables expand inside the container.
# shellcheck disable=SC2016
exec docker run -it --rm \
  --name "$CONTAINER_NAME" \
  --network=host --shm-size=1g \
  "${DEVICE_ARGS[@]}" \
  "${MOUNT_ARGS[@]}" \
  -e MASTER_ADDR \
  -e MASTER_PORT \
  -e NPU_DEVICES \
  -w /work/vllm \
  --entrypoint /bin/bash "$IMAGE" -c '
set -e
container_python=$(command -v python3 || command -v python)
exec bash examples/online_serving/hetero_pp/check_cpu_gloo.sh \
  --python "$container_python" \
  --head "$MASTER_ADDR" --port "$MASTER_PORT" --devices "$NPU_DEVICES"
'
