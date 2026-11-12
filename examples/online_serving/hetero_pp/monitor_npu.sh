#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Run on either host. tee creates the CSV as the host user, not container root.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
container="" output=""
forward=()
die() { echo "错误：$*" >&2; exit 2; }
while (($#)); do
  case "$1" in
    --help|-h)
      echo '用法：bash monitor_npu.sh [--container NAME] [--duration 60] [--interval 2] [--output /host/file.csv]'
      echo '默认检测唯一的 hetero-pp-head/worker；Ctrl+C 停止，CSV 保留在本机 HOME/vllm_hetero_pp_logs。'
      echo '可选 --python /container/python --cann-env /container/set_env.sh --samples N'
      exit 0 ;;
    --container|--output|--duration|--interval|--samples|--python|--cann-env)
      (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || die "$1 缺少参数"
      case "$1" in
        --container) container="$2" ;; --output) output="$2" ;;
        *) forward+=("$1" "$2") ;;
      esac
      shift 2 ;;
    *) die "未知参数：$1" ;;
  esac
done
command -v docker >/dev/null || die '找不到 docker'
if [[ -z "$container" ]]; then
  names="$(docker ps --format '{{.Names}}')"
  matches=()
  while IFS= read -r name; do
    case "$name" in hetero-pp-head|hetero-pp-worker) matches+=("$name") ;; esac
  done <<< "$names"
  ((${#matches[@]} == 1)) || die '请用 --container NAME 指定正在运行的本次测试容器'
  container="${matches[0]}"
fi
umask 077
if [[ -z "$output" ]]; then
  directory="${HOST_LOG_DIR:-${HOME:?需要 HOME 或 HOST_LOG_DIR}/vllm_hetero_pp_logs}"
  mkdir -p -- "$directory"
  output="$(mktemp "$directory/npu_monitor_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX.csv")"
else
  (set -o noclobber; : > "$output") || die '输出文件已存在或路径不可写；不覆盖旧数据'
fi
echo "监控 $container；本机 CSV：$output；Ctrl+C 停止监控，不停止服务。" >&2
# Docker exec without a TTY does not reliably forward Ctrl+C to its process.
# A unique marker stops only this collector, never the model or another monitor.
stop_directory="/tmp/hetero_npu_monitor_${output##*/}_$$"
request_stop() {
  docker exec "$container" sh -c '
    if [ -d "$1" ]; then : > "$1/stop"; fi
  ' npu-monitor-stop "$stop_directory" >/dev/null 2>&1 || true
}
trap request_stop EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
docker exec -i "$container" bash -c '
  stop_directory="$1"; shift
  umask 077
  mkdir -- "$stop_directory" || exit 2
  candidate=""
  for name in python python3; do
    if candidate="$(command -v "$name")"; then break; fi
  done
  [[ -n "$candidate" ]] || exit 2
  exec "$candidate" -S /work/vllm/examples/online_serving/hetero_pp/monitor_npu.py \
    --stop-directory "$stop_directory" "$@"
' npu-monitor "$stop_directory" "${forward[@]}" | tee "$output"
