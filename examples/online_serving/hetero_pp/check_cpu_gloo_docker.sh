#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Run on each Linux host; the existing helper performs the actual validation.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

usage() {
  cat <<'EOF'
两机 Docker 通信验证（每台两张卡，不需要 Ray 或模型）
用法：
  bash check_cpu_gloo_docker.sh --soc 910b --head 192.168.1.10
  bash check_cpu_gloo_docker.sh --soc 310p --head 192.168.1.10
  bash check_cpu_gloo_docker.sh --image 管理员镜像:标签 --head 192.168.1.10
选项：
  --soc 910b|310p       选择 v0.23.0 镜像（不带 post）
  --image IMAGE        使用指定镜像，优先于 --soc
  --head IPv4          两端填写相同的、属于其中一台宿主机的 IPv4
  --devices 0,1        本机两个 /dev/davinci 设备号（默认 0,1）
  --port 19090         master 端口，另使用 port+1 和 Gloo 动态 TCP 端口
  --timeout 600        容器内测试时间上限（秒，不含镜像拉取时间）
  --interface eth0     容器内/宿主机使用的网卡名
  --cann-env PATH      指定容器内已有 SDK 环境脚本
  --dry-run            仅显示 Docker 命令，不拉镜像、不启动容器
EOF
}

die() { echo "错误：$*" >&2; exit 2; }
soc="" image="" head="" devices="0,1" port=19090 limit=600
interface="" cann_env="" dry_run=false
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --dry-run) dry_run=true; shift ;;
    --soc|--image|--head|--devices|--port|--timeout|--interface|--cann-env)
      (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || die "$1 缺少参数"
      case "$1" in
        --soc) soc="$2" ;;
        --image) image="$2" ;;
        --head) head="$2" ;;
        --devices) devices="$2" ;;
        --port) port="$2" ;;
        --timeout) limit="$2" ;;
        --interface) interface="$2" ;;
        --cann-env) cann_env="$2" ;;
      esac
      shift 2 ;;
    *) die "未知参数：$1；使用 --help 查看用法" ;;
  esac
done
[[ -n "$head" ]] || die "需要 --head，两端填同一个 IP"
[[ -z "$soc" || "$soc" == 910b || "$soc" == 310p ]] || die "--soc 只能是 910b 或 310p"
if [[ -z "$image" ]]; then
  case "$soc" in
    910b) image=quay.io/ascend/vllm-ascend:v0.23.0 ;;
    310p) image=quay.io/ascend/vllm-ascend:v0.23.0-310p ;;
    *) die "需要 --soc 或 --image" ;;
  esac
fi
[[ "$image" != -* ]] || die "镜像名不能以 - 开头"
[[ "$devices" =~ ^(0|[1-9][0-9]*),(0|[1-9][0-9]*)$ ]] || die "--devices 需为两个设备号，例如 0,1"
first="${devices%,*}" second="${devices#*,}"
[[ "$first" != "$second" ]] || die "两个设备号不能相同"
if ! [[ "$port" =~ ^[1-9][0-9]{3,4}$ ]] || ((port < 1024 || port > 65534)); then
  die "端口需在 1024..65534"
fi
if ! [[ "$limit" =~ ^[1-9][0-9]{1,3}$ ]] || ((limit < 30 || limit > 3600)); then
  die "timeout 需在 30..3600 秒"
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../../.." && pwd)"
[[ "$repo_dir" != *,* ]] || die "Docker bind mount 不支持此项目路径中的逗号"

command=(docker run --rm --network host --shm-size=1g)
if [[ -t 0 && -t 1 ]]; then command+=(-it); fi
for device in "davinci$first" "davinci$second" davinci_manager devmm_svm hisi_hdc; do
  command+=(--device "/dev/$device")
done
for path in /usr/local/dcmi /usr/local/Ascend/driver/lib64 \
  /usr/local/Ascend/driver/version.info /etc/ascend_install.info; do
  command+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
for smi in /usr/local/bin/npu-smi /usr/local/sbin/npu-smi; do
  if [[ -f "$smi" ]]; then
    command+=(--mount "type=bind,src=$smi,dst=/usr/local/bin/npu-smi,readonly")
    break
  fi
done
command+=(--mount "type=bind,src=$repo_dir,dst=/work/vllm" -w /work/vllm)
# Expanded by the container shell, not the host shell.
# shellcheck disable=SC2016
command+=(--entrypoint /bin/bash "$image" -c '
set -e
container_python=$(command -v python3 || command -v python) || {
  echo "No Python in container; use a matching vLLM-Ascend image." >&2
  exit 2
}
exec bash examples/online_serving/hetero_pp/check_cpu_gloo.sh \
  --python "$container_python" "$@"
' comm-check --head "$head" --devices "$devices")
if [[ "$port" != 19090 ]]; then command+=(--port "$port"); fi
if [[ "$limit" != 600 ]]; then command+=(--timeout "$limit"); fi
if [[ -n "$interface" ]]; then command+=(--interface "$interface"); fi
if [[ -n "$cann_env" ]]; then command+=(--cann-env "$cann_env"); fi
if "$dry_run"; then
  printf '%q ' "${command[@]}"
  echo
  exit 0
fi
command -v docker >/dev/null || die "宿主机找不到 docker"
echo "镜像：${image}；本机设备：${devices}；head：${head}" >&2
echo "结果保存在项目 output/comm_check_*/；请同时在另一台机器运行。" >&2
exec "${command[@]}"
