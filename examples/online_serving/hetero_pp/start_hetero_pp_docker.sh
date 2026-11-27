#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Host entrypoint. Declared node SoCs select images; TorchNPU verifies them inside.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

# You can fill only these IP suffixes here, or pass --head / --peer below.
IP_PREFIX="${IP_PREFIX:-141.147.135}"
head="${HEAD_IP:-${HEAD_LAST:-}}"
peer="${PEER_IP:-${PEER_LAST:-}}"
head_soc=910b peer_soc="" transport=gloo
model="${HOST_MODEL_DIR:-}"
log_dir="${HOST_LOG_DIR:-${HOME:?需要 HOME 或 HOST_LOG_DIR}/vllm_hetero_pp_logs}"
image_910b="${IMAGE_910B:-quay.io/ascend/vllm-ascend:v0.23.0-openeuler}"
image_310p="${IMAGE_310P:-quay.io/ascend/vllm-ascend:v0.23.0-310p-openeuler}"
image="${IMAGE:-}"
devices="${HOST_NPU_DEVICES:-0,1}"
ray_port="${RAY_PORT:-16379}" api_port="${SERVICE_PORT:-8000}"
master_port="${MASTER_PORT:-29501}"
layers="${PP_LAYERS:-}"
npu_core_fractions=""
interface="" dry_run=false debug_shell=false profile=false verify_overlap=false trace_root=""
forward=()

usage() {
  cat <<'EOF'
两边宿主机使用同一命令（不依赖宿主机 npu-smi / TorchNPU）：
  bash start_hetero_pp_docker.sh --head 53 --peer 54 --model /host/model
--head / --peer 可填 IP 最后一个数字；默认前缀 141.147.135，也可填完整 IPv4。
先在 head 运行，等待时在 peer 运行；两边可使用不同宿主机模型路径。
选项：
  --head VALUE         主端 IP 或末段；也可设置 HEAD_LAST / HEAD_IP
  --peer VALUE         另一个节点 IP 或末段；也可设置 PEER_LAST / PEER_IP
  --head-soc 910b|310p  head 两张卡型号，默认 910b；两边填相同配置
  --peer-soc 910b|310p  peer 型号：gloo 默认 310p，hccl 默认 910b
  --transport gloo|hccl 默认 gloo + Ray；hccl 为原生 mp，仅支持两边 910B
  --model DIR          本机完整模型目录，挂载到容器 /workspace/model
  --log-dir DIR        宿主机可读日志根目录，默认 $HOME/vllm_hetero_pp_logs
  --devices 0,1        宿主机两张卡设备文件编号，只影响 Docker 映射
  --interface NAME     默认按到对端的路由自动识别
  --image IMAGE        覆盖本机镜像；也可分别设置 IMAGE_910B / IMAGE_310P
  --name NAME          本机容器名，默认 hetero-pp-head / hetero-pp-worker
  --ray-port PORT      默认 16379，两端一致；其他 Ray/Gloo 端口仍动态分配
  --master-port PORT   HCCL/mp 主端端口，默认 29501，两边一致；不启动 Ray
  --api-port PORT      默认 8000；gloo 在 head 回环监听，hccl 在所有网卡监听
  --layers A,B,C,D     原 910B→310P 默认 18,18,6,6；其他组合按模型总层数均分
  --npu-core-fractions A,B,C,D
                       按 PP stage 0–3 指定核心比例，例如 1,1,1/2,1/4
  --profile           Gloo 或 HCCL：head 采集/打印统计/生成 DP，不启动 API
  --verify-overlap    仅 Gloo 在线：head 自动验证 Stage 0/1 NPU 批次重叠
  --async-scheduling / --no-async-scheduling
  --enable-chunked-prefill / --no-enable-chunked-prefill
                       可选，显式设定 HCCL 普通服务的调度与 chunked prefill
  --model-runner-v1    HCCL 回测使用与采集一致的 V1 runner
  --seed N            HCCL 普通服务的模型随机种子
  --expected-checkpoint-sha256 SHA
                       HCCL 回测所使用的 checkpoint 内容 SHA256
  --trace-root DIR    可选，本机结果目录；不需要共享，默认随本轮日志保存
                       head 通过 worker RPC 收回并校验 peer 记录
  --profile-timeout N profiling 全流程上限秒数，默认 3600
  --profile-compute-timing event|host-sync
                       host-sync 为四个 stage 使用逐层同步 CPU 计时，绕开 profiling Event
  --block-size N / --num-gpu-blocks-override N
                       可选，显式 KV 配置；profiling 默认 128 / 64
  --startup-timeout N  转发到容器内启动入口
  --max-model-len N / --max-num-batched-tokens N / --max-num-seqs N
  --gpu-memory-utilization N / --cann-env PATH / --python PATH
                       可选，转发给容器内入口；路径是容器内路径
                       普通服务容量/组批参数省略时使用 vLLM 默认，不预设 512/256/4
  --chat-template codellama-instruct / PATH
                       内置 CodeLlama-Instruct 模板或容器内文件路径；省略沿用 tokenizer
  --dry-run            只显示命令，不启动 Docker，不初始化 NPU
  --shell              只进 Bash，不启动服务/预检，不创建实验日志目录
                       无需 --model；型号参数选择镜像，IP 参数选择角色/网卡
                       两种角色都禁用 seccomp 过滤，覆盖镜像 ENTRYPOINT
                       退出保留容器，可用 docker start -ai 容器名再次进入
EOF
}
die() { echo "错误：$*" >&2; exit 2; }
name="${CONTAINER_NAME:-}"
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --dry-run) dry_run=true; shift ;;
    --shell) debug_shell=true; shift ;;
    --profile) profile=true; shift ;;
    --verify-overlap) verify_overlap=true; shift ;;
    --async-scheduling|--no-async-scheduling|--enable-chunked-prefill|--no-enable-chunked-prefill|--model-runner-v1)
      forward+=("$1"); shift ;;
    --head|--peer|--head-soc|--peer-soc|--transport|--model|--log-dir|--devices|--interface|--image|--name|--ray-port|--master-port|--api-port|--layers|--npu-core-fractions|--startup-timeout|--max-model-len|--max-num-batched-tokens|--max-num-seqs|--gpu-memory-utilization|--cann-env|--python|--chat-template|--trace-root|--profile-timeout|--profile-compute-timing|--block-size|--num-gpu-blocks-override|--seed|--expected-checkpoint-sha256)
      (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || die "$1 缺少参数"
      case "$1" in
        --head) head="$2" ;; --peer) peer="$2" ;; --model) model="$2" ;;
        --head-soc) head_soc="$2" ;; --peer-soc) peer_soc="$2" ;;
        --transport) transport="$2" ;; --master-port) master_port="$2" ;;
        --log-dir) log_dir="$2" ;;
        --devices) devices="$2" ;; --interface) interface="$2" ;;
        --image) image="$2" ;; --name) name="$2" ;;
        --ray-port) ray_port="$2" ;; --api-port) api_port="$2" ;;
        --layers) layers="$2" ;; --trace-root) trace_root="$2" ;;
        --npu-core-fractions) npu_core_fractions="$2" ;;
        *) forward+=("$1" "$2") ;;
      esac
      shift 2 ;;
    *) die "未知参数：$1；使用 --help 查看" ;;
  esac
done

normalize_soc() {
  case "$1" in
    910b|910B) echo 910b ;;
    310p|310P) echo 310p ;;
    *) die '--head-soc / --peer-soc 只能是 910b 或 310p' ;;
  esac
}
head_soc="$(normalize_soc "$head_soc")"
[[ "$transport" == gloo || "$transport" == hccl ]] || die '--transport 只能是 gloo 或 hccl'
if [[ -z "$peer_soc" ]]; then
  if [[ "$transport" == hccl ]]; then peer_soc=910b; else peer_soc=310p; fi
fi
peer_soc="$(normalize_soc "$peer_soc")"
if [[ -n "$npu_core_fractions" ]]; then
  if "$debug_shell"; then
    die '--npu-core-fractions 需要服务或 profiling，不能和 --shell 一起用'
  fi
  forward+=(--npu-core-fractions "$npu_core_fractions")
fi
if "$verify_overlap"; then
  if [[ "$transport" != gloo ]] || "$profile" || "$debug_shell"; then
    die '--verify-overlap 仅用于 Gloo 在线验证'
  fi
  forward+=(--verify-overlap)
fi
if [[ "$transport" == hccl ]]; then
  [[ "$head_soc" == 910b && "$peer_soc" == 910b ]] || die 'HCCL 只支持两边 910B；异构请用 gloo'
fi

if "$profile"; then
  "$debug_shell" && die '--profile 和 --shell 不能同时使用'
  if [[ -n "$trace_root" ]]; then
    [[ "$trace_root" == /* && -d "$trace_root" && -w "$trace_root" && -x "$trace_root" ]] ||
      die '--trace-root 需要已有、可写的本地目录绝对路径'
    trace_root="$(cd -- "$trace_root" && pwd -P)"
    [[ "$trace_root" != *,* ]] || die '本地结果目录路径不能含逗号'
  fi
elif [[ -n "$trace_root" ]]; then
  die '--trace-root 仅用于 --profile'
fi

expand_ip() {
  local value="$1" part result="" parts=()
  [[ "$value" != "" ]] || die "需要填写 --head 和 --peer 的 IP/末段"
  if [[ "$value" =~ ^[0-9]{1,3}$ ]]; then value="$IP_PREFIX.$value"; fi
  [[ "$value" =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]] || die "无效 IPv4：$value"
  IFS=. read -r -a parts <<< "$value"
  for part in "${parts[@]}"; do
    ((10#$part <= 255)) || die "无效 IPv4：$value"
    result+="${result:+.}$((10#$part))"
  done
  [[ "${parts[0]}" != 127 && "$result" != 0.0.0.0 ]] || die "不能使用回环或未指定地址"
  echo "$result"
}
head="$(expand_ip "$head")" peer="$(expand_ip "$peer")"
[[ "$head" != "$peer" ]] || die "head 和 peer 不能相同"
[[ "$devices" =~ ^(0|[1-9][0-9]*),(0|[1-9][0-9]*)$ ]] || die "devices 需为两个设备号，例如 0,1"
first="${devices%,*}" second="${devices#*,}"
[[ "$first" != "$second" ]] || die "两张卡设备号不能相同"
coordination_port="$ray_port"
if [[ "$transport" == hccl ]]; then coordination_port="$master_port"; fi
for port in "$coordination_port" "$api_port"; do
  if ! [[ "$port" =~ ^[1-9][0-9]{3,4}$ ]] || ((port < 1024 || port > 65535)); then
    die "端口需在 1024..65535"
  fi
done
[[ "$coordination_port" != "$api_port" ]] || die "主端通信和 API 不能使用同一端口"
if [[ -n "$layers" ]]; then
  [[ "$layers" =~ ^[1-9][0-9]*,[1-9][0-9]*,[1-9][0-9]*,[1-9][0-9]*$ ]] || die "layers 需要四个正整数"
fi
if ! "$debug_shell"; then
  [[ -d "$model" ]] || die "模型目录不存在：$model"
  model="$(cd -- "$model" && pwd -P)"
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "$script_dir/../../.." && pwd)"
[[ "$repo" != *,* && "$model" != *,* ]] || die "Docker bind mount 路径不能含逗号"

command -v ip >/dev/null || die "需要 ip 命令识别本机 IPv4/路由；与 npu-smi 无关"
addresses="$(ip -o -4 addr show)" || die "无法查询本机 IPv4"
owns_head=false owns_peer=false
while read -r _ nic family cidr _; do
  [[ "$family" == inet ]] || continue
  address="${cidr%/*}"
  if [[ "$address" == "$head" ]]; then owns_head=true; fi
  if [[ "$address" == "$peer" ]]; then owns_peer=true; fi
done <<< "$addresses"
if "$owns_head" && ! "$owns_peer"; then
  role=head local_ip="$head" target="$peer" local_soc="$head_soc"
elif "$owns_peer" && ! "$owns_head"; then
  role=worker local_ip="$peer" target="$head" local_soc="$peer_soc"
else
  die "本机必须且只能拥有 head/peer 中一个 IP；不猜测设备型号"
fi
if [[ "$local_soc" == 910b ]]; then
  image="${image:-$image_910b}"
else
  image="${image:-$image_310p}"
fi
[[ -n "$image" && "$image" != -* ]] || die "无效镜像名"
if [[ -z "$interface" ]]; then
  route="$(ip -4 route get "$target" from "$local_ip")" || die "无法查找到对端的路由"
  read -r -a words <<< "$route"
  for ((i=0; i+1<${#words[@]}; i++)); do
    if [[ "${words[i]}" == dev ]]; then interface="${words[i+1]}"; break; fi
  done
fi
valid_interface=false
while read -r _ nic family cidr _; do
  if [[ "${nic%%@*}" == "$interface" && "$family" == inet && "${cidr%/*}" == "$local_ip" ]]; then
    valid_interface=true
  fi
done <<< "$addresses"
"$valid_interface" && [[ "$interface" != lo ]] || die "所选网卡不拥有本机 IP：$interface / $local_ip"

if "$debug_shell"; then
  name="${name:-hetero-pp-debug-${role}-$(date +%Y%m%d_%H%M%S)-$$}"
  command=(docker run --name "$name" --net=host --shm-size=1g
    --security-opt seccomp=unconfined -i)
  if [[ -t 0 && -t 1 ]]; then command+=(-t); fi
else
if [[ "$log_dir" != /* ]]; then log_dir="$(pwd -P)/$log_dir"; fi
[[ "$log_dir" != *,* ]] || die "日志 bind mount 路径不能含逗号"
if "$dry_run"; then
  run_dir="$log_dir/preview-$role"
else
  # The invoking host user owns these private parents, even when Docker is root.
  (umask 077; mkdir -p -- "$log_dir") || die "无法创建日志目录；用 --log-dir 指定你自己的可写路径"
  log_dir="$(cd -- "$log_dir" && pwd -P)"
  [[ -w "$log_dir" && -x "$log_dir" ]] || die "日志目录不可写/不可进入：$log_dir"
  run_dir="$(umask 077; mktemp -d "$log_dir/hetero_pp_${role}_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
fi
name="${name:-hetero-pp-$role}"
command=(docker run --rm --name "$name" --net=host --shm-size=1g)
if [[ "$local_soc" == 310p ]]; then command+=(--security-opt seccomp=unconfined); fi
if [[ -t 0 && -t 1 ]]; then command+=(-it); fi
fi
for device in "davinci$first" "davinci$second" davinci_manager hisi_hdc; do
  command+=(--device "/dev/$device")
done
# Some 310P installations do not expose this node; never create a substitute.
if [[ -c /dev/devmm_svm ]]; then command+=(--device /dev/devmm_svm); fi
for path in /usr/local/dcmi /usr/local/Ascend/driver/lib64 \
  /usr/local/Ascend/driver/version.info /etc/ascend_install.info; do
  command+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
command+=(--mount "type=bind,src=$repo,dst=/work/vllm" -w /work/vllm
  -e "GLOO_SOCKET_IFNAME=$interface" -e "VLLM_HOST_IP=$local_ip")
if [[ "$transport" == hccl ]]; then
  command+=(-e "HCCL_IF_IP=$local_ip" -e "TP_SOCKET_IFNAME=$interface")
fi
if "$debug_shell"; then
  command+=(--entrypoint /bin/bash "$image")
else
command+=(--mount "type=bind,src=$run_dir,dst=/work/hetero_logs"
  --mount "type=bind,src=$model,dst=/workspace/model,readonly"
  -e "HETERO_PP_ROLE=$role" -e "HETERO_PP_TRANSPORT=$transport")
if "$profile" && [[ -n "$trace_root" ]]; then
  command+=(--mount "type=bind,src=$trace_root,dst=/work/hetero_profiles"
    -e HETERO_PP_MODE=profile)
fi
# Retain the image's entrypoint so custom image initialization is not bypassed.
# Expanded only in the container. A joined worker must keep PID 1 alive for Ray.
# shellcheck disable=SC2016
command+=("$image" /bin/bash -c '
set -u
umask 022
if [[ "$HETERO_PP_ROLE" == head || "$HETERO_PP_TRANSPORT" == hccl ]]; then
  exec bash /work/vllm/examples/online_serving/hetero_pp/start_hetero_pp.sh "$@"
fi
bash /work/vllm/examples/online_serving/hetero_pp/start_hetero_pp.sh "$@"
status=$?
if ((status != 0)); then exit "$status"; fi
echo "peer Ray 已加入；保持此容器运行。head 完成本轮服务/采集后 Ctrl+C 停止本容器。"
trap "exit 0" INT TERM HUP
while true; do sleep 5 & wait $!; done
' hetero-pp --head "$head" --peer "$peer" --model /workspace/model
  --head-soc "$head_soc" --peer-soc "$peer_soc"
  --transport "$transport" --master-port "$master_port"
  --node-ip "$local_ip" --interface "$interface"
  --ray-port "$ray_port" --api-port "$api_port" --output /work/hetero_logs/start)
if [[ -n "$layers" ]]; then command+=(--layers "$layers"); fi
if "$profile"; then
  profile_root=/work/hetero_logs/profiles
  if [[ -n "$trace_root" ]]; then profile_root=/work/hetero_profiles; fi
  command+=(--profile --trace-root "$profile_root")
fi
# Bash 3 treats an empty array as unset under nounset.
if [[ -n "${forward[*]-}" ]]; then command+=("${forward[@]}"); fi
fi
if "$dry_run"; then
  printf '%q ' "${command[@]}"
  echo
  exit 0
fi
command -v docker >/dev/null || die "找不到 docker 命令"
echo "本机角色=$role，型号=$local_soc，传输=$transport，IP=$local_ip，网卡=$interface，镜像=$image" >&2
if "$debug_shell"; then
  echo "只进 Bash：跳过镜像入口/自动环境初始化、NPU 预检、Ray 和模型启动；不挂载模型或实验日志。" >&2
  echo "调试容器禁用 seccomp 过滤；退出后保留，可用 docker start -ai $name 再次进入。" >&2
  echo "另开终端进入：docker exec -it $name /bin/bash" >&2
else
echo "选择镜像依据节点型号配置；容器内将实际校验两张 NPU 型号及初始化。" >&2
if [[ "$transport" == hccl ]]; then
  echo "原生 HCCL/mp，主端端口=$master_port，API端口=$api_port；需跨机 HCCL 网络可用。" >&2
else
  echo "Ray端口=$ray_port，API端口=$api_port；不删除已有容器、不重新设置 NPU 可见性。" >&2
fi
echo "宿主机本轮日志：$run_dir/start；目录和日志文件将允许宿主机用户读取。" >&2
fi
exec "${command[@]}"
