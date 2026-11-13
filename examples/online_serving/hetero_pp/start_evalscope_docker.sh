#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Host entrypoint: prepare a CPU-only HTTP benchmark client, then open Bash.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

usage() {
  cat <<'EOF'
用法：bash start_evalscope_docker.sh [选项]
默认用已有 v0.23.0 openEuler 镜像新建客户端容器，安装 evalscope[perf]，
复制压测脚本和项目内 GSM8K 测试集，然后进入 Bash。不自动发请求。
不映射 NPU、不挂载驱动、不启动/停止 Ray 或模型，不安装宿主机依赖。

  --image IMAGE          默认 quay.io/ascend/vllm-ascend:v0.23.0-openeuler
  --name NAME            默认 evalscope-client-时间-PID；不删除已有容器
  --output-dir DIR       新建宿主机结果目录，默认 HOME/vllm_hetero_pp_logs 下
  --python COMMAND       容器内用于创建 venv 的 Python，默认 python3.12
  --evalscope-version V  可选，指定版本；默认安装包索引中的最新版
  --index-url URL        包索引，默认 https://pypi.org/simple
  -h, --help             显示帮助

首次安装需要联网，退出保留容器；docker start -ai 容器名可再次进入。
客户端与服务同宿主机时使用 http://127.0.0.1:8000，只启动一个客户端。
EOF
}
die() { echo "错误：$*" >&2; exit 2; }
image="quay.io/ascend/vllm-ascend:v0.23.0-openeuler"
name="evalscope-client-$(date +%Y%m%d-%H%M%S)-$$" output=""
bootstrap_python="python3.12" version="" index_url="https://pypi.org/simple"
while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --image|--name|--output-dir|--python|--evalscope-version|--index-url)
      (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || die "$1 缺少参数"
      case "$1" in
        --image) image="$2" ;; --name) name="$2" ;; --output-dir) output="$2" ;;
        --python) bootstrap_python="$2" ;; --evalscope-version) version="$2" ;;
        --index-url) index_url="$2" ;;
      esac
      shift 2 ;;
    *) die "未知参数：$1" ;;
  esac
done
[[ "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || die '无效容器名'
[[ -z "$version" || "$version" =~ ^[0-9][0-9A-Za-z.+_-]*$ ]] || die '无效版本号'
[[ "$image" != -* && "$bootstrap_python" != -* ]] || die '无效镜像或 Python'
case "$index_url" in http://*|https://*) ;; *) die '无效包索引 URL' ;; esac
directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ -s "$directory/data/gsm8k/test.jsonl" ]] || die '缺少项目 GSM8K 数据，请更新完整项目'
command -v docker >/dev/null || die '找不到 docker 命令'
# Preserve the image Python path, never replace it with the host PATH.
if ! image_env="$(docker image inspect "$image" --format '{{range .Config.Env}}{{println .}}{{end}}')"; then
  docker pull "$image"
  image_env="$(docker image inspect "$image" --format '{{range .Config.Env}}{{println .}}{{end}}')"
fi
image_path="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
while IFS= read -r entry; do
  if [[ "$entry" == PATH=* ]]; then image_path="${entry#PATH=}"; fi
done <<< "$image_env"
umask 077
if [[ -z "$output" ]]; then
  parent="${HOME:?需要 HOME}/vllm_hetero_pp_logs"
  mkdir -p -- "$parent"
  output="$(mktemp -d "$parent/evalscope_client_XXXXXX")"
else
  [[ ! -e "$output" && ! -L "$output" ]] || die '结果目录已存在，请用新目录'
  mkdir -p -- "$output"
fi
output="$(cd -- "$output" && pwd -P)"
[[ "$output" != *,* ]] || die '结果路径不能含逗号'
command=(docker create -i --name "$name" --net=host --shm-size=1g
  --user "$(id -u):$(id -g)"
  --mount "type=bind,src=$output,dst=/results"
  -e HOME=/tmp/evalscope-client
  -e "PATH=/tmp/evalscope-client/.venv/bin:/tmp/evalscope-client/tools:$image_path"
  -e PYTHONPATH= -e PYTHONNOUSERSITE=1 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0
  -e "EVALSCOPE_BOOTSTRAP_PYTHON=$bootstrap_python"
  -e "EVALSCOPE_VERSION=$version" -e "EVALSCOPE_INDEX_URL=$index_url")
if [[ -t 0 && -t 1 ]]; then command+=(-t); fi
command+=(--entrypoint /bin/bash "$image" /tmp/evalscope-setup.sh
  /tmp/benchmark_gsm8k.sh /tmp/evalscope-gsm8k /results)
echo "客户端容器：${name}；宿主机结果目录：$output"
"${command[@]}"
docker cp "$directory/evalscope_container_setup.sh" "$name:/tmp/evalscope-setup.sh"
docker cp "$directory/benchmark_gsm8k.sh" "$name:/tmp/benchmark_gsm8k.sh"
docker cp "$directory/data/gsm8k" "$name:/tmp/evalscope-gsm8k"
echo "首次安装失败可查看 docker logs ${name}，再用 docker start -ai $name 重试。"
echo "以后进入：docker start -ai ${name}；运行中另开终端：docker exec -it $name /bin/bash"
exec docker start -ai "$name"
