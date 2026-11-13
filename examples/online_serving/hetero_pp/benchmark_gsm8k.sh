#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# HTTP client only: does not start/stop Ray, Docker, vLLM, or NPU processes.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

usage() {
  cat <<'EOF'
用法：bash benchmark_gsm8k.sh --dataset-path /path/to/gsm8k_test.jsonl [选项]

默认依次测试并发 8、16、32、64，每轮 512 个正式请求 + 64 个预热请求。
输入为 JSONL，每行包含 question 字段；使用 EvalScope openqa 读取器，
实际请求内容来自本地 GSM8K 文件，不会下载默认 OpenQA 数据。

  --url URL             服务根地址或 /v1/chat/completions
                        默认 http://127.0.0.1:8000
  --model NAME          API 的 served-model-name，默认 hetero-model
  --parallel N [...]    并发列表，默认 8 16 32 64
  --number N            每个并发的正式请求数，默认 512
  --warmup-num N        每轮预热数，默认 64；0 关闭，否则至少等于最大并发
  --max-tokens N        每请求输出上限，默认 128；保留正常 EOS 停止
  --tokenizer-path DIR  可选，本机模型 tokenizer 路径
  --evalscope PATH      已安装的 EvalScope 可执行文件，默认 evalscope
  --output-dir DIR      新建结果目录；默认 HOME/vllm_hetero_pp_logs 下唯一目录
  --dry-run             只显示命令，不发请求、不创建结果目录
  -h, --help            显示帮助

须已有可用的 EvalScope perf 和 vLLM API；脚本不安装依赖、不启动服务。
这是持续补发的 HTTP 并发压测，不是整批结束再发下一批。
指标为客户端 TTFT/TPOT/E2E，包含网络和服务器排队等开销。
EOF
}

die() { echo "错误：$*" >&2; exit 2; }
positive() { [[ "$1" =~ ^[1-9][0-9]{0,8}$ ]]; }

url="http://127.0.0.1:8000" model="hetero-model" dataset=""
number=512 warmup=64 max_tokens=128 tokenizer=""
evalscope_bin="evalscope" output_dir="" dry_run=0
parallel=(8 16 32 64)

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) dry_run=1; shift ;;
    --parallel)
      shift
      parallel=()
      while (($#)) && [[ "$1" != --* ]]; do
        positive "$1" || die '--parallel 需要正整数，例如 8 16 32 64'
        parallel+=("$1")
        shift
      done
      ((${#parallel[@]})) || die '--parallel 缺少并发数'
      ;;
    --dataset-path|--url|--model|--number|--warmup-num|--max-tokens|--tokenizer-path|--evalscope|--output-dir)
      (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || die "$1 缺少参数"
      case "$1" in
        --dataset-path) dataset="$2" ;; --url) url="$2" ;;
        --model) model="$2" ;; --number) number="$2" ;;
        --warmup-num) warmup="$2" ;; --max-tokens) max_tokens="$2" ;;
        --tokenizer-path) tokenizer="$2" ;; --evalscope) evalscope_bin="$2" ;;
        --output-dir) output_dir="$2" ;;
      esac
      shift 2 ;;
    *) die "未知参数：$1；使用 --help 查看用法" ;;
  esac
done

[[ -n "$dataset" && -f "$dataset" && -r "$dataset" && -s "$dataset" ]] ||
  die '请用 --dataset-path 指定可读、非空的 GSM8K JSONL 文件'
positive "$number" || die '--number 需要正整数'
positive "$max_tokens" || die '--max-tokens 需要正整数'
[[ "$warmup" == 0 ]] || positive "$warmup" || die '--warmup-num 需要非负整数'
max_parallel=0
for count in "${parallel[@]}"; do
  if ((count > max_parallel)); then max_parallel="$count"; fi
done
((number >= max_parallel)) || die '--number 不能小于最大并发，否则达不到指定并发'
((warmup == 0 || warmup >= max_parallel)) ||
  die '--warmup-num 应至少等于最大并发，或设为 0 明确关闭预热'

case "$url" in http://*|https://*) ;; *) die '--url 需要 http:// 或 https://' ;; esac
url="${url%/}"
case "$url" in
  */v1/chat/completions) ;;
  */v1) url="$url/chat/completions" ;;
  *) url="$url/v1/chat/completions" ;;
esac
[[ -z "$tokenizer" || -d "$tokenizer" ]] || die '--tokenizer-path 必须是本机可读目录'
[[ -z "$output_dir" || (! -e "$output_dir" && ! -L "$output_dir") ]] ||
  die '--output-dir 已存在；请换一个新目录，避免覆盖旧结果'

if ((dry_run)); then
  output_dir="${output_dir:-${HOME:?需要 HOME}/vllm_hetero_pp_logs/evalscope_preview}"
else
  command -v "$evalscope_bin" >/dev/null || die '找不到 EvalScope；请激活已有环境或使用 --evalscope /path/to/evalscope'
  if ! perf_help="$("$evalscope_bin" perf --help 2>&1)"; then
    die 'EvalScope perf 无法运行；请检查当前环境及 perf 依赖'
  fi
  for flag in --parallel --number --warmup-num --outputs-dir --stream; do
    [[ "$perf_help" == *"$flag"* ]] || die "当前 EvalScope 不支持 $flag；请核对版本"
  done
  umask 077
  if [[ -z "$output_dir" ]]; then
    parent="${HOME:?需要 HOME}/vllm_hetero_pp_logs"
    mkdir -p -- "$parent"
    output_dir="$(mktemp -d "$parent/evalscope_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
  else
    mkdir -p -- "$(dirname -- "$output_dir")"
    mkdir -- "$output_dir" || die '无法新建结果目录'
  fi
fi

numbers=()
for _ in "${parallel[@]}"; do numbers+=("$number"); done
command=("$evalscope_bin" perf
  --url "$url" --model "$model" --api openai
  --dataset openqa --dataset-path "$dataset"
  --parallel "${parallel[@]}" --number "${numbers[@]}"
  --warmup-num "$warmup" --rate -1 --stream
  --temperature 0 --seed 42 --max-tokens "$max_tokens"
  --connect-timeout 10 --read-timeout 600 --total-timeout 1800
  --outputs-dir "$output_dir/results")
if [[ -n "$tokenizer" ]]; then command+=(--tokenizer-path "$tokenizer"); fi

if ((dry_run)); then printf '%q ' "${command[@]}"; echo; exit 0; fi
{ echo '#!/usr/bin/env bash'; printf '%q ' "${command[@]}"; echo; } > "$output_dir/command.sh"
"$evalscope_bin" --version > "$output_dir/evalscope_version.txt" 2>&1 || true
echo "结果目录：$output_dir"
echo "并发：${parallel[*]}；每轮正式请求 ${number}，预热 ${warmup}；输出上限 ${max_tokens} token。"
echo '请确认服务 context 容量覆盖题目、chat template 和输出上限。'
"${command[@]}" 2>&1 | tee "$output_dir/run.log"
echo "EvalScope 命令已结束；请检查报告中的成功/失败请求数。结果：$output_dir/results"
