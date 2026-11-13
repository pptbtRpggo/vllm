#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Internal container entrypoint. All package installation stays in its own venv.
set -euo pipefail
[[ $# == 3 ]] || { echo '需要压测脚本、数据目录和结果目录三个参数' >&2; exit 2; }
benchmark="$1" data="$2" results="$3"
mkdir -p "$HOME/tools/data/gsm8k"
cp "$benchmark" "$HOME/tools/benchmark_gsm8k.sh"
cp -R "$data/." "$HOME/tools/data/gsm8k/"
chmod u+x "$HOME/tools/benchmark_gsm8k.sh"
export PATH="$HOME/.venv/bin:$HOME/tools:$PATH"
if [[ ! -e "$HOME/vllm_hetero_pp_logs" && ! -L "$HOME/vllm_hetero_pp_logs" ]]; then
  ln -s "$results" "$HOME/vllm_hetero_pp_logs"
fi
if [[ ! -e "$HOME/.bashrc" ]]; then
cat > "$HOME/.bashrc" <<'EOF'
export PATH="$HOME/.venv/bin:$HOME/tools:$PATH"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
cd "$HOME"
EOF
fi
if [[ ! -f "$HOME/.evalscope-ready" ]]; then
  bootstrap_python="${EVALSCOPE_BOOTSTRAP_PYTHON:-python3.12}"
  command -v "$bootstrap_python" >/dev/null || { echo "找不到镜像 Python：$bootstrap_python" >&2; exit 2; }
  "$bootstrap_python" -c 'import sys; assert sys.version_info >= (3,11), "客户端需要 Python >= 3.11"'
  if [[ ! -x "$HOME/.venv/bin/python" ]]; then
    "$bootstrap_python" -m venv "$HOME/.venv"
  fi
  package='evalscope[perf]'
  if [[ -n "${EVALSCOPE_VERSION:-}" ]]; then package+="==$EVALSCOPE_VERSION"; fi
  "$HOME/.venv/bin/python" -m pip --isolated install --upgrade \
    --timeout 30 --retries 1 --index-url "${EVALSCOPE_INDEX_URL:-https://pypi.org/simple}" \
    "$package" 2>&1 | tee -a "$results/evalscope_install.log"
  "$HOME/.venv/bin/python" -m pip freeze > "$results/client_environment.txt"
fi
if ! help_text="$(evalscope perf --help 2>&1)"; then
  echo "$help_text" >&2
  exit 1
fi
for flag in --parallel --number --warmup-num --outputs-dir --stream; do
  [[ "$help_text" == *"$flag"* ]] || { echo "已安装 EvalScope 缺少参数：$flag" >&2; exit 1; }
done
touch "$HOME/.evalscope-ready"
evalscope --version
echo '准备完成。已复制压测脚本和 GSM8K；尚未发出任何推理请求。'
echo '小测试：benchmark_gsm8k.sh --parallel 1 --number 4 --warmup-num 0 --max-tokens 32'
echo '完整压测：benchmark_gsm8k.sh'
echo "默认地址 http://127.0.0.1:8000，模型 hetero-model；结果写入 $results 对应的宿主机目录。"
cd "$HOME"
exec /bin/bash -i
