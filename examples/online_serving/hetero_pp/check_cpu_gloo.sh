#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Both hosts use this entrypoint; the Python helper selects the NPU environment.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../../.." && pwd)"
candidates=()
previous=""
help_only=false
for argument in "$@"; do
  case "$argument" in --help|-h) help_only=true ;; esac
  if [[ "$previous" == --python ]]; then candidates+=("$argument"); fi
  case "$argument" in --python=*) candidates+=("${argument#--python=}") ;; esac
  previous="$argument"
done
bootstrap_limit=(env)
if limiter="$(command -v timeout || command -v gtimeout)"; then
  bootstrap_limit=("$limiter" --signal=TERM --kill-after=2s 10s)
elif ! "$help_only"; then
  echo '需要 coreutils timeout 来限制 Python 环境探测时间；请检查 PATH。' >&2
  exit 2
fi
for root in "${VIRTUAL_ENV:-}" "${CONDA_PREFIX:-}" "$repo_dir/.venv"; do
  if [[ -n "$root" ]]; then candidates+=("$root/bin/python"); fi
done
candidates=("${COMM_CHECK_BOOTSTRAP_PYTHON:-}" "${candidates[@]}")
for name in python python3; do
  if location="$(command -v "$name")"; then candidates+=("$location"); fi
done
for candidate in "${candidates[@]}"; do
  if [[ -x "$candidate" ]] && "${bootstrap_limit[@]}" "$candidate" -S -c \
    'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
    exec "$candidate" -S "$script_dir/check_cpu_gloo.py" "$@"
  fi
done
echo '找不到 Python 3.9+；请激活已有环境或使用 --python /path/to/venv/bin/python。' >&2
exit 2
