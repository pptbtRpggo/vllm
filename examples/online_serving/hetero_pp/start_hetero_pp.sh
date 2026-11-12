#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Run inside the already working container on each host; do not replace its entrypoint.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../../.." && pwd)"
candidates=()
previous=""
for argument in "$@"; do
  if [[ "$previous" == --python ]]; then candidates+=("$argument"); fi
  case "$argument" in --python=*) candidates+=("${argument#--python=}") ;; esac
  previous="$argument"
done
for root in "${VIRTUAL_ENV:-}" "${CONDA_PREFIX:-}" "$repo_dir/.venv"; do
  if [[ -n "$root" ]]; then candidates+=("$root/bin/python"); fi
done
for name in python python3; do
  if location="$(command -v "$name")"; then candidates+=("$location"); fi
done
for candidate in "${candidates[@]}"; do
  if [[ -x "$candidate" ]]; then
    exec "$candidate" -S "$script_dir/start_hetero_pp.py" "$@"
  fi
done
echo '找不到已有 Python；请用 --python /path/to/python 指定。' >&2
exit 2
