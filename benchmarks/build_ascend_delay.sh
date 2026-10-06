#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
repo_root=$(cd "$(dirname "$0")/.." && pwd)
output_dir=${1:?usage: build_ascend_delay.sh OUTPUT_DIRECTORY}
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)
cann_root=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
compiler=${cann_root}/bin/bisheng
"$compiler" --cce-aicore-arch=dav-c220 -O2 -std=c++17 -xcce -fPIC \
  -I"$cann_root/compiler/tikcpp" \
  -I"$cann_root/compiler/tikcpp/tikcfw" \
  -I"$cann_root/compiler/tikcpp/tikcfw/impl" \
  -I"$cann_root/compiler/tikcpp/tikcfw/interface" \
  -I"$cann_root/include" \
  -c "$repo_root/csrc/ascend/hetero_delay.cpp" -o "$output_dir/hetero_delay.o"
"$compiler" --cce-fatobj-link -L"$cann_root/lib64" \
  "$output_dir/hetero_delay.o" --shared \
  -lruntime -lstdc++ -lascendcl -lm -ldl \
  -o "$output_dir/libhetero_delay.so"
echo "$output_dir/libhetero_delay.so"
