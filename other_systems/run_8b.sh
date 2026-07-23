#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
KERNEL_OUT="${KERNEL_OUT:-examples/evaluate/eval-guidellm/results/other_systems_nm_kernel_8b_${STAMP}}"
LAYER_OUT="${LAYER_OUT:-examples/evaluate/eval-guidellm/results/other_systems_nm_layer_8b_${STAMP}}"

conda run --no-capture-output -n spec python -u other_systems/verify.py
MPLCONFIGDIR=temp/matplotlib \
  conda run --no-capture-output -n spec python -u \
  other_systems/bench_kernel.py --output-root "${KERNEL_OUT}"
MPLCONFIGDIR=temp/matplotlib \
  conda run --no-capture-output -n spec python -u \
  other_systems/bench_layer.py --output-root "${LAYER_OUT}"

printf '%s\n' "${KERNEL_OUT}" "${LAYER_OUT}"
