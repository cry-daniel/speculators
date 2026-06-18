#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${EVAL_ROOT}/results/lm_eval}"
ENV_OUT="${ENV_OUT:-${RESULTS_DIR}/environment.txt}"

mkdir -p "${RESULTS_DIR}"

conda run -n spec python -m pip install --upgrade-strategy only-if-needed \
  "lm-eval[api,tasks,longbench,math]==0.4.12"

{
  echo "# lm-eval environment"
  date --iso-8601=seconds
  echo
  echo "## pip check"
  conda run -n spec python -m pip check || true
  echo
  echo "## Python"
  conda run -n spec python -c "import sys; print(sys.version)"
  echo
  echo "## Package versions"
  conda run -n spec python -c "import importlib.metadata as m; print('lm-eval', m.version('lm-eval')); print('torch', m.version('torch')); print('vllm', m.version('vllm')); print('guidellm', m.version('guidellm'))"
  echo
  echo "## lm-eval help"
  conda run -n spec lm-eval --help
  echo
  echo "## selected tasks"
  conda run -n spec lm-eval ls tasks | grep -Ei 'logiqa|gsm8k_cot|hendrycks_math|mmlu_generative|humaneval_instruct|longbench2' || true
  echo
  echo "## selected groups"
  conda run -n spec lm-eval ls groups | grep -Ei 'hendrycks_math|mmlu|longbench2' || true
} >"${ENV_OUT}" 2>&1

echo "${ENV_OUT}"
