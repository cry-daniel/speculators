#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EVAL_DIR}/../../.." && pwd)"

timestamp="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EVAL_DIR}/results/speclink_cv_${timestamp}}"
PORT="${PORT:-8050}"
PYTHON_BIN="${PYTHON_BIN:-/ACALAB/stu1/miniconda3/envs/spec/bin/python}"
BENCHMARK_MODE="${BENCHMARK_MODE:-steady_state}"
MAX_REQUESTS="${MAX_REQUESTS:-80}"
MAX_TOKENS="${MAX_TOKENS:-128}"
STEADY_STATE_WARMUP_S="${STEADY_STATE_WARMUP_S:-30}"
STEADY_STATE_MEASUREMENT_S="${STEADY_STATE_MEASUREMENT_S:-120}"
STEADY_STATE_COOLDOWN_S="${STEADY_STATE_COOLDOWN_S:-30}"
STEADY_STATE_BUCKET_S="${STEADY_STATE_BUCKET_S:-1}"
STEADY_STATE_MAX_PROMPTS="${STEADY_STATE_MAX_PROMPTS:-${MAX_REQUESTS}}"
STEADY_STATE_IGNORE_EOS="${STEADY_STATE_IGNORE_EOS:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
PROFILE_MAX_EVENTS="${PROFILE_MAX_EVENTS:-120}"
LOG_MAX_EVENTS="${LOG_MAX_EVENTS:-240}"
ANALYSIS_PROFILE_MAX_ROWS="${ANALYSIS_PROFILE_MAX_ROWS:-1000}"
CASE_OFFSET="${CASE_OFFSET:-0}"
CASE_LIMIT="${CASE_LIMIT:-0}"
DRY_RUN="${DRY_RUN:-0}"
ANALYZE_ONLY="${ANALYZE_ONLY:-0}"
SKIP_UNIT_TESTS="${SKIP_UNIT_TESTS:-0}"
DISABLE_VLLM_ASYNC_SCHEDULING="${DISABLE_VLLM_ASYNC_SCHEDULING:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
ALLOW_BATCHED_PREFIX_VERIFICATION="${ALLOW_BATCHED_PREFIX_VERIFICATION:-1}"
ALLOW_BATCHED_SUFFIX="${ALLOW_BATCHED_SUFFIX:-${ALLOW_BATCHED_PREFIX_VERIFICATION}}"

if [[ "${BENCHMARK_MODE}" == "steady_state" && "${MAX_TOKENS}" == "0" ]]; then
  MAX_TOKENS="${STEADY_STATE_MAX_TOKENS:-1024}"
fi

extra_args=()
if [[ "${DISABLE_VLLM_ASYNC_SCHEDULING}" == "1" ]]; then
  extra_args+=(--disable-vllm-async-scheduling)
fi
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  extra_args+=(--enforce-eager)
fi
if [[ "${CASE_OFFSET}" != "0" ]]; then
  extra_args+=(--case-offset "${CASE_OFFSET}")
fi
if [[ "${CASE_LIMIT}" != "0" ]]; then
  extra_args+=(--case-limit "${CASE_LIMIT}")
fi
if [[ "${ALLOW_BATCHED_PREFIX_VERIFICATION}" == "1" ]]; then
  extra_args+=(--allow-batched-prefix-verification)
fi
extra_args+=(--env "SPECLINK_CV_ALLOW_BATCHED_SUFFIX=${ALLOW_BATCHED_SUFFIX}")
if [[ "${STEADY_STATE_IGNORE_EOS}" == "0" ]]; then
  extra_args+=(--no-steady-state-ignore-eos)
else
  extra_args+=(--steady-state-ignore-eos)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  extra_args+=(--dry-run)
fi
if [[ "${ANALYZE_ONLY}" == "1" ]]; then
  extra_args+=(--analyze-only)
fi
if [[ "${SKIP_UNIT_TESTS}" == "1" ]]; then
  extra_args+=(--skip-unit-tests)
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -u examples/evaluate/eval-guidellm/scripts/run_speclink_cv_guidellm_matrix.py \
  --benchmark-mode "${BENCHMARK_MODE}" \
  --models qwen3_8b,llama3_1_8b \
  --datasets math,mtbench \
  --ks 8,12 \
  --batch-sizes 8,16,32 \
  --methods pure_vllm,eagle3_oneshot,cv_half_sync_simple,cv_half_sync_roofline,cv_half_async_simple,cv_half_async_roofline,cv_half_async_staged_simple \
  --max-requests "${MAX_REQUESTS}" \
  --max-tokens "${MAX_TOKENS}" \
  --steady-state-warmup-s "${STEADY_STATE_WARMUP_S}" \
  --steady-state-measurement-s "${STEADY_STATE_MEASUREMENT_S}" \
  --steady-state-cooldown-s "${STEADY_STATE_COOLDOWN_S}" \
  --steady-state-bucket-s "${STEADY_STATE_BUCKET_S}" \
  --steady-state-max-prompts "${STEADY_STATE_MAX_PROMPTS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --profile-max-events "${PROFILE_MAX_EVENTS}" \
  --log-max-events "${LOG_MAX_EVENTS}" \
  --analysis-profile-max-rows "${ANALYSIS_PROFILE_MAX_ROWS}" \
  --port "${PORT}" \
  --resume \
  --output-root "${OUTPUT_ROOT}" \
  "${extra_args[@]}"
