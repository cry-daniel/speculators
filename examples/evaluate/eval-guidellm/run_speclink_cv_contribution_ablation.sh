#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

timestamp="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/temp/speclink_cv_contribution_ablation_${timestamp}}"
PYTHON_BIN="${PYTHON_BIN:-/ACALAB/stu1/miniconda3/envs/spec/bin/python}"
MODEL_LIST="${MODEL_LIST:-qwen3_8b,llama3_1_8b}"
DATASET_LIST="${DATASET_LIST:-math}"
K_LIST="${K_LIST:-12}"
BATCH_SIZE_LIST="${BATCH_SIZE_LIST:-16}"
MAX_REQUESTS="${MAX_REQUESTS:-32}"
MAX_TOKENS="${MAX_TOKENS:-0}"
PORT="${PORT:-8096}"
BENCHMARK_MODE="${BENCHMARK_MODE:-steady_state}"
STEADY_STATE_WARMUP_S="${STEADY_STATE_WARMUP_S:-30}"
STEADY_STATE_MEASUREMENT_S="${STEADY_STATE_MEASUREMENT_S:-120}"
STEADY_STATE_COOLDOWN_S="${STEADY_STATE_COOLDOWN_S:-30}"
STEADY_STATE_MAX_PROMPTS="${STEADY_STATE_MAX_PROMPTS:-${MAX_REQUESTS}}"
STEADY_STATE_IGNORE_EOS="${STEADY_STATE_IGNORE_EOS:-1}"

if [[ "${BENCHMARK_MODE}" == "steady_state" && "${MAX_TOKENS}" == "0" ]]; then
  MAX_TOKENS="${STEADY_STATE_MAX_TOKENS:-1024}"
fi

COMMON_ENV=(
  MODEL_LIST="${MODEL_LIST}"
  DATASET_LIST="${DATASET_LIST}"
  K_LIST="${K_LIST}"
  BATCH_SIZE_LIST="${BATCH_SIZE_LIST}"
  MAX_REQUESTS="${MAX_REQUESTS}"
  MAX_TOKENS="${MAX_TOKENS}"
  BENCHMARK_MODE="${BENCHMARK_MODE}"
  STEADY_STATE_WARMUP_S="${STEADY_STATE_WARMUP_S}"
  STEADY_STATE_MEASUREMENT_S="${STEADY_STATE_MEASUREMENT_S}"
  STEADY_STATE_COOLDOWN_S="${STEADY_STATE_COOLDOWN_S}"
  STEADY_STATE_MAX_PROMPTS="${STEADY_STATE_MAX_PROMPTS}"
  STEADY_STATE_IGNORE_EOS="${STEADY_STATE_IGNORE_EOS}"
  FORCE_PREFIX_LEN="${FORCE_PREFIX_LEN:-0}"
  BATCH_INVARIANT="${BATCH_INVARIANT:-0}"
  ALLOW_CV_CUDAGRAPH="${ALLOW_CV_CUDAGRAPH:-1}"
  DENSE_REALIGN_STEPS="${DENSE_REALIGN_STEPS:-0}"
  SUFFIX_REPLAY_ONE_SHOT_SHAPE="${SUFFIX_REPLAY_ONE_SHOT_SHAPE:-0}"
  PROFILE_MAX_EVENTS="${PROFILE_MAX_EVENTS:-1800}"
  LOG_MAX_EVENTS="${LOG_MAX_EVENTS:-1400}"
  ANALYSIS_PROFILE_MAX_ROWS="${ANALYSIS_PROFILE_MAX_ROWS:-12000}"
  EXTRACT_OUTPUTS="${EXTRACT_OUTPUTS:-1}"
)

echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Running batched live CV ablation group"
env "${COMMON_ENV[@]}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}/batched" \
  PORT="${PORT}" \
  METHOD_LIST="${BATCHED_METHOD_LIST:-eagle3_oneshot,cv_half_async_simple,cv_half_async_staged_simple}" \
  MAX_VERIFY_SEQS_PER_STEP="${MAX_VERIFY_SEQS_PER_STEP:-32}" \
  ALLOW_BATCHED_PREFIX_VERIFICATION=1 \
  ALLOW_BATCHED_SUFFIX=1 \
  bash "${SCRIPT_DIR}/run_speclink_cv_math_quality.sh"

echo "[INFO] Running singleton-live CV ablation group"
env "${COMMON_ENV[@]}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}/singleton_live" \
  PORT="$((PORT + 1))" \
  METHOD_LIST="${SINGLETON_METHOD_LIST:-cv_half_async_staged_simple}" \
  MAX_VERIFY_SEQS_PER_STEP=1 \
  ALLOW_BATCHED_PREFIX_VERIFICATION=1 \
  ALLOW_BATCHED_SUFFIX=0 \
  bash "${SCRIPT_DIR}/run_speclink_cv_math_quality.sh"

echo "[INFO] Summarizing contribution ablation"
"${PYTHON_BIN}" "${REPO_ROOT}/tools/speclink_cv/analyze_contribution_ablation.py" \
  --batched-root "${OUTPUT_ROOT}/batched" \
  --singleton-root "${OUTPUT_ROOT}/singleton_live" \
  --output-dir "${OUTPUT_ROOT}/09_reports"

"${PYTHON_BIN}" "${REPO_ROOT}/tools/speclink_cv/analyze_steady_state_throughput.py" \
  "${OUTPUT_ROOT}/batched" \
  "${OUTPUT_ROOT}/singleton_live" \
  --output-dir "${OUTPUT_ROOT}/09_reports" \
  --running-fraction "${STEADY_STATE_RUNNING_FRACTION:-0.8}"

echo "[INFO] Report: ${OUTPUT_ROOT}/09_reports/contribution_ablation.md"
