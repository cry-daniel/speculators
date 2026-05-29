#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

timestamp="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/temp/speclink_cv_math_quality_${timestamp}}"
PORT="${PORT:-8078}"
PYTHON_BIN="${PYTHON_BIN:-/ACALAB/stu1/miniconda3/envs/spec/bin/python}"
MODEL_LIST="${MODEL_LIST:-qwen3_8b}"
DATASET_LIST="${DATASET_LIST:-math}"
K_LIST="${K_LIST:-16}"
BATCH_SIZE_LIST="${BATCH_SIZE_LIST:-8,16,32}"
METHOD_LIST="${METHOD_LIST:-eagle3_oneshot,cv_half_async_simple}"
MAX_REQUESTS="${MAX_REQUESTS:-32}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
BENCHMARK_MODE="${BENCHMARK_MODE:-guidellm}"
STEADY_STATE_WARMUP_S="${STEADY_STATE_WARMUP_S:-30}"
STEADY_STATE_MEASUREMENT_S="${STEADY_STATE_MEASUREMENT_S:-120}"
STEADY_STATE_COOLDOWN_S="${STEADY_STATE_COOLDOWN_S:-30}"
STEADY_STATE_BUCKET_S="${STEADY_STATE_BUCKET_S:-1}"
STEADY_STATE_MAX_PROMPTS="${STEADY_STATE_MAX_PROMPTS:-${MAX_REQUESTS}}"
STEADY_STATE_IGNORE_EOS="${STEADY_STATE_IGNORE_EOS:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
FORCE_PREFIX_LEN="${FORCE_PREFIX_LEN:-8}"
DENSE_REALIGN_STEPS="${DENSE_REALIGN_STEPS:-0}"
MAX_VERIFY_SEQS_PER_STEP="${MAX_VERIFY_SEQS_PER_STEP:-32}"
PROFILE_MAX_EVENTS="${PROFILE_MAX_EVENTS:-300}"
LOG_MAX_EVENTS="${LOG_MAX_EVENTS:-400}"
ANALYSIS_PROFILE_MAX_ROWS="${ANALYSIS_PROFILE_MAX_ROWS:-3000}"
GPU_UTIL_SAMPLING_MS="${GPU_UTIL_SAMPLING_MS:-500}"
ALLOW_BATCHED_PREFIX_VERIFICATION="${ALLOW_BATCHED_PREFIX_VERIFICATION:-1}"
ALLOW_BATCHED_SUFFIX="${ALLOW_BATCHED_SUFFIX:-${ALLOW_BATCHED_PREFIX_VERIFICATION}}"
SUFFIX_REPLAY_ONE_SHOT_SHAPE="${SUFFIX_REPLAY_ONE_SHOT_SHAPE:-0}"
BATCH_INVARIANT="${BATCH_INVARIANT:-0}"
NSYS_PROFILE="${NSYS_PROFILE:-0}"
NSYS_OUTPUT_NAME="${NSYS_OUTPUT_NAME:-vllm_guidellm_profile}"
NSYS_STATS="${NSYS_STATS:-1}"
KEEP_NSYS_SQLITE="${KEEP_NSYS_SQLITE:-0}"
EXTRACT_OUTPUTS="${EXTRACT_OUTPUTS:-1}"
ANALYZE_PERF_GAP="${ANALYZE_PERF_GAP:-1}"
ANALYZE_STEADY_STATE="${ANALYZE_STEADY_STATE:-1}"
STEADY_STATE_RUNNING_FRACTION="${STEADY_STATE_RUNNING_FRACTION:-0.8}"
ALLOW_CV_CUDAGRAPH="${ALLOW_CV_CUDAGRAPH:-0}"
PREFIX_FULL_CUDAGRAPH="${PREFIX_FULL_CUDAGRAPH:-${ALLOW_CV_CUDAGRAPH}}"
ENFORCE_EAGER="${ENFORCE_EAGER:-}"

if [[ -z "${ENFORCE_EAGER}" ]]; then
  if [[ "${ALLOW_CV_CUDAGRAPH}" == "1" ]]; then
    ENFORCE_EAGER=0
  else
    ENFORCE_EAGER=1
  fi
fi

extra_args=()
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  extra_args+=(--enforce-eager)
fi
if [[ "${ALLOW_CV_CUDAGRAPH}" == "1" ]]; then
  extra_args+=(--allow-cv-cudagraph)
fi
if [[ "${ALLOW_BATCHED_PREFIX_VERIFICATION}" == "1" ]]; then
  extra_args+=(--allow-batched-prefix-verification)
fi
if [[ "${BATCH_INVARIANT}" == "1" ]]; then
  extra_args+=(--env VLLM_BATCH_INVARIANT=1)
fi
extra_args+=(--env "SPECLINK_CV_ALLOW_BATCHED_SUFFIX=${ALLOW_BATCHED_SUFFIX}")
extra_args+=(--env "SPECLINK_CV_SUFFIX_REPLAY_ONE_SHOT_SHAPE=${SUFFIX_REPLAY_ONE_SHOT_SHAPE}")
if [[ "${NSYS_PROFILE}" == "1" ]]; then
  extra_args+=(--nsys-profile --nsys-output-name "${NSYS_OUTPUT_NAME}")
  if [[ "${NSYS_STATS}" == "0" ]]; then
    extra_args+=(--no-nsys-stats)
  fi
  if [[ "${KEEP_NSYS_SQLITE}" == "1" ]]; then
    extra_args+=(--keep-nsys-sqlite)
  fi
fi
if [[ "${STEADY_STATE_IGNORE_EOS}" == "0" ]]; then
  extra_args+=(--no-steady-state-ignore-eos)
else
  extra_args+=(--steady-state-ignore-eos)
fi

cd "${REPO_ROOT}"

"${PYTHON_BIN}" -u examples/evaluate/eval-guidellm/scripts/run_speclink_cv_guidellm_matrix.py \
  --benchmark-mode "${BENCHMARK_MODE}" \
  --models "${MODEL_LIST}" \
  --datasets "${DATASET_LIST}" \
  --ks "${K_LIST}" \
  --batch-sizes "${BATCH_SIZE_LIST}" \
  --methods "${METHOD_LIST}" \
  --max-requests "${MAX_REQUESTS}" \
  --max-tokens "${MAX_TOKENS}" \
  --steady-state-warmup-s "${STEADY_STATE_WARMUP_S}" \
  --steady-state-measurement-s "${STEADY_STATE_MEASUREMENT_S}" \
  --steady-state-cooldown-s "${STEADY_STATE_COOLDOWN_S}" \
  --steady-state-bucket-s "${STEADY_STATE_BUCKET_S}" \
  --steady-state-max-prompts "${STEADY_STATE_MAX_PROMPTS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --port "${PORT}" \
  --disable-vllm-async-scheduling \
  --allow-shape-drift-chunking \
  --max-verify-seqs-per-step "${MAX_VERIFY_SEQS_PER_STEP}" \
  --env "SPECLINK_CV_PREFIX_FULL_CUDAGRAPH=${PREFIX_FULL_CUDAGRAPH}" \
  --env "SPECLINK_CV_DENSE_REALIGN_STEPS=${DENSE_REALIGN_STEPS}" \
  --env "SPECLINK_CV_FORCE_PREFIX_LEN=${FORCE_PREFIX_LEN}" \
  --profile-max-events "${PROFILE_MAX_EVENTS}" \
  --log-max-events "${LOG_MAX_EVENTS}" \
  --analysis-profile-max-rows "${ANALYSIS_PROFILE_MAX_ROWS}" \
  --gpu-util-sampling-ms "${GPU_UTIL_SAMPLING_MS}" \
  --skip-unit-tests \
  --resume \
  --output-root "${OUTPUT_ROOT}" \
  "${extra_args[@]}"

if [[ "${ANALYZE_PERF_GAP}" == "1" ]]; then
  "${PYTHON_BIN}" -u tools/speclink_cv/analyze_performance_gap.py \
    "${OUTPUT_ROOT}" \
    --output-dir "${OUTPUT_ROOT}/09_reports"
fi

if [[ "${ANALYZE_STEADY_STATE}" == "1" && "${BENCHMARK_MODE}" != "steady_state" ]]; then
  "${PYTHON_BIN}" -u tools/speclink_cv/analyze_steady_state_throughput.py \
    "${OUTPUT_ROOT}" \
    --output-dir "${OUTPUT_ROOT}/09_reports" \
    --running-fraction "${STEADY_STATE_RUNNING_FRACTION}"
fi

if [[ "${EXTRACT_OUTPUTS}" == "1" && "${BENCHMARK_MODE}" != "steady_state" ]]; then
  IFS=',' read -r -a models <<< "${MODEL_LIST}"
  IFS=',' read -r -a datasets <<< "${DATASET_LIST}"
  IFS=',' read -r -a batches <<< "${BATCH_SIZE_LIST}"
  IFS=',' read -r -a ks <<< "${K_LIST}"
  IFS=',' read -r -a methods <<< "${METHOD_LIST}"
  case_args=()
  if [[ "${FORCE_PREFIX_LEN}" == "0" ]]; then
    h_label="half"
  else
    h_label="h${FORCE_PREFIX_LEN}"
  fi
  for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
      if [[ "${dataset}" != "math" ]]; then
        continue
      fi
      for k in "${ks[@]}"; do
        for bs in "${batches[@]}"; do
          eagle3_dir="${OUTPUT_ROOT}/runs/${model}_${dataset}_k${k}_bs${bs}_eagle3_oneshot"
          for method in "${methods[@]}"; do
            if [[ "${method}" == cv_* ]]; then
              cv_dir="${OUTPUT_ROOT}/runs/${model}_${dataset}_k${k}_bs${bs}_${method}"
              if [[ -f "${eagle3_dir}/guidellm_results.json" && -f "${cv_dir}/guidellm_results.json" ]]; then
                case_args+=(--case "${model}_${dataset}_k${k}_${h_label}_bs${bs}_${method}::${eagle3_dir}::${cv_dir}")
              fi
            fi
          done
        done
      done
    done
  done
  if [[ "${#case_args[@]}" -gt 0 ]]; then
    "${PYTHON_BIN}" examples/evaluate/eval-guidellm/scripts/extract_math_outputs_by_qid.py \
      --mode cv-wrong \
      "${case_args[@]}" \
      --out-md "${OUTPUT_ROOT}/09_reports/math_cv_wrong_outputs.md" \
      --out-json "${OUTPUT_ROOT}/09_reports/math_cv_wrong_outputs.json"
    "${PYTHON_BIN}" examples/evaluate/eval-guidellm/scripts/extract_math_outputs_by_qid.py \
      --mode cv-drop \
      "${case_args[@]}" \
      --out-md "${OUTPUT_ROOT}/09_reports/math_cv_drop_outputs.md" \
      --out-json "${OUTPUT_ROOT}/09_reports/math_cv_drop_outputs.json"
  fi
fi

echo "[INFO] SpecLink-CV math quality output root: ${OUTPUT_ROOT}"
