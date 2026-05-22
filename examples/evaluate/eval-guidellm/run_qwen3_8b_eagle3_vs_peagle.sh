#!/usr/bin/env bash
# Compare Qwen3-8B EAGLE3 and P-EAGLE throughput with GuideLLM.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs"
RUN_EVALUATION="${SCRIPT_DIR}/run_evaluation.sh"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"

DEFAULT_MODELS_DIR="/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models"
EAGLE3_MODEL="${EAGLE3_SPECULATOR_MODEL:-${DEFAULT_MODELS_DIR}/qwen3-8b-eagle3-speculator}"
PEAGLE_MODEL="${PEAGLE_SPECULATOR_MODEL:-${DEFAULT_MODELS_DIR}/qwen3-8b-peagle-speculator}"
OUTPUT_ROOT="${RESULTS_DIR}/qwen3_8b_eagle3_vs_peagle_$(date +%Y%m%d_%H%M%S)"
NUM_SPEC_TOKENS=""
PORT=""

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Optional:
  --eagle3-model MODEL       EAGLE3 speculator checkpoint path or HuggingFace ID
                             (default: ${DEFAULT_MODELS_DIR}/qwen3-8b-eagle3-speculator)
  --peagle-model MODEL       P-EAGLE speculator checkpoint path or HuggingFace ID
                             (default: ${DEFAULT_MODELS_DIR}/qwen3-8b-peagle-speculator)
  --output-root DIR          Output root directory
                             (default: ${RESULTS_DIR}/qwen3_8b_eagle3_vs_peagle_TIMESTAMP)
  --num-spec-tokens N        Override NUM_SPEC_TOKENS for both runs
  --port PORT                Override vLLM server port for both runs
  -h, --help                 Show this help message

Example:
  $0 --num-spec-tokens 3
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --eagle3-model)
            EAGLE3_MODEL="$2"
            shift 2
            ;;
        --peagle-model)
            PEAGLE_MODEL="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --num-spec-tokens)
            NUM_SPEC_TOKENS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            show_usage
            exit 1
            ;;
    esac
done

COMMON_ARGS=()
[[ -n "${NUM_SPEC_TOKENS}" ]] && COMMON_ARGS+=(--num-spec-tokens "${NUM_SPEC_TOKENS}")
[[ -n "${PORT}" ]] && COMMON_ARGS+=(--port "${PORT}")

mkdir -p "${OUTPUT_ROOT}"

echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Running Qwen3-8B EAGLE3 baseline..."
"${RUN_EVALUATION}" \
    -c "${CONFIG_DIR}/qwen3-8b-eagle3.env" \
    -s "${EAGLE3_MODEL}" \
    -o "${OUTPUT_ROOT}/eagle3" \
    --results-dir "${RESULTS_DIR}" \
    "${COMMON_ARGS[@]}"

echo "[INFO] Running Qwen3-8B P-EAGLE with parallel drafting..."
"${RUN_EVALUATION}" \
    -c "${CONFIG_DIR}/qwen3-8b-peagle.env" \
    -s "${PEAGLE_MODEL}" \
    -o "${OUTPUT_ROOT}/peagle" \
    --results-dir "${RESULTS_DIR}" \
    "${COMMON_ARGS[@]}"

echo "[INFO] Comparison complete."
echo "[INFO]   EAGLE3 results: ${OUTPUT_ROOT}/eagle3"
echo "[INFO]   P-EAGLE results: ${OUTPUT_ROOT}/peagle"
