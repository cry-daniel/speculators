#!/usr/bin/env bash
# Run threshold-tradeoff study from speculator trace data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
JITTER_SCRIPT="${EVAL_ROOT}/run_acceptance_jitter.sh"
THRESHOLD_SCRIPT="${SCRIPT_DIR}/threshold_tradeoff.py"

CONDA_ENV="${CONDA_ENV:-spec}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-${EVAL_ROOT}/results}"
TEMP_DIR="${TEMP_DIR:-${EVAL_ROOT}/temp}"

TRACE_ROOT="${TRACE_ROOT:-${TEMP_DIR}/threshold_tradeoff_trace_${TIMESTAMP}}"
JITTER_OUTPUT_ROOT="${JITTER_OUTPUT_ROOT:-${TEMP_DIR}/threshold_jitter_intermediate_${TIMESTAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULTS_DIR}/threshold_tradeoff_${TIMESTAMP}}"

CASES="${CASES:-qwen3_8b:peagle,qwen3_8b:eagle3,llama3_1_8b:eagle3}"
WORKLOADS="${WORKLOADS:-math,mtbench}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-8,12,16}"
MATH_PROMPTS="${MATH_PROMPTS:-80}"
MTBENCH_PROMPTS="${MTBENCH_PROMPTS:-80}"
THRESHOLDS="${THRESHOLDS:-0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60}"
MODELS_FILTER="${MODELS_FILTER:-}"
METHODS_FILTER="${METHODS_FILTER:-}"
K_FILTER="${K_FILTER:-}"

DRY_RUN=0
ANALYZE_ONLY=""

to_space_list() {
    printf '%s\n' "${1//,/ }"
}

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run speculative-decoding threshold tradeoff analysis from trace data.

Default experiments are math+mtbench and cases (qwen3 peagle/eagle3, llama3.1 eagle3)
for K={8,12,16}.

Options:
  --trace-root DIR             Existing trace root to analyze (skip generation)
  --output-root DIR            Final threshold output root
  --workloads CSV              Trace workloads, comma-separated (default: ${WORKLOADS})
  --cases CSV                  Cases, comma-separated model:method pairs (run + trace generation)
  --num-spec-tokens CSV        K values for run generation (default: ${NUM_SPEC_TOKENS})
  --thresholds CSV             Threshold grid (default: ${THRESHOLDS})
  --models-filter CSV          Filter model labels for analysis only
  --methods-filter CSV         Filter methods for analysis only
  --k-filter CSV               Filter K values for analysis only
  --analyze-only DIR           Analyze existing trace root only
  --dry-run                    Print generation commands without running
  -h, --help                  Show this message

Environment:
  CONDA_ENV=${CONDA_ENV}
  TRACE_ROOT=${TRACE_ROOT}
  OUTPUT_ROOT=${OUTPUT_ROOT}
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --trace-root)
            ANALYZE_ONLY="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --workloads)
            WORKLOADS="$2"
            shift 2
            ;;
        --cases)
            CASES="$2"
            shift 2
            ;;
        --num-spec-tokens)
            NUM_SPEC_TOKENS="$2"
            shift 2
            ;;
        --thresholds)
            THRESHOLDS="$2"
            shift 2
            ;;
        --models-filter)
            MODELS_FILTER="$2"
            shift 2
            ;;
        --methods-filter)
            METHODS_FILTER="$2"
            shift 2
            ;;
        --k-filter)
            K_FILTER="$2"
            shift 2
            ;;
        --analyze-only)
            ANALYZE_ONLY="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
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

check_python_env() {
    local exe
    exe="$(python -c 'import sys; print(sys.executable)')"
    if [[ "${exe}" != *"/envs/${CONDA_ENV}/"* ]]; then
        cat >&2 << EOF
[ERROR] This script must run inside conda env '${CONDA_ENV}'.
[ERROR] Current python: ${exe}
[ERROR] Use: conda run -n ${CONDA_ENV} bash ${SCRIPT_DIR}/run_threshold_tradeoff.sh ...
EOF
        exit 1
    fi
}

run_tradeoff_analysis() {
    local workload_csv="$1"
    local models="$2"
    local methods="$3"
    local ks="$4"
    python "${THRESHOLD_SCRIPT}" \
        "${TRACE_ROOT}" \
        --output-root "${OUTPUT_ROOT}" \
        --workloads "${workload_csv}" \
        --thresholds "${THRESHOLDS}" \
        --models "${models}" \
        --methods "${methods}" \
        --num-spec-tokens "${ks}"
}

main() {
    cd "${EVAL_ROOT}"
    check_python_env

    if [[ -n "${ANALYZE_ONLY}" ]]; then
        TRACE_ROOT="${ANALYZE_ONLY}"
        if [[ ! -d "${TRACE_ROOT}" ]]; then
            echo "[ERROR] trace root not found: ${TRACE_ROOT}" >&2
            exit 1
        fi
        run_tradeoff_analysis "${WORKLOADS}" "${MODELS_FILTER}" "${METHODS_FILTER}" "${K_FILTER}"
        echo "[INFO] Analysis complete: ${OUTPUT_ROOT}"
        exit 0
    fi

    local workloads_space num_spec_tokens_space cases_space
    workloads_space="$(to_space_list "${WORKLOADS}")"
    num_spec_tokens_space="$(to_space_list "${NUM_SPEC_TOKENS}")"
    cases_space="$(to_space_list "${CASES}")"

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[INFO] Threshold root: ${OUTPUT_ROOT}"
        echo "[INFO] Trace root: ${TRACE_ROOT}"
        echo "[INFO] Will run:"
        echo "  CASES=${CASES}"
        echo "  WORKLOADS=${WORKLOADS}"
        echo "  K values=${NUM_SPEC_TOKENS}"
        echo "  conda run -n ${CONDA_ENV} bash ${JITTER_SCRIPT} --output-root ${JITTER_OUTPUT_ROOT} --work-root ${TRACE_ROOT}"
        echo "  python ${THRESHOLD_SCRIPT} ${TRACE_ROOT} --output-root ${OUTPUT_ROOT} --workloads ${WORKLOADS} --thresholds ${THRESHOLDS}"
        exit 0
    fi

    CONDA_ENV="${CONDA_ENV}" \
    CASES="${cases_space}" \
    WORKLOADS="${workloads_space}" \
    NUM_SPEC_TOKENS_LIST="${num_spec_tokens_space}" \
    MATH_PROMPTS="${MATH_PROMPTS}" \
    MTBENCH_PROMPTS="${MTBENCH_PROMPTS}" \
    conda run -n "${CONDA_ENV}" bash "${JITTER_SCRIPT}" \
        --output-root "${JITTER_OUTPUT_ROOT}" \
        --work-root "${TRACE_ROOT}"

    run_tradeoff_analysis "${WORKLOADS}" "${MODELS_FILTER}" "${METHODS_FILTER}" "${K_FILTER}"
    echo "[INFO] Threshold output root: ${OUTPUT_ROOT}"
    echo "[INFO] Trace root: ${TRACE_ROOT}"
}

main "$@"
