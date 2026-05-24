#!/usr/bin/env bash
# Unified SpecLink method runner: vLLM + GuideLLM + accuracy harness.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ORIGINAL_ARGS=("$@")
LOCAL_NO_PROXY="localhost,127.0.0.1"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${LOCAL_NO_PROXY}"
export no_proxy="${no_proxy:+${no_proxy},}${LOCAL_NO_PROXY}"

METHOD=""
NUM_SPEC_TOKENS=""
GUIDELLM_RATE="${GUIDELLM_RATE:-1}"
REQUEST_TYPE="${REQUEST_TYPE:-chat_completions}"
PORT=""
DATASET="data/math_reasoning.jsonl"
OUTPUT_DIR=""
MAX_TOKENS="512"
ACCURACY_LIMIT=""
BENCHMARK_LIMIT=""
REPEAT_ID="0"
ACCURACY_CONCURRENCY="1"
DENSE_REFERENCE_JSONL=""

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
EAGLE3_SPECULATOR_MODEL="${EAGLE3_SPECULATOR_MODEL:-/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/qwen3-8b-eagle3-speculator}"
PEAGLE_SPECULATOR_MODEL="${PEAGLE_SPECULATOR_MODEL:-/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/qwen3-8b-peagle-speculator}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-1800}"

show_usage() {
    cat << EOF
Usage: $0 --method dense|eagle3|peagle|speclink --output-dir DIR [OPTIONS]

Options:
  --num-spec-tokens N       Speculative length K; dense should use 0
  --guidellm-rate N         GuideLLM throughput max concurrency
  --request-type TYPE       GuideLLM request type (default: chat_completions)
  --port PORT               vLLM port
  --dataset PATH            Dataset JSONL
  --max-tokens N            Max generated tokens for GuideLLM and accuracy
  --accuracy-limit N        Accuracy sample limit
  --benchmark-limit N       GuideLLM sample limit via temporary JSONL subset
  --accuracy-concurrency N  Accuracy harness concurrency
  --dense-reference-jsonl P Dense accuracy_outputs.jsonl for equivalence
  --repeat-id N             Repeat id recorded in env.txt
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method) METHOD="$2"; shift 2 ;;
        --num-spec-tokens) NUM_SPEC_TOKENS="$2"; shift 2 ;;
        --guidellm-rate) GUIDELLM_RATE="$2"; shift 2 ;;
        --request-type) REQUEST_TYPE="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --accuracy-limit) ACCURACY_LIMIT="$2"; shift 2 ;;
        --benchmark-limit) BENCHMARK_LIMIT="$2"; shift 2 ;;
        --accuracy-concurrency) ACCURACY_CONCURRENCY="$2"; shift 2 ;;
        --dense-reference-jsonl) DENSE_REFERENCE_JSONL="$2"; shift 2 ;;
        --repeat-id) REPEAT_ID="$2"; shift 2 ;;
        -h|--help) show_usage; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1" >&2; show_usage; exit 1 ;;
    esac
done

if [[ -z "${METHOD}" || -z "${OUTPUT_DIR}" ]]; then
    echo "[ERROR] --method and --output-dir are required" >&2
    show_usage
    exit 1
fi

case "${METHOD}" in
    dense)
        NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-0}"
        PORT="${PORT:-8009}"
        VLLM_METHOD="dense"
        SPECULATOR_MODEL=""
        PARALLEL_DRAFTING="false"
        ;;
    eagle3)
        NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-8}"
        PORT="${PORT:-8010}"
        VLLM_METHOD="eagle3"
        SPECULATOR_MODEL="${EAGLE3_SPECULATOR_MODEL}"
        PARALLEL_DRAFTING="false"
        ;;
    peagle)
        NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-8}"
        PORT="${PORT:-8011}"
        VLLM_METHOD="eagle3"
        SPECULATOR_MODEL="${PEAGLE_SPECULATOR_MODEL}"
        PARALLEL_DRAFTING="true"
        ;;
    speclink)
        NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-8}"
        PORT="${PORT:-8012}"
        VLLM_METHOD="eagle3"
        SPECULATOR_MODEL="${SPECLINK_SPECULATOR_MODEL:-${PEAGLE_SPECULATOR_MODEL}}"
        PARALLEL_DRAFTING="${SPECLINK_PARALLEL_DRAFTING:-true}"
        export SPECLINK_ENABLE="${SPECLINK_ENABLE:-1}"
        export SPECLINK_MODE="${SPECLINK_MODE:-plan_only}"
        export SPECLINK_LAYOUT="${SPECLINK_LAYOUT:-speclink_prob}"
        export SPECLINK_METHOD="${SPECLINK_METHOD:-speclink}"
        export SPECLINK_PROFILE="${SPECLINK_PROFILE:-1}"
        export SPECLINK_TRACE_OUT="${SPECLINK_TRACE_OUT:-${OUTPUT_DIR}/live_sparse_trace.jsonl}"
        export SPECLINK_PROFILE_OUT="${SPECLINK_PROFILE_OUT:-${OUTPUT_DIR}/profile_events.jsonl}"
        ;;
    *)
        echo "[ERROR] unknown method: ${METHOD}" >&2
        exit 1
        ;;
esac

mkdir -p "${OUTPUT_DIR}"
FAILURES="${OUTPUT_DIR}/failures.md"
SERVER_LOG="${OUTPUT_DIR}/vllm_server.log"
SERVER_PID="${OUTPUT_DIR}/vllm_server.pid"
GUIDELLM_LOG="${OUTPUT_DIR}/guidellm_output.log"
GUIDELLM_RESULTS="${OUTPUT_DIR}/guidellm_results.json"
ACCEPTANCE_RESULTS="${OUTPUT_DIR}/acceptance_analysis.txt"
ACCURACY_OUTPUTS="${OUTPUT_DIR}/accuracy_outputs.jsonl"
ACCURACY_SUMMARY="${OUTPUT_DIR}/accuracy_summary.json"

if [[ -f "${FAILURES}" ]]; then
    mv "${FAILURES}" "${OUTPUT_DIR}/failures.previous.$(date +%Y%m%d-%H%M%S).md"
fi

archive_stale_outputs() {
    local stamp
    stamp="$(date +%Y%m%d-%H%M%S)"
    local archive_dir="${OUTPUT_DIR}/previous_${stamp}"
    local moved="false"
    local path
    for path in \
        "${SERVER_LOG}" \
        "${SERVER_PID}" \
        "${GUIDELLM_LOG}" \
        "${GUIDELLM_RESULTS}" \
        "${ACCEPTANCE_RESULTS}" \
        "${ACCURACY_OUTPUTS}" \
        "${ACCURACY_SUMMARY}" \
        "${SPECLINK_TRACE_OUT:-}" \
        "${SPECLINK_PROFILE_OUT:-}"
    do
        [[ -n "${path}" && -f "${path}" ]] || continue
        if [[ "${moved}" == "false" ]]; then
            mkdir -p "${archive_dir}"
            moved="true"
        fi
        mv "${path}" "${archive_dir}/$(basename "${path}")"
    done
}

archive_stale_outputs

record_failure() {
    local exit_code="$1"
    local line_no="$2"
    local command="$3"
    {
        echo "## ${METHOD} failure"
        echo
        echo "- exit_code: ${exit_code}"
        echo "- line: ${line_no}"
        echo "- command: \`${command}\`"
        echo "- output_dir: ${OUTPUT_DIR}"
        echo
        echo "### vllm_server.log tail"
        echo '```text'
        tail -n 80 "${SERVER_LOG}" 2>/dev/null || true
        echo '```'
        echo
        echo "### guidellm_output.log tail"
        echo '```text'
        tail -n 80 "${GUIDELLM_LOG}" 2>/dev/null || true
        echo '```'
        echo
    } >> "${FAILURES}"
}

cleanup() {
    "${SCRIPT_DIR}/vllm_stop.sh" --pid-file "${SERVER_PID}" 2>/dev/null || true
}

on_error() {
    local exit_code="$1"
    local line_no="$2"
    local command="$3"
    record_failure "${exit_code}" "${line_no}" "${command}"
}

trap 'on_error $? $LINENO "$BASH_COMMAND"' ERR
trap cleanup EXIT INT TERM

{
    printf 'cwd=%s\n' "${EVAL_DIR}"
    printf 'method=%s\n' "${METHOD}"
    printf 'num_spec_tokens=%s\n' "${NUM_SPEC_TOKENS}"
    printf 'guidellm_rate=%s\n' "${GUIDELLM_RATE}"
    printf 'request_type=%s\n' "${REQUEST_TYPE}"
    printf 'port=%s\n' "${PORT}"
    printf 'dataset=%s\n' "${DATASET}"
    printf 'max_tokens=%s\n' "${MAX_TOKENS}"
    printf 'accuracy_limit=%s\n' "${ACCURACY_LIMIT:-full}"
    printf 'benchmark_limit=%s\n' "${BENCHMARK_LIMIT:-full}"
    printf 'repeat_id=%s\n' "${REPEAT_ID}"
    if [[ "${METHOD}" == "speclink" ]]; then
        printf 'speclink_mode=%s\n' "${SPECLINK_MODE:-plan_only}"
        printf 'speclink_layout=%s\n' "${SPECLINK_LAYOUT:-speclink_prob}"
        printf 'speclink_method=%s\n' "${SPECLINK_METHOD:-speclink}"
        printf 'speclink_block_size=%s\n' "${SPECLINK_BLOCK_SIZE:-}"
        printf 'speclink_topk_per_token=%s\n' "${SPECLINK_TOPK_PER_TOKEN:-}"
        printf 'speclink_shared_budget=%s\n' "${SPECLINK_SHARED_BUDGET:-}"
        printf 'speclink_private_min=%s\n' "${SPECLINK_PRIVATE_MIN:-}"
        printf 'speclink_private_max=%s\n' "${SPECLINK_PRIVATE_MAX:-}"
        printf 'speclink_alpha=%s\n' "${SPECLINK_ALPHA:-}"
        printf 'speclink_beta=%s\n' "${SPECLINK_BETA:-}"
        printf 'speclink_lambda_risk=%s\n' "${SPECLINK_LAMBDA_RISK:-}"
        printf 'speclink_risk_threshold=%s\n' "${SPECLINK_RISK_THRESHOLD:-}"
        printf 'speclink_parallel_drafting=%s\n' "${SPECLINK_PARALLEL_DRAFTING:-}"
        printf 'speclink_speculator_model=%s\n' "${SPECLINK_SPECULATOR_MODEL:-}"
    fi
} > "${OUTPUT_DIR}/env.txt"
printf '%q ' "$0" "${ORIGINAL_ARGS[@]}" > "${OUTPUT_DIR}/command.txt"
printf '\n' >> "${OUTPUT_DIR}/command.txt"

BENCHMARK_DATASET="${DATASET}"
if [[ -n "${BENCHMARK_LIMIT}" && "${BENCHMARK_LIMIT}" != "0" ]]; then
    BENCHMARK_DATASET="${OUTPUT_DIR}/benchmark_limit_${BENCHMARK_LIMIT}.jsonl"
    python "${SCRIPT_DIR}/speclink_subset_jsonl.py" \
        --input "${DATASET}" \
        --output "${BENCHMARK_DATASET}" \
        --limit "${BENCHMARK_LIMIT}"
fi

SERVE_ARGS=(
    -b "${BASE_MODEL}"
    --num-spec-tokens "${NUM_SPEC_TOKENS}"
    --method "${VLLM_METHOD}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --port "${PORT}"
    --health-check-timeout "${HEALTH_CHECK_TIMEOUT}"
    --log-file "${SERVER_LOG}"
    --pid-file "${SERVER_PID}"
)
[[ -n "${SPECULATOR_MODEL}" ]] && SERVE_ARGS+=(-s "${SPECULATOR_MODEL}")
[[ "${PARALLEL_DRAFTING}" == "true" ]] && SERVE_ARGS+=(--parallel-drafting)

"${SCRIPT_DIR}/vllm_serve.sh" "${SERVE_ARGS[@]}"

"${SCRIPT_DIR}/run_guidellm.sh" \
    -d "${BENCHMARK_DATASET}" \
    --target "http://localhost:${PORT}" \
    --output-file "${GUIDELLM_RESULTS}" \
    --log-file "${GUIDELLM_LOG}" \
    --temperature "${TEMPERATURE:-0.0}" \
    --top-p "${TOP_P:-1.0}" \
    --top-k "${TOP_K:--1}" \
    --max-tokens "${MAX_TOKENS}" \
    --rate "${GUIDELLM_RATE}" \
    --request-type "${REQUEST_TYPE}"

ACCURACY_ARGS=(
    --base-url "http://localhost:${PORT}"
    --model "${BASE_MODEL}"
    --dataset "${DATASET}"
    --temperature 0
    --max-tokens "${MAX_TOKENS}"
    --concurrency "${ACCURACY_CONCURRENCY}"
    --out-jsonl "${ACCURACY_OUTPUTS}"
    --summary-json "${ACCURACY_SUMMARY}"
)
[[ -n "${ACCURACY_LIMIT}" && "${ACCURACY_LIMIT}" != "0" ]] && ACCURACY_ARGS+=(--limit "${ACCURACY_LIMIT}")
[[ -n "${DENSE_REFERENCE_JSONL}" ]] && ACCURACY_ARGS+=(--dense-reference-jsonl "${DENSE_REFERENCE_JSONL}")
python "${SCRIPT_DIR}/speclink_eval_math_accuracy.py" "${ACCURACY_ARGS[@]}"

if [[ "${METHOD}" == "dense" || "${NUM_SPEC_TOKENS}" == "0" ]]; then
    echo "dense run: no speculative acceptance metrics" > "${ACCEPTANCE_RESULTS}"
else
    python "${SCRIPT_DIR}/parse_logs.py" "${SERVER_LOG}" -o "${ACCEPTANCE_RESULTS}"
fi

echo "[INFO] SpecLink method run complete: ${OUTPUT_DIR}"
