#!/usr/bin/env bash
# Run the accepted-draft-count jitter experiment from vLLM speculative traces.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
SPECULATORS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SPECLINK_ROOT="$(cd "${SPECULATORS_ROOT}/.." && pwd)"
DEFAULT_MODELS_DIR="${SPECLINK_ROOT}/models"

CONDA_ENV="${CONDA_ENV:-spec}"
REPO_VLLM_DIR="${SPECULATORS_ROOT}/vllm"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
TEMP_DIR="${TEMP_DIR:-${SCRIPT_DIR}/temp}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULTS_DIR}/accepted_count_jitter_${TIMESTAMP}}"
WORK_ROOT="${WORK_ROOT:-${TEMP_DIR}/accepted_count_jitter_work_${TIMESTAMP}}"

CASES="${CASES:-qwen3_8b:peagle qwen3_8b:eagle3 llama3_1_8b:eagle3}"
WORKLOADS="${WORKLOADS:-math mtbench synthetic_1000x1000}"
NUM_SPEC_TOKENS_LIST="${NUM_SPEC_TOKENS_LIST:-8 12 16}"

QWEN3_8B_BASE_MODEL="${QWEN3_8B_BASE_MODEL:-${QWEN3_8B_MODEL:-Qwen/Qwen3-8B}}"
LLAMA3_1_8B_BASE_MODEL="${LLAMA3_1_8B_BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
QWEN3_8B_EAGLE3_SPECULATOR_MODEL="${QWEN3_8B_EAGLE3_SPECULATOR_MODEL:-${DEFAULT_MODELS_DIR}/qwen3-8b-eagle3-speculator}"
QWEN3_8B_PEAGLE_SPECULATOR_MODEL="${QWEN3_8B_PEAGLE_SPECULATOR_MODEL:-${DEFAULT_MODELS_DIR}/qwen3-8b-peagle-speculator}"
LLAMA3_1_8B_EAGLE3_SPECULATOR_MODEL="${LLAMA3_1_8B_EAGLE3_SPECULATOR_MODEL:-${DEFAULT_MODELS_DIR}/llama-3.1-8b-eagle3-speculator}"

MATH_DATASET="${MATH_DATASET:-${SCRIPT_DIR}/data/math_reasoning.jsonl}"
MTBENCH_DATASET="${MTBENCH_DATASET:-${SCRIPT_DIR}/data/mt_bench.jsonl}"
MATH_PROMPTS="${MATH_PROMPTS:-80}"
MTBENCH_PROMPTS="${MTBENCH_PROMPTS:-80}"
SYNTHETIC_PROMPTS="${SYNTHETIC_PROMPTS:-8}"
REAL_MAX_TOKENS="${REAL_MAX_TOKENS:-128}"
SYNTHETIC_PROMPT_TOKENS="${SYNTHETIC_PROMPT_TOKENS:-1000}"
SYNTHETIC_MAX_TOKENS="${SYNTHETIC_MAX_TOKENS:-1000}"

PORT="${PORT:-8040}"
REQUEST_CONCURRENCY="${REQUEST_CONCURRENCY:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-1800}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-0}"
SEED="${SEED:-42}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

DRY_RUN=0
SMOKE_ONLY=0
ANALYZE_ONLY=""

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Runs accepted-count jitter cases:
  cases: ${CASES}
  workloads: ${WORKLOADS}
  NUM_SPEC_TOKENS: ${NUM_SPEC_TOKENS_LIST}

Options:
  --output-root DIR       Final output root (default: ${OUTPUT_ROOT})
  --work-root DIR         Intermediate trace/log root (default: ${WORK_ROOT})
  --port PORT             vLLM server port (default: ${PORT})
  --dry-run               Print cases without launching vLLM
  --smoke-only            Run one small Qwen3 EAGLE3/math/K=8 smoke into temp/
  --analyze-only DIR      Analyze an existing work/output root
  -h, --help              Show this message

Environment overrides:
  CASES="${CASES}"
  WORKLOADS="${WORKLOADS}"
  NUM_SPEC_TOKENS_LIST="${NUM_SPEC_TOKENS_LIST}"
  MATH_PROMPTS=${MATH_PROMPTS}
  MTBENCH_PROMPTS=${MTBENCH_PROMPTS}
  SYNTHETIC_PROMPTS=${SYNTHETIC_PROMPTS}
  SYNTHETIC_PROMPT_TOKENS=${SYNTHETIC_PROMPT_TOKENS}
  SYNTHETIC_MAX_TOKENS=${SYNTHETIC_MAX_TOKENS}
  REQUEST_CONCURRENCY=${REQUEST_CONCURRENCY}
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --work-root)
            WORK_ROOT="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --smoke-only)
            SMOKE_ONLY=1
            shift
            ;;
        --analyze-only)
            ANALYZE_ONLY="$2"
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

LOCAL_NO_PROXY="localhost,127.0.0.1"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${LOCAL_NO_PROXY}"
export no_proxy="${no_proxy:+${no_proxy},}${LOCAL_NO_PROXY}"

check_python_env() {
    local exe
    exe="$(python -c 'import sys; print(sys.executable)')"
    if [[ "${exe}" != *"/envs/${CONDA_ENV}/"* ]]; then
        cat >&2 << EOF
[ERROR] This script must run inside conda env '${CONDA_ENV}'.
[ERROR] Current python: ${exe}
[ERROR] Use: conda run -n ${CONDA_ENV} bash ./run_acceptance_jitter.sh
EOF
        exit 1
    fi
}

verify_repo_vllm() {
    python - "${REPO_VLLM_DIR}" <<'PY'
import pathlib
import sys
import vllm

expected = pathlib.Path(sys.argv[1]).resolve()
origin_value = getattr(vllm, "__file__", None)
if origin_value is None:
    raise SystemExit("vllm resolved as a namespace package")
origin = pathlib.Path(origin_value).resolve()
print(f"[INFO] vLLM import path: {origin}")
if expected not in origin.parents:
    raise SystemExit(f"vLLM is not imported from {expected}")
PY
}

base_model_for() {
    case "$1" in
        qwen3_8b) printf '%s\n' "${QWEN3_8B_BASE_MODEL}" ;;
        llama3_1_8b) printf '%s\n' "${LLAMA3_1_8B_BASE_MODEL}" ;;
        *) echo "[ERROR] Unknown model label: $1" >&2; exit 1 ;;
    esac
}

speculator_for() {
    local model_label="$1"
    local method="$2"
    case "${model_label}:${method}" in
        qwen3_8b:eagle3) printf '%s\n' "${QWEN3_8B_EAGLE3_SPECULATOR_MODEL}" ;;
        qwen3_8b:peagle) printf '%s\n' "${QWEN3_8B_PEAGLE_SPECULATOR_MODEL}" ;;
        llama3_1_8b:eagle3) printf '%s\n' "${LLAMA3_1_8B_EAGLE3_SPECULATOR_MODEL}" ;;
        *) echo "[ERROR] Unsupported case: ${model_label}:${method}" >&2; exit 1 ;;
    esac
}

dataset_for_workload() {
    case "$1" in
        math) printf '%s\n' "${MATH_DATASET}" ;;
        mtbench) printf '%s\n' "${MTBENCH_DATASET}" ;;
        synthetic_1000x1000) printf '\n' ;;
        *) echo "[ERROR] Unknown workload: $1" >&2; exit 1 ;;
    esac
}

prompts_for_workload() {
    case "$1" in
        math) printf '%s\n' "${MATH_PROMPTS}" ;;
        mtbench) printf '%s\n' "${MTBENCH_PROMPTS}" ;;
        synthetic_1000x1000) printf '%s\n' "${SYNTHETIC_PROMPTS}" ;;
        *) echo "[ERROR] Unknown workload: $1" >&2; exit 1 ;;
    esac
}

max_tokens_for_workload() {
    case "$1" in
        math|mtbench) printf '%s\n' "${REAL_MAX_TOKENS}" ;;
        synthetic_1000x1000) printf '%s\n' "${SYNTHETIC_MAX_TOKENS}" ;;
        *) echo "[ERROR] Unknown workload: $1" >&2; exit 1 ;;
    esac
}

append_command() {
    local root="$1"
    shift
    printf '%q ' "$@" >> "${root}/commands.sh"
    printf '\n' >> "${root}/commands.sh"
}

write_repro_command() {
    local root="$1"
    mkdir -p "${root}"
    {
        echo "#!/usr/bin/env bash"
        echo "set -euo pipefail"
        printf 'cd %q\n' "${SCRIPT_DIR}"
        printf 'conda run -n %q bash %q --output-root %q --work-root %q\n' \
            "${CONDA_ENV}" "${SCRIPT_PATH}" "${OUTPUT_ROOT}" "${WORK_ROOT}"
    } > "${root}/commands.sh"
    chmod +x "${root}/commands.sh"
}

write_manifest() {
    local root="$1"
    mkdir -p "${root}"
    {
        echo "# Acceptance Jitter Run"
        echo
        echo "- date: $(date -Iseconds)"
        echo "- output root: ${OUTPUT_ROOT}"
        echo "- work root: ${WORK_ROOT}"
        echo "- cases: ${CASES}"
        echo "- workloads: ${WORKLOADS}"
        echo "- num_spec_tokens: ${NUM_SPEC_TOKENS_LIST}"
        echo "- math prompts: ${MATH_PROMPTS}"
        echo "- MTBench prompts: ${MTBENCH_PROMPTS}"
        echo "- synthetic prompts: ${SYNTHETIC_PROMPTS}"
        echo "- synthetic prompt/output tokens: ${SYNTHETIC_PROMPT_TOKENS}/${SYNTHETIC_MAX_TOKENS}"
        echo "- request concurrency: ${REQUEST_CONCURRENCY}"
        echo "- max num batched tokens: ${MAX_NUM_BATCHED_TOKENS}"
        echo "- temperature/top_p/top_k: ${TEMPERATURE}/${TOP_P}/${TOP_K}"
        echo "- qwen base: ${QWEN3_8B_BASE_MODEL}"
        echo "- llama base: ${LLAMA3_1_8B_BASE_MODEL}"
        echo "- qwen eagle3 speculator: ${QWEN3_8B_EAGLE3_SPECULATOR_MODEL}"
        echo "- qwen peagle speculator: ${QWEN3_8B_PEAGLE_SPECULATOR_MODEL}"
        echo "- llama eagle3 speculator: ${LLAMA3_1_8B_EAGLE3_SPECULATOR_MODEL}"
        echo "- git commit: $(git -C "${SPECULATORS_ROOT}" rev-parse HEAD 2>/dev/null || true)"
        echo "- git diff summary:"
        git -C "${SPECULATORS_ROOT}" diff --stat 2>/dev/null | sed 's/^/  /' || true
    } > "${root}/run_manifest.md"
}

prepare_inputs() {
    if [[ ! -f "${MATH_DATASET}" ]]; then
        echo "[ERROR] Math dataset not found: ${MATH_DATASET}" >&2
        exit 1
    fi
    if [[ ! -f "${MTBENCH_DATASET}" ]]; then
        echo "[INFO] MTBench dataset not found; preparing ${MTBENCH_DATASET}"
        python "${SCRIPT_DIR}/prepare_mt_bench_dataset.py" \
            --output "${MTBENCH_DATASET}" \
            --raw-out "${SCRIPT_DIR}/data/mt_bench_raw.jsonl"
    fi
}

run_requests() {
    local run_dir="$1"
    local workload="$2"
    local model_label="$3"
    local base_model="$4"
    local method="$5"
    local k="$6"
    local prompts="$7"
    local max_tokens="$8"
    local dataset_path="$9"
    local responses="${run_dir}/responses/responses.jsonl"
    local -a cmd=(
        python "${SCRIPT_DIR}/scripts/send_speclink_confidence_requests.py"
        --target "http://localhost:${PORT}"
        --model "${base_model}"
        --model-label "${model_label}"
        --dataset-label "${workload}"
        --method "${method}"
        --num-spec-tokens "${k}"
        --max-prompts "${prompts}"
        --max-tokens "${max_tokens}"
        --temperature "${TEMPERATURE}"
        --top-p "${TOP_P}"
        --top-k "${TOP_K}"
        --seed "${SEED}"
        --concurrency "${REQUEST_CONCURRENCY}"
        --timeout "${REQUEST_TIMEOUT}"
        --output-jsonl "${responses}"
    )
    if [[ "${workload}" == "synthetic_1000x1000" ]]; then
        cmd+=(--synthetic-prompt-tokens "${SYNTHETIC_PROMPT_TOKENS}")
    else
        cmd+=(--dataset "${dataset_path}")
    fi
    append_command "${WORK_ROOT}" "${cmd[@]}"
    "${cmd[@]}" | tee "${run_dir}/client.log"
}

run_case() {
    local workload="$1"
    local model_label="$2"
    local method="$3"
    local k="$4"
    local base_model
    local speculator_model
    local dataset_path
    local prompts
    local max_tokens
    local case_name
    local run_dir
    local trace_file
    local responses_file
    local server_log
    local server_pid
    local -a serve_cmd
    local rc=0

    base_model="$(base_model_for "${model_label}")"
    speculator_model="$(speculator_for "${model_label}" "${method}")"
    dataset_path="$(dataset_for_workload "${workload}")"
    prompts="$(prompts_for_workload "${workload}")"
    max_tokens="$(max_tokens_for_workload "${workload}")"
    case_name="${workload}_${model_label}_${method}_k${k}"
    run_dir="${WORK_ROOT}/runs/${case_name}"
    trace_file="${run_dir}/trace/trace.jsonl"
    responses_file="${run_dir}/responses/responses.jsonl"
    server_log="${run_dir}/vllm_server.log"
    server_pid="${run_dir}/vllm_server.pid"

    mkdir -p "${run_dir}/trace" "${run_dir}/responses"
    if [[ "${SKIP_EXISTING}" == "1" && -s "${trace_file}" && -s "${responses_file}" ]]; then
        echo "[INFO] Skipping existing ${case_name}"
        return 0
    fi
    if [[ ! -e "${speculator_model}" ]]; then
        echo "[ERROR] Speculator model not found: ${speculator_model}" >&2
        exit 1
    fi

    : > "${trace_file}"
    serve_cmd=(
        "${SCRIPT_DIR}/scripts/vllm_serve.sh"
        -b "${base_model}"
        -s "${speculator_model}"
        --num-spec-tokens "${k}"
        --method eagle3
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
        --max-model-len "${MAX_MODEL_LEN}"
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
        --max-num-seqs "${REQUEST_CONCURRENCY}"
        --port "${PORT}"
        --health-check-timeout "${HEALTH_CHECK_TIMEOUT}"
        --log-file "${server_log}"
        --pid-file "${server_pid}"
        --enforce-eager
    )
    if [[ "${method}" == "peagle" ]]; then
        serve_cmd+=(--parallel-drafting)
    fi

    append_command "${WORK_ROOT}" env SPECLINK_TRACE_CONFIDENCE=1 \
        SPECLINK_TRACE_OUTPUT="${trace_file}" \
        SPECLINK_TRACE_RUN_ID="${case_name}" \
        SPECLINK_TRACE_DATASET_LABEL="${workload}" \
        SPECLINK_TRACE_MODEL_LABEL="${model_label}" \
        SPECLINK_TRACE_METHOD="${method}" \
        SPECLINK_TRACE_NUM_SPEC_TOKENS="${k}" \
        "${serve_cmd[@]}"

    echo "[INFO] Running ${workload}/${model_label}/${method}/K=${k} prompts=${prompts} max_tokens=${max_tokens}"
    set +e
    (
        set -euo pipefail
        SPECLINK_TRACE_CONFIDENCE=1 \
        SPECLINK_TRACE_OUTPUT="${trace_file}" \
        SPECLINK_TRACE_RUN_ID="${case_name}" \
        SPECLINK_TRACE_DATASET_LABEL="${workload}" \
        SPECLINK_TRACE_MODEL_LABEL="${model_label}" \
        SPECLINK_TRACE_METHOD="${method}" \
        SPECLINK_TRACE_NUM_SPEC_TOKENS="${k}" \
            "${serve_cmd[@]}"
        run_requests "${run_dir}" "${workload}" "${model_label}" "${base_model}" \
            "${method}" "${k}" "${prompts}" "${max_tokens}" "${dataset_path}"
        if ! python "${SCRIPT_DIR}/scripts/parse_logs.py" "${server_log}" \
            -o "${run_dir}/acceptance_analysis.txt" \
            > "${run_dir}/parse_logs_stdout.log" \
            2> "${run_dir}/parse_logs_stderr.log"; then
            echo "[WARN] No aggregate SpecDecoding metrics parsed for ${case_name}; step trace is still analyzed."
        fi
    )
    rc=$?
    set -e
    "${SCRIPT_DIR}/scripts/vllm_stop.sh" --pid-file "${server_pid}" || true
    if [[ "${rc}" -ne 0 ]]; then
        echo "[ERROR] Failed ${case_name}; see ${run_dir}" >&2
        return "${rc}"
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${workload}" "${model_label}" "${method}" "${k}" "ok" "${run_dir}" \
        >> "${WORK_ROOT}/status.tsv"
}

print_cases() {
    for workload in ${WORKLOADS}; do
        for case_spec in ${CASES}; do
            IFS=: read -r model_label method <<< "${case_spec}"
            for k in ${NUM_SPEC_TOKENS_LIST}; do
                printf 'case: %s/%s/%s/K=%s prompts=%s max_tokens=%s\n' \
                    "${workload}" "${model_label}" "${method}" "${k}" \
                    "$(prompts_for_workload "${workload}")" \
                    "$(max_tokens_for_workload "${workload}")"
            done
        done
    done
}

analyze_results() {
    mkdir -p "${OUTPUT_ROOT}"
    append_command "${WORK_ROOT}" python \
        "${SCRIPT_DIR}/scripts/analyze_acceptance_jitter.py" \
        "${WORK_ROOT}" --output-root "${OUTPUT_ROOT}"
    python "${SCRIPT_DIR}/scripts/analyze_acceptance_jitter.py" \
        "${WORK_ROOT}" --output-root "${OUTPUT_ROOT}"
    write_repro_command "${OUTPUT_ROOT}"
    write_manifest "${OUTPUT_ROOT}"
}

main() {
    cd "${SCRIPT_DIR}"
    check_python_env

    if [[ "${SMOKE_ONLY}" == "1" ]]; then
        CASES="qwen3_8b:eagle3"
        WORKLOADS="math"
        NUM_SPEC_TOKENS_LIST="8"
        MATH_PROMPTS=2
        REAL_MAX_TOKENS=16
        OUTPUT_ROOT="${TEMP_DIR}/accepted_count_jitter_smoke_${TIMESTAMP}"
        WORK_ROOT="${TEMP_DIR}/accepted_count_jitter_smoke_work_${TIMESTAMP}"
    fi

    if [[ -n "${ANALYZE_ONLY}" ]]; then
        mkdir -p "${WORK_ROOT}"
        : > "${WORK_ROOT}/commands.sh"
        python "${SCRIPT_DIR}/scripts/analyze_acceptance_jitter.py" \
            "${ANALYZE_ONLY}" --output-root "${OUTPUT_ROOT}"
        write_repro_command "${OUTPUT_ROOT}"
        write_manifest "${OUTPUT_ROOT}"
        exit 0
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[INFO] Output root: ${OUTPUT_ROOT}"
        echo "[INFO] Work root: ${WORK_ROOT}"
        print_cases
        exit 0
    fi

    verify_repo_vllm
    prepare_inputs
    mkdir -p "${WORK_ROOT}"
    : > "${WORK_ROOT}/commands.sh"
    chmod +x "${WORK_ROOT}/commands.sh"
    printf "workload\tmodel_label\tmethod\tnum_spec_tokens\tstatus\trun_dir\n" \
        > "${WORK_ROOT}/status.tsv"

    for workload in ${WORKLOADS}; do
        for case_spec in ${CASES}; do
            IFS=: read -r model_label method <<< "${case_spec}"
            for k in ${NUM_SPEC_TOKENS_LIST}; do
                run_case "${workload}" "${model_label}" "${method}" "${k}"
            done
        done
    done

    analyze_results
    echo "[INFO] Final output root: ${OUTPUT_ROOT}"
    echo "[INFO] Intermediate work root: ${WORK_ROOT}"
}

main "$@"
