#!/usr/bin/env bash
# Run Qwen3-8B EAGLE3/P-EAGLE motivation breakdown experiments.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPECULATORS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SPECLINK_ROOT="$(cd "${SPECULATORS_ROOT}/.." && pwd)"
DEFAULT_MODELS_DIR="${SPECLINK_ROOT}/models"

CONDA_ENV="${CONDA_ENV:-spec}"
REPO_VLLM_DIR="${SPECULATORS_ROOT}/vllm"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULTS_DIR}/motivation_breakdown_$(date +%Y%m%d_%H%M%S)}"

BASE_MODEL="${BASE_MODEL:-${QWEN3_8B_MODEL:-Qwen/Qwen3-8B}}"
EAGLE3_SPECULATOR_MODEL="${EAGLE3_SPECULATOR_MODEL:-${DEFAULT_MODELS_DIR}/qwen3-8b-eagle3-speculator}"
PEAGLE_SPECULATOR_MODEL="${PEAGLE_SPECULATOR_MODEL:-${DEFAULT_MODELS_DIR}/qwen3-8b-peagle-speculator}"

ALGOS="${ALGOS:-eagle3 peagle}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16}"
NUM_SPEC_TOKENS_LIST="${NUM_SPEC_TOKENS_LIST:-8 16 24}"
REQUESTS_PER_RUN="${REQUESTS_PER_RUN:-32}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-4}"
PROMPT_TOKENS="${PROMPT_TOKENS:-1000}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-1000}"

PORT="${PORT:-8020}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-1800}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
SPECLINK_BREAKDOWN_SYNC="${SPECLINK_BREAKDOWN_SYNC:-1}"

DRY_RUN=0
SUMMARIZE_ONLY=""

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Runs the Qwen3-8B motivation breakdown matrix:
  algos={eagle3,peagle}
  batch_size={1,2,4,8,16}
  NUM_SPEC_TOKENS={8,16,24}
  synthetic prompt/output tokens=${PROMPT_TOKENS}/${OUTPUT_TOKENS}

Options:
  --output-root DIR       Output root (default: ${OUTPUT_ROOT})
  --port PORT             vLLM server port (default: ${PORT})
  --dry-run               Print the matrix without starting vLLM
  --summarize-only DIR    Regenerate CSV/XLSX/SVG for an existing output root
  -h, --help              Show this help

Environment overrides:
  ALGOS="${ALGOS}"
  BATCH_SIZES="${BATCH_SIZES}"
  NUM_SPEC_TOKENS_LIST="${NUM_SPEC_TOKENS_LIST}"
  REQUESTS_PER_RUN=${REQUESTS_PER_RUN}
  WARMUP_REQUESTS=${WARMUP_REQUESTS}
  MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}
  BASE_MODEL=${BASE_MODEL}

Examples:
  conda run -n spec bash ./motivation_breakdown.sh
  ALGOS=eagle3 BATCH_SIZES=1 NUM_SPEC_TOKENS_LIST=8 REQUESTS_PER_RUN=2 \\
    conda run -n spec bash ./motivation_breakdown.sh
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-root)
            OUTPUT_ROOT="$2"
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
        --summarize-only)
            SUMMARIZE_ONLY="$2"
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

python_bin() {
    python - "$@" <<'PY'
import sys
print(sys.executable)
PY
}

check_python_env() {
    local exe
    exe="$(python_bin)"
    if [[ "${exe}" != *"/envs/${CONDA_ENV}/"* ]]; then
        cat >&2 << EOF
[ERROR] This script must run inside the '${CONDA_ENV}' conda environment.
[ERROR] Current python: ${exe}
[ERROR] Use: conda run -n ${CONDA_ENV} bash ./motivation_breakdown.sh
EOF
        exit 1
    fi
}

verify_repo_vllm() {
    local imported
    imported="$(
        python - "${REPO_VLLM_DIR}" <<'PY'
import importlib.metadata
import pathlib
import sys
import vllm

vllm_dir = pathlib.Path(sys.argv[1]).resolve()
origin_value = getattr(vllm, "__file__", None)
if origin_value is None:
    raise SystemExit("vllm resolved as a namespace package; run from eval-guidellm or reinstall editable")
origin = pathlib.Path(origin_value).resolve()
print(f"{importlib.metadata.version('vllm')} {origin}")
if vllm_dir not in origin.parents:
    raise SystemExit(2)
PY
    )" || {
        cat >&2 << EOF
[ERROR] vLLM is not imported from the vendored repository checkout.
[ERROR] Expected under: ${REPO_VLLM_DIR}
[ERROR] Install it with: conda run -n ${CONDA_ENV} python -m pip install -e ${REPO_VLLM_DIR} --no-deps --no-build-isolation
EOF
        exit 1
    }
    echo "[INFO] vLLM import path: ${imported}"
}

speculator_for_algo() {
    case "$1" in
        eagle3)
            printf '%s\n' "${EAGLE3_SPECULATOR_MODEL}"
            ;;
        peagle)
            printf '%s\n' "${PEAGLE_SPECULATOR_MODEL}"
            ;;
        *)
            echo "[ERROR] Unknown algo: $1" >&2
            exit 1
            ;;
    esac
}

parallel_flag_for_algo() {
    if [[ "$1" == "peagle" ]]; then
        printf '%s\n' "--parallel-drafting"
    fi
    return 0
}

write_metadata() {
    local path="$1"
    local algo="$2"
    local batch_size="$3"
    local num_spec_tokens="$4"
    local status="$5"
    python - "$path" "$algo" "$batch_size" "$num_spec_tokens" "$status" <<PY
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = {
    "algo": sys.argv[2],
    "batch_size": int(sys.argv[3]),
    "num_spec_tokens": int(sys.argv[4]),
    "status": sys.argv[5],
    "base_model": "${BASE_MODEL}",
    "prompt_tokens": int("${PROMPT_TOKENS}"),
    "output_tokens": int("${OUTPUT_TOKENS}"),
    "requests": int("${REQUESTS_PER_RUN}"),
    "warmup_requests": int("${WARMUP_REQUESTS}"),
    "port": int("${PORT}"),
    "max_num_batched_tokens": int("${MAX_NUM_BATCHED_TOKENS}"),
    "max_num_seqs": int(sys.argv[3]),
    "vllm_source": "${REPO_VLLM_DIR}",
    "vllm_install": "editable",
}
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
}

run_guidellm_case() {
    local target="$1"
    local batch_size="$2"
    local max_requests="$3"
    local output_json="$4"
    local output_log="$5"

    guidellm benchmark run \
        --target "${target}" \
        --request-type text_completions \
        --model "${BASE_MODEL}" \
        --processor "${BASE_MODEL}" \
        --data "prompt_tokens=${PROMPT_TOKENS},prompt_tokens_min=${PROMPT_TOKENS},prompt_tokens_max=${PROMPT_TOKENS},output_tokens=${OUTPUT_TOKENS},output_tokens_min=${OUTPUT_TOKENS},output_tokens_max=${OUTPUT_TOKENS}" \
        --profile concurrent \
        --rate "${batch_size}" \
        --max-requests "${max_requests}" \
        --output-path "${output_json}" \
        --backend-args "{\"extras\": {\"body\": {\"temperature\":${TEMPERATURE}, \"top_p\":${TOP_P}, \"top_k\":${TOP_K}, \"ignore_eos\": true}}}" \
        | tee "${output_log}"
}

run_case() {
    local algo="$1"
    local batch_size="$2"
    local num_spec_tokens="$3"
    local run_dir="${OUTPUT_ROOT}/runs/${algo}_bs${batch_size}_k${num_spec_tokens}"
    local server_log="${run_dir}/vllm_server.log"
    local server_pid="${run_dir}/vllm_server.pid"
    local events_file="${run_dir}/breakdown_events.jsonl"
    local guidellm_json="${run_dir}/guidellm_results.json"
    local guidellm_log="${run_dir}/guidellm_output.log"
    local warmup_json="${run_dir}/warmup_guidellm_results.json"
    local warmup_log="${run_dir}/warmup_guidellm_output.log"
    local acceptance_file="${run_dir}/acceptance_analysis.txt"
    local speculator_model
    local extra_parallel_flag
    local rc=0

    mkdir -p "${run_dir}"
    write_metadata "${run_dir}/metadata.json" "${algo}" "${batch_size}" \
        "${num_spec_tokens}" "running"
    : > "${events_file}"

    speculator_model="$(speculator_for_algo "${algo}")"
    extra_parallel_flag="$(parallel_flag_for_algo "${algo}")"

    echo "[INFO] Starting ${algo} bs=${batch_size} K=${num_spec_tokens}"
    set +e
    (
        set -uo pipefail
        SPECLINK_BREAKDOWN=1 \
        SPECLINK_BREAKDOWN_OUT="${events_file}" \
        SPECLINK_BREAKDOWN_ALGO="${algo}" \
        SPECLINK_BREAKDOWN_BATCH_SIZE="${batch_size}" \
        SPECLINK_BREAKDOWN_NUM_SPEC_TOKENS="${num_spec_tokens}" \
        SPECLINK_BREAKDOWN_SYNC="${SPECLINK_BREAKDOWN_SYNC}" \
        "${SCRIPT_DIR}/scripts/vllm_serve.sh" \
            -b "${BASE_MODEL}" \
            -s "${speculator_model}" \
            --num-spec-tokens "${num_spec_tokens}" \
            --method eagle3 \
            --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
            --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
            --max-num-seqs "${batch_size}" \
            --port "${PORT}" \
            --health-check-timeout "${HEALTH_CHECK_TIMEOUT}" \
            --log-file "${server_log}" \
            --pid-file "${server_pid}" \
            ${extra_parallel_flag} || exit $?

        if [[ "${WARMUP_REQUESTS}" -gt 0 ]]; then
            echo "[INFO] Warmup ${algo} bs=${batch_size} K=${num_spec_tokens}"
            run_guidellm_case "http://localhost:${PORT}" "${batch_size}" \
                "${WARMUP_REQUESTS}" "${warmup_json}" "${warmup_log}" || exit $?
            : > "${events_file}"
        fi

        echo "[INFO] Measuring ${algo} bs=${batch_size} K=${num_spec_tokens}"
        run_guidellm_case "http://localhost:${PORT}" "${batch_size}" \
            "${REQUESTS_PER_RUN}" "${guidellm_json}" "${guidellm_log}" || exit $?

        if [[ ! -s "${guidellm_json}" ]]; then
            echo "[ERROR] Missing GuideLLM result JSON: ${guidellm_json}" >&2
            exit 1
        fi
        if [[ ! -s "${events_file}" ]]; then
            echo "[ERROR] Missing breakdown events: ${events_file}" >&2
            exit 1
        fi

        python "${SCRIPT_DIR}/scripts/parse_logs.py" "${server_log}" \
            -o "${acceptance_file}" || true
    )
    rc=$?
    set -e

    "${SCRIPT_DIR}/scripts/vllm_stop.sh" --pid-file "${server_pid}" || true

    if [[ "${rc}" -eq 0 ]]; then
        write_metadata "${run_dir}/metadata.json" "${algo}" "${batch_size}" \
            "${num_spec_tokens}" "ok"
        printf "%s\t%s\t%s\tok\t%s\n" "${algo}" "${batch_size}" \
            "${num_spec_tokens}" "${run_dir}" >> "${OUTPUT_ROOT}/status.tsv"
        echo "[INFO] Completed ${algo} bs=${batch_size} K=${num_spec_tokens}"
    else
        write_metadata "${run_dir}/metadata.json" "${algo}" "${batch_size}" \
            "${num_spec_tokens}" "failed"
        printf "%s\t%s\t%s\tfailed\t%s\n" "${algo}" "${batch_size}" \
            "${num_spec_tokens}" "${run_dir}" >> "${OUTPUT_ROOT}/status.tsv"
        echo "[ERROR] Failed ${algo} bs=${batch_size} K=${num_spec_tokens}; see ${run_dir}" >&2
    fi
    return "${rc}"
}

summarize() {
    local root="$1"
    python "${SCRIPT_DIR}/scripts/summarize_motivation_breakdown.py" "${root}"
}

main() {
    cd "${SCRIPT_DIR}"
    check_python_env

    if [[ -n "${SUMMARIZE_ONLY}" ]]; then
        summarize "${SUMMARIZE_ONLY}"
        exit 0
    fi

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "[INFO] Dry-run matrix:"
        for algo in ${ALGOS}; do
            for batch_size in ${BATCH_SIZES}; do
                for num_spec_tokens in ${NUM_SPEC_TOKENS_LIST}; do
                    echo "  ${algo} bs=${batch_size} K=${num_spec_tokens}"
                done
            done
        done
        exit 0
    fi

    verify_repo_vllm

    mkdir -p "${OUTPUT_ROOT}/runs"
    printf "algo\tbatch_size\tnum_spec_tokens\tstatus\trun_dir\n" \
        > "${OUTPUT_ROOT}/status.tsv"

    local failures=0
    for algo in ${ALGOS}; do
        for batch_size in ${BATCH_SIZES}; do
            for num_spec_tokens in ${NUM_SPEC_TOKENS_LIST}; do
                if ! run_case "${algo}" "${batch_size}" "${num_spec_tokens}"; then
                    failures=$((failures + 1))
                fi
            done
        done
    done

    summarize "${OUTPUT_ROOT}"

    echo "[INFO] Motivation breakdown output root: ${OUTPUT_ROOT}"
    if [[ "${failures}" -gt 0 ]]; then
        echo "[ERROR] ${failures} run(s) failed. Check ${OUTPUT_ROOT}/status.tsv" >&2
        exit 1
    fi
}

main "$@"
