#!/usr/bin/env bash
# Run SpecLink DLM-confidence vs TLM-acceptance trace experiment.

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
OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULTS_DIR}/speclink_confidence_acceptance_datasets_${TIMESTAMP}}"
SMOKE_ROOT="${SMOKE_ROOT:-${TEMP_DIR}/speclink_confidence_acceptance_smoke_${TIMESTAMP}}"
FOUR_WAY_WORK_ROOT="${FOUR_WAY_WORK_ROOT:-${TEMP_DIR}/speclink_confidence_acceptance_reproduce_${TIMESTAMP}}"
SPECLINK_SINGLE_CASE="${SPECLINK_SINGLE_CASE:-0}"

BASE_MODEL="${BASE_MODEL:-${QWEN3_8B_MODEL:-Qwen/Qwen3-8B}}"
MODEL_LABEL="${MODEL_LABEL:-qwen3_8b}"
DATASET="${DATASET:-${SCRIPT_DIR}/data/math_reasoning.jsonl}"
DATASET_LABEL="${DATASET_LABEL:-math}"
EAGLE3_SPECULATOR_MODEL="${EAGLE3_SPECULATOR_MODEL:-${DEFAULT_MODELS_DIR}/qwen3-8b-eagle3-speculator}"
PEAGLE_SPECULATOR_MODEL="${PEAGLE_SPECULATOR_MODEL:-${DEFAULT_MODELS_DIR}/qwen3-8b-peagle-speculator}"
QWEN3_8B_BASE_MODEL="${QWEN3_8B_BASE_MODEL:-${QWEN3_8B_MODEL:-Qwen/Qwen3-8B}}"
LLAMA3_1_8B_BASE_MODEL="${LLAMA3_1_8B_BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
QWEN3_8B_EAGLE3_SPECULATOR_MODEL="${QWEN3_8B_EAGLE3_SPECULATOR_MODEL:-${DEFAULT_MODELS_DIR}/qwen3-8b-eagle3-speculator}"
LLAMA3_1_8B_EAGLE3_SPECULATOR_MODEL="${LLAMA3_1_8B_EAGLE3_SPECULATOR_MODEL:-${DEFAULT_MODELS_DIR}/llama-3.1-8b-eagle3-speculator}"
MATH_DATASET="${MATH_DATASET:-${SCRIPT_DIR}/data/math_reasoning.jsonl}"
MTBENCH_DATASET="${MTBENCH_DATASET:-${SCRIPT_DIR}/data/mt_bench.jsonl}"

METHODS="${METHODS:-eagle3 peagle}"
MAIN_NUM_SPEC_TOKENS="${MAIN_NUM_SPEC_TOKENS:-4 8}"
MAIN_PROMPTS="${MAIN_PROMPTS:-512}"
MAIN_MAX_TOKENS="${MAIN_MAX_TOKENS:-128}"
SMOKE_PROMPTS="${SMOKE_PROMPTS:-16}"
SMOKE_MAX_TOKENS="${SMOKE_MAX_TOKENS:-64}"
SMOKE_NUM_SPEC_TOKENS="${SMOKE_NUM_SPEC_TOKENS:-4}"
REQUEST_CONCURRENCY="${REQUEST_CONCURRENCY:-1}"

PORT="${PORT:-8030}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-1800}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-0}"
SEED="${SEED:-42}"
REPRO_NUM_SPEC_TOKENS="${REPRO_NUM_SPEC_TOKENS:-8}"
REPRO_PROMPTS="${REPRO_PROMPTS:-80}"
REPRO_MAX_TOKENS="${REPRO_MAX_TOKENS:-128}"
REPRO_REQUEST_CONCURRENCY="${REPRO_REQUEST_CONCURRENCY:-1}"
QWEN_PORT="${QWEN_PORT:-8036}"
LLAMA_PORT="${LLAMA_PORT:-8037}"

RUN_SMOKE=1
RUN_MAIN=1
DRY_RUN=0
ANALYZE_ONLY=""

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
  --single-case           Run one configurable case instead of the default four-way report
  --output-root DIR       Main output root (default: ${OUTPUT_ROOT})
  --smoke-root DIR        Smoke output root (default: ${SMOKE_ROOT})
  --smoke-only            Run only smoke into temp/
  --main-only             Skip smoke and run main only
  --analyze-only DIR      Regenerate parsed/calibration/figures/report for DIR
  --port PORT             vLLM server port (default: ${PORT})
  --dry-run               Print cases without launching vLLM
  -h, --help              Show this message

Environment overrides:
  SPECLINK_SINGLE_CASE=${SPECLINK_SINGLE_CASE}
  FOUR_WAY_WORK_ROOT=${FOUR_WAY_WORK_ROOT}
  REPRO_NUM_SPEC_TOKENS=${REPRO_NUM_SPEC_TOKENS}
  REPRO_PROMPTS=${REPRO_PROMPTS}
  REPRO_MAX_TOKENS=${REPRO_MAX_TOKENS}
  METHODS="${METHODS}"
  MAIN_NUM_SPEC_TOKENS="${MAIN_NUM_SPEC_TOKENS}"
  MAIN_PROMPTS=${MAIN_PROMPTS}
  MAIN_MAX_TOKENS=${MAIN_MAX_TOKENS}
  REQUEST_CONCURRENCY=${REQUEST_CONCURRENCY}
  MODEL_LABEL=${MODEL_LABEL}
  DATASET_LABEL=${DATASET_LABEL}
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --single-case)
            SPECLINK_SINGLE_CASE=1
            shift
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --smoke-root)
            SMOKE_ROOT="$2"
            shift 2
            ;;
        --smoke-only)
            RUN_MAIN=0
            shift
            ;;
        --main-only)
            RUN_SMOKE=0
            shift
            ;;
        --analyze-only)
            ANALYZE_ONLY="$2"
            RUN_SMOKE=0
            RUN_MAIN=0
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
[ERROR] Use: conda run -n ${CONDA_ENV} bash ./run_speclink_confidence_acceptance.sh
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

speculator_for_method() {
    case "$1" in
        eagle3) printf '%s\n' "${EAGLE3_SPECULATOR_MODEL}" ;;
        peagle) printf '%s\n' "${PEAGLE_SPECULATOR_MODEL}" ;;
        *) echo "[ERROR] Unknown method: $1" >&2; exit 1 ;;
    esac
}

append_command() {
    local root="$1"
    shift
    printf '%q ' "$@" >> "${root}/commands.sh"
    printf '\n' >> "${root}/commands.sh"
}

write_env_report() {
    local root="$1"
    mkdir -p "${root}"
    {
        echo "# Environment Report"
        echo
        echo "- date: $(date -Iseconds)"
        echo "- hostname: $(hostname)"
        echo "- cwd: ${SCRIPT_DIR}"
        echo "- git branch: $(git -C "${SPECULATORS_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
        echo "- git commit: $(git -C "${SPECULATORS_ROOT}" rev-parse HEAD 2>/dev/null || true)"
        echo "- git diff summary:"
        git -C "${SPECULATORS_ROOT}" diff --stat 2>/dev/null | sed 's/^/  /' || true
        echo "- python: $(python -c 'import sys; print(sys.executable)')"
        echo "- conda env: ${CONDA_DEFAULT_ENV:-unknown}"
        python - <<'PY'
import importlib.metadata
import pathlib
import torch
import vllm

print(f"- torch: {torch.__version__}")
print(f"- torch cuda: {torch.version.cuda}")
print(f"- vLLM: {importlib.metadata.version('vllm')}")
try:
    print(f"- GuideLLM: {importlib.metadata.version('guidellm')}")
except Exception as exc:
    print(f"- GuideLLM: unavailable ({exc})")
print(f"- vLLM path: {pathlib.Path(vllm.__file__).resolve()}")
PY
        echo "- nvidia-smi:"
        nvidia-smi | sed 's/^/  /' || true
        echo "- model label: ${MODEL_LABEL}"
        echo "- dataset label: ${DATASET_LABEL}"
        echo "- base model: ${BASE_MODEL}"
        echo "- EAGLE3 speculator exists: $(test -e "${EAGLE3_SPECULATOR_MODEL}" && echo yes || echo no) (${EAGLE3_SPECULATOR_MODEL})"
        echo "- P-EAGLE speculator exists: $(test -e "${PEAGLE_SPECULATOR_MODEL}" && echo yes || echo no) (${PEAGLE_SPECULATOR_MODEL})"
        echo "- dataset exists: $(test -f "${DATASET}" && echo yes || echo no) (${DATASET})"
        echo "- PEAGLE compatibility:"
        python - <<'PY' | sed 's/^/  /'
from vllm.transformers_utils.configs.speculators import algos
print("supported", sorted(algos.SUPPORTED_SPECULATORS_TYPES))
PY
    } > "${root}/env_report.md"
}

prepare_root() {
    local root="$1"
    mkdir -p "${root}/trace" "${root}/parsed" "${root}/calibration" \
        "${root}/figures" "${root}/runs"
    : > "${root}/commands.sh"
    chmod +x "${root}/commands.sh"
    write_env_report "${root}"
}

run_requests() {
    local root="$1"
    local method="$2"
    local k="$3"
    local max_prompts="$4"
    local max_tokens="$5"
    local run_dir="$6"
    local responses="${run_dir}/responses.jsonl"
    local cmd=(
        python "${SCRIPT_DIR}/scripts/send_speclink_confidence_requests.py"
        --target "http://localhost:${PORT}"
        --model "${BASE_MODEL}"
        --model-label "${MODEL_LABEL}"
        --dataset-label "${DATASET_LABEL}"
        --dataset "${DATASET}"
        --method "${method}"
        --num-spec-tokens "${k}"
        --max-prompts "${max_prompts}"
        --max-tokens "${max_tokens}"
        --temperature "${TEMPERATURE}"
        --top-p "${TOP_P}"
        --top-k "${TOP_K}"
        --seed "${SEED}"
        --concurrency "${REQUEST_CONCURRENCY}"
        --output-jsonl "${responses}"
    )
    append_command "${root}" "${cmd[@]}"
    "${cmd[@]}" | tee "${run_dir}/client.log"
}

run_case() {
    local root="$1"
    local method="$2"
    local k="$3"
    local max_prompts="$4"
    local max_tokens="$5"
    local run_dir="${root}/runs/${DATASET_LABEL}_${MODEL_LABEL}_${method}_k${k}"
    local server_log="${run_dir}/vllm_server.log"
    local server_pid="${run_dir}/vllm_server.pid"
    local trace_file="${root}/trace/${DATASET_LABEL}_${MODEL_LABEL}_${method}_trace.jsonl"
    local speculator_model
    local -a serve_cmd
    local rc=0

    mkdir -p "${run_dir}"
    speculator_model="$(speculator_for_method "${method}")"
    serve_cmd=(
        "${SCRIPT_DIR}/scripts/vllm_serve.sh"
        -b "${BASE_MODEL}"
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

    append_command "${root}" env SPECLINK_TRACE_CONFIDENCE=1 \
        SPECLINK_TRACE_OUTPUT="${trace_file}" \
        SPECLINK_TRACE_RUN_ID="${DATASET_LABEL}_${MODEL_LABEL}_${method}_k${k}" \
        SPECLINK_TRACE_DATASET_LABEL="${DATASET_LABEL}" \
        SPECLINK_TRACE_MODEL_LABEL="${MODEL_LABEL}" \
        SPECLINK_TRACE_METHOD="${method}" \
        SPECLINK_TRACE_NUM_SPEC_TOKENS="${k}" \
        "${serve_cmd[@]}"

    set +e
    (
        set -euo pipefail
        SPECLINK_TRACE_CONFIDENCE=1 \
        SPECLINK_TRACE_OUTPUT="${trace_file}" \
        SPECLINK_TRACE_RUN_ID="${DATASET_LABEL}_${MODEL_LABEL}_${method}_k${k}" \
        SPECLINK_TRACE_DATASET_LABEL="${DATASET_LABEL}" \
        SPECLINK_TRACE_MODEL_LABEL="${MODEL_LABEL}" \
        SPECLINK_TRACE_METHOD="${method}" \
        SPECLINK_TRACE_NUM_SPEC_TOKENS="${k}" \
            "${serve_cmd[@]}"
        run_requests "${root}" "${method}" "${k}" "${max_prompts}" \
            "${max_tokens}" "${run_dir}"
        if ! python "${SCRIPT_DIR}/scripts/parse_logs.py" "${server_log}" \
            -o "${run_dir}/acceptance_analysis.txt" \
            > "${run_dir}/parse_logs_stdout.log" \
            2> "${run_dir}/parse_logs_stderr.log"; then
            echo "[WARN] No aggregate SpecDecoding metrics parsed for ${method} K=${k}; token-level trace is still analyzed."
        fi
    )
    rc=$?
    set -e
    "${SCRIPT_DIR}/scripts/vllm_stop.sh" --pid-file "${server_pid}" || true
    if [[ "${rc}" -ne 0 ]]; then
        echo "[ERROR] Failed ${method} K=${k}; see ${run_dir}" >&2
        return "${rc}"
    fi
}

analyze_root() {
    local root="$1"
    append_command "${root}" python \
        "${SCRIPT_DIR}/scripts/analyze_speclink_confidence_acceptance.py" "${root}"
    python "${SCRIPT_DIR}/scripts/analyze_speclink_confidence_acceptance.py" "${root}"
}

run_matrix() {
    local root="$1"
    local prompts="$2"
    local max_tokens="$3"
    shift 3
    local ks=("$@")
    prepare_root "${root}"
    for method in ${METHODS}; do
        : > "${root}/trace/${DATASET_LABEL}_${MODEL_LABEL}_${method}_trace.jsonl"
        for k in "${ks[@]}"; do
            echo "[INFO] Running ${method} K=${k}, prompts=${prompts}, max_tokens=${max_tokens}"
            run_case "${root}" "${method}" "${k}" "${prompts}" "${max_tokens}"
        done
    done
    analyze_root "${root}"
    echo "[INFO] Output root: ${root}"
}

run_repro_case() {
    local dataset_label="$1"
    local dataset_path="$2"
    local model_label="$3"
    local base_model="$4"
    local speculator_model="$5"
    local port="$6"
    local case_root="${FOUR_WAY_WORK_ROOT}/${dataset_label}_${model_label}_eagle3_k${REPRO_NUM_SPEC_TOKENS}"
    local -a cmd=(
        env
        SPECLINK_SINGLE_CASE=1
        OUTPUT_ROOT="${case_root}"
        DATASET_LABEL="${dataset_label}"
        DATASET="${dataset_path}"
        MODEL_LABEL="${model_label}"
        BASE_MODEL="${base_model}"
        EAGLE3_SPECULATOR_MODEL="${speculator_model}"
        METHODS=eagle3
        MAIN_NUM_SPEC_TOKENS="${REPRO_NUM_SPEC_TOKENS}"
        MAIN_PROMPTS="${REPRO_PROMPTS}"
        MAIN_MAX_TOKENS="${REPRO_MAX_TOKENS}"
        REQUEST_CONCURRENCY="${REPRO_REQUEST_CONCURRENCY}"
        PORT="${port}"
        bash "${SCRIPT_PATH}" --single-case --main-only
    )

    echo "[INFO] Reproducing ${dataset_label}/${model_label}/eagle3/K=${REPRO_NUM_SPEC_TOKENS}"
    printf '%q ' "${cmd[@]}" >> "${FOUR_WAY_WORK_ROOT}/commands.sh"
    printf '\n' >> "${FOUR_WAY_WORK_ROOT}/commands.sh"
    "${cmd[@]}"
}

write_four_way_commands() {
    local root="$1"
    {
        echo "#!/usr/bin/env bash"
        echo "set -euo pipefail"
        printf 'cd %q\n' "${SCRIPT_DIR}"
        printf 'conda run -n %q bash ./run_speclink_confidence_acceptance.sh\n' "${CONDA_ENV}"
    } > "${root}/commands.sh"
    chmod +x "${root}/commands.sh"
}

write_four_way_report() {
    local root="$1"
    {
        echo "# Four-Way Confidence Acceptance Reproduction"
        echo
        echo "- date: $(date -Iseconds)"
        echo "- work root: ${FOUR_WAY_WORK_ROOT}"
        echo "- output root: ${root}"
        echo "- num_spec_tokens: ${REPRO_NUM_SPEC_TOKENS}"
        echo "- prompts per case: ${REPRO_PROMPTS}"
        echo "- max tokens per request: ${REPRO_MAX_TOKENS}"
        echo "- request concurrency: ${REPRO_REQUEST_CONCURRENCY}"
        echo "- math dataset: ${MATH_DATASET}"
        echo "- mtbench dataset: ${MTBENCH_DATASET}"
        echo "- qwen base: ${QWEN3_8B_BASE_MODEL}"
        echo "- llama base: ${LLAMA3_1_8B_BASE_MODEL}"
        echo "- qwen eagle3 speculator: ${QWEN3_8B_EAGLE3_SPECULATOR_MODEL}"
        echo "- llama eagle3 speculator: ${LLAMA3_1_8B_EAGLE3_SPECULATOR_MODEL}"
    } > "${root}/repro_report.md"
}

run_four_way_reproduction() {
    local combine_root="${OUTPUT_ROOT}"
    local -a combine_cmd

    mkdir -p "${FOUR_WAY_WORK_ROOT}"
    : > "${FOUR_WAY_WORK_ROOT}/commands.sh"
    chmod +x "${FOUR_WAY_WORK_ROOT}/commands.sh"

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

    run_repro_case "math" "${MATH_DATASET}" "qwen3_8b" \
        "${QWEN3_8B_BASE_MODEL}" "${QWEN3_8B_EAGLE3_SPECULATOR_MODEL}" "${QWEN_PORT}"
    run_repro_case "math" "${MATH_DATASET}" "llama3_1_8b" \
        "${LLAMA3_1_8B_BASE_MODEL}" "${LLAMA3_1_8B_EAGLE3_SPECULATOR_MODEL}" "${LLAMA_PORT}"
    run_repro_case "mtbench" "${MTBENCH_DATASET}" "qwen3_8b" \
        "${QWEN3_8B_BASE_MODEL}" "${QWEN3_8B_EAGLE3_SPECULATOR_MODEL}" "${QWEN_PORT}"
    run_repro_case "mtbench" "${MTBENCH_DATASET}" "llama3_1_8b" \
        "${LLAMA3_1_8B_BASE_MODEL}" "${LLAMA3_1_8B_EAGLE3_SPECULATOR_MODEL}" "${LLAMA_PORT}"

    combine_cmd=(
        python "${SCRIPT_DIR}/scripts/combine_speclink_confidence_results.py"
        --output-root "${combine_root}"
        --source "qwen3_8b:math:${FOUR_WAY_WORK_ROOT}/math_qwen3_8b_eagle3_k${REPRO_NUM_SPEC_TOKENS}"
        --source "llama3_1_8b:math:${FOUR_WAY_WORK_ROOT}/math_llama3_1_8b_eagle3_k${REPRO_NUM_SPEC_TOKENS}"
        --source "qwen3_8b:mtbench:${FOUR_WAY_WORK_ROOT}/mtbench_qwen3_8b_eagle3_k${REPRO_NUM_SPEC_TOKENS}"
        --source "llama3_1_8b:mtbench:${FOUR_WAY_WORK_ROOT}/mtbench_llama3_1_8b_eagle3_k${REPRO_NUM_SPEC_TOKENS}"
        --method eagle3
        --num-spec-tokens "${REPRO_NUM_SPEC_TOKENS}"
        --analyze
    )
    printf '%q ' "${combine_cmd[@]}" >> "${FOUR_WAY_WORK_ROOT}/commands.sh"
    printf '\n' >> "${FOUR_WAY_WORK_ROOT}/commands.sh"
    "${combine_cmd[@]}"
    write_four_way_commands "${combine_root}"
    write_four_way_report "${combine_root}"
    echo "[INFO] Four-way output root: ${combine_root}"
    echo "[INFO] Intermediate case roots are under: ${FOUR_WAY_WORK_ROOT}"
}

main() {
    cd "${SCRIPT_DIR}"
    check_python_env

    if [[ -n "${ANALYZE_ONLY}" ]]; then
        analyze_root "${ANALYZE_ONLY}"
        exit 0
    fi

    verify_repo_vllm

    if [[ "${SPECLINK_SINGLE_CASE}" -ne 1 ]]; then
        if [[ "${DRY_RUN}" -eq 1 ]]; then
            echo "[INFO] Four-way output root: ${OUTPUT_ROOT}"
            echo "[INFO] Four-way intermediate root: ${FOUR_WAY_WORK_ROOT}"
            echo "case: math/qwen3_8b/eagle3/K=${REPRO_NUM_SPEC_TOKENS} prompts=${REPRO_PROMPTS}"
            echo "case: math/llama3_1_8b/eagle3/K=${REPRO_NUM_SPEC_TOKENS} prompts=${REPRO_PROMPTS}"
            echo "case: mtbench/qwen3_8b/eagle3/K=${REPRO_NUM_SPEC_TOKENS} prompts=${REPRO_PROMPTS}"
            echo "case: mtbench/llama3_1_8b/eagle3/K=${REPRO_NUM_SPEC_TOKENS} prompts=${REPRO_PROMPTS}"
            exit 0
        fi
        run_four_way_reproduction
        exit 0
    fi

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "[INFO] Smoke root: ${SMOKE_ROOT}"
        echo "[INFO] Main root:  ${OUTPUT_ROOT}"
        [[ "${RUN_SMOKE}" -eq 1 ]] && echo "smoke: METHODS=${METHODS} K=${SMOKE_NUM_SPEC_TOKENS} prompts=${SMOKE_PROMPTS}"
        [[ "${RUN_MAIN}" -eq 1 ]] && echo "main:  METHODS=${METHODS} K=${MAIN_NUM_SPEC_TOKENS} prompts=${MAIN_PROMPTS}"
        exit 0
    fi

    if [[ ! -f "${DATASET}" ]]; then
        echo "[ERROR] Dataset not found: ${DATASET}" >&2
        exit 1
    fi

    if [[ "${RUN_SMOKE}" -eq 1 ]]; then
        run_matrix "${SMOKE_ROOT}" "${SMOKE_PROMPTS}" "${SMOKE_MAX_TOKENS}" \
            "${SMOKE_NUM_SPEC_TOKENS}"
    fi
    if [[ "${RUN_MAIN}" -eq 1 ]]; then
        read -r -a main_ks <<< "${MAIN_NUM_SPEC_TOKENS}"
        run_matrix "${OUTPUT_ROOT}" "${MAIN_PROMPTS}" "${MAIN_MAX_TOKENS}" \
            "${main_ks[@]}"
    fi
}

main "$@"
