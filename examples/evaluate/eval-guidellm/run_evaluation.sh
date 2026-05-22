#!/usr/bin/env bash
# Main controller script for evaluating speculator models with guidellm

set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE=""
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
LOCAL_NO_PROXY="localhost,127.0.0.1"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${LOCAL_NO_PROXY}"
export no_proxy="${no_proxy:+${no_proxy},}${LOCAL_NO_PROXY}"

# Variables (precedence: CLI args > config file > defaults)
BASE_MODEL=""
SPECULATOR_MODEL=""
NUM_SPEC_TOKENS=""
METHOD=""
PARALLEL_DRAFTING=""
DATASET=""
TENSOR_PARALLEL_SIZE=""
MAX_MODEL_LEN=""
GPU_MEMORY_UTILIZATION=""
PORT=""
HEALTH_CHECK_TIMEOUT=""
OUTPUT_DIR=""
TEMPERATURE=""
TOP_P=""
TOP_K=""
TOKENIZER_MODE=""
NO_CHUNKED_PREFILL=""

# ==============================================================================
# Helper Functions
# ==============================================================================

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Required (use one):
  -c, --config FILE    Config file (e.g., configs/llama-eagle3.env)
  -b BASE_MODEL        Base model path or HuggingFace ID
     -d DATASET        Dataset for benchmarking

Optional:
  -s SPECULATOR_MODEL  Speculator model (omit for built-in MTP heads)
  -o OUTPUT_DIR        Output directory (default: RESULTS_DIR/eval_results_TIMESTAMP)
  --results-dir DIR    Default output root (default: ${SCRIPT_DIR}/results)
  --num-spec-tokens N  Number of speculative tokens (default: 3)
  --port PORT          vLLM server port (default: 8000)
  --parallel-drafting  Enable vLLM parallel drafting for P-EAGLE
  -h, --help           Show this help message

Examples:
  $0 -c configs/llama-3.3-70b-eagle3.env              # EAGLE3 via config file
  $0 -c configs/qwen3-8b-peagle.env                    # P-EAGLE via config file
  $0 -c configs/qwen3-next-80b-mtp.env                 # MTP via config file
  $0 -b "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic" \\
     -s "RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3" \\
     -d "emulated"                                     # EAGLE3 via command line
  $0 -c configs/llama-eagle3.env -d "other.jsonl"     # Override dataset
EOF
}

check_dependencies() {
    local missing_deps=()

    for cmd in vllm guidellm python curl hf; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        echo "[ERROR] Missing required dependencies: ${missing_deps[*]}" >&2
        echo "[ERROR] Install with: pip install vllm guidellm huggingface-hub" >&2
        return 1
    fi

    return 0
}

cleanup() {
    local exit_code=$?

    echo "[INFO] Cleaning up..."
    "${SCRIPT_DIR}/scripts/vllm_stop.sh" --pid-file "${OUTPUT_DIR}/vllm_server.pid" 2>/dev/null || true

    exit "${exit_code}"
}

# ==============================================================================
# Parse Command Line Arguments
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -b)
            BASE_MODEL="$2"
            shift 2
            ;;
        -s)
            SPECULATOR_MODEL="$2"
            shift 2
            ;;
        -d)
            DATASET="$2"
            shift 2
            ;;
        -o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
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
        --parallel-drafting)
            PARALLEL_DRAFTING="true"
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

# ==============================================================================
# Load Configuration
# ==============================================================================

if [[ -n "${CONFIG_FILE}" ]]; then
    if [[ -f "${CONFIG_FILE}" ]]; then
        echo "[INFO] Loading configuration from: ${CONFIG_FILE}"
        # Source config file, but only if variables are not already set
        while IFS='=' read -r key value; do
            # Skip comments and empty lines
            [[ "$key" =~ ^#.*$ ]] && continue
            [[ -z "$key" ]] && continue

            # Remove quotes from value
            value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")

            # Only set if not already set via command line
            if [[ -z "${!key:-}" ]]; then
                eval "${key}=\"${value}\""
            fi
        done < "${CONFIG_FILE}"
    else
        echo "[ERROR] Config file not found: ${CONFIG_FILE}" >&2
        exit 1
    fi
fi

# ==============================================================================
# Apply Defaults
# ==============================================================================

# Apply defaults for any variables not set by CLI args or config file
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"
METHOD="${METHOD:-eagle3}"
PARALLEL_DRAFTING="${PARALLEL_DRAFTING:-false}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
PORT="${PORT:-8000}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_DIR}/eval_results_$(date +%Y%m%d_%H%M%S)}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"

# ==============================================================================
# Validate Configuration
# ==============================================================================

if [[ -z "${BASE_MODEL}" ]]; then
    echo "[ERROR] BASE_MODEL is required (set in config file or via -b)" >&2
    show_usage
    exit 1
fi

if [[ -z "${DATASET}" ]]; then
    echo "[ERROR] DATASET is required (set in config file or via -d)" >&2
    show_usage
    exit 1
fi

if ! check_dependencies; then
    exit 1
fi

# eagle3 requires an external speculator; mtp uses the built-in head
if [[ "${METHOD}" == "eagle3" && -z "${SPECULATOR_MODEL}" ]]; then
    echo "[ERROR] METHOD=eagle3 requires SPECULATOR_MODEL to be set (use -s or set it in the config file)" >&2
    exit 1
fi

if [[ "${PARALLEL_DRAFTING}" == "true" && -z "${SPECULATOR_MODEL}" ]]; then
    echo "[ERROR] PARALLEL_DRAFTING=true requires SPECULATOR_MODEL to be set" >&2
    exit 1
fi

# Setup cleanup handler
trap cleanup EXIT INT TERM

# ==============================================================================
# Setup Output Directory
# ==============================================================================

if ! mkdir -p "${OUTPUT_DIR}"; then
    echo "[ERROR] Failed to create output directory: ${OUTPUT_DIR}" >&2
    exit 1
fi

echo "[INFO] Output directory: ${OUTPUT_DIR}"

# Define output file paths
SERVER_LOG="${OUTPUT_DIR}/vllm_server.log"
SERVER_PID="${OUTPUT_DIR}/vllm_server.pid"
GUIDELLM_LOG="${OUTPUT_DIR}/guidellm_output.log"
GUIDELLM_RESULTS="${OUTPUT_DIR}/guidellm_results.json"
ACCEPTANCE_RESULTS="${OUTPUT_DIR}/acceptance_analysis.txt"

# ==============================================================================
# Start vLLM Server
# ==============================================================================

echo "[INFO] Starting vLLM server..."

SERVE_ARGS=(
    -b "${BASE_MODEL}"
    --num-spec-tokens "${NUM_SPEC_TOKENS}"
    --method "${METHOD}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --port "${PORT}"
    --health-check-timeout "${HEALTH_CHECK_TIMEOUT}"
    --log-file "${SERVER_LOG}"
    --pid-file "${SERVER_PID}"
)
[[ -n "${SPECULATOR_MODEL}" ]] && SERVE_ARGS+=(-s "${SPECULATOR_MODEL}")
[[ -n "${TOKENIZER_MODE}" ]] && SERVE_ARGS+=(--tokenizer-mode "${TOKENIZER_MODE}")
[[ "${NO_CHUNKED_PREFILL}" == "true" ]] && SERVE_ARGS+=(--no-enable-chunked-prefill)
[[ "${PARALLEL_DRAFTING}" == "true" ]] && SERVE_ARGS+=(--parallel-drafting)

"${SCRIPT_DIR}/scripts/vllm_serve.sh" "${SERVE_ARGS[@]}"

# ==============================================================================
# Run GuideLLM Benchmark
# ==============================================================================

echo "[INFO] Running benchmark..."

"${SCRIPT_DIR}/scripts/run_guidellm.sh" \
    -d "${DATASET}" \
    --target "http://localhost:${PORT}" \
    --output-file "${GUIDELLM_RESULTS}" \
    --log-file "${GUIDELLM_LOG}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}"

# ==============================================================================
# Parse Acceptance Lengths
# ==============================================================================

echo "[INFO] Parsing acceptance lengths..."

PARSER_SCRIPT="${SCRIPT_DIR}/scripts/parse_logs.py"

if [[ ! -f "${PARSER_SCRIPT}" ]]; then
    echo "[ERROR] Parser script not found: ${PARSER_SCRIPT}" >&2
    exit 1
fi

if ! python "${PARSER_SCRIPT}" "${SERVER_LOG}" -o "${ACCEPTANCE_RESULTS}"; then
    echo "[ERROR] Failed to parse acceptance lengths" >&2
    exit 1
fi

# ==============================================================================
# Summary
# ==============================================================================

echo ""
echo "[INFO] Evaluation complete!"
echo "[INFO] Results saved to: ${OUTPUT_DIR}"
echo "[INFO]   - Server log:        ${SERVER_LOG}"
echo "[INFO]   - GuideLLM output:   ${GUIDELLM_LOG}"
echo "[INFO]   - GuideLLM results:  ${GUIDELLM_RESULTS}"
echo "[INFO]   - Acceptance stats:  ${ACCEPTANCE_RESULTS}"
