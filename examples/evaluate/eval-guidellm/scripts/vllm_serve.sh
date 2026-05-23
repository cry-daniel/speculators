#!/usr/bin/env bash
# Start vLLM server for speculator model evaluation

set -euo pipefail

LOCAL_NO_PROXY="localhost,127.0.0.1"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${LOCAL_NO_PROXY}"
export no_proxy="${no_proxy:+${no_proxy},}${LOCAL_NO_PROXY}"

# ==============================================================================
# Configuration Variables
# ==============================================================================

BASE_MODEL=""
SPECULATOR_MODEL=""
NUM_SPEC_TOKENS=""
METHOD=""
PARALLEL_DRAFTING=""
TENSOR_PARALLEL_SIZE=""
MAX_MODEL_LEN=""
GPU_MEMORY_UTILIZATION=""
MAX_NUM_BATCHED_TOKENS=""
MAX_NUM_SEQS=""
PERFORMANCE_MODE=""
USE_LOCAL_ARGMAX_REDUCTION=""
REJECTION_SAMPLE_METHOD=""
DISABLE_SPECULATIVE_DECODING=""
SPECULATIVE_TOKEN_TREE=""
PORT=""
HEALTH_CHECK_TIMEOUT=""
SERVER_LOG=""
PID_FILE=""
TOKENIZER_MODE=""
NO_CHUNKED_PREFILL=""

readonly SLEEP_INTERVAL=5

# ==============================================================================
# Helper Functions
# ==============================================================================

show_usage() {
    cat << EOF
Usage: $0 -b BASE_MODEL [OPTIONS]

Required:
  -b BASE_MODEL              Base model path or HuggingFace ID

Optional:
  -s SPECULATOR_MODEL            Speculator model path or HuggingFace ID
                                 (omit for built-in MTP heads)
  --num-spec-tokens TOKENS       Number of speculative tokens (default: 3)
  --method METHOD                Speculative decoding method (default: eagle3)
  --parallel-drafting            Enable parallel drafting for P-EAGLE
  --tensor-parallel-size SIZE    Number of GPUs (default: 1)
  --max-model-len LENGTH         Max model length (default: 24000)
  --gpu-memory-utilization UTIL  GPU memory fraction (default: 0.85)
  --max-num-batched-tokens N     vLLM scheduler token budget per iteration
  --max-num-seqs N               Maximum active sequences
  --performance-mode MODE        vLLM performance mode: balanced, interactivity, throughput
  --use-local-argmax-reduction   Enable vLLM draft-token local argmax fast path
  --rejection-sample-method METHOD
                                 Rejection sampler: strict, probabilistic, synthetic
  --speculative-token-tree TREE  Python literal tree choices for tree speculative decoding
  --no-speculative-decoding       Serve the base model without speculative decoding
  --port PORT                    Server port (default: 8000)
  --health-check-timeout SECS    Health check timeout (default: 300)
  --log-file FILE                Log file path (default: vllm_server.log)
  --pid-file FILE                PID file path (default: vllm_server.pid)
  --tokenizer-mode MODE          Tokenizer mode passed to vllm (e.g. auto)
  --no-enable-chunked-prefill    Pass --no-enable-chunked-prefill to vllm
  -h, --help                     Show this help message

Examples:
  # Eagle3 (separate speculator)
  $0 -b "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic" \\
     -s "RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3" \\
     --num-spec-tokens 3 --method eagle3

  # P-EAGLE (EAGLE3-compatible speculator with parallel drafting)
  $0 -b "Qwen/Qwen3-8B" \\
     -s "/path/to/qwen3-8b-peagle" \\
     --num-spec-tokens 3 --method eagle3 --parallel-drafting

  # MTP (built-in head)
  $0 -b "Qwen/Qwen3-Next-80B-A3B-Instruct" \\
     --num-spec-tokens 2 --method mtp \\
     --tokenizer-mode auto --no-enable-chunked-prefill
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -b)
            BASE_MODEL="$2"
            shift 2
            ;;
        -s)
            SPECULATOR_MODEL="$2"
            shift 2
            ;;
        --num-spec-tokens)
            NUM_SPEC_TOKENS="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --parallel-drafting)
            PARALLEL_DRAFTING="true"
            shift
            ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --max-num-batched-tokens)
            MAX_NUM_BATCHED_TOKENS="$2"
            shift 2
            ;;
        --max-num-seqs)
            MAX_NUM_SEQS="$2"
            shift 2
            ;;
        --performance-mode)
            PERFORMANCE_MODE="$2"
            shift 2
            ;;
        --use-local-argmax-reduction)
            USE_LOCAL_ARGMAX_REDUCTION="true"
            shift
            ;;
        --rejection-sample-method)
            REJECTION_SAMPLE_METHOD="$2"
            shift 2
            ;;
        --speculative-token-tree)
            SPECULATIVE_TOKEN_TREE="$2"
            shift 2
            ;;
        --no-speculative-decoding)
            DISABLE_SPECULATIVE_DECODING="true"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --health-check-timeout)
            HEALTH_CHECK_TIMEOUT="$2"
            shift 2
            ;;
        --log-file)
            SERVER_LOG="$2"
            shift 2
            ;;
        --pid-file)
            PID_FILE="$2"
            shift 2
            ;;
        --tokenizer-mode)
            TOKENIZER_MODE="$2"
            shift 2
            ;;
        --no-enable-chunked-prefill)
            NO_CHUNKED_PREFILL="true"
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
# Apply Defaults
# ==============================================================================

# Apply defaults for any arguments not provided
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"
METHOD="${METHOD:-eagle3}"
PARALLEL_DRAFTING="${PARALLEL_DRAFTING:-false}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
PORT="${PORT:-8000}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"
SERVER_LOG="${SERVER_LOG:-vllm_server.log}"
PID_FILE="${PID_FILE:-vllm_server.pid}"

# ==============================================================================
# Validate Arguments
# ==============================================================================

if [[ -z "${BASE_MODEL}" ]]; then
    echo "[ERROR] Missing required argument: -b BASE_MODEL" >&2
    show_usage
    exit 1
fi

if [[ "${DISABLE_SPECULATIVE_DECODING}" != "true" && "${PARALLEL_DRAFTING}" == "true" && -z "${SPECULATOR_MODEL}" ]]; then
    echo "[ERROR] --parallel-drafting requires -s SPECULATOR_MODEL" >&2
    exit 1
fi

# ==============================================================================
# Start Server
# ==============================================================================

if [[ "${DISABLE_SPECULATIVE_DECODING}" == "true" ]]; then
    echo "[INFO] Starting vLLM server without speculative decoding"
else
    echo "[INFO] Starting vLLM server with speculative decoding"
fi
echo "[INFO]   Base model: ${BASE_MODEL}"
if [[ "${DISABLE_SPECULATIVE_DECODING}" != "true" ]]; then
    echo "[INFO]   Speculator model: ${SPECULATOR_MODEL:-(built-in MTP head)}"
    echo "[INFO]   Num speculative tokens: ${NUM_SPEC_TOKENS}"
    echo "[INFO]   Method: ${METHOD}"
    echo "[INFO]   Parallel drafting: ${PARALLEL_DRAFTING}"
fi
echo "[INFO]   Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "[INFO]   Max model length: ${MAX_MODEL_LEN}"
echo "[INFO]   GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
[[ -n "${MAX_NUM_BATCHED_TOKENS}" ]] && echo "[INFO]   Max num batched tokens: ${MAX_NUM_BATCHED_TOKENS}"
[[ -n "${MAX_NUM_SEQS}" ]] && echo "[INFO]   Max num seqs: ${MAX_NUM_SEQS}"
[[ -n "${PERFORMANCE_MODE}" ]] && echo "[INFO]   Performance mode: ${PERFORMANCE_MODE}"
[[ "${USE_LOCAL_ARGMAX_REDUCTION}" == "true" ]] && echo "[INFO]   Local argmax reduction: enabled"
[[ -n "${REJECTION_SAMPLE_METHOD}" ]] && echo "[INFO]   Rejection sample method: ${REJECTION_SAMPLE_METHOD}"
[[ -n "${SPECULATIVE_TOKEN_TREE}" ]] && echo "[INFO]   Speculative token tree: ${SPECULATIVE_TOKEN_TREE}"
echo "[INFO]   Port: ${PORT}"
echo "[INFO]   Log file: ${SERVER_LOG}"
[[ -n "${TOKENIZER_MODE}" ]] && echo "[INFO]   Tokenizer mode: ${TOKENIZER_MODE}"
[[ "${NO_CHUNKED_PREFILL}" == "true" ]] && echo "[INFO]   Chunked prefill: disabled"

SPEC_CONFIG=""
if [[ "${DISABLE_SPECULATIVE_DECODING}" != "true" ]]; then
    # Build speculative-config JSON:
    #   With external speculator (Eagle): include model + max_model_len fields
    #   Without speculator (MTP built-in): method + num_speculative_tokens only
    if [[ -n "${SPECULATOR_MODEL}" ]]; then
        SPEC_CONFIG="{\"model\": \"${SPECULATOR_MODEL}\", \"num_speculative_tokens\": ${NUM_SPEC_TOKENS}, \"method\": \"${METHOD}\", \"max_model_len\": ${MAX_MODEL_LEN}"
    else
        SPEC_CONFIG="{\"method\": \"${METHOD}\", \"num_speculative_tokens\": ${NUM_SPEC_TOKENS}"
    fi
    [[ "${PARALLEL_DRAFTING}" == "true" ]] && SPEC_CONFIG="${SPEC_CONFIG}, \"parallel_drafting\": true"
    [[ "${USE_LOCAL_ARGMAX_REDUCTION}" == "true" ]] && SPEC_CONFIG="${SPEC_CONFIG}, \"use_local_argmax_reduction\": true"
    [[ -n "${REJECTION_SAMPLE_METHOD}" ]] && SPEC_CONFIG="${SPEC_CONFIG}, \"rejection_sample_method\": \"${REJECTION_SAMPLE_METHOD}\""
    [[ -n "${SPECULATIVE_TOKEN_TREE}" ]] && SPEC_CONFIG="${SPEC_CONFIG}, \"speculative_token_tree\": \"${SPECULATIVE_TOKEN_TREE}\""
    SPEC_CONFIG="${SPEC_CONFIG}}"
fi

# Build optional extra flags
EXTRA_FLAGS=()
[[ -n "${TOKENIZER_MODE}" ]] && EXTRA_FLAGS+=(--tokenizer-mode "${TOKENIZER_MODE}")
[[ "${NO_CHUNKED_PREFILL}" == "true" ]] && EXTRA_FLAGS+=(--no-enable-chunked-prefill)
[[ -n "${MAX_NUM_BATCHED_TOKENS}" ]] && EXTRA_FLAGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
[[ -n "${MAX_NUM_SEQS}" ]] && EXTRA_FLAGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
[[ -n "${PERFORMANCE_MODE}" ]] && EXTRA_FLAGS+=(--performance-mode "${PERFORMANCE_MODE}")

# Fail fast if the port is already in use rather than burying the error in vLLM logs
if ss -tlnp 2>/dev/null | grep -q ":${PORT} " || nc -z 127.0.0.1 "${PORT}" 2>/dev/null; then
    echo "[ERROR] Port ${PORT} is already in use. Stop the existing process or set PORT to a different value." >&2
    exit 1
fi

VLLM_ARGS=(
    serve "${BASE_MODEL}"
    --seed 42
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --port "${PORT}"
)
[[ "${DISABLE_SPECULATIVE_DECODING}" != "true" ]] && VLLM_ARGS+=(--speculative-config "${SPEC_CONFIG}")

vllm "${VLLM_ARGS[@]}" "${EXTRA_FLAGS[@]}" > "${SERVER_LOG}" 2>&1 &

VLLM_PID=$!
echo "${VLLM_PID}" > "${PID_FILE}"
echo "[INFO] vLLM server started (PID: ${VLLM_PID})"

# ==============================================================================
# Wait for Server to be Ready
# ==============================================================================

echo "[INFO] Waiting for server to be ready (timeout: ${HEALTH_CHECK_TIMEOUT}s)..."

elapsed=0

while [[ ${elapsed} -lt ${HEALTH_CHECK_TIMEOUT} ]]; do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "[INFO] Server ready!"
        exit 0
    fi

    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[ERROR] vLLM server died during startup" >&2
        echo "[ERROR] Check logs: ${SERVER_LOG}" >&2
        tail -n 50 "${SERVER_LOG}" >&2
        rm -f "${PID_FILE}"
        exit 1
    fi

    sleep "${SLEEP_INTERVAL}"
    elapsed=$((elapsed + SLEEP_INTERVAL))
done

echo "[ERROR] Server failed to start within ${HEALTH_CHECK_TIMEOUT}s" >&2
kill -TERM "${VLLM_PID}" 2>/dev/null || true
rm -f "${PID_FILE}"
exit 1
