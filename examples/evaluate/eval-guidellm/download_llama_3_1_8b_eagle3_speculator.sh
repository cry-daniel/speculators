#!/usr/bin/env bash
# Download the Llama-3.1-8B EAGLE3 speculator checkpoint.

set -euo pipefail

MODELS_DIR="/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models"
LLAMA_EAGLE3_DIR=""
HF_BIN="/ACALAB/stu1/miniconda3/envs/spec/bin/hf"
MAX_WORKERS=8

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Optional:
  --models-dir DIR       Directory that will contain the checkpoint
                         (default: ${MODELS_DIR})
  --output-dir DIR       Local Llama EAGLE3 checkpoint directory
                         (default: MODELS_DIR/llama-3.1-8b-eagle3-speculator)
  --hf PATH              Hugging Face CLI executable
                         (default: ${HF_BIN})
  --max-workers N        Hugging Face download worker count (default: 8)
  -h, --help             Show this help message

Downloads:
  RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models-dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            LLAMA_EAGLE3_DIR="$2"
            shift 2
            ;;
        --hf)
            HF_BIN="$2"
            shift 2
            ;;
        --max-workers)
            MAX_WORKERS="$2"
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

LLAMA_EAGLE3_DIR="${LLAMA_EAGLE3_DIR:-${MODELS_DIR}/llama-3.1-8b-eagle3-speculator}"

if [[ ! -x "${HF_BIN}" ]]; then
    echo "[ERROR] Hugging Face CLI not found or not executable: ${HF_BIN}" >&2
    exit 1
fi

mkdir -p "${LLAMA_EAGLE3_DIR}"

echo "[INFO] Downloading Llama-3.1-8B EAGLE3 speculator to: ${LLAMA_EAGLE3_DIR}"
"${HF_BIN}" download RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3 \
    --local-dir "${LLAMA_EAGLE3_DIR}" \
    --max-workers "${MAX_WORKERS}" \
    --include ".gitattributes" \
    --include "README.md" \
    --include "assets/*" \
    --include "config.json" \
    --include "eagle3.py" \
    --include "generation_config.json" \
    --include "model.safetensors"

echo "[INFO] Llama EAGLE3 checkpoint ready: ${LLAMA_EAGLE3_DIR}"
