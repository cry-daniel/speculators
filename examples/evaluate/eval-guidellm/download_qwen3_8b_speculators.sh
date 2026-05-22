#!/usr/bin/env bash
# Download local Qwen3-8B EAGLE3 and P-EAGLE speculator checkpoints.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODELS_DIR="/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models"
EAGLE3_DIR=""
PEAGLE_DIR=""
PYTHON_BIN="/ACALAB/stu1/miniconda3/envs/spec/bin/python"
HF_BIN="/ACALAB/stu1/miniconda3/envs/spec/bin/hf"
MAX_WORKERS=8
INSTALL_LOCAL_SPECULATORS=true

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Optional:
  --models-dir DIR       Directory that will contain both checkpoints
                         (default: ${MODELS_DIR})
  --eagle3-dir DIR       Local EAGLE3 checkpoint directory
  --peagle-dir DIR       Local P-EAGLE checkpoint directory
  --python PATH          Python executable for the spec conda environment
                         (default: ${PYTHON_BIN})
  --hf PATH              Hugging Face CLI executable
                         (default: ${HF_BIN})
  --max-workers N        Hugging Face download worker count (default: 8)
  --skip-install         Do not install the local speculators package into spec
  -h, --help             Show this help message

Downloads:
  EAGLE3: RedHatAI/Qwen3-8B-speculator.eagle3
  P-EAGLE: nm-testing/qwen3-8b-peagle-speculators
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --models-dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        --eagle3-dir)
            EAGLE3_DIR="$2"
            shift 2
            ;;
        --peagle-dir)
            PEAGLE_DIR="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
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
        --skip-install)
            INSTALL_LOCAL_SPECULATORS=false
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

EAGLE3_DIR="${EAGLE3_DIR:-${MODELS_DIR}/qwen3-8b-eagle3-speculator}"
PEAGLE_DIR="${PEAGLE_DIR:-${MODELS_DIR}/qwen3-8b-peagle-speculator}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] Python executable not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -x "${HF_BIN}" ]]; then
    echo "[ERROR] Hugging Face CLI not found or not executable: ${HF_BIN}" >&2
    exit 1
fi

mkdir -p "${EAGLE3_DIR}" "${PEAGLE_DIR}"

if [[ "${INSTALL_LOCAL_SPECULATORS}" == "true" ]]; then
    echo "[INFO] Installing local speculators package into: ${PYTHON_BIN}"
    "${PYTHON_BIN}" -m pip install -e "${REPO_ROOT}" --no-deps
fi

echo "[INFO] Downloading EAGLE3 speculator to: ${EAGLE3_DIR}"
"${HF_BIN}" download RedHatAI/Qwen3-8B-speculator.eagle3 \
    --local-dir "${EAGLE3_DIR}" \
    --max-workers "${MAX_WORKERS}" \
    --include ".gitattributes" \
    --include "README.md" \
    --include "assets/*" \
    --include "config.json" \
    --include "eagle3.py" \
    --include "generation_config.json" \
    --include "model.safetensors"

echo "[INFO] Downloading P-EAGLE speculator to: ${PEAGLE_DIR}"
"${HF_BIN}" download nm-testing/qwen3-8b-peagle-speculators \
    --local-dir "${PEAGLE_DIR}" \
    --max-workers "${MAX_WORKERS}" \
    --include ".gitattributes" \
    --include "config.json" \
    --include "config.py" \
    --include "model.safetensors" \
    --include "val_metrics.json"

if [[ -e "${PEAGLE_DIR}/optimizer_state_dict.pt" ]]; then
    echo "[ERROR] Unexpected training optimizer state found in PEAGLE directory: ${PEAGLE_DIR}/optimizer_state_dict.pt" >&2
    exit 1
fi

echo "[INFO] Local speculator checkpoints ready:"
echo "[INFO]   EAGLE3: ${EAGLE3_DIR}"
echo "[INFO]   P-EAGLE: ${PEAGLE_DIR}"
