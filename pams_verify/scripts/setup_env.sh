#!/usr/bin/env bash
set -euo pipefail

echo "PAMS-Verify uses the existing conda environment when available:"
echo "  conda activate spec"
echo
echo "Required packages:"
echo "  vllm[bench] torch transformers accelerate datasets pandas numpy scipy scikit-learn matplotlib tqdm pyyaml psutil pynvml triton"
echo
if [[ "${PAMS_INSTALL:-0}" == "1" ]]; then
  python -m pip install 'vllm[bench]' torch transformers accelerate datasets pandas numpy scipy scikit-learn matplotlib tqdm pyyaml psutil pynvml triton
else
  echo "Dry run only. Set PAMS_INSTALL=1 to install packages."
fi

