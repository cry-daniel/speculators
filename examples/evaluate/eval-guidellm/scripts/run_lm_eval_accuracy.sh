#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

cd "${REPO_ROOT}"
conda run --no-capture-output -n spec python -u \
  examples/evaluate/eval-guidellm/scripts/run_lm_eval_accuracy.py "$@"
