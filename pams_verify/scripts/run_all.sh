#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
bash scripts/run_smoke.sh
python scripts/run_dense_baselines.py --target-model Qwen/Qwen3-8B --workloads short_chat short_mtbench_like --max-model-len 4096 --concurrency 1 2 4 --num-prompts 32
python scripts/run_vllm_integration_a.py --target-model Qwen/Qwen3-8B
python scripts/run_vllm_integration_b.py --target-model Qwen/Qwen3-8B
python scripts/run_vllm_integration_c.py --target-model Qwen/Qwen3-8B
python scripts/run_end2end_matrix.py --target-model Qwen/Qwen3-8B --draft-model Qwen/Qwen3-0.6B --workloads short_chat short_mtbench_like medium_sharegpt_like long_rag_4k long_output mixed_5090_safe --concurrency 1 2 4 8 --respect-memory-estimator true
python scripts/parse_results.py --root experiments
python scripts/plot_results.py --root experiments
python -m pams.report --root experiments --output reports/final_report.md

