#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python scripts/detect_hardware.py
python scripts/estimate_memory.py --model Qwen/Qwen3-8B --dtype bfloat16
python scripts/download_models.py
python scripts/run_trace_collection.py --target-model Qwen/Qwen3-8B --draft-model Qwen/Qwen3-0.6B --workloads short_chat medium_sharegpt_like long_rag_4k --max-model-len 8192 --splits calibration validation test
python scripts/run_offline_mask_analysis.py --trace-dir experiments/02_trace_collection/raw --output-dir experiments/05_mask_planner_offline
python scripts/run_sparse_kernel_bench.py --model-config Qwen/Qwen3-8B --seq-lens 512 2048 --draft-lens 2 4 --block-sizes 16 32
python scripts/run_exactness_check.py
python scripts/parse_results.py --root experiments
python scripts/plot_results.py --root experiments
python -m pams.report --root experiments --output reports/final_report.md

