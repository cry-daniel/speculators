#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-experiments/00_env/raw/gpu_metrics.csv}"
INTERVAL="${PAMS_GPU_METRICS_INTERVAL:-1}"
DURATION="${PAMS_GPU_METRICS_DURATION:-30}"
mkdir -p "$(dirname "$OUT")"
echo "timestamp,name,utilization.gpu,memory.used,memory.total,power.draw" > "$OUT"
end=$((SECONDS + DURATION))
while [[ $SECONDS -lt $end ]]; do
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits >> "$OUT" || true
  sleep "$INTERVAL"
done

