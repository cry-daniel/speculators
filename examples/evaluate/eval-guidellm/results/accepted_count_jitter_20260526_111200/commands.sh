#!/usr/bin/env bash
set -euo pipefail
cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm
conda run -n spec bash /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/run_acceptance_jitter.sh --output-root ./results/accepted_count_jitter_20260526_111200 --work-root ./temp/accepted_count_jitter_work_20260526_111200
