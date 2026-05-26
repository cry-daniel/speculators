# Acceptance Jitter Report

This report isolates per-decode-step accepted draft-token counts. It is meant to answer whether a fixed boundary such as 2 or 4 is stable enough before considering end-to-end latency.

- decode steps: 92620
- workloads: math, mtbench, synthetic_1000x1000
- cases: Llama3 EAGLE3, Qwen3 EAGLE3, Qwen3 P-EAGLE
- K values: 8, 12, 16

## Key Columns

- `p_accepted_lt_2`: fraction of decode steps where fewer than 2 draft tokens were accepted.
- `p_accepted_lt_4`: fraction of decode steps where fewer than 4 draft tokens were accepted.
- `mean_abs_step_delta`: average absolute change in accepted count between adjacent decode steps within the same request.

## Fixed Boundary Reading

A fixed h is theoretically fragile when `p_accepted_lt_h` is high: the verifier would often discover rejection before the suffix boundary, so a planned suffix forward would not be useful on those steps.

## Compact Summary

- math / Llama3 EAGLE3 / K=8: mean=1.41, std=1.93, P(<2)=0.66, P(<4)=0.87, P(full)=0.03, steps=4326
- math / Llama3 EAGLE3 / K=12: mean=1.43, std=2.06, P(<2)=0.66, P(<4)=0.87, P(full)=0.00, steps=4298
- math / Llama3 EAGLE3 / K=16: mean=1.44, std=2.11, P(<2)=0.66, P(<4)=0.87, P(full)=0.00, steps=4276
- math / Qwen3 EAGLE3 / K=8: mean=1.60, std=1.81, P(<2)=0.58, P(<4)=0.87, P(full)=0.02, steps=4005
- math / Qwen3 EAGLE3 / K=12: mean=1.61, std=1.93, P(<2)=0.59, P(<4)=0.87, P(full)=0.00, steps=3985
- math / Qwen3 EAGLE3 / K=16: mean=1.60, std=1.96, P(<2)=0.58, P(<4)=0.87, P(full)=0.00, steps=4008
- math / Qwen3 P-EAGLE / K=8: mean=1.66, std=1.43, P(<2)=0.53, P(<4)=0.86, P(full)=0.00, steps=3924
- math / Qwen3 P-EAGLE / K=12: mean=1.65, std=1.43, P(<2)=0.53, P(<4)=0.86, P(full)=0.00, steps=3927
- math / Qwen3 P-EAGLE / K=16: mean=1.64, std=1.41, P(<2)=0.52, P(<4)=0.86, P(full)=0.00, steps=3941
- mtbench / Llama3 EAGLE3 / K=8: mean=0.91, std=1.48, P(<2)=0.79, P(<4)=0.94, P(full)=0.01, steps=5427
- mtbench / Llama3 EAGLE3 / K=12: mean=0.92, std=1.55, P(<2)=0.79, P(<4)=0.94, P(full)=0.00, steps=5410
- mtbench / Llama3 EAGLE3 / K=16: mean=0.92, std=1.58, P(<2)=0.79, P(<4)=0.94, P(full)=0.00, steps=5410
- mtbench / Qwen3 EAGLE3 / K=8: mean=1.26, std=1.53, P(<2)=0.66, P(<4)=0.93, P(full)=0.01, steps=4596
- mtbench / Qwen3 EAGLE3 / K=12: mean=1.27, std=1.62, P(<2)=0.66, P(<4)=0.93, P(full)=0.00, steps=4577
- mtbench / Qwen3 EAGLE3 / K=16: mean=1.25, std=1.63, P(<2)=0.67, P(<4)=0.93, P(full)=0.00, steps=4607
- mtbench / Qwen3 P-EAGLE / K=8: mean=1.29, std=1.28, P(<2)=0.64, P(<4)=0.92, P(full)=0.00, steps=4540
- mtbench / Qwen3 P-EAGLE / K=12: mean=1.28, std=1.27, P(<2)=0.65, P(<4)=0.92, P(full)=0.00, steps=4545
- mtbench / Qwen3 P-EAGLE / K=16: mean=1.28, std=1.27, P(<2)=0.64, P(<4)=0.92, P(full)=0.00, steps=4536
- synthetic_1000x1000 / Llama3 EAGLE3 / K=8: mean=6.88, std=1.22, P(<2)=0.01, P(<4)=0.02, P(full)=0.48, steps=1025
- synthetic_1000x1000 / Llama3 EAGLE3 / K=12: mean=6.88, std=1.22, P(<2)=0.01, P(<4)=0.02, P(full)=0.00, steps=1025
- synthetic_1000x1000 / Llama3 EAGLE3 / K=16: mean=6.88, std=1.22, P(<2)=0.01, P(<4)=0.02, P(full)=0.00, steps=1025
- synthetic_1000x1000 / Qwen3 EAGLE3 / K=8: mean=7.01, std=0.74, P(<2)=0.00, P(<4)=0.00, P(full)=0.28, steps=1008
- synthetic_1000x1000 / Qwen3 EAGLE3 / K=12: mean=7.01, std=3.66, P(<2)=0.00, P(<4)=0.27, P(full)=0.27, steps=1008
- synthetic_1000x1000 / Qwen3 EAGLE3 / K=16: mean=9.72, std=3.78, P(<2)=0.00, P(<4)=0.00, P(full)=0.00, steps=753
- synthetic_1000x1000 / Qwen3 P-EAGLE / K=8: mean=2.73, std=0.93, P(<2)=0.07, P(<4)=0.89, P(full)=0.00, steps=2148
- synthetic_1000x1000 / Qwen3 P-EAGLE / K=12: mean=2.74, std=0.93, P(<2)=0.07, P(<4)=0.88, P(full)=0.00, steps=2145
- synthetic_1000x1000 / Qwen3 P-EAGLE / K=16: mean=2.74, std=0.95, P(<2)=0.07, P(<4)=0.88, P(full)=0.00, steps=2145

## Files

- `step_level_acceptance.csv`: one row per decode step.
- `summary.csv`: fixed-boundary and jitter metrics by workload/case/K.
- `accepted_count_distribution.csv`: empirical distribution over accepted count.
- `figures/*_accepted_count_jitter.png`: workload-specific decode-step curves.
