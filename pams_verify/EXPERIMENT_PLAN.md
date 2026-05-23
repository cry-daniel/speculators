# Experiment Plan

## Registered Phases

0. Environment and hardware detection.
1. Novelty audit.
2. Deterministic workloads.
3. Dense and standard speculative baselines.
4. Trace collection.
5. Union-growth analysis.
6. Acceptance-prior calibration.
7. Mask planner implementation.
8. Offline mask quality and speed proxy.
9. Sparse attention reference microbenchmark.
10. vLLM integration attempts A, B, and C.
11. End-to-end matrix.
12. Correctness and quality.
13. Ablations.
14. Figures.
15. Final report.

## Default Safety Rules

- Start with `gpu_memory_utilization=0.85`.
- Start smoke tests at `max_model_len=4096`.
- Try `8192` only after smoke passes.
- Try `16384` only if the memory estimator reports at least 20% headroom.
- Never run `32768` context by default on RTX 5090.
- Record OOMs and unsupported vLLM options under `experiments/13_failures_oom` or the relevant experiment folder.

## Split Policy

Calibration, validation, and test traces are disjoint prompt-id splits. The
temperature calibrator is fit on calibration, inspected on validation, and final
offline metrics are reported on test.

## Claim Policy

Simulation, microbenchmark, and patched-vLLM results are separate labels. GO
requires a patched-vLLM end-to-end PAMS result and cannot be satisfied by
offline or reference-kernel results.

