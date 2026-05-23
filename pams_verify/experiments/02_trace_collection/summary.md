# Phase 4 Trace Collection

Current run used deterministic synthetic fallback traces because online vLLM/HF draft attention extraction is not available in the sandboxed process.
Dense target labels in these traces are synthetic audit labels and are not used by the online planner.

Trace files:
- calibration: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/pams_verify/experiments/02_trace_collection/raw/trace_calibration.jsonl` rows=9744
- validation: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/pams_verify/experiments/02_trace_collection/raw/trace_validation.jsonl` rows=7296
- test: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/pams_verify/experiments/02_trace_collection/raw/trace_test.jsonl` rows=7536
