# Acceptance Jitter Run

- date: 2026-05-26T12:07:00+08:00
- output root: ./results/accepted_count_jitter_20260526_111200
- work root: ./temp/accepted_count_jitter_work_20260526_111200
- cases: qwen3_8b:peagle qwen3_8b:eagle3 llama3_1_8b:eagle3
- workloads: math mtbench synthetic_1000x1000
- num_spec_tokens: 8 12 16
- math prompts: 80
- MTBench prompts: 80
- synthetic prompts: 8
- synthetic prompt/output tokens: 1000/1000
- request concurrency: 1
- max num batched tokens: 8192
- temperature/top_p/top_k: 0/1.0/0
- qwen base: Qwen/Qwen3-8B
- llama base: meta-llama/Llama-3.1-8B-Instruct
- qwen eagle3 speculator: /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/qwen3-8b-eagle3-speculator
- qwen peagle speculator: /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/qwen3-8b-peagle-speculator
- llama eagle3 speculator: /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/llama-3.1-8b-eagle3-speculator
- git commit: 5016fe1c4bf9c906b2499ddfff48396dc9464b20
- git diff summary:
   AGENTS.md                                          | 70 +++++++++++++++++-
   .../scripts/send_speclink_confidence_requests.py   | 83 ++++++++++++++++++++--
   2 files changed, 146 insertions(+), 7 deletions(-)
