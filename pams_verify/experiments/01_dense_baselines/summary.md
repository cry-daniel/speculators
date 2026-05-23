# Live vLLM Baseline Smoke

This smoke run starts a real vLLM OpenAI-compatible server and measures it with `vllm bench serve` on a short random workload.

- Target model: `Qwen/Qwen3-8B`
- max_model_len: `2048`
- max_num_seqs: `4`
- prompts: `4`
- completed methods: `['dense_no_spec', 'ngram_4', 'ngram_8']`
- failed methods: `[]`

## dense_no_spec

- request throughput: `5.288704850532784`
- output throughput: `84.61927760852454`
- mean TTFT ms: `19.75426683202386`
- mean ITL ms: `10.577683373412583`
- result file: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/pams_verify/experiments/01_dense_baselines/raw/live_dense_no_spec/dense_no_spec.json`

## ngram_4

- request throughput: `5.082521450706967`
- output throughput: `81.32034321131147`
- mean TTFT ms: `19.940570055041462`
- mean ITL ms: `11.045143386581913`
- result file: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/pams_verify/experiments/01_dense_baselines/raw/live_ngram_4/ngram_4.json`

## ngram_8

- request throughput: `5.080696788943743`
- output throughput: `81.29114862309989`
- mean TTFT ms: `19.41956125665456`
- mean ITL ms: `11.082133998570498`
- result file: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/pams_verify/experiments/01_dense_baselines/raw/live_ngram_8/ngram_8.json`
