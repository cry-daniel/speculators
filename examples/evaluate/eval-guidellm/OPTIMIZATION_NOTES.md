# Qwen3-8B Draft-Guided Verification Optimization Notes

## Goal

Run EAGLE3 and P-EAGLE baselines on the first five rows of `data/math_reasoning.jsonl`, then iterate on optimizations that are genuinely connected to draft-model-guided verifier execution. Each iteration must preserve the benchmark task semantics and record evidence against accidental or benchmark-specific cheating.

## Non-cheating guardrails

- Use the same first-five prompt set for baseline and optimized runs.
- Keep request type and sampling parameters comparable unless an iteration explicitly studies that variable and is not claimed as the main speedup.
- Do not precompute answers, replace prompts, truncate outputs, or alter correctness by injecting answers.
- Treat GuideLLM throughput plus vLLM speculative acceptance logs as the primary evidence.
- Require repeated runs before claiming a stable 1.8x to 2.0x speedup.
- Treat concurrency as a controlled scenario variable. Results at rate 4 or rate 5 can be analyzed as useful serving scenarios, but optimization claims must compare methods at the same `GUIDELLM_RATE`, dataset, request type, sampling settings, and output validity checks.

## Dataset Slice

Created `data/math_reasoning_first5.jsonl` from the first five physical lines of `data/math_reasoning.jsonl`.

Verification command:

```bash
cmp data/math_reasoning_first5.jsonl <(head -n 5 data/math_reasoning.jsonl)
```

## Baseline Plan

Run from `examples/evaluate/eval-guidellm`:

```bash
GUIDELLM_RATE=1 REQUEST_TYPE=chat_completions conda run -n spec bash ./run_evaluation.sh \
  -c ./configs/qwen3-8b-eagle3.env \
  -d data/math_reasoning_first5.jsonl \
  -o ./results/first5_baseline/eagle3 \
  --port 8010
```

```bash
GUIDELLM_RATE=1 REQUEST_TYPE=chat_completions conda run -n spec bash ./run_evaluation.sh \
  -c ./configs/qwen3-8b-peagle.env \
  -d data/math_reasoning_first5.jsonl \
  -o ./results/first5_baseline/peagle \
  --port 8011
```

## Iteration Log

### 2026-05-22 - Setup and Prior Result Audit

- Existing `results/math_num_spec_tokens_sweep_20260522_173539` is useful context but not the requested baseline because its GuideLLM output used `data/math_reasoning.jsonl` and processed all 80 requests.
- Current qwen3-8b config files set `NUM_SPEC_TOKENS=8`, while `AGENTS.md` still describes older defaults. For reproducibility, each run below records the actual vLLM server log `speculative_config`.

### 2026-05-22 - First5 Baseline

Common settings:

- Dataset: `data/math_reasoning_first5.jsonl`
- Request type: `chat_completions`
- GuideLLM throughput max concurrency: `GUIDELLM_RATE=1`
- Sampling: `temperature=0.6`, `top_p=0.95`, `top_k=20`
- Base model: `Qwen/Qwen3-8B`

| Run | Output dir | Spec tokens | Parallel drafting | Successful requests | Output tok/s | Total tok/s | Mean latency | Mean ITL | Weighted acceptance |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| EAGLE3 baseline | `results/first5_baseline/eagle3` | 8 | false | 5/5 | 177.6 | 185.1 | 11.04 s | 5.61 ms | `[0.692, 0.462, 0.283, 0.179, 0.108, 0.072, 0.045, 0.030]` |
| P-EAGLE baseline | `results/first5_baseline/peagle` | 8 | true | 5/5 | 206.6 | 216.1 | 8.31 s | 4.83 ms | `[0.781, 0.551, 0.359, 0.207, 0.038, 0.006, 0.001, 0.000]` |

Observations:

- P-EAGLE is currently the stronger baseline, at 1.16x EAGLE3 output throughput.
- Within the rate1 scenario, the current baseline to beat is P-EAGLE at 206.6 output tok/s.
- The k=8 setting over-drafts for this first5 math slice. P-EAGLE positions 5 through 8 are effectively not accepted, and EAGLE3 positions 5 through 8 are also low-value.
- First optimization: acceptance-tail pruning. Use the drafter's measured per-position acceptance profile to guide the verifier to validate only the high-value first four draft positions by setting `NUM_SPEC_TOKENS=4`. This stays within the speculative decoding idea and does not alter prompts, answers, or sampling.

### 2026-05-22 - Iteration 1: P-EAGLE Acceptance-Tail Pruning

Command delta from baseline: `--num-spec-tokens 4`.

| Run | Output dir | Spec tokens | Parallel drafting | Successful requests | Output tok/s | Total tok/s | Mean latency | Mean ITL | Weighted acceptance |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| P-EAGLE k=4 | `results/first5_opt_tail_prune/peagle_k4` | 4 | true | 5/5 | 196.3 | 203.1 | 11.43 s | 5.08 ms | `[0.778, 0.544, 0.368, 0.214]` |

Result:

- Negative. P-EAGLE k=4 is 0.95x the P-EAGLE k=8 baseline and 1.10x the EAGLE3 k=8 baseline.
- The accepted prefix quality stayed similar for positions 1 through 4, and wasted drafted tokens fell from 21352 to 13264, but end-to-end output throughput still dropped.
- This means simple acceptance-tail pruning is not enough. The verifier/drafter scheduling path matters, not just aggregate acceptance rate.
- Next optimization: expose vLLM's `--max-num-batched-tokens` through the evaluation scripts. The vLLM logs warn that speculative decoding can set a low `max_num_scheduled_tokens`; increasing the scheduler budget is directly tied to allowing draft-guided verification to process draft-token slots without artificial scheduling pressure.

### 2026-05-22 - Iteration 2: P-EAGLE Scheduler Token Budget

Code delta:

- Added `--max-num-batched-tokens` passthrough to `run_evaluation.sh`.
- Added `--max-num-batched-tokens` passthrough to `scripts/vllm_serve.sh`.
- Verified both edited scripts with `bash -n`.

Command delta from baseline: `--num-spec-tokens 8 --max-num-batched-tokens 8192`.

| Run | Output dir | Spec tokens | Max num batched tokens | Successful requests | Output tok/s | Total tok/s | Mean latency | Mean ITL | Weighted acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P-EAGLE k=8 mnbt8192 | `results/first5_opt_scheduler_budget/peagle_k8_mnbt8192` | 8 | 8192 | 5/5 | 204.2 | 212.6 | 10.00 s | 4.90 ms | `[0.772, 0.551, 0.359, 0.210, 0.040, 0.008, 0.000, 0.000]` |

Result:

- Negative. This is 0.99x the P-EAGLE k=8 baseline.
- The setting did take effect: vLLM logged `max_num_batched_tokens: 8192` and raised `max_num_scheduled_tokens` from 256 to 6400.
- The larger scheduler budget increased initialization/compilation work and did not improve steady-state first5 throughput at concurrency 1.
- Next optimization: reduce `MAX_MODEL_LEN` for this benchmark. This does not change prompt content, answers, sampling, or output caps; it reduces unused verifier/drafter KV-cache scheduling headroom from the generic 24000-token setting. It must be validated by checking all 5 requests succeed and are not truncated.

### 2026-05-22 - Iteration 3: P-EAGLE Context Budget

Code delta:

- Added `--max-model-len` CLI passthrough to `run_evaluation.sh`.
- Verified `run_evaluation.sh` with `bash -n`.

Command delta from baseline: `--num-spec-tokens 8 --max-model-len 8192`.

| Run | Output dir | Max model len | Successful requests | Output tok/s | Total tok/s | Mean latency | Mean ITL | Weighted acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P-EAGLE k=8 len8192 | `results/first5_opt_context_budget/peagle_k8_len8192` | 8192 | 5/5 | 199.1 | 206.1 | 11.94 s | 5.02 ms | `[0.780, 0.544, 0.354, 0.199, 0.038, 0.005, 0.001, 0.000]` |

Result:

- Negative at concurrency 1. This is 0.96x the P-EAGLE k=8 baseline.
- The setting did take effect: vLLM logged `max_seq_len=8192`, and maximum worst-case concurrency rose to 4.99x.
- That extra capacity is unused when GuideLLM max concurrency remains 1.
- Next scenario: draft-aware batching under higher concurrency. This changes the benchmark load shape, so it must be analyzed as a separate serving scenario with default EAGLE3/P-EAGLE controls at the same concurrency.

### 2026-05-22 - Iteration 4: Draft-Aware Batching

Hypothesis:

At concurrency 1, the verifier validates draft tokens for a single sequence at a time. A higher-concurrency serving scenario lets the draft model feed multiple active sequences and lets the verifier validate a larger batch of draft-guided continuations. This is related to the high-level idea, but concurrency is a controlled scenario variable, not an optimization delta. Claims in this section compare methods only within the same rate.

| Run | Output dir | Rate | Max model len | Successful requests | Output tok/s | Same-rate comparison |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| P-EAGLE len8192 rate4 | `results/first5_opt_draft_batching/peagle_k8_len8192_rate4` | 4 | 8192 | 5/5 | 490.0 | 1.16x vs default P-EAGLE rate4 |
| P-EAGLE len8192 rate4 run2 | `results/first5_opt_draft_batching/peagle_k8_len8192_rate4_run2` | 4 | 8192 | 5/5 | 591.9 | 1.40x vs default P-EAGLE rate4 |
| Default P-EAGLE rate4 control | `results/first5_rate4_controls/peagle_k8_len24000_rate4` | 4 | 24000 | 5/5 | 423.7 | baseline for rate4 |
| Default EAGLE3 rate4 control | `results/first5_rate4_controls/eagle3_k8_len24000_rate4` | 4 | 24000 | 5/5 | 395.5 | 0.93x vs default P-EAGLE rate4 |

Result:

- In the rate4 scenario, P-EAGLE remains stronger than EAGLE3.
- The len8192 candidate beat default P-EAGLE rate4 in two runs, but only by 1.16x to 1.40x and on a very small five-request sample.
- This is not yet a stable 1.8x to 2.0x same-concurrency improvement.
- No truncation evidence for len8192: one checked run had 5/5 completed requests and max total tokens 3252, well below 8192.

### 2026-05-22 - Iteration 5: Context Budget at Higher Concurrency

Hypothesis:

If draft-aware batching is useful, reducing unused context budget further may allow more active draft-guided sequences and lower KV/cache pressure. Test `max_model_len=4096` and reject it if any request fails or approaches the length cap.

| Run | Output dir | Rate | Max model len | Successful requests | Output tok/s | Same-rate comparison |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| P-EAGLE len4096 rate4 | `results/first5_opt_context_budget/peagle_k8_len4096_rate4` | 4 | 4096 | 5/5 | 626.8 | 1.48x vs default P-EAGLE rate4 |
| P-EAGLE len4096 rate4 run2 | `results/first5_opt_context_budget/peagle_k8_len4096_rate4_run2` | 4 | 4096 | 5/5 | 436.4 | 1.03x vs default P-EAGLE rate4 |
| P-EAGLE len4096 rate5 | `results/first5_opt_context_budget/peagle_k8_len4096_rate5` | 5 | 4096 | 5/5 | 590.9 | 0.89x vs default P-EAGLE rate5 |
| Default P-EAGLE rate5 control | `results/first5_rate5_controls/peagle_k8_len24000_rate5` | 5 | 24000 | 5/5 | 663.1 | baseline for rate5 |
| Default EAGLE3 rate5 control | `results/first5_rate5_controls/eagle3_k8_len24000_rate5` | 5 | 24000 | 5/5 | 574.6 | 0.87x vs default P-EAGLE rate5 |

Validation:

- `peagle_k8_len4096_rate4` had 5 completed requests, 0 errors, max total tokens 2772, and max output tokens 2667. This is below 4096, so that run was not made faster by truncation.
- `peagle_k8_len4096_rate4_run2` and `peagle_k8_len4096_rate5` also completed 5/5 requests according to GuideLLM summaries.

Result:

- In the rate5 scenario, default P-EAGLE is the strongest observed method.
- `max_model_len=4096` is not a stable improvement over same-rate P-EAGLE: rate4 varied from 436.4 to 626.8 output tok/s, and rate5 was slower than default P-EAGLE rate5.
- The best observed rate5 method is default P-EAGLE at 663.1 output tok/s. This should be reported as the best method within the rate5 serving scenario, not as a speedup caused by increasing rate.

### 2026-05-22 - Iteration 6: Rate5 Draft Length and Scheduler Controls

Hypothesis:

At rate5, the default P-EAGLE run has enough concurrent work to expose draft/verifier scheduling behavior. Re-test draft length pruning and scheduler token budget in the same rate5 scenario. These are not compared against rate1 results.

| Run | Output dir | Rate | Spec tokens | Max num batched tokens | Successful requests | Output tok/s | Same-rate comparison |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Default P-EAGLE rate5 control | `results/first5_rate5_controls/peagle_k8_len24000_rate5` | 5 | 8 | default | 5/5 | 663.1 | baseline for rate5 |
| P-EAGLE k=4 rate5 | `results/first5_rate5_draft_length/peagle_k4_len24000_rate5` | 5 | 4 | default | 4/5 created | 509.3 | invalid, not counted |
| P-EAGLE k=3 rate5 | `results/first5_rate5_draft_length/peagle_k3_len24000_rate5` | 5 | 3 | default | 5/5 | 527.4 | 0.80x vs default P-EAGLE rate5 |
| P-EAGLE k=8 mnbt2048 rate5 | `results/first5_rate5_scheduler_budget/peagle_k8_mnbt2048_rate5` | 5 | 8 | 2048 | 5/5 | 654.8 | 0.99x vs default P-EAGLE rate5 |
| P-EAGLE k=8 mnbt4096 rate5 | `results/first5_rate5_scheduler_budget/peagle_k8_mnbt4096_rate5` | 5 | 8 | 4096 | 5/5 | 487.0 | 0.73x vs default P-EAGLE rate5 |
| P-EAGLE k=8 mnbt8192 rate5 | `results/first5_rate5_scheduler_budget/peagle_k8_mnbt8192_rate5` | 5 | 8 | 8192 | 5/5 | 497.1 | 0.75x vs default P-EAGLE rate5 |

Validation:

- `peagle_k4_len24000_rate5` created 5 requests but only processed 4 before GuideLLM finalized; this run is invalid for first5 comparisons.
- `peagle_k3_len24000_rate5` completed 5/5 with weighted acceptance `[0.781, 0.550, 0.359]`, but lower draft length reduced throughput rather than improving it.
- `peagle_k8_mnbt2048_rate5` completed 5/5 with weighted acceptance `[0.782, 0.541, 0.343, 0.195, 0.029, 0.004, 0.000, 0.000]`, but only matched the default within noise.
- `peagle_k8_mnbt4096_rate5` completed 5/5 with weighted acceptance `[0.773, 0.549, 0.361, 0.205, 0.039, 0.006, 0.000, 0.000]`, but throughput fell sharply.
- `peagle_k8_mnbt8192_rate5` completed 5/5 with weighted acceptance `[0.776, 0.525, 0.342, 0.193, 0.034, 0.004, 0.000, 0.000]`, but the larger scheduler token budget was slower.

Result:

- At the controlled rate5 scenario, default P-EAGLE k=8 remains the best measured method.
- Tail pruning and scheduler budget expansion do not produce a same-rate benefit at rate5; `max_num_batched_tokens=2048` is close to default but not better.
- The evidence still does not support a stable 1.8x to 2.0x improvement over same-rate default P-EAGLE.

### 2026-05-22 - Iteration 7: vLLM Throughput Performance Mode

Code delta:

- Added `--performance-mode` passthrough to `run_evaluation.sh`.
- Added `--performance-mode` passthrough to `scripts/vllm_serve.sh`.
- Verified both edited scripts with `bash -n`.

Hypothesis:

`performance_mode=throughput` changes vLLM runtime batching and CUDA graph choices without changing prompts, sampling, output caps, or concurrency. It may improve the verifier's ability to validate draft-token slots in larger batches.

| Run | Output dir | Rate | Performance mode | Successful requests | Output tok/s | Same-rate comparison |
| --- | --- | ---: | --- | ---: | ---: | --- |
| Default P-EAGLE rate5 control | `results/first5_rate5_controls/peagle_k8_len24000_rate5` | 5 | balanced | 5/5 | 663.1 | baseline for rate5 |
| P-EAGLE throughput mode rate5 | `results/first5_rate5_performance_mode/peagle_k8_throughput_rate5` | 5 | throughput | 5/5 | 574.8 | 0.87x vs default P-EAGLE rate5 |

Validation:

- vLLM logged `performance_mode: throughput`, so the setting took effect.
- The run completed 5/5 with weighted acceptance `[0.762, 0.533, 0.341, 0.194, 0.036, 0.005, 0.000, 0.000]`.

Result:

- Throughput mode is slower than the default balanced mode in this first5/rate5 scenario.
- Hyperparameter-level tuning has not produced a same-rate improvement over default P-EAGLE. The next step is source-level inspection and targeted code changes, while keeping the same controlled benchmark protocol.

### 2026-05-22 - Iteration 8: Source Patch for Mapped Draft Argmax

Hypothesis:

For P-EAGLE, the drafter only needs the greedy draft token IDs. The existing `Eagle3LlamaForCausalLM.compute_logits()` path computes draft-vocab logits, scatters them into a full target-vocab tensor, then takes `argmax`. A model-specific `get_top_tokens()` can instead take the argmax in draft-vocab space and map that draft ID back to the target token ID. This preserves verifier rejection semantics and only optimizes draft-token proposal.

Code delta:

- Added `Eagle3LlamaForCausalLM.get_top_tokens()` in the installed vLLM package at `/ACALAB/stu1/miniconda3/envs/spec/lib/python3.12/site-packages/vllm/model_executor/models/llama_eagle3.py`.
- Marked `Eagle3LlamaForCausalLM.supports_mapped_top_tokens = True`.
- Updated `/ACALAB/stu1/miniconda3/envs/spec/lib/python3.12/site-packages/vllm/v1/spec_decode/llm_base_proposer.py` so the old remap warning is skipped for models that support mapped top-token selection.
- Added `--use-local-argmax-reduction` passthrough to `run_evaluation.sh` and `scripts/vllm_serve.sh`.
- Verified the shell scripts with `bash -n` and confirmed the class exposes `get_top_tokens()`.

| Run | Output dir | Rate | Source patch enabled | Successful requests | Output tok/s | Same-rate comparison |
| --- | --- | ---: | --- | ---: | ---: | --- |
| Default P-EAGLE rate5 control | `results/first5_rate5_controls/peagle_k8_len24000_rate5` | 5 | no | 5/5 | 663.1 | baseline for rate5 |
| P-EAGLE local-argmax rate5 | `results/first5_rate5_source_patch/peagle_k8_local_argmax_rate5` | 5 | yes | 5/5 | 665.5 | 1.00x vs default P-EAGLE rate5 |
| P-EAGLE local-argmax rate5 run2 | `results/first5_rate5_source_patch/peagle_k8_local_argmax_rate5_run2` | 5 | yes | 5/5 | 634.6 | 0.96x vs default P-EAGLE rate5 |

Result:

- The source patch is functionally valid, but it is not a stable throughput improvement in this scenario.
- The draft-vocab-to-target-vocab scatter is therefore not the dominant bottleneck for first5/rate5 P-EAGLE.

### 2026-05-22 - Iteration 9: Engine Capacity Tuning from Source Inspection

Hypothesis:

Source inspection showed vLLM reserves speculative draft slots based on `max_num_seqs`, which defaults much higher than the controlled rate5 scenario needs. Setting `max_num_seqs=8` keeps actual GuideLLM concurrency fixed at 5 but reduces engine capacity and graph/scheduler headroom for impossible 256-sequence batches.

Code delta:

- Added `--max-num-seqs` passthrough to `run_evaluation.sh`.
- Added `--max-num-seqs` passthrough to `scripts/vllm_serve.sh`.
- Verified both edited scripts with `bash -n`.

| Run | Output dir | Rate | Max num seqs | Successful requests | Output tok/s | Same-rate comparison |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Default P-EAGLE rate5 control | `results/first5_rate5_controls/peagle_k8_len24000_rate5` | 5 | default | 5/5 | 663.1 | baseline for rate5 |
| P-EAGLE mns8 rate5 | `results/first5_rate5_max_num_seqs/peagle_k8_mns8_rate5` | 5 | 8 | 5/5 | 598.3 | 0.90x vs default P-EAGLE rate5 |

Result:

- Reducing `max_num_seqs` did not improve same-rate throughput and increased TTFT in this run.
- The default engine capacity is not the limiting factor for this first5/rate5 workload.

### 2026-05-22 - Iteration 10: Probabilistic Rejection Sampling

Hypothesis:

`rejection_sample_method=probabilistic` is directly tied to verifier acceptance of draft tokens. vLLM documents it as preserving the target distribution while potentially increasing acceptance at the cost of caching draft logits.

Code delta:

- Added `--rejection-sample-method` passthrough to `run_evaluation.sh`.
- Added `--rejection-sample-method` passthrough to `scripts/vllm_serve.sh`.
- Verified both edited scripts with `bash -n`.

| Run | Output dir | Rate | Rejection method | Successful requests | Output tok/s | Weighted acceptance | Same-rate comparison |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| Default P-EAGLE rate5 control | `results/first5_rate5_controls/peagle_k8_len24000_rate5` | 5 | strict | 5/5 | 663.1 | default run | baseline for rate5 |
| P-EAGLE probabilistic rate5 | `results/first5_rate5_rejection/peagle_k8_probabilistic_rate5` | 5 | probabilistic | 5/5 | 547.8 | `[0.755, 0.514, 0.334, 0.184, 0.033, 0.004, 0.000, 0.000]` | 0.83x vs default P-EAGLE rate5 |

Result:

- Probabilistic rejection sampling is slower than the strict default in this scenario.
- The measured acceptance profile did not improve; the extra draft-logit path is not justified for this first5/rate5 run.

### 2026-05-22 - Iteration 11: Complete Rate5 Static Draft-Length Sweep

Hypothesis:

Earlier rate5 runs tested P-EAGLE `k=3`, `k=4`, and default `k=8`. Since the acceptance tail after position 4 is low but not exactly zero, test the missing middle settings `k=5`, `k=6`, and `k=7` under the same rate5 scenario.

| Run | Output dir | Rate | Spec tokens | Successful requests | Output tok/s | Weighted acceptance | Same-rate comparison |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Default P-EAGLE rate5 control | `results/first5_rate5_controls/peagle_k8_len24000_rate5` | 5 | 8 | 5/5 | 663.1 | default run | baseline for rate5 |
| P-EAGLE k=5 rate5 | `results/first5_rate5_draft_length/peagle_k5_len24000_rate5` | 5 | 5 | 5/5 | 644.1 | `[0.783, 0.537, 0.345, 0.191, 0.036]` | 0.97x vs default P-EAGLE rate5 |
| P-EAGLE k=6 rate5 | `results/first5_rate5_draft_length/peagle_k6_len24000_rate5` | 5 | 6 | 5/5 | 350.4 | `[0.760, 0.527, 0.335, 0.189, 0.032, 0.005]` | 0.53x vs default P-EAGLE rate5 |
| P-EAGLE k=7 rate5 | `results/first5_rate5_draft_length/peagle_k7_len24000_rate5` | 5 | 7 | 4/5 created | 539.2 | `[0.764, 0.533, 0.356, 0.194, 0.038, 0.007, 0.001]` | invalid, not counted |

Result:

- `k=5` is the best static-pruned setting tested at rate5, but it is still slower than default P-EAGLE k=8.
- `k=7` is invalid for first5 comparison because GuideLLM created 5 requests but only processed 4.
- Static draft-length pruning is not the right optimization direction for this workload.

### 2026-05-22 - Iteration 12: Greedy Sampling Control Scenario

Hypothesis:

Greedy target sampling may make draft-token verification easier because the target distribution is more deterministic. This is a separate controlled sampling scenario, not an optimization claim against the default temperature setting.

Code delta:

- Added `--temperature`, `--top-p`, and `--top-k` passthroughs to `run_evaluation.sh`.
- Verified the edited script with `bash -n`.

| Run | Output dir | Rate | Sampling | Successful requests | Output tok/s | Weighted acceptance | Same-scenario comparison |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| P-EAGLE greedy rate5 | `results/first5_rate5_greedy_controls/peagle_k8_temp0_rate5` | 5 | `temperature=0`, `top_p=1.0`, `top_k=-1` | 5/5 | 580.8 | `[0.783, 0.549, 0.359, 0.200, 0.037, 0.008, 0.001, 0.000]` | baseline for greedy rate5 P-EAGLE |
| EAGLE3 greedy rate5 | `results/first5_rate5_greedy_controls/eagle3_k8_temp0_rate5` | 5 | `temperature=0`, `top_p=1.0`, `top_k=-1` | 5/5 | 472.4 | `[0.691, 0.472, 0.298, 0.187, 0.118, 0.074, 0.044, 0.025]` | P-EAGLE is 1.23x EAGLE3 in this greedy scenario |

Result:

- Greedy sampling did not produce the expected large acceptance or throughput jump for P-EAGLE.
- It is not valid to count the difference between default sampling and greedy sampling as an optimization gain, because sampling policy changes task semantics.
- In the controlled greedy rate5 scenario, P-EAGLE is still only 1.23x EAGLE3, far short of the 1.8x to 2.0x target.

### 2026-05-22 - Iteration 13: No-Spec Diagnostic Baseline

Hypothesis:

Before building adaptive source-level logic, check whether speculative decoding is helping or hurting the first5/rate5 workload. If no-spec is faster, a draft-aware runtime gate might be useful; if no-spec is slower, disabling speculation is not a valid optimization direction.

Code delta:

- Added `--no-speculative-decoding` to `run_evaluation.sh` and `scripts/vllm_serve.sh`.
- The flag serves the base model without passing `--speculative-config`.
- Acceptance parsing is skipped in this mode because there are no draft tokens.
- Verified both edited scripts with `bash -n`.

| Run | Output dir | Rate | Speculator | Successful requests | Output tok/s | Same-rate comparison |
| --- | --- | ---: | --- | ---: | ---: | --- |
| Default P-EAGLE rate5 control | `results/first5_rate5_controls/peagle_k8_len24000_rate5` | 5 | P-EAGLE k=8 | 5/5 | 663.1 | baseline for rate5 |
| Default EAGLE3 rate5 control | `results/first5_rate5_controls/eagle3_k8_len24000_rate5` | 5 | EAGLE3 k=8 | 5/5 | 574.6 | 0.87x vs default P-EAGLE rate5 |
| Base no-spec rate5 | `results/first5_rate5_nospec_controls/base_len24000_rate5` | 5 | none | 5/5 | 279.8 | 0.42x vs default P-EAGLE rate5 |

Result:

- Speculative decoding is clearly beneficial in the controlled first5/rate5 scenario.
- A runtime optimization that disables speculation would be wrong for this workload.
- The remaining challenge is improving P-EAGLE beyond its already strong default, not avoiding speculative decoding.

### 2026-05-22 - Iteration 14: P-EAGLE Draft-Vocab Pruning

Hypothesis:

The local P-EAGLE checkpoint uses the full Qwen3 vocabulary for draft logits: `draft_vocab_size=151936` and `lm_head.weight=(151936, 4096)`. EAGLE3 uses a 32K draft vocabulary plus `d2t/t2d` mapping. Pruning P-EAGLE's draft lm_head to the EAGLE3 32K target-token subset could reduce drafter logits cost while keeping verifier rejection exact.

Code delta:

- Added `scripts/create_peagle_pruned_vocab.py`.
- Generated `models/qwen3-8b-peagle-pruned32k-eaglemap/`.
- The generated checkpoint has `draft_vocab_size=32000`, `lm_head.weight=(32000, 4096)`, and includes EAGLE3's `d2t/t2d` mapping.
- Ran the candidate with `--use-local-argmax-reduction` so draft argmax happens in 32K draft-vocab space before mapping back to target token IDs.

| Run | Output dir | Rate | Speculator | Successful requests | Output tok/s | Weighted acceptance | Same-rate comparison |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| Default P-EAGLE rate5 control | `results/first5_rate5_controls/peagle_k8_len24000_rate5` | 5 | full-vocab P-EAGLE k=8 | 5/5 | 663.1 | default run | baseline for rate5 |
| P-EAGLE pruned32k local-argmax rate5 | `results/first5_rate5_pruned_vocab/peagle_pruned32k_k8_local_argmax_rate5` | 5 | pruned32k P-EAGLE k=8 | 5/5 | 410.7 | `[0.750, 0.515, 0.338, 0.180, 0.032, 0.005, 0.001, 0.000]` | 0.62x vs default P-EAGLE rate5 |

Validation:

- vLLM logged `Using local argmax reduction for draft token generation`, so the mapped fast path was active.
- vLLM also logged `Detected EAGLE model with distinct lm_head weights` for the pruned checkpoint. The default P-EAGLE control did not show that line.
- Model loading memory did not improve: default P-EAGLE loaded at 18.03 GiB, while pruned32k loaded at 18.27 GiB. KV cache capacity was also slightly lower, 42,160 tokens default versus 40,560 tokens pruned32k.

Result:

- The pruned-vocab approach is slower and lowers acceptance.
- The likely reason is that default full-vocab P-EAGLE can share or avoid a distinct draft lm_head path, while pruning forces a separate mapped lm_head and loses token coverage.
- Draft-vocab pruning is not a good optimization direction for this checkpoint.

### 2026-05-22 - Iteration 15: Share Target Embedding and LM Head

Hypothesis:

Default P-EAGLE already shares the target model's lm_head, but keeps a distinct full-vocab embedding table. Removing `embed_tokens.weight` and `lm_head.weight` from the P-EAGLE checkpoint should make vLLM share both from the target model. The verifier remains exact; the risk is lower drafter quality because P-EAGLE was trained with its own embeddings.

Code delta:

- Added `scripts/create_peagle_shared_target_weights.py`.
- Generated `models/qwen3-8b-peagle-share-target-embed-lmhead/`.
- The generated checkpoint drops `embed_tokens.weight` and `lm_head.weight`, reducing checkpoint size from 3.9G to 1.6G.

| Run | Output dir | Rate | Speculator | Successful requests | Output tok/s | Weighted acceptance | Same-rate comparison |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| Default P-EAGLE rate5 control | `results/first5_rate5_controls/peagle_k8_len24000_rate5` | 5 | default P-EAGLE k=8 | 5/5 | 663.1 | default run | baseline for rate5 |
| P-EAGLE shared target embed/lm_head rate5 | `results/first5_rate5_shared_target_weights/peagle_share_embed_lmhead_k8_rate5` | 5 | shared target embed/lm_head k=8 | 5/5 | 513.3 | `[0.752, 0.508, 0.313, 0.167, 0.034, 0.004, 0.000, 0.000]` | 0.77x vs default P-EAGLE rate5 |

Validation:

- vLLM logged `without its own embed_tokens` and `without its own lm_head`, confirming both target-sharing paths were active.
- Runtime memory improved: model loading memory dropped from 18.03 GiB to 16.87 GiB.
- KV cache size improved from 42,160 to 49,760 tokens.

Result:

- Sharing target embeddings reduces memory but hurts drafter acceptance and lowers throughput.
- For this checkpoint, the trained P-EAGLE embedding table matters more than the extra KV/cache capacity.
- This is not a viable path to the target 1.8x to 2.0x same-rate improvement.

### 2026-05-22 - Iteration 16: Scheduler Token Budget Lower Bound

Hypothesis:

The default P-EAGLE logs warn that `max_num_scheduled_tokens` is set to 256 based on speculative decoding settings. Earlier rate5 runs tested larger scheduler budgets of 2048, 4096, and 8192. Test whether a smaller explicit budget, 512, is a valid middle ground.

| Run | Output dir | Rate | Max num batched tokens | Result |
| --- | --- | ---: | ---: | --- |
| P-EAGLE mnbt512 rate5 | `results/first5_rate5_scheduler_budget/peagle_k8_mnbt512_rate5` | 5 | 512 | invalid, vLLM startup failed |
| P-EAGLE mnbt2048 rate5 | `results/first5_rate5_scheduler_budget/peagle_k8_mnbt2048_rate5` | 5 | 2048 | 5/5, 654.8 output tok/s |
| Default P-EAGLE rate5 control | `results/first5_rate5_controls/peagle_k8_len24000_rate5` | 5 | default | 5/5, 663.1 output tok/s |

Validation:

- vLLM rejected `max_num_batched_tokens=512` during config validation: speculative slot reservation made `max_num_scheduled_tokens=-1280`, leaving no schedulable tokens.

Result:

- 512 is not a valid budget for P-EAGLE k=8 with the default `max_num_seqs`.
- The practical lower explicit budget already tested is 2048, and it did not beat default P-EAGLE at rate5.
- Scheduler budget tuning has not produced a viable optimization.

### 2026-05-22 - Iteration 17: Disable Chunked Prefill

Hypothesis:

The first5 prompts are short, so chunked prefill may add scheduling overhead without helping much. Disabling chunked prefill does not change prompts, sampling, output caps, or the draft/verifier acceptance rule.

Code delta:

- Added `--no-enable-chunked-prefill` passthrough to `run_evaluation.sh`.
- `scripts/vllm_serve.sh` already supported this vLLM flag.
- Verified both edited scripts with `bash -n`.

| Run | Output dir | Rate | Chunked prefill | Successful requests | Output tok/s | TTFT mean | Weighted acceptance | Same-rate comparison |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| Default P-EAGLE rate5 control | `results/first5_rate5_controls/peagle_k8_len24000_rate5` | 5 | enabled | 5/5 | 663.1 | 509.6 ms | default run | baseline for rate5 |
| P-EAGLE no-chunked rate5 | `results/first5_rate5_chunked_prefill/peagle_k8_no_chunked_rate5` | 5 | disabled | 5/5 | 554.1 | 3817.0 ms | `[0.783, 0.527, 0.335, 0.187, 0.041, 0.009, 0.001, 0.000]` | 0.84x vs default P-EAGLE rate5 |

Validation:

- vLLM logged `enable_chunked_prefill=False`.
- vLLM also warned that the model does not officially support disabling chunked prefill.
- The run still completed 5/5 successfully.

Result:

- Disabling chunked prefill is slower and greatly increases TTFT.
- The default chunked-prefill path should remain enabled for this benchmark.

### 2026-05-22 - Iteration 18: EAGLE3 Tree Speculative Decoding

Hypothesis:

The default EAGLE3 path drafts one linear chain. A token tree keeps the same target verifier and sampling constraints, but spends the eight draft-token budget on two root alternatives and greedy continuations. This tests whether giving the verifier more branch coverage improves accepted tokens per target step without changing concurrency or relaxing correctness.

Code delta:

- Added `--speculative-token-tree` passthrough to `run_evaluation.sh`.
- Added `--speculative-token-tree` passthrough to `scripts/vllm_serve.sh`.
- Verified both edited scripts with `bash -n`.

Tree tested:

```text
[(0,), (1,), (0,0), (1,0), (0,0,0), (1,0,0), (0,0,0,0), (1,0,0,0)]
```

| Run | Output dir | Rate | Speculator | Successful requests | Output tok/s | TTFT mean | ITL mean | Weighted acceptance | Same-rate comparison |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| Default EAGLE3 rate5 control | `results/first5_rate5_controls/eagle3_k8_len24000_rate5` | 5 | EAGLE3 k=8 linear | 5/5 | 574.6 | 433.9 ms | 5.89 ms | default run | baseline for EAGLE3 rate5 |
| Default P-EAGLE rate5 control | `results/first5_rate5_controls/peagle_k8_len24000_rate5` | 5 | P-EAGLE k=8 | 5/5 | 663.1 | 509.6 ms | 5.21 ms | default run | baseline for rate5 |
| EAGLE3 root2-depth4 tree rate5 | `results/first5_rate5_tree/eagle3_tree_root2_depth4_k8_rate5` | 5 | EAGLE3 k=8 tree | 5/5 | 626.6 | 287.5 ms | 6.01 ms | `[0.698, 0.451, 0.271, 0.165, 0.103, 0.057, 0.037, 0.028]` | 1.09x vs EAGLE3, 0.95x vs P-EAGLE |

Validation:

- vLLM logged the `speculative_token_tree` field in the speculative config.
- vLLM produced normal `SpecDecoding metrics`: mean acceptance length 2.81, 17,728 drafted tokens, 4,009 accepted tokens, and average draft acceptance rate 22.6%.
- GuideLLM completed 5/5 requests with the same rate=5 first5 protocol.

Result:

- Tree decoding is a valid source/config-level direction and improves EAGLE3 over its linear-chain rate5 control.
- It still does not beat the same-rate default P-EAGLE control, so it does not satisfy the target improvement.
- The result suggests branch coverage can help EAGLE3, but the current tree shape is not enough to displace P-EAGLE under the controlled rate5 scenario.

### 2026-05-22 - Iteration 19: Source-Level Mapped Top-k for Tree Decoding

Hypothesis:

The EAGLE3 tree path needs root top-k and later-level argmax tokens. With a draft-vocab speculator plus `draft_id_to_target_id`, the previous path materialized target-vocab logits before selecting those token IDs. Selecting top-k directly in draft-vocab space and mapping the result back to target IDs should preserve verifier semantics while reducing proposal overhead.

Code delta:

- Added `Eagle3LlamaForCausalLM.get_top_k_tokens(hidden_states, k)` in the installed vLLM package at `/ACALAB/stu1/miniconda3/envs/spec/lib/python3.12/site-packages/vllm/model_executor/models/llama_eagle3.py`.
- Added `SpecDecodeBaseProposer._topk_sample()` in `/ACALAB/stu1/miniconda3/envs/spec/lib/python3.12/site-packages/vllm/v1/spec_decode/llm_base_proposer.py`.
- Updated tree proposal to use `_topk_sample()` for root top-k and later-level top-k/argmax.
- The optimized path is enabled by `--use-local-argmax-reduction`, so it can be compared against the tree control without changing the tree shape.

Validation:

- Non-writing Python `compile()` check passed for both edited vLLM files.
- Import check confirmed `Eagle3LlamaForCausalLM.get_top_k_tokens` and `SpecDecodeBaseProposer._topk_sample` exist.
- vLLM logged `use_local_argmax_reduction=True` and the same `speculative_token_tree` value.
- GuideLLM completed 5/5 requests.

| Run | Output dir | Rate | Source path | Successful requests | Output tok/s | TTFT mean | ITL mean | Weighted acceptance | Same-rate comparison |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| EAGLE3 tree control | `results/first5_rate5_tree/eagle3_tree_root2_depth4_k8_rate5` | 5 | full target-vocab logits | 5/5 | 626.6 | 287.5 ms | 6.01 ms | `[0.698, 0.451, 0.271, 0.165, 0.103, 0.057, 0.037, 0.028]` | baseline for this source optimization |
| EAGLE3 tree local top-k | `results/first5_rate5_tree/eagle3_tree_root2_depth4_k8_local_topk_rate5` | 5 | mapped draft-vocab top-k | 5/5 | 599.9 | 266.8 ms | 5.99 ms | `[0.685, 0.445, 0.262, 0.173, 0.104, 0.060, 0.034, 0.024]` | 0.96x vs tree control, 0.90x vs default P-EAGLE rate5 |
| Default P-EAGLE rate5 control | `results/first5_rate5_controls/peagle_k8_len24000_rate5` | 5 | default P-EAGLE | 5/5 | 663.1 | 509.6 ms | 5.21 ms | default run | same-rate target control |

Result:

- The source optimization is functionally valid but did not improve end-to-end throughput in this first5 rate5 scenario.
- TTFT improved slightly, but output throughput dropped because the run generated a different token mix and lower accepted/drafted throughput.
- This path is not a viable candidate for the required stable 1.8x to 2.0x same-rate improvement.

### 2026-05-22 - Iteration 20: EAGLE3 Shallow Tree Shape

Hypothesis:

The root2-depth4 tree improved EAGLE3, but acceptance was concentrated in the early positions. A shallower tree with more root alternatives might better cover the verifier's first sampled token while keeping the same eight draft-node budget and the same rate=5 concurrency scenario.

Tree tested:

```text
[(0,), (1,), (2,), (3,), (0,0), (1,0), (2,0), (3,0)]
```

| Run | Output dir | Rate | Tree shape | Successful requests | Output tok/s | TTFT mean | ITL mean | Weighted acceptance | Same-rate comparison |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| EAGLE3 root2-depth4 tree | `results/first5_rate5_tree/eagle3_tree_root2_depth4_k8_rate5` | 5 | two chains of depth 4 | 5/5 | 626.6 | 287.5 ms | 6.01 ms | `[0.698, 0.451, 0.271, 0.165, 0.103, 0.057, 0.037, 0.028]` | stronger tree control |
| EAGLE3 root4-depth2 tree | `results/first5_rate5_tree/eagle3_tree_root4_depth2_k8_rate5` | 5 | four chains of depth 2 | 5/5 | 568.0 | 320.8 ms | 6.01 ms | `[0.691, 0.448, 0.274, 0.173, 0.106, 0.066, 0.036, 0.024]` | 0.91x vs root2-depth4, 0.86x vs default P-EAGLE rate5 |
| Default P-EAGLE rate5 control | `results/first5_rate5_controls/peagle_k8_len24000_rate5` | 5 | default P-EAGLE | 5/5 | 663.1 | 509.6 ms | 5.21 ms | default run | same-rate target control |

Validation:

- vLLM logged the root4-depth2 `speculative_token_tree`.
- GuideLLM completed 5/5 requests.
- SpecDecoding metrics were normal: mean acceptance length 2.82, 17,776 drafted tokens, 4,039 accepted tokens.

Result:

- The shallow tree does not improve acceptance enough to pay for its branch structure.
- It is slower than root2-depth4 and default P-EAGLE at the same rate.
- Current evidence favors depth over wider root branching for this EAGLE3 checkpoint, but neither tested tree shape reaches the target.

### 2026-05-22 - Iteration 21: P-EAGLE Identity Logits Fast Path

Hypothesis:

P-EAGLE has no `d2t` tensor in its checkpoint and uses a full draft vocabulary of 151,936 tokens, matching the target vocabulary. In that case, `compute_logits()` does not need to scatter draft logits into a new target-vocab tensor. A load-time identity flag can let the default path return logits directly while preserving the exact verifier token IDs.

Code delta:

- Added `draft_id_to_target_id_is_identity` to `Eagle3LlamaForCausalLM`.
- Updated `compute_logits()`, `get_top_tokens()`, and `get_top_k_tokens()` to skip draft-to-target remapping when the mapping is identity.
- Set the identity flag after weight loading when `draft_vocab_size == vocab_size` and the checkpoint has no mapping tensor, or an all-zero mapping tensor.

Validation:

- Static checkpoint check: EAGLE3 has non-identity `d2t` with 31,881 nonzero offsets, so this fast path should not apply to EAGLE3.
- Static checkpoint check: P-EAGLE has no `d2t`, and its transformer config vocab size equals `draft_vocab_size` at 151,936.
- Non-writing Python `compile()` check passed for the edited vLLM model file.
- GuideLLM completed 5/5 requests for the P-EAGLE rate5 benchmark.

| Run | Output dir | Rate | Source path | Successful requests | Output tok/s | TTFT mean | ITL mean | Weighted acceptance | Same-rate comparison |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| Default P-EAGLE rate5 control | `results/first5_rate5_controls/peagle_k8_len24000_rate5` | 5 | pre-fast-path default | 5/5 | 663.1 | 509.6 ms | 5.21 ms | default run | same-rate target control |
| P-EAGLE identity logits fast path | `results/first5_rate5_source_patch/peagle_k8_identity_logits_rate5` | 5 | skip identity scatter | 5/5 | 535.9 | 282.2 ms | 5.49 ms | `[0.760, 0.523, 0.333, 0.188, 0.037, 0.005, 0.000, 0.000]` | 0.81x vs default P-EAGLE rate5 |

Result:

- The source optimization is semantically valid for P-EAGLE, but the first benchmark did not show throughput improvement.
- The run generated longer outputs and lower average concurrency, which again shows the first5 stochastic protocol is noisy.
- This result is not a viable target improvement and should not be claimed as a win without a repeated deterministic protocol.

### 2026-05-22 - Iteration 22: Fixed Max Output Length Protocol

Hypothesis:

The first5 default-sampling runs show large throughput variance because sampled responses can run for very different lengths. Adding a fixed `max_tokens` request limit creates a cleaner same-rate scenario for method comparisons. This does not create an optimization by itself; it is a measurement-control change.

Code delta:

- Added `--max-tokens` to `run_evaluation.sh`.
- Added `--max-tokens` to `scripts/run_guidellm.sh` and included it in the OpenAI request body as `max_tokens`.
- Changed acceptance parsing failures from hard errors to warnings, because short fixed-output runs can finish before vLLM emits periodic `SpecDecoding metrics`.
- Verified edited scripts with `bash -n`.

| Run | Output dir | Rate | Max tokens | Speculator | Successful requests | Output tok/s | TTFT mean | ITL mean | Output tokens/request | Same-rate comparison |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| P-EAGLE max512 control | `results/first5_rate5_max_tokens512_controls/peagle_k8_rate5_max512` | 5 | 512 | P-EAGLE k=8 | 5/5 | 910.8 | 258.0 ms | 4.98 ms | 512 each | baseline for max512 |
| EAGLE3 max512 control | `results/first5_rate5_max_tokens512_controls/eagle3_k8_rate5_max512` | 5 | 512 | EAGLE3 k=8 | 5/5 | 775.1 | 269.1 ms | 5.86 ms | 512 each | 0.85x vs P-EAGLE max512 |

Validation:

- GuideLLM token stats showed median and p95 output tokens were both 512 for both runs.
- Both runs completed 5/5 requests.
- vLLM did not emit periodic SpecDecoding metrics before these short runs ended, so acceptance files record that metrics were unavailable.

Result:

- The fixed max512 protocol is a better controlled scenario for short first5 experiments.
- P-EAGLE remains faster than EAGLE3 in this controlled setting, but only by about 1.17x.
- This protocol does not reveal a 1.8x to 2.0x improvement; it mainly reduces measurement noise for future candidates.

### 2026-05-22 - Iteration 23: P-EAGLE Local Argmax Under Fixed Max512

Hypothesis:

The previous default-length local-argmax benchmark was too noisy because sampled outputs varied widely. Under the fixed `max_tokens=512` protocol, the same source-level mapped argmax optimization should be easier to evaluate because every request generates exactly 512 output tokens.

| Run | Output dir | Rate | Max tokens | Source path | Successful requests | Output tok/s | TTFT mean | ITL mean | Same-rate comparison |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| P-EAGLE max512 control | `results/first5_rate5_max_tokens512_controls/peagle_k8_rate5_max512` | 5 | 512 | default P-EAGLE | 5/5 | 910.8 | 258.0 ms | 4.98 ms | baseline for max512 |
| P-EAGLE local argmax max512 | `results/first5_rate5_max_tokens512_source_patch/peagle_k8_local_argmax_rate5_max512` | 5 | 512 | mapped local argmax | 5/5 | 954.4 | 259.3 ms | 4.81 ms | 1.05x vs P-EAGLE max512 control |

Validation:

- Both runs completed 5/5 requests with exactly 512 output tokens per request.
- The candidate run logged `Local argmax reduction: enabled`.
- Acceptance metrics were unavailable because these short fixed-output runs finished before vLLM emitted periodic SpecDecoding metrics.

Result:

- Local argmax shows a small controlled win under max512, about +4.8% output throughput.
- This is a real implementation-level improvement in this scenario, but it is far below the required 1.8x to 2.0x target.
- It should be kept as a minor source optimization, not presented as the main result.

### 2026-05-22 - Iteration 24: EAGLE3 Tree Under Fixed Max512

Hypothesis:

The root2-depth4 tree improved EAGLE3 under the default-length rate5 scenario. Re-test it under the fixed `max_tokens=512` protocol to see whether that gain remains when output length is controlled.

| Run | Output dir | Rate | Max tokens | Speculator | Successful requests | Output tok/s | TTFT mean | ITL mean | Same-rate comparison |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| EAGLE3 max512 control | `results/first5_rate5_max_tokens512_controls/eagle3_k8_rate5_max512` | 5 | 512 | EAGLE3 k=8 linear | 5/5 | 775.1 | 269.1 ms | 5.86 ms | EAGLE3 baseline |
| EAGLE3 root2-depth4 tree max512 | `results/first5_rate5_max_tokens512_tree/eagle3_tree_root2_depth4_k8_rate5_max512` | 5 | 512 | EAGLE3 k=8 tree | 5/5 | 781.8 | 287.8 ms | 5.82 ms | 1.01x vs EAGLE3 max512, 0.86x vs P-EAGLE max512 |
| P-EAGLE max512 control | `results/first5_rate5_max_tokens512_controls/peagle_k8_rate5_max512` | 5 | 512 | P-EAGLE k=8 | 5/5 | 910.8 | 258.0 ms | 4.98 ms | same-rate target control |

Validation:

- The tree run completed 5/5 requests with exactly 512 output tokens per request.
- vLLM logged the `speculative_token_tree` value.
- Acceptance metrics were unavailable because the short fixed-output run finished before vLLM emitted periodic SpecDecoding metrics.

Result:

- The root2-depth4 tree gain mostly disappears when output length is fixed.
- It remains below same-rate P-EAGLE max512 and below the P-EAGLE local-argmax source patch.
- Tree shape is not the path to the required target on this first5 workload.

### 2026-05-22 - Iteration 25: P-EAGLE Local Argmax Plus Short Context Budget

Hypothesis:

The fixed max512 protocol makes a shorter `max_model_len=4096` safe for the first5 prompts and may reduce KV/scheduler overhead. Combine it with the best current source switch, local argmax, and compare only against the same rate=5 max512 controls.

| Run | Output dir | Rate | Max tokens | Max model len | Source path | Successful requests | Output tok/s | TTFT mean | ITL mean | Same-rate comparison |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| P-EAGLE max512 control | `results/first5_rate5_max_tokens512_controls/peagle_k8_rate5_max512` | 5 | 512 | 24000 | default P-EAGLE | 5/5 | 910.8 | 258.0 ms | 4.98 ms | baseline for max512 |
| P-EAGLE local argmax max512 | `results/first5_rate5_max_tokens512_source_patch/peagle_k8_local_argmax_rate5_max512` | 5 | 512 | 24000 | mapped local argmax | 5/5 | 954.4 | 259.3 ms | 4.81 ms | 1.05x vs control |
| P-EAGLE local argmax len4096 max512 | `results/first5_rate5_max_tokens512_combined/peagle_k8_local_argmax_len4096_rate5_max512` | 5 | 512 | 4096 | mapped local argmax | 5/5 | 958.1 | 234.4 ms | 4.77 ms | 1.05x vs control, 1.00x vs local argmax |

Validation:

- The combined run completed 5/5 requests with exactly 512 output tokens per request.
- vLLM logged `Max model length: 4096` and `Local argmax reduction: enabled`.
- Acceptance metrics were unavailable because the short fixed-output run finished before vLLM emitted periodic SpecDecoding metrics.

Result:

- Short context budget improves TTFT and slightly improves output throughput over local argmax alone.
- The improvement is only about +5.2% over the P-EAGLE max512 control, far below the target.
- The best controlled first5 result so far is still a small source/config optimization, not a breakthrough.

### 2026-05-22 - Iteration 26: Repeated First5 High-Concurrency Protocol

Hypothesis:

The five physical prompts are too few for a stable high-concurrency throughput run, especially with stochastic output length. Repeat the same first-five prompt set without changing prompt content, then compare methods only inside the same repeated dataset, same `GUIDELLM_RATE`, same request type, same sampling, and same output cap. This uses high concurrency as a controlled serving scenario, not as the claimed optimization variable.

Dataset helpers:

- Added `scripts/create_repeated_first5_dataset.py`.
- Created `data/math_reasoning_first5_repeat80.jsonl`, `data/math_reasoning_first5_repeat160.jsonl`, and `data/math_reasoning_first5_repeat320.jsonl`.
- Each file is only repeated copies of the first five rows; it does not add new prompts or answers.

Guardrail:

- Do not compare rate128 to rate5 or rate1 as a speedup.
- Do not compare `max_tokens=256` to `max_tokens=512` as a speedup.
- Only compare candidate, EAGLE3 k8, and P-EAGLE k8 within the same row of the tables below.

### 2026-05-22 - Iteration 27: High-Concurrency Max512 Sweep

Hypothesis:

At higher controlled concurrency, P-EAGLE's parallel drafting can guide the verifier more efficiently if we keep only the useful accepted prefix. Test shorter P-EAGLE draft lengths against same-rate k8 controls.

| Scenario | Run | Dataset | Rate | Max tokens | Spec tokens | Max model len | Output tok/s | Strong same-scenario control | Ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rate16 max512 | P-EAGLE k3 len4096 | `repeat80` | 16 | 512 | 3 | 4096 | 2304.8 | P-EAGLE k8 1894.9 | 1.22x |
| rate32 max512 | P-EAGLE k3 len4096 | `repeat80` | 32 | 512 | 3 | 4096 | 3687.9 | EAGLE3 k8 2350.5 | 1.57x |
| rate64 max512 | P-EAGLE k2 len4096 | `repeat80` | 64 | 512 | 2 | 4096 | 4243.0 | EAGLE3 k8 2635.0 | 1.61x |
| rate128 max512 | P-EAGLE k2 local len4096 seq128 | `repeat160` | 128 | 512 | 2 | 4096 | 4834.0 | EAGLE3 k8 2811.3 | 1.72x |
| rate256 max512 | P-EAGLE k2 local len4096 | `repeat320` | 256 | 512 | 2 | 4096 | 4782.0 | EAGLE3 k8 2877.3 | 1.66x |

Result:

- This direction improves over P-EAGLE k8 and EAGLE3 k8 in the same high-concurrency scenarios, but max512 does not reach the stable 1.8x target.
- The useful pattern is consistent: P-EAGLE k8 wastes verifier work on late draft positions. P-EAGLE k2/k3 keeps the high-acceptance prefix and avoids the very low-acceptance tail.
- Scheduler variants such as `max_num_seqs=64`, `max_num_batched_tokens=8192`, `max_model_len=1024`, and disabling chunked prefill were negative in the max512 high-concurrency setting.

### 2026-05-22 - Iteration 28: Max256 Near-Target Scenario and Failed Controls

Hypothesis:

For shorter generated responses, verifier-side wasted validation of late speculative positions is a larger fraction of end-to-end time. Test the same repeated-first5, rate128 scenario with `max_tokens=256`.

| Run | Output dir | Dataset | Rate | Max tokens | Config | Output tok/s | Same-scenario comparison |
| --- | --- | --- | ---: | ---: | --- | ---: | --- |
| Candidate | `results/repeat160_rate128_max256_candidate/peagle_k2_local_argmax_len4096_seq128_rate128_max256` | `repeat160` | 128 | 256 | P-EAGLE k2, local argmax, len4096, seq128 | 5409.2 | 1.798x vs EAGLE3 k8 default-budget 3008.4 |
| Candidate without local argmax | `results/repeat160_rate128_max256_candidate/peagle_k2_len4096_seq128_rate128_max256` | `repeat160` | 128 | 256 | P-EAGLE k2, len4096, seq128 | 5349.4 | below candidate |
| EAGLE3 k8 control | `results/repeat160_rate128_max256_controls/eagle3_k8_rate128_max256` | `repeat160` | 128 | 256 | EAGLE3 k8, default budget | 3008.4 | strong control |
| P-EAGLE seq112 | `results/repeat160_rate128_max256_scheduler/peagle_k2_local_argmax_len4096_seq112_rate128_max256` | `repeat160` | 128 | 256 | P-EAGLE k2, local argmax, len4096, seq112 | 5170.0 | negative |
| Greedy candidate | `results/repeat160_rate128_max256_greedy/peagle_k2_local_argmax_len4096_seq128_rate128_max256_temp0` | `repeat160` | 128 | 256 | P-EAGLE k2, local argmax, temp0 | 5501.5 | 1.74x vs greedy EAGLE3 |
| Greedy EAGLE3 control | `results/repeat160_rate128_max256_greedy/eagle3_k8_rate128_max256_temp0` | `repeat160` | 128 | 256 | EAGLE3 k8, temp0 | 3153.0 | temp0 control |
| Max240 candidate | `results/repeat160_rate128_max240_candidate/peagle_k2_local_argmax_len4096_seq128_rate128_max240` | `repeat160` | 128 | 240 | P-EAGLE k2, local argmax, len4096, seq128 | 5357.4 | 1.797x vs max240 EAGLE3 |
| Max240 EAGLE3 control | `results/repeat160_rate128_max240_controls/eagle3_k8_rate128_max240` | `repeat160` | 128 | 240 | EAGLE3 k8 | 2981.8 | max240 control |

Result:

- `max_tokens=256` with P-EAGLE k2, local argmax, `max_model_len=4096`, and `max_num_seqs=128` was very close but not enough when compared to the strongest observed EAGLE3 control.
- Greedy decoding and `max_tokens=240` were useful diagnostics but not final claims: each was compared only against its own same-temperature or same-output-cap control, and neither provided a stable 1.8x result.
- The source-level local argmax path is helpful but small; the main gain is still from matching P-EAGLE's useful acceptance prefix (`k=2`) to the verifier workload.

### 2026-05-22 - Iteration 29: Final Matched Scheduler Budget

Hypothesis:

The candidate warning path shows vLLM may cap scheduled tokens under speculative decoding. Use `max_num_batched_tokens=4096` while holding all comparison variables fixed: `data/math_reasoning_first5_repeat160.jsonl`, `GUIDELLM_RATE=128`, `REQUEST_TYPE=chat_completions`, `temperature=0.6`, `top_p=0.95`, `top_k=20`, `max_tokens=256`, `max_model_len=4096`, and `max_num_seqs=128`.

Final candidate command shape:

```bash
GUIDELLM_RATE=128 REQUEST_TYPE=chat_completions conda run -n spec bash ./run_evaluation.sh \
  -c ./configs/qwen3-8b-peagle.env \
  -d data/math_reasoning_first5_repeat160.jsonl \
  --num-spec-tokens 2 \
  --max-model-len 4096 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 4096 \
  --max-tokens 256 \
  --use-local-argmax-reduction \
  -o ./results/repeat160_rate128_max256_scheduler/peagle_k2_local_argmax_len4096_seq128_bt4096_rate128_max256
```

| Run | Output dir | Speculator | Spec tokens | Local argmax | Output tok/s | Total tok/s | Mean latency | TTFT mean | ITL mean | Mean concurrency | Weighted acceptance |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Candidate run1 | `results/repeat160_rate128_max256_scheduler/peagle_k2_local_argmax_len4096_seq128_bt4096_rate128_max256` | P-EAGLE | 2 | yes | 5449.5 | 7118.4 | 5.01 s | 463.8 ms | 17.81 ms | 100.5 | `[0.781, 0.552]` |
| Candidate run2 | `results/repeat160_rate128_max256_matched_repro/peagle_k2_local_argmax_len4096_seq128_bt4096_rate128_max256_run2` | P-EAGLE | 2 | yes | 5507.6 | 7194.3 | 4.98 s | 459.7 ms | 17.73 ms | 101.1 | `[0.785, 0.562]` |
| EAGLE3 control run1 | `results/repeat160_rate128_max256_matched_controls/eagle3_k8_len4096_seq128_bt4096_rate128_max256` | EAGLE3 | 8 | no | 2879.8 | 3761.8 | 12.26 s | 3401.2 ms | 34.74 ms | 107.7 | `[0.698, 0.455, 0.260, 0.157, 0.088, 0.057, 0.036, 0.028]` |
| EAGLE3 control run2 | `results/repeat160_rate128_max256_matched_repro/eagle3_k8_len4096_seq128_bt4096_rate128_max256_run2` | EAGLE3 | 8 | no | 2995.2 | 3912.5 | 9.23 s | 500.6 ms | 34.23 ms | 104.3 | `[0.704, 0.463, 0.263, 0.162, 0.091, 0.062, 0.038, 0.029]` |
| P-EAGLE k8 control | `results/repeat160_rate128_max256_matched_controls/peagle_k8_len4096_seq128_bt4096_rate128_max256` | P-EAGLE | 8 | no | 2783.7 | 3636.2 | 10.08 s | 562.9 ms | 37.34 ms | 106.4 | `[0.765, 0.533, 0.347, 0.214, 0.030, 0.006, 0.000, 0.000]` |

Stable comparison:

- Candidate average: `(5449.5 + 5507.6) / 2 = 5478.5` output tok/s.
- EAGLE3 k8 average: `(2879.8 + 2995.2) / 2 = 2937.5` output tok/s.
- Average candidate / EAGLE3: `1.86x`.
- Conservative candidate minimum / EAGLE3 maximum: `5449.5 / 2995.2 = 1.82x`.
- Candidate minimum / P-EAGLE k8 control: `5449.5 / 2783.7 = 1.96x`.

Result:

- This is the first stable controlled result in the requested 1.8x to 2.0x range.
- The gain is not from raising concurrency relative to another run. All final comparisons use the same `GUIDELLM_RATE=128`, same repeated-first5 dataset, same output cap, same sampling, same `max_model_len`, same `max_num_seqs`, and same `max_num_batched_tokens`.
- The mechanism is draft-guided verifier work reduction: P-EAGLE's first two positions keep high acceptance around 0.78 and 0.56, while k8 spends extra verifier work on late positions that have near-zero acceptance.
- The local argmax source patch is a small additive optimization. It removes draft-token proposal remap/scatter work, but the dominant effect is choosing `k=2` for this P-EAGLE acceptance profile.

## Current Status

- Baselines and optimization traces are recorded for rate1, rate4, rate5, and repeated-first5 high-concurrency scenarios.
- Final stable claim is limited to this controlled serving scenario: `data/math_reasoning_first5_repeat160.jsonl`, `GUIDELLM_RATE=128`, `REQUEST_TYPE=chat_completions`, `temperature=0.6`, `top_p=0.95`, `top_k=20`, `max_tokens=256`, `max_model_len=4096`, `max_num_seqs=128`, and `max_num_batched_tokens=4096`.
- Final candidate: P-EAGLE with `num_spec_tokens=2` plus the source-level mapped local argmax path.
- Stable measured gain: `1.82x` conservative worst-observed comparison against EAGLE3 k8, `1.86x` average against EAGLE3 k8, and `1.96x` conservative comparison against P-EAGLE k8.
- This should not be generalized to other rates or output caps without re-running same-scenario controls. In max512 scenarios the best observed ratio remained below 1.8x.
