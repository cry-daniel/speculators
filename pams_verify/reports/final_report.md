# PAMS-Verify Final Report

## 1. Executive Summary

Decision: **NO-GO**

Strongest result: Offline PAMS mask planning improves the synthetic accepted-token-per-loaded-block proxy.
Weakest result: No live patched-vLLM sparse verifier path compiled or ran.
End-to-end vLLM speedup achieved: no.

## 2. Hardware and Model Setup

GPU: `NVIDIA GeForce RTX 5090`
VRAM GB: `31.84`
Torch CUDA available in escalated preflight: `True`
vLLM version: `0.20.0`
Recommended max_model_len: `4096`

# Phase 0 Memory Estimate

- Model: `Qwen/Qwen3-8B`
- Dtype: `bfloat16`
- GPU memory utilization: `0.85`
- Estimated weight memory GB: `15.274`
- KV bytes/token: `147456`
- Max KV tokens under budget: `58032`
- Requested KV tokens: `65536`
- Requested headroom ratio: `-0.115`
- Recommended max_model_len: `4096`
- Recommended max_num_seqs: `14`

OOM degrade order: max_num_seqs, max_model_len, num_prompts, dtype_or_kv_cache_dtype.

## 3. Correctness Policy

Exact modes must compare token IDs or dense verifier decisions. Approximate sparse modes report false accept and false reject separately. The current exactness evidence is offline synthetic only.

Correctness metrics: `{"approximate_sparse": {"decision_match_rate": 0.7229299363057324, "dense_fallback_rate": 0.0, "false_accept_rate": 0.14132165605095542, "false_reject_rate": 0.1357484076433121}, "exact_fallback": {"decision_match_rate": 1.0, "dense_fallback_rate": 1.0, "false_accept_rate": 0.0, "false_reject_rate": 0.0}, "false_accept_examples": [{"acceptance_prior": 0.7756282464891082, "block_id": "short_chat_0116:0:bs16", "dense_accept": false, "prompt_id": "short_chat_0116", "sparse_accept": true, "token_index": 2}, {"acceptance_prior": 0.8354232187693759, "block_id": "short_chat_0116:0:bs16", "dense_accept": false, "prompt_id": "short_chat_0116", "sparse_accept": true, "token_index": 3}, {"acceptance_prior": 0.7756282464891082, "block_id": "short_chat_0116:0:bs32", "dense_accept": false, "prompt_id": "short_chat_0116", "sparse_accept": true, "token_index": 2}, {"acceptance_prior": 0.8354232187693759, "block_id": "short_chat_0116:0:bs32", "dense_accept": false, "prompt_id": "short_chat_0116", "sparse_accept": true, "token_index": 3}, {"acceptance_prior": 0.7756282464891082, "block_id": "short_chat_0116:0:bs64", "dense_accept": false, "prompt_id": "short_chat_0116", "sparse_accept": true, "token`

## 4. Experiment 1: Union Problem

Union metrics: `[{"accepted_token_weighted_recall": 0.4809182286803607, "accepted_tokens_per_loaded_block": 0.2831975150020762, "average_mean_token_blocks": 49.5291932059448, "average_union_blocks": 49.5291932059448, "decision_match_rate": 1.0, "dense_fallback_rate": 0.0, "dense_fallback_ratio": 0.0, "estimated_hbm_bytes_per_speculative_block": 205837305.477707, "false_accept_rate": 0.0, "false_reject_rate": 0.0, "mask_jaccard_overlap": 1.0, "method": "dense_all_blocks", "num_speculative_blocks": 1884, "target_attention_top_block_recall": 1.0, "union_growth_ratio": 1.0}, {"accepted_token_weighted_recall": 0.4114566403031581, "accepted_tokens_per_loaded_block": 0.3291669363073003, "average_mean_token_blocks": 7.121284501061571, "average_union_blocks": 14.007961783439491, "decision_match_rate": 0.7023619957537155, "dense_fallback_rate": 0.0, "dense_fallback_ratio": 0.0, "estimated_hbm_bytes_per_speculative_block": 67970015.38853504, "false_accept_rate": 0.25384819532908703, "false_reject_rate": 0.04378980891719745, "mask_jaccard_overlap": 0.5804033506793592, "method": "independent_topk", "num_speculative_blocks": 1884, "target_attention_top_block_recall": 0.7644141454352441, "union_growth_ratio": 1.`

## 5. Experiment 2: Acceptance Prior

Acceptance metrics: `{"calibration": {"auroc_accept": 0.765730444418969, "bias": -0.7, "brier": 0.18774577567774217, "ece": 0.01238643186643796, "temperature": 0.9758064516129032}, "test": {"auroc_accept": 0.7627384104936682, "auroc_accept_from_prior": 0.7627384104936682, "auroc_useful_from_rho": 0.7886071297317163, "bias": -0.7, "brier": 0.18373704772496013, "ece": 0.03760528256484607, "rho_useful_correlation": 0.49459094174966056, "temperature": 0.9758064516129032}, "validation": {"auroc_accept": 0.7595409292035398, "bias": -0.7, "brier": 0.18717168068955806, "ece": 0.01625983999807396, "temperature": 0.9758064516129032}}`

## 6. Experiment 3: Offline PAMS Mask Planning

Offline mask metrics: `[{"accepted_token_weighted_recall": 0.4809182286803607, "accepted_tokens_per_loaded_block": 0.2831975150020762, "average_mean_token_blocks": 49.5291932059448, "average_union_blocks": 49.5291932059448, "decision_match_rate": 1.0, "dense_fallback_rate": 0.0, "dense_fallback_ratio": 0.0, "estimated_hbm_bytes_per_speculative_block": 205837305.477707, "false_accept_rate": 0.0, "false_reject_rate": 0.0, "mask_jaccard_overlap": 1.0, "method": "dense_all_blocks", "num_speculative_blocks": 1884, "target_attention_top_block_recall": 1.0, "union_growth_ratio": 1.0}, {"accepted_token_weighted_recall": 0.4114566403031581, "accepted_tokens_per_loaded_block": 0.3291669363073003, "average_mean_token_blocks": 7.121284501061571, "average_union_blocks": 14.007961783439491, "decision_match_rate": 0.7023619957537155, "dense_fallback_rate": 0.0, "dense_fallback_ratio": 0.0, "estimated_hbm_bytes_per_speculative_block": 67970015.38853504, "false_accept_rate": 0.25384819532908703, "false_reject_rate": 0.04378980891719745, "mask_jaccard_overlap": 0.5804033506793592, "method": "independent_topk", "num_speculative_blocks": 1884, "target_attention_top_block_recall": 0.7644141454352441, "union_growth_ratio": 1.`

## 7. Experiment 4: Sparse Kernel Microbenchmark

Kernel metrics: `{"device": "cpu", "mean_latency_ms_by_method": {"dense": 0.1058023141619439, "independent_sparse": 0.3476667043287307, "pams": 0.28033107325124246, "shared_fixed_residual": 0.29383497409677756, "shared_only": 0.2604509451581786}, "rows": 240, "triton": {"available": false, "label": "reference_only", "reason": "custom Triton kernel was not implemented; sparse_attention_ref is used for measured reference overhead"}}`

## 8. Experiment 5: vLLM Integration Attempts

Integration A registered an exact scheduler policy offline but did not apply a live vLLM hook.
Integration B inspected vLLM and wrote a proposed diff, but arbitrary verifier block masks were unsupported and no patch was applied.
Integration C evaluated fallback policies offline only.

Integration B evidence: `{"apply_requested": false, "arbitrary_block_mask_supported": false, "compiled": false, "integration": "B_attention_patch_sparse_verifier", "limitation": "Installed vLLM does not expose an arbitrary verifier block-mask path; patching site-packages was not performed from the shared sandbox.", "outcome": "unsupported_installed_backend_no_patch_applied", "pams_feature_flag_present": false, "patched_vllm": false, "ran_live_vllm": false, "vllm_inspection": {"arbitrary_block_mask_supported": false, "attention_modules_checked": ["/ACALAB/stu1/miniconda3/envs/spec/lib/python3.12/site-packages/vllm/v1/attention/__init__.py", "/ACALAB/stu1/miniconda3/envs/spec/lib/python3.12/site-packages/vllm/v1/attention/backends/flash_attn.py", "/ACALAB/stu1/miniconda3/envs/spec/lib/python3.12/site-packages/vllm/v1/attention/backends/flex_attention.py", "/ACALAB/stu1/miniconda3/envs/spec/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py", "/ACALAB/stu1/miniconda3/envs/spec/lib/python3.12/site-packages/vllm/model_executor/layers/attention/__init__.py", "/ACALAB/stu1/miniconda3/envs/spec/lib/python3.12/site-packages/vllm/transformers_utils/configs/speculators/__init__.py", "/ACAL`

## 9. End-to-End Results

Live standard-baseline smoke results were collected for Qwen3-8B at `max_model_len=2048`, `max_num_seqs=4`, four random short prompts, and concurrency 1. These are smoke measurements, not the full matrix.

Live baseline smoke: `[{"bench_log": "/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/pams_verify/experiments/01_dense_baselines/raw/live_dense_no_spec/vllm_bench_serve.log", "benchmark_json": {"backend": "openai-chat", "burstiness": 1.0, "completed": 4, "date": "20260523-114606", "duration": 0.7563288391102105, "endpoint_type": "openai-chat", "failed": 0, "label": null, "max_concurrency": 1, "max_concurrent_requests": 4, "max_output_tokens_per_s": 68.0, "mean_e2el_ms": 189.00654336903244, "mean_itl_ms": 10.577683373412583, "mean_ttft_ms": 19.75426683202386, "median_e2el_ms": 189.3227535765618, "median_itl_ms": 11.261725099757314, "median_ttft_ms": 20.076504559256136, "model_id": "Qwen/Qwen3-8B", "num_prompts": 4, "output_throughput": 84.61927760852454, "p50_e2el_ms": 189.3227535765618, "p50_itl_ms": 11.261725099757314, "p50_ttft_ms": 20.076504559256136, "p95_e2el_ms": 189.58817801903933, "p95_itl_ms": 11.562294338364154, "p95_ttft_ms": 20.149416127242148, "p99_e2el_ms": 189.59762213286012, "p99_itl_ms": 11.630019263830036, "p99_ttft_ms": 20.15113212633878, "request_goodput": null, "request_rate": "inf", "request_throughput": 5.288704850532784, "rtfx": 0.0, "std_e2el_ms": 0.7303610518983091, "std_itl_ms": 2.7320878496715344, "std_ttft_ms": 0.6037764473467924, "tokenizer_id": "Qwen/Qwen3-8B", "total_input_tokens": 288, "total_output_tokens": 64, "total_token_throughput": 465.40602684688497}, "method": "dense_no_spec", "result_file": "/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/pams_verify/experiments/01_dense_baselines/raw/live_dense_no_spec/dense_no_spec.json", "server_log": "/`

End-to-end matrix: `{"cuda_available_in_process": true, "matrix": [{"concurrency": 1, "method": "dense_no_spec", "status": "registered_not_executed", "workload": "short_chat"}, {"concurrency": 1, "method": "vllm_ngram_4", "status": "registered_not_executed", "workload": "short_chat"}, {"concurrency": 1, "method": "vllm_ngram_8", "status": "registered_not_executed", "workload": "short_chat"}, {"concurrency": 1, "method": "model_draft_fixed_4", "status": "registered_not_executed", "workload": "short_chat"}, {"concurrency": 1, "method": "model_draft_fixed_8", "status": "registered_not_executed", "workload": "short_chat"}, {"concurrency": 1, "method": "independent_sparse_verifier", "status": "blocked_no_patched_vllm_sparse_verifier", "workload": "short_chat"}, {"concurrency": 1, "method": "shared_fixed_residual", "status": "blocked_no_patched_vllm_sparse_verifier", "workload": "short_chat"}, {"concurrency": 1, "method": "pams", "status": "blocked_no_patched_vllm_sparse_verifier", "workload": "short_chat"}, {"concurrency": 1, "method": "pams_fallback_exact", "status": "blocked_no_patched_vllm_sparse_verifier", "workload": "short_chat"}, {"concurrency": 1, "method": "pams_fallback_approximate", "status": "b`

No PAMS end-to-end throughput or ITL claim is made.

## 10. Ablations

# Phase 13 Ablations

Offline ablations were run on the synthetic test trace. End-to-end ablations remain unavailable until vLLM integration succeeds.

## 11. Failure Log

# Failure Log

- `07_vllm_integration_a_scheduler_hook`: `attempted_no_live_patch`
- `08_vllm_integration_b_attention_patch`: `attempted_unsupported_backend`
- `09_vllm_integration_c_fallback_prefilter`: `attempted_offline_prefilter_only`
- `10_end2end`: `blocked_no_patched_vllm_sparse_verifier`


## 12. Final Judgment

This is currently a NO-GO for a systems paper claim. The cleanest claim is an offline research prototype showing the union-growth motivation and a concrete implementation plan for shared+residual mask planning. The next engineering step is an editable vLLM source checkout with a minimal exact scheduler hook first, then an attention backend prototype that can consume verifier block masks.
