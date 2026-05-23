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
