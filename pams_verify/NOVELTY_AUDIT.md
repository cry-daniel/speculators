# Novelty Audit

This is a working audit for positioning PAMS-Verify. It is intentionally
conservative: if a capability is not implemented and measured in patched vLLM,
PAMS must not claim it.

| Work | What it does | Similarity to PAMS-Verify | What PAMS must not claim | What PAMS uniquely tests |
| --- | --- | --- | --- | --- |
| STS | Speculative/tree-style serving and acceptance-aware decoding ideas. | Shares the speculative verification setting. | Do not claim generic speculative scheduling or tree construction as novel. | Whether shared+residual sparse verifier masks improve accepted tokens per loaded KV block. |
| SpecSA | Sparse attention for speculative decoding/verification. | Closest to draft-guided sparse verifier attention. | Do not claim draft-guided sparse verification itself. | Union growth across multiple draft-token masks and memory-coherent planning. |
| MagicDec | Uses efficient speculative decoding mechanisms for serving. | Related serving-level speculative baseline. | Do not claim general speculative acceleration. | Acceptance-prior allocation of verifier sparse memory budget. |
| SpecAttn | Speculative or sparse attention acceleration. | Related attention-side optimization. | Do not claim arbitrary sparse attention as novel. | Prefix-risk-aware dense fallback for sparse verifier correctness. |
| QSpec | Quantization-oriented speculative decoding. | Related efficiency objective. | Do not claim quantized verification or quantization speedup. | Accepted tokens per loaded verifier KV block under sparse verification. |
| QuantSpec | Quantization for speculative decoding or verifier efficiency. | Shares target of reducing memory/compute. | Do not claim compression-only benefits. | Joint probability and memory planning rather than pure quantization. |
| MatX blockwise sparse speculative decoding | Blockwise sparse speculative decoding. | Similar block granularity. | Do not claim blockwise sparse speculative decoding broadly. | Shared global block set plus token-private residual allocation. |
| FASER | Fast speculative or sparse execution strategy. | Related systems optimization. | Do not claim generic runtime routing or early exit. | Validation of memory-coherent masks in a patched vLLM path, if implemented. |
| CAST | Cache/attention-aware speculative technique. | Related cache-aware decoding. | Do not claim cache-aware speculative decoding generally. | Measurement of union growth and fallback overhead with Qwen3-8B on RTX 5090. |
| TransKV | KV-cache transformation/compression/reuse. | Shares KV-memory motivation. | Do not claim transactional KV cache or pure KV compression. | Sparse verifier block loading per speculative block. |
| Batch Speculative Decoding Done Right | Batch/scheduler treatment of speculative decoding. | Related scheduler baseline. | Do not claim batching policy as PAMS novelty. | Probability-weighted verifier sparse-memory budget allocation. |
| vLLM built-in speculative decoding | Production standard speculative decoding modes. | Required baseline. | Do not claim speedup without comparing against vLLM standard speculative baselines. | End-to-end PAMS integration over best standard vLLM baseline, if achieved. |

Boundary statement:

PAMS-Verify is not claiming draft attention predicts target sparse masks in
general, transactional KV cache, generic adaptive speculative length, generic
cost-aware tree construction, early-exit verification, or pure KV compression.

PAMS-Verify specifically tests:

- Whether independent per-draft-token sparse verifier masks cause KV-block union growth.
- Whether shared+residual sparse masks reduce loaded KV blocks.
- Whether prefix acceptance priors allocate sparse verifier memory budget better than fixed residual budgets.
- Whether prefix-risk-aware dense fallback preserves verifier correctness/quality.
- Whether these mechanisms produce actual end-to-end improvement in patched vLLM on Qwen3-8B / RTX 5090.

