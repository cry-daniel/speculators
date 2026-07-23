# vLLM integration feasibility

## Verdict

All three BF16 paths can be made into vLLM linear backends for static N:M
weights, but the current adapters are benchmark integrations rather than a
production vLLM integration.  SpInfer is the cleanest first candidate.
Flash-LLM needs the most restrictive batch bucketing, while the SparTA artifact
path always launches two dependent GEMMs and keeps two compressed streams.

| Path | Static model-load compression | CUDA Graph capture in this repo | Main serving obstacle | Feasibility |
|---|---:|---:|---|---|
| Flash-LLM | yes | yes | token rows must be 8/16/32/64/128 or a multiple of 128; output N also has tile-alignment constraints | conditional |
| SpInfer | yes | yes | same token-row buckets; workspace/output allocation must move out of forward | good |
| SparTA artifact | yes | yes | cuSPARSELt base and SpInfer residual are serial dependencies; two formats and workspaces must be managed | good but higher overhead |

## What is already proven

- Model weights are compressed once and reused across `M=512/1024/2048`.
- All three calls can be captured and replayed in a PyTorch CUDA Graph.
- Qwen3-8B and Llama-3.1-8B `qkv/o/gate_up/down` shapes pass BF16 numerical
  checks in the one-layer path.
- The canonical TP=1 shape table also includes Qwen3-14B, Qwen3-32B, and
  Llama3-70B.  Their N and K dimensions satisfy the current 64/128/256
  alignment requirements.

These checks do not prove end-to-end vLLM throughput or model quality.  N:M
weight selection changes the model, so accuracy calibration/evaluation is a
separate requirement.

## Required production work

1. Implement a vLLM `LinearMethodBase`-style loader that stores the compressed
   tensors in model parameters/buffers and performs offline N:M conversion at
   load time (or loads precompressed checkpoints).
2. Register each launch through `torch.library` with FakeTensor/meta support.
   The current lazy pybind modules are sufficient for eager benchmarking but
   are not a stable `torch.compile` contract.
3. Preallocate output and split-K workspace per CUDA-Graph bucket.  The current
   wrapper allocates tensors inside `forward`; graph capture makes the replay
   address stable, but eager serving would still pay allocator overhead.
4. Bucket and pad dynamic token rows to the kernel-supported set.  Padding must
   be included in throughput accounting.  Very small decode batches that do
   not map efficiently to 8/16/32/64/128 should fall back to cuBLAS.
5. Tune split-K by `(GPU, dtype, N, K, token bucket, N:M format)` offline and
   serialize the choice.  No online compilation or tuning belongs in the
   serving hot path.
6. For tensor parallelism, prune after sharding or guarantee that checkpoint
   shards preserve K-axis M-group boundaries.  Revalidate each local N/K
   alignment; TP=1 shape validity alone is insufficient.
7. Add vLLM end-to-end correctness, CUDA Graph, mixed prefill/decode, and
   throughput tests before enabling a backend by default.

## Recommended order

Start with SpInfer as an opt-in static N:M vLLM method, using graph buckets for
`M=128,256,512,1024,2048` and cuBLAS fallback elsewhere.  Add Flash-LLM only
if the formal kernel sweep demonstrates a worthwhile regime.  Treat SparTA as
an ablation: it is useful for comparing a structured base plus residual, but
its two serial weight streams make it less attractive as the primary serving
backend.
