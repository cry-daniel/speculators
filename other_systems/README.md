# External BF16 N:M baselines

This directory contains reproducible RTX 5090 (`sm_120`) BF16 adapters for
Flash-LLM, the SparTA decomposition shipped with the SpInfer artifact, and
SpInfer.  These baselines implement static weight N:M sparsity only.  They do
not implement SpecLink's token-dense/token-sparse hybrid policy.

## Source lock and layout

`sources.lock.json` pins the exact upstream commits. Unmodified source
snapshots and licenses live in `upstream/`; runnable BF16 ports live in
`flash_llm_bf16/` and `spinfer_bf16/`. For SparTA, the lock records both the
MIT-licensed Microsoft project and the Apache-2.0 `sparTA.h` artifact actually
used from SpInfer.

| Method label | Weight representation | Timed computation |
|---|---|---|
| `flash_llm` | Tiled-CSL value/index records | Flash-LLM sparse CUDA kernel |
| `spinfer` | bitmap + compressed values | SpInfer bitmap CUDA kernel |
| `sparta` | at-most-2:4 base + remaining residual | BF16 cuSPARSELt, then SpInfer residual accumulated in its epilogue |

The SpInfer artifact's `sparTA.h` uses the same decomposition but combines
FP16 cuSPARSELt with an FP16 Sputnik residual kernel.  This repository is
BF16-only and its model activations are contiguous `[tokens,K]`; therefore the
port uses the artifact's SpInfer bitmap kernel for the residual rather than
introducing an FP16-only Sputnik path and a timed activation transpose.  It
retains two-kernel SparTA semantics and has no third add kernel, but it is not a
bit-for-bit Sputnik port.

## Porting changes

- All operand and compressed-value types are BF16.  Tensor-core PTX uses
  `mma.sync...bf16.bf16.f32`.
- Flash-LLM's register staging capacity was generalized from the upstream
  high-sparsity limit to the full 64x64 tile bound.  Its two 128-row halves now
  have disjoint, explicit register bounds; this is required for 5:8 and 3:4.
- SpInfer dynamic shared memory reserves the full 4096-value 64x64 tile bound
  instead of the upstream 13B-specific value 2304.  This prevents 3:4 and
  denser formats from overwriting activation staging.
- SparTA reverses the artifact's accumulation order because PyTorch's BF16
  cuSPARSELt binding does not expose matrix `beta=1`: cuSPARSELt writes the
  base result first, then SpInfer adds the residual in its epilogue.
- Compression is an offline setup cost.  Timed regions contain GEMM launches
  only and retain no dense copy inside an external prepared representation.

`apply_nm_mask` supports arbitrary exact `N:M`.  Formal defaults are 5:8 and
3:4.  Those two formats use a deterministic balanced distribution over K4
subgroups so all three methods receive the identical sparse matrix and SparTA
has a meaningful 2:4 base.  Mask construction and validation are row-chunked,
so the same code can prepare 14B/32B/70B shapes without multi-gigabyte top-k
temporaries.

## Correctness

Compile both CUDA extensions and check sparse and dense edge formats:

```bash
cd /path/to/speculators
conda run -n spec python -u other_systems/verify.py
```

The extension cache defaults to `temp/other_systems_build/`.  Set
`SPECLINK_OTHER_SYSTEMS_VERBOSE_BUILD=1` for compiler output or
`SPECLINK_OTHER_SYSTEMS_BUILD_DIR` to move the cache.

## Kernel benchmark

Formal defaults cover Qwen3-8B and Llama-3.1-8B, all
`qkv/o/gate_up/down` linears, and `M=512/1024/2048`:

```bash
MPLCONFIGDIR=temp/matplotlib conda run -n spec python -u \
  other_systems/bench_kernel.py \
  --output-root examples/evaluate/eval-guidellm/results/other_systems_nm_kernel_8b_TIMESTAMP
```

Both formal scripts query `nvidia-smi` before CUDA initialization and abort if
another compute process is present.  A one-command full reproduction is:

```bash
bash other_systems/run_8b.sh
```

Each split-K choice is screened outside the formal interval.  Formal timing
uses 100 graph warmups and ten independent 1000-replay CUDA Event intervals.
A 256 MiB eviction is issued before every independent interval.  The report
contains the median, P10, P90, raw trials, selected split-K, correctness error,
and speedup relative to BF16 cuBLAS operating on the identical N:M values.

Useful variants:

```bash
# Fast compilation and correctness smoke, written under temp/.
conda run -n spec python -u other_systems/bench_kernel.py --smoke

# Any exact N:M accepted by the adapters.
conda run -n spec python -u other_systems/bench_kernel.py \
  --formats 1:4,5:8,3:4,4:4 --models qwen3_8b --m-values 512

# The shape table is already general to the larger requested models.
conda run -n spec python -u other_systems/bench_kernel.py \
  --models qwen3_14b,qwen3_32b,llama3_70b --formats 5:8,3:4
```

## One-layer benchmark

The layer benchmark replaces all four target linears with one selected
external method while retaining RMSNorm, Q/K normalization where applicable,
RoPE, non-fused GQA attention, softmax, SiLU, and residual additions.  It uses
context length 128, seven draft tokens plus one current token per request, and
batch sizes 64/128/256, hence `M=512/1024/2048`.

```bash
MPLCONFIGDIR=temp/matplotlib conda run -n spec python -u \
  other_systems/bench_layer.py \
  --output-root examples/evaluate/eval-guidellm/results/other_systems_nm_layer_8b_TIMESTAMP
```

`bench_layer.py` accepts the same larger model labels.  Its current defaults
remain the two 8B models, as requested.  Generated smoke artifacts go to
`examples/evaluate/eval-guidellm/temp/`; formal outputs go to the explicitly
selected results directory.

## Five-model full-layer comparison and SpecLink D1 breakdown

The complete five-model comparison combines the external 5:8 baselines with
cuBLAS, SpecLink D1, and the pure-2:4 upper bound:

```bash
MPLCONFIGDIR=temp/matplotlib conda run -n spec python -u \
  other_systems/bench_full_layer_five_models.py
```

The concurrent D1 breakdown uses the formal SpecLink row above as its E2E
reference. It preserves the complement-first multi-stream CUDA Graph and
reports non-additive `GEMM`, `Attention`, `Gather/Scatter`, and `Others`
active time:

```bash
MPLCONFIGDIR=temp/matplotlib conda run -n spec python -u \
  other_systems/bench_speclink_d1_full_layer_breakdown.py \
  --reference-root \
  examples/evaluate/eval-guidellm/results_final/five_model_full_layer_5_8_vs_speclink_d1_20260723
```

Use `--reference-root` and `--output-root` to point at a differently named
formal run. The generated benchmark artifacts remain ignored by Git.

See `VLLM_FEASIBILITY.md` for the serving integration boundary.
