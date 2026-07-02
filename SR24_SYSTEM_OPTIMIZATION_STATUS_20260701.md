# SR24 Systems Optimization Status - 2026-07-01

## Goal

The active target is still: make SR24/SpecLink faster than dense EAGLE3 by
about `1.2x` for most batch sizes `8/16/32/64` and datasets, while allowing an
absolute accuracy drop within `8 percentage points`. Quick quality gates should
use GSM8K with `limit >= 50`; small throughput probes can use
`math_reasoning` before expanding to the full matrix.

All runs below were kept under:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/
```

## Code Changes in This Pass

The main correctness and diagnosis changes were:

- Llama SR24 hooks now check the runtime SR24 flag as well as the import-time
  flag. This avoids missing hooks when SR24 is enabled after import.
- SR24 sparse output now fails fast if a module is marked SR24-enabled but the
  sparse path cannot produce an output. It no longer silently falls back to a
  dense `Linear` on sparse storage.
- The direct cuSPARSELt path now fails fast if it receives an unpacked sparse
  tensor instead of silently using `F.linear`.
- The `criticalprefix4_bucket16_directcslt` preset is aligned to the current
  serving interpretation: one extra low-priority row, no residual bucket
  priority, and default vLLM compile enabled.
- Two temporary lossy candidates were added to the sweep runner for bucket12
  and active-only bucket probes.
- The row-routed gate/up and down fallback paths now initialize `bias` before
  the min-base dense fallback. This fixes a runtime `bias` reference hazard in
  underfilled-branch fallback checks.
- The lossy sweep runner now includes all-MLP prefix1/prefix0 probes and
  down-proj-only prefix2/prefix1/prefix0 probes, so the 8pp budget can be
  tested without forcing every draft row through dense correction.

These changes are meant to expose wrong execution paths early. They are not a
new speed path by themselves.

## Current Evidence

### 1. Base-only 2:4 has a real speed upper bound but fails quality

Throughput root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_upperbound_bs64_math256_20260701
```

Quality root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_upperbound_quality50_20260701
```

On Llama-3.1-8B, EAGLE3 K=8, `math_reasoning`, bs64, max tokens 256:

| method | total tok/s | full-batch tok/s | accepted draft/step |
|---|---:|---:|---:|
| dense EAGLE3 | 2786.130 | 3174.794 | 1.735 |
| base-only 2:4 | 3855.742 | 4830.710 | 3.427 |

This is `1.384x` total and `1.522x` full-batch speedup. The quality loss is
too large, though: GSM8K-50 drops from `0.72` to `0.22` (`-50pp`). Base-only is
therefore an upper-bound signal, not a usable method.

### 2. Layer-scoped sparse/dense mixing passes quality in eager but conflicts
with vLLM default compile

Quality-safe eager root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_res16_25_base26_31_smallrow160_enforceeager_quality50_20260701
```

Default-compile failure root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_split_csltopaque_freshcache_quality50_20260701
```

The layer-scoped candidate keeps `gate_up_proj` layers 16-25 corrected and
makes layers 26-31 base-only. In enforce-eager mode it passes GSM8K-50:

```text
dense 0.72 -> SR24 0.72
```

But it does not survive default vLLM compile. The generated Inductor graph
contains dense `extern_kernels.mm(... reinterpret_tensor(arg5_1, (4096, 28672),
...))` for a sparse `gate_up_proj` weight. Example file:

```text
/tmp/sr24_vllm_cache_csltopaque_20260701/torch_compile_cache/torch_aot_compile/386fca4901990323f62c804e6de9a4e863535645c7e9be3f55110717b18b8ed2/inductor_cache/p2/cp2txdy5p523jcjgfc54km3fyom3zr2bswarba25vpieto7y5c2o.py
```

Read: default vLLM compile reuses a shared decoder-layer graph across layers.
If some layers have dense/residual storage and other layers are base-only sparse
storage, the compiled graph can be specialized to the wrong branch. Avoid
layer-heterogeneous SR24 formats under default compile unless the model runner
gets separate graph keys or the operator data format becomes uniform.

### 3. Homogeneous channel-pair split avoids the compile issue, but it is not a
speed path

Quality roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_channel_pair_quality50_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_channel_pair_dense75_quality50_20260701
```

Throughput root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_channel_pair_dense75_direct_graph_throughput_bs64_math256_20260701
```

The channel-pair split keeps a fraction of gate/up intermediate channels dense
and routes the rest through a full-row 2:4 branch. It is homogeneous across
layers, so it avoids the layer-mixed compile problem.

Quality on GSM8K-50:

| gate/up dense channel fraction | dense acc | SR24 acc | delta |
|---:|---:|---:|---:|
| 25% | 0.72 | 0.50 | -22pp |
| 50% | 0.72 | 0.56 | -16pp |
| 75% | 0.72 | 0.68 | -4pp |

The 75% dense-channel case passes the 8pp gate, but throughput is negative on
bs64/math/max256 with direct cuSPARSELt and graph enabled:

| method | total tok/s | full-batch tok/s | avg GPU util |
|---|---:|---:|---:|
| dense EAGLE3 | 2795.986 | 3176.820 | 92.9% |
| channel-pair SR24 | 2235.064 | 2707.800 | 79.7% |

That is only `0.799x` total and `0.852x` full-batch speed. The extra
branching, activation/concat work, and lower utilization outweigh the sparse
gate/up savings.

### 4. Bucket/active-only fixes are quality-safe but near parity

The aligned bucket12/bucket16 and active-only probes show that removing obvious
duplicate dense work and preserving default compile improves the path, but not
enough:

| root | read |
|---|---|
| `sr24_bucket12_defaultcompile_true_throughputonly_bs64_math256_20260701` | about `1.051x` total, `0.990x` full-batch |
| `sr24_activeonly_probe_bs64_math256_20260701` | GSM8K-50 unchanged, throughput about parity |
| `sr24_bucket12_activefused_throughputonly_bs64_math256_20260701` | about `1.054x` total, `0.992x` full-batch |

These are useful cleanups, but not a route to `1.2x`.

### 5. Per-leaf dense fallback improves tail behavior, not steady-state
operator throughput

Current rerun root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup384_planner_quality50_bs32_64_math512_20260701_rerun_fixedblock
```

This probe adds one planner knob:

```text
SPECLINK_SR24_ROUTE_MIN_BASE_ROWS_BY_LEAF=gate_up_proj=384
```

The policy is still `lossy_prefix2_rowrouted_mlp`: first two draft rows plus
the verifier bonus row are important/dense; later draft rows are sparse-only.
The new knob only says that a gate/up sparse branch with fewer than 384 base
rows should fall back to the dense MLP path instead of launching an underfilled
2:4 sparse branch. It does not change the quality policy.

Quality gate:

| task | dense | SR24 | delta |
|---|---:|---:|---:|
| GSM8K-CoT, limit 50, max new tokens 512 | 0.7400 | 0.7400 | 0.0pp |

Throughput on `math_reasoning`, K=8, max tokens 512:

| bs | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 2791.845 | 2854.784 | 1.022x | 3451.616 | 3463.623 | 1.003x |
| 64 | 2486.932 | 3024.591 | 1.216x | 4448.570 | 4236.282 | 0.952x |

This is the first current-tree gateup384 point that reaches `1.2x` on a total
tokens/s row, but it does not reach `1.2x` in the full-batch steady-state
metric. The `bs64` total gain is therefore best read as improved completion
tail behavior and slightly higher accepted draft length, not as proof that the
mixed MLP operator is fast enough.

Code inspection also matters here: the fixed-block row-routed MLP already uses
disjoint dense and sparse inputs. Important rows go through dense gate/up and
dense/exact down, while unimportant rows go through the 2:4 sparse branch. The
old `reuse_base_output` path can compute sparse for all rows and overwrite the
important rows, but that is not the current fixed-prefix path. The remaining
waste is not "dense after sparse for unimportant tokens"; it is fragmented
dense/sparse GEMMs, separate activation branches, assemble/copy overhead, and
incomplete graph-stable fill.

### 6. Triton fixed-block assembly is not enough

Throughput-only root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup384_tritonassemble_throughput_bs32_64_math512_20260701
```

This repeats the `gate_up_proj=384` fallback and adds:

```text
SPECLINK_SR24_TRITON_ROUTE_ASSEMBLY=1
```

It does not change the quality policy; it only replaces the fixed-block
PyTorch output-copy assembly with a Triton assembly kernel. The result is
negative:

| variant | bs | total speedup | full-batch speedup | accepted draft/step | avg GPU util |
|---|---:|---:|---:|---:|---:|
| gateup384 | 32 | 1.022x | 1.003x | 2.492 | 94.7% |
| gateup384 | 64 | 1.216x | 0.952x | 2.506 | 93.3% |
| gateup384 + Triton assembly | 32 | 0.963x | 0.995x | 2.427 | 94.0% |
| gateup384 + Triton assembly | 64 | 1.211x | 0.954x | 2.472 | 93.6% |

Read: output assembly alone is not the main limiter. Replacing only the final
copy/scatter step cannot make the mixed MLP a `1.2x` steady-state path. The
next operator work has to avoid materializing separate gathered dense/sparse
inputs or fuse/group the branch GEMMs and activation work more deeply.

### 7. Current all-corrected compressed_dense check

Current no-fastpath exact-path root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_compressed_current_gateup_bs64_math64_20260701
```

Setup: Llama-3.1-8B EAGLE3 K=8, `math_reasoning`, bs64, max tokens 64,
`all_corrected_24`, `target_leafs=gate_up_proj`,
`residual_backend=compressed_dense`, `residual_device=cuda`, and the
all-corrected dense fastpath disabled.

| method | total tok/s | full-batch tok/s | avg GPU util | graph modes |
|---|---:|---:|---:|---|
| dense baseline | 1823.281 | 2780.108 | 78.6% | |
| all_corrected compressed_dense | 508.216 | 894.658 | 87.1% | `NONE=59` |

The SR24 stats prove this is not CPU residual execution:

```text
sr24_residual_backend_counts={"compressed_dense": 32}
sr24_residual_device_counts={"cuda:0": 32}
sr24_compressed_residual_runtime_on_gpu=True
sr24_residual_cuda_module_count=32
sr24_residual_cpu_module_count=0
```

Default vLLM compile does not rescue this path. The matching compile run at:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_compressed_current_gateup_compile_bs64_math64_20260701
```

failed during engine startup. Inductor emitted dense
`extern_kernels.mm(... reinterpret_tensor(arg5_1, (4096, 28672), ...))` on the
sparse semi-structured gate/up weight and crashed with:

```text
RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet.
```

Read: no-fastpath `all_corrected_24` remains an operator diagnostic. The
optimized exact control is still the densefastpath. A useful exact sparse path
requires a graph-safe fused packed base+residual operator, not more
`compressed_dense` wrapper tuning.

## 8pp Down-Only and Prefix Sweep

Latest small-scale sweep root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_8pp_downonly_prefix_sweep_bs32_64_math512_20260701
```

Setup: Llama-3.1-8B EAGLE3 K=8, GSM8K-CoT `limit=50` quality gate with
`max_new_tokens=512`, and `math_reasoning` throughput with bs32/64,
`max_tokens=512`, fixed 64 requests. The intent was to test the user's lossy
target directly: important rows use dense, unimportant rows use 2:4 sparse, and
unimportant rows are not corrected by dense residual work.

All candidates passed the quick quality gate exactly:

| candidate | GSM8K dense -> SR24 | best total speedup | best full-batch speedup |
|---|---:|---:|---:|
| `lossy_prefix2_noverify_sparse_gateup384_compile` | 0.74 -> 0.74 | 1.214x | 1.000x |
| `lossy_prefix1_mlp_noverify_sparse_minbase128_compile` | 0.74 -> 0.74 | 0.994x | 0.994x |
| `lossy_prefix0_mlp_noverify_sparse_minbase128_compile` | 0.74 -> 0.74 | 0.955x | 1.001x |
| `lossy_prefix2_down_only_noverify_sparse_compile` | 0.74 -> 0.74 | 1.320x | 1.000x |
| `lossy_prefix1_down_only_noverify_sparse_compile` | 0.74 -> 0.74 | 0.998x | 1.001x |
| `lossy_prefix0_down_only_noverify_sparse_compile` | 0.74 -> 0.74 | 0.998x | 0.996x |

The important detailed rows:

| candidate | bs | dense total | SR24 total | total speedup | dense full | SR24 full | full speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gateup384` | 32 | 2798.5 | 2873.3 | 1.027x | 3454.1 | 3455.4 | 1.000x |
| `gateup384` | 64 | 2486.9 | 3018.0 | 1.214x | 4448.8 | 4219.1 | 0.948x |
| `down_only_prefix2` | 32 | 2783.6 | 2734.3 | 0.982x | 3447.7 | 3448.3 | 1.000x |
| `down_only_prefix2` | 64 | 2487.3 | 3283.7 | 1.320x | 4449.0 | 4437.4 | 0.997x |
| `down_only_prefix1` | 32 | 2786.0 | 2780.4 | 0.998x | 3450.5 | 3444.5 | 0.998x |
| `down_only_prefix1` | 64 | 3358.0 | 3201.8 | 0.954x | 4428.7 | 4432.6 | 1.001x |
| `down_only_prefix0` | 32 | 2815.4 | 2775.2 | 0.986x | 3455.7 | 3437.1 | 0.995x |
| `down_only_prefix0` | 64 | 2485.8 | 2481.0 | 0.998x | 4446.0 | 4428.6 | 0.996x |

Read:

- The current row-routed paths already satisfy the desired disjoint semantics:
  important rows are dense, unimportant rows are 2:4 sparse, and sparse-only
  rows are not followed by dense residual correction.
- Reducing the number of dense-important draft rows from prefix2 to prefix1 or
  prefix0 does not create steady-state speedup. This means "important token
  count is too high" is not the current dominant bottleneck.
- Restricting SR24 to `down_proj` avoids the worst gate/up row-split overhead
  and can improve total/request-tail behavior, especially at bs64, but
  full-batch throughput is still parity. Because gate/up remains dense, this is
  not enough to reach a 1.2x steady-state target.
- The strongest total-speed rows should not be interpreted as operator
  speedups. They come from run-window/tail behavior and small acceptance-length
  variation; the full-batch rows show the mixed operator is still not faster.

## 8pp Down-Proj Layer Scope and Overlap Check

Latest roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_downproj_baseonly_quality50_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_downproj_tail8_baseonly_quality50_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_downproj_tail16_baseonly_quality50_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_downproj_tail20_baseonly_quality50_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_downproj_tail24_baseonly_quality50_20260701
```

Quality gate: Llama-3.1-8B, EAGLE3 K=8, GSM8K-CoT `limit=50`,
`max_new_tokens=512`. Dense reference from the paired full-down run is `0.72`.

| base-only down_proj scope | GSM8K-50 acc | delta vs dense | read |
|---|---:|---:|---|
| all layers 0-31 | 0.48 | -24pp | fails |
| tail8, layers 24-31 | 0.76 | +4pp | passes |
| tail16, layers 16-31 | 0.72 | 0pp | passes |
| tail20, layers 12-31 | 0.64 | -8pp | exactly on the allowed boundary |
| tail24, layers 8-31 | 0.56 | -16pp | fails |

Throughput roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_downproj_tail8_baseonly_throughput_bs32_64_math512_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_downproj_tail16_baseonly_throughput_bs32_64_math512_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_downproj_tail20_baseonly_throughput_bs32_64_math512_20260701
```

`tail20` is the largest layer scope that satisfies the 8pp gate, but it is not
a steady-state speed path:

| bs | dense total | tail20 total | total speedup | dense full | tail20 full | full speedup | SR24 GPU util |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 2818.415 | 2201.929 | 0.781x | 3454.557 | 2915.857 | 0.844x | 73.1% |
| 64 | 2486.948 | 2739.567 | 1.102x | 4448.917 | 4430.974 | 0.996x | 83.5% |

The accepted draft length went up, not down:

| bs | dense accepted draft/step | tail20 accepted draft/step |
|---:|---:|---:|
| 32 | 2.456 | 2.782 |
| 64 | 2.429 | 2.659 |

Read: accepted length is not the limiting factor. The live base-only sparse
down path loses GPU utilization and graph efficiency enough to erase the sparse
compute saving.

A narrow code probe added optional CUDA-stream overlap for `row_routed_down`:
when `SPECLINK_SR24_ROUTE_OVERLAP_STREAMS=1`, dense-important rows and
2:4-sparse base rows can run on separate streams before assembly. The change is
gated and does not affect normal vLLM/EAGLE3 paths. Clean throughput root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_downonly_prefix2_overlap_throughput_bs32_64_math512_20260701
```

It did not produce a visible speedup:

| bs | dense full | down-only overlap full | full speedup |
|---:|---:|---:|---:|
| 32 | 3449.922 | 3415.701 | 0.990x |
| 64 | 4445.933 | 4433.942 | 0.997x |

This matches the packed microbench direction: simple Python-level stream
overlap is not enough. The useful case is grouping real verifier work until
the dense and sparse branches both reach effective batch around 64, then
executing the grouped route in a graph-stable operator.

One more live probe tested whether the tail20 base-only quality boundary could
recover CUDA Graph replay by making the data format more uniform. Instead of
attaching only `down_proj=12-31`, all `down_proj` layers were SR24-attached;
layers `0-11` used `base_only_dense_verify_max_rows=8192` plus
`base_only_dense_verify_layer_ids_by_leaf=down_proj=0-11`, while layers
`12-31` remained sparse base-only. This preserves the same intended math as
tail20 but avoids leaving early layers as untouched dense modules.

Root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_downproj_tail20_uniform_attach_densefallback_throughput_bs32_64_math512_20260701
```

It still stayed in CUDA Graph `NONE` and did not improve full-batch throughput:

| bs | dense full | uniform-attach tail20 full | full speedup | SR24 graph |
|---:|---:|---:|---:|---|
| 32 | 3445.486 | 2887.410 | 0.838x | `{"NONE": 398}` |
| 64 | 4447.739 | 4427.921 | 0.996x | `{"NONE": 237}` |

Read: uniform attach plus per-layer Python dense fallback is still not a
graph-stable operator. The graph-safe route must use one uniform branch shape
inside the operator, with route descriptors selecting dense-important and
sparse-only rows, rather than relying on per-layer module attributes to switch
between dense and sparse implementations.

### 15. Relaxing the protection prefix still does not create 1.2x

The current goal allows up to 8 percentage points absolute GSM8K accuracy loss,
so I also tested a more aggressive gate-up-only candidate: protect only draft
position 0 plus the verifier bonus row with dense `gate_up_proj`, leave all
other draft/no-verify gate/up rows 2:4 sparse-only, and require at least 128
base rows before splitting. This directly tests whether "important token count
is too high" is the limiter.

Root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_aggressive_prefix_quality50_speed_b8_64_math512_20260701
```

Quality gate:

| candidate | dense GSM8K-50 | SR24 GSM8K-50 | delta |
|---|---:|---:|---:|
| `lossy_prefix1_gateup_only_noverify_sparse_minbase128_compile` | 0.7400 | 0.7400 | 0.0pp |

Throughput on Llama-3.1-8B, EAGLE3 K=8, `math_reasoning`, max tokens 512:

| bs | dense total | SR24 total | total speedup | dense full | SR24 full | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1395.4 | 1443.5 | 1.035x | 1550.7 | 1546.8 | 0.997x |
| 16 | 2128.2 | 2215.2 | 1.041x | 2571.0 | 2485.2 | 0.967x |
| 32 | 2850.0 | 2779.0 | 0.975x | 3453.4 | 3443.1 | 0.997x |
| 64 | 3201.6 | 2485.6 | 0.776x | 4433.8 | 4448.7 | 1.003x |

Read: even after relaxing the accuracy target and reducing dense-protected
draft positions, steady-state throughput is still parity. The optimization
cannot be obtained by another prefix/threshold sweep in the current Python split
operator. The only positive total-throughput rows are tail effects; full-batch
throughput stays around 1.0x.

### 16. Fixed-capacity grouped bucket planner check

I refreshed the packed verifier-block MLP microbenchmark with the batch sizes
the current target cares about: `8/16/32/64`, EAGLE3 K=8, dense-important prefix
1 or 2, fixed bucket capacity multiple 64, and useful-block coalescing factors
`1/2/4/8`.

Root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_packed_grouped_bucket_k8_bs8_64_prefix12_20260701
```

Planner report:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_packed_grouped_bucket_k8_bs8_64_prefix12_20260701/operator_planner.md
```

For prefix 2, which keeps the first two draft rows plus the verifier bonus row
dense and routes the remaining draft rows to the 2:4 branch, the local
operator requirement is:

| bs | required coalesce | effective bs | mixed local speedup | read |
|---:|---:|---:|---:|---|
| 8 | 8 | 64 | 1.240x | needs grouping |
| 16 | 4 | 64 | 1.238x | needs grouping |
| 32 | 2 | 64 | 1.238x | needs grouping |
| 64 | 1 | 64 | 1.240x | ready |

`local speedup` compares the mixed dense/sparse operator against a dense
operator over the same grouped fixed bucket. This is the real operator-level
requirement for replacing dense verifier MLP. The report also includes
`serial speedup`, which compares against running the original ungrouped dense
block repeatedly; that is only an optimistic scheduler upper bound and should
not be used as a serving claim unless the live scheduler can safely coalesce
those rows.

Read: useful row fill, not the accuracy threshold, is the blocking system
problem. Low batch sizes become viable only if the scheduler/operator can group
real useful verifier rows until the dense-important and sparse-only branches
look like effective batch around 64. If that condition is not met, the live
planner should use dense fallback rather than launching underfilled mixed
kernels. Sparse/dense stream overlap should be applied only to these filled
buckets; the earlier Python-level stream overlap on underfilled live rows stayed
at parity.

### 17. 2026-07-01 follow-up: fallback and token-policy probes

New code added a disabled-by-default scheduler policy knob:

```text
SPECLINK_SR24_SCHEDULER_POLICY_DENSE_BYPASS=1
--sr24-scheduler-policy-dense-bypass
```

The intent was to test whether underfilled fixed-block mixed MLP steps should
return to the original vLLM dense MLP instead of using the hook-internal dense
fallback. This is a useful negative result: with SR24-attached modules,
returning `None` from the MLP hook is not a free dense bypass. The subsequent
default Llama MLP path still sees SR24 sparse storage and residual
post-processing hooks, so it can do sparse base plus dense correction rather
than one original dense MLP.

Probe root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_scheduler_policy_dense_bypass_math_bs8_64_20260701
```

Compared with the previous policy gate
`sr24_scheduler_policy_live_gate_math_bs8_64_20260701`, dense-bypass worsened
the SR24 rows:

| bs | policy-gate total speedup | dense-bypass total speedup | policy-gate full speedup | dense-bypass full speedup |
|---:|---:|---:|---:|---:|
| 8 | 0.876x | 0.852x | 0.968x | 0.955x |
| 16 | 0.801x | 0.815x | 0.936x | 0.952x |
| 32 | 0.992x | 0.789x | 1.005x | 0.948x |
| 64 | 1.036x | 0.861x | 0.995x | 0.902x |

Read: do not use this knob as an optimization path. If a true dense fallback is
needed, it has to be a first-class operator path that directly uses dense
weights and suppresses later SR24 residual hooks. The existing hook-internal
`_row_routed_mlp_full_dense_output` is closer to correct than returning to the
default SR24-attached module path.

I also tested the aggressive "bonus-only dense" policy: keep only the verifier
bonus row dense and route every draft token through 2:4 base rows.

Root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bonus_only_mlp_speed_bs64_20260701
```

It was slower:

| method | bs | total tok/s | full tok/s | accepted draft/step |
|---|---:|---:|---:|---:|
| dense EAGLE3 | 64 | 2249.153 | 2814.215 | 1.402 |
| bonus-only SR24 | 64 | 1667.007 | 2454.044 | 1.396 |

Read: important-token count is not currently the limiting factor. Making more
rows sparse increases the slow 2:4 base branch and hurts throughput.

Finally, I reran a focused eager CUDA-event breakdown. A first 32-request
bs64 run accidentally capped active requests below 64, so it mostly measured
`route_min_base_rows=384` dense fallback. With 64 requests, the run hit only
one mixed fixed-block step because the short max-token run mostly stopped after
prompt/prefill; still, it exposed the startup/prefill cost shape:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedblock_breakdown_bs64_eager_64req_20260701
```

Top CUDA-event rows in that eager diagnostic were dense no-verify MLP calls:

| event | calls | avg ms |
|---|---:|---:|
| `noverify_dense_mlp_gate_up_cuda_ms` | 128 | 4.135 |
| `noverify_dense_mlp_down_cuda_ms` | 128 | 2.149 |
| `noverify_dense_mlp_act_cuda_ms` | 128 | 0.213 |

Read: short-output breakdowns can be dominated by prompt/no-verify dense work
and are easy to misread. For operator diagnosis, use long enough generations
and enough requests to keep active verifier blocks full; otherwise
`route_min_base_rows` fallback and no-verify paths hide the fixed-block mixed
operator.

## Current Optimization Direction

The current code already has the desired logical dataflow for the fixed-prefix
mixed MLP: important rows use dense branch inputs, sparse-only rows use 2:4
branch inputs, and sparse-only rows are not followed by dense residual
correction. The remaining gap is a systems/operator gap:

1. Keep a graph-stable route-table data format: fixed bucket capacity,
   active mask, dense-prefix width, sparse-base width, and output permutation.
2. Group real verifier blocks until the effective operator batch reaches the
   microbench sweet spot, around 64 useful request blocks for K=8/prefix2.
3. Run dense-important and 2:4 base branches as one packed operator family,
   preferably with branch overlap inside the operator rather than Python-level
   stream management.
4. Use dense fallback for underfilled groups; do not force mixed sparse kernels
   for bs8/16/32 until the queue can coalesce enough useful work.
5. Keep the 8pp accuracy budget as a scheduler/policy guard, but stop treating
   threshold/prefix sweeps as the main speed lever. Current evidence shows
   threshold changes mostly shift quality/acceptance while full-batch speed
   stays near parity.

The next implementation step should therefore be a live grouped verifier-bucket
adapter for the fixed-block MLP route, not another token threshold sweep.

### 17. Prefix-fill and operator-guard check

I then scanned every dense-important prefix from 0 to 8 at batch sizes
`8/16/32/64`, EAGLE3 K=8, and fixed bucket capacity multiple 64. This directly
tests the question of whether "important token count is too small": when the
dense-important branch is too small, we can promote lower-priority rows into
dense to fill a Tensor Core tile, but that only helps if the mixed operator
beats dense at the same fixed capacity.

Root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_packed_prefix_fill_k8_bs8_64_20260701
```

Planner report:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_packed_prefix_fill_k8_bs8_64_20260701/operator_planner.md
```

Best single-block local choices:

| bs | best prefix | best mixed local speedup | planner action |
|---:|---:|---:|---|
| 8 | 8 | 0.970x | dense fallback |
| 16 | 8 | 0.975x | dense fallback |
| 32 | 0 | 0.993x | dense fallback |
| 64 | 0 | 1.383x | use mixed single block |

Read: dense-fill promotion does not solve low-batch performance. For bs8/16/32,
even the best prefix choice is slower than dense. For bs64, sparse-heavy
prefixes are locally profitable, and prefix2 still reaches about `1.241x`.
Therefore the live serving planner should not spend dense work merely to fill
tiles at small batch. It should use dense fallback until useful rows can be
grouped to effective batch around 64.

I added an explicit guard preset for that policy:

```text
lossy_prefix2_rowrouted_mlp_operator_guard
lossy_prefix2_rowrouted_mlp_operator_guard_compile
```

The preset keeps the fixed-prefix2 row-routed MLP semantics, disables dense-fill
promotion with `SPECLINK_SR24_ROW_ROUTED_MLP_MIN_DENSE_ROWS=0`, and requires
`SPECLINK_SR24_ROUTE_MIN_BASE_ROWS=384` before taking the mixed branch. It is a
guardrail, not the final 1.2x operator.

Tiny throughput-only sanity:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/throughput/lossy_prefix2_rowrouted_mlp_operator_guard_compile/report.md
```

This run used only 8 total requests, `max_tokens=128`, and no full-batch window,
so it is a functional sanity check rather than a serving claim:

| bs | dense total tok/s | guard total tok/s | speedup |
|---:|---:|---:|---:|
| 32 | 691.1 | 706.7 | 1.023x |
| 64 | 726.2 | 705.9 | 0.972x |

Read: the guard avoids the worst known underfilled branch shape but does not
create the missing speedup by itself. The next implementation step is still a
fixed-capacity grouped/fused MLP path that coalesces real useful verifier rows
before launching dense-important and 2:4 sparse branches.

### 18. Fixed-prefix route descriptor implementation

I added a conservative runtime data-format step in `vllm/vllm/speclink_sr24.py`:
`VerifyResidualPlan` now carries an optional `FixedPrefixRouteDescriptor` with
the fixed route shape:

```text
active_count, scheduled_width, valid_width, prefix, dense_width, base_width
```

The old `(residual_rows, base_rows)` tensors are still produced and passed
through unchanged. The row-routed fixed-block MLP now checks whether the
descriptor matches the row counts; if it matches, it uses the descriptor shape
directly instead of re-deriving the block layout from row tensor sizes. This is
not yet a fused operator, but it is the data-format boundary needed before the
hot path can stop depending on Python-built row lists.

Diagnostic smoke:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedprefix_descriptor_smoke_20260701_rerun
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/temp/sr24_fixedprefix_descriptor_smoke_20260701_rerun/speclink_t08/bs8/rep1/speclink_sr24_breakdown.json
```

The first attempt exposed an `UnboundLocalError` in the descriptor-hit counter;
that was fixed by moving `linear_breakdown` initialization before the descriptor
check. The rerun succeeded. Key counters:

| counter | value |
|---|---:|
| `scheduler_fixed_prefix_route_vectorized_builds` | 14 |
| `scheduler_fixed_prefix_route_contiguous_builds` | 14 |
| `row_routed_mlp_fixed_block_descriptor_hits` | 320 |
| `row_routed_mlp_fixed_block_descriptor_misses` | 0 |
| `row_routed_mlp_fixed_block_calls` | 320 |

Read: the descriptor is live in the real vLLM/SR24 fixed-block path and is
compatible with CUDA Graph FULL decode in the smoke. It does not claim a speedup
yet; it is a prerequisite for replacing row tensors with fixed-capacity route
slots and a grouped/fused dense-important plus 2:4 sparse MLP operator.

### 19. Fixed-prefix descriptor-only route plan

I added the first opt-in row-list removal step:

```text
SPECLINK_SR24_FIXED_PREFIX_ROUTE_DESCRIPTOR_ONLY=1
--sr24-fixed-prefix-route-descriptor-only
```

This flag is wired through both the GuideLLM throughput matrix runner and the
lm-eval accuracy runner. The `lossy_prefix2_rowrouted_mlp_operator_guard_compile`
candidate also carries the flag now, so future speed/quality checks use the same
data-format path. For descriptor-compatible fixed-prefix
`route_all_residual_rows + row_routed_mlp` plans, the scheduler passes only the
compact `FixedPrefixRouteDescriptor` through `VerifyResidualPlan` and skips
building residual/base row-index tensors and the direct bucket. The fixed-block
MLP then reconstructs dense/base slices from the descriptor shape.

Diagnostic smoke:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedprefix_descriptor_only_smoke_20260701_cli
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/temp/sr24_fixedprefix_descriptor_only_smoke_20260701_cli/speclink_t08/bs8/rep1/speclink_sr24_breakdown.json
```

Key counters from the `counts` object:

| counter | value |
|---|---:|
| `scheduler_fixed_prefix_route_descriptor_only_builds` | 12 |
| `scheduler_fixed_prefix_route_descriptor_only_plan_hits` | 12 |
| `scheduler_fixed_prefix_route_descriptor_only_skip_bucket` | 12 |
| `scheduler_fixed_prefix_route_vectorized_builds` | 0 |
| `cached_residual_rows_available_steps` | 0 |
| `cached_base_rows_available_steps` | 0 |
| `row_routed_mlp_fixed_block_descriptor_hits` | 320 |
| `row_routed_mlp_fixed_block_descriptor_misses` | 0 |

Read: the service now has a live descriptor-only fixed-prefix route path; the
old row-index tensors are not flowing through the verify plan for this shape.
This is still a data-format and scheduler-overhead step, not the final systems
optimization. The remaining speed path is to make the fixed descriptor feed a
graph-stable grouped/fused operator that can coalesce real useful dense and 2:4
sparse rows to roughly effective batch 64, while falling back to dense when the
branches are underfilled.

### 20. Descriptor reuse-base ablation

I extended the fixed-block descriptor path so
`SPECLINK_SR24_ROW_ROUTED_MLP_REUSE_BASE_OUTPUT=1` can run with descriptor-only
fixed-prefix plans. For CUDA Graph capture, descriptor-only fixed-prefix plans
now keep the residual mask but no longer store residual/base row-index tensors
in `VerifyResidualPlan`. The fixed-block MLP can then run the 2:4 sparse base
MLP on the full verifier block and overwrite only the dense-important
prefix/bonus rows.

Path smoke:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedprefix_reusebase_descriptor_smoke2_20260701
```

Key counters:

| counter | value |
|---|---:|
| `cudagraph_capture_fixed_prefix_descriptor_only_rows_skipped` | 10 |
| `scheduler_fixed_prefix_route_descriptor_only_plan_hits` | 14 |
| `row_routed_mlp_fixed_block_descriptor_hits` | 320 |
| `row_routed_mlp_fixed_block_reuse_base_output_calls` | 320 |
| `cached_residual_rows_available_steps` | 0 |
| `cached_base_rows_available_steps` | 0 |

Clean math_reasoning A/B, Llama-3.1-8B, K=8, max tokens 128, 64 total requests:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_descriptor_reusebase_clean_ab_bs32_64_math128_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_descriptor_disjoint_clean_bs32_64_math128_20260701
```

| bs | dense total/full tok/s | disjoint total/full tok/s | disjoint speedup | reuse-base total/full tok/s | reuse-base speedup |
|---:|---:|---:|---:|---:|---:|
| 32 | 1536.607 / 2155.389 | 1929.127 / 2318.133 | 1.255x / 1.075x | 1908.134 / 2261.575 | 1.242x / 1.049x |
| 64 | 2178.452 / 3025.653 | 1562.576 / 2910.591 | 0.717x / 0.962x | 1292.451 / 2620.169 | 0.593x / 0.866x |

Read: increasing sparse-base M by running sparse on the whole verifier block is
not the missing operator optimization. It is slightly worse than the disjoint
fixed-block path at bs32 and much worse at bs64. Keep reuse-base as an explicit
negative ablation only; do not use it as the next serving candidate. The useful
signal is that the disjoint descriptor path can improve total tokens/s at bs32
but still does not reach the 1.2x steady full-batch target, so the remaining
work is not just row-fill. It needs a real grouped/fused operator or scheduling
change that reduces branch launches and activation/assembly overhead without
duplicating sparse work on dense-important rows.

### 21. Compile-cache isolation and fixed-block input buffer

I fixed a benchmark correctness issue before trusting dense-vs-SR24 A/B rows:
SR24 and non-SR24 vLLM processes now use separate `VLLM_CACHE_ROOT`
fingerprints in both the GuideLLM matrix runner and the lm-eval accuracy
runner. This matters because vLLM's default compile-cache key does not include
`SPECLINK_SR24_*` env branches. A dense baseline launched after an SR24
`--sr24-default-vllm-compile` run could replay a graph that expects SR24-only
attributes such as `_speclink_sr24_sparse_base_weight`.

I also added `SPECLINK_SR24_FIXED_BLOCK_INPUT_BUFFER=1` for the fixed-block
row-routed MLP path. Instead of allocating and concatenating new dense/base
input tensors every call, it reuses fixed-capacity buffers and copies prefix,
promoted, bonus, and base rows into the appropriate ranges.

Path smoke:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedblock_inputbuf_smoke_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/temp/sr24_fixedblock_inputbuf_smoke_20260701/speclink_t08/bs8/rep1/speclink_sr24_breakdown.json
```

Key counters:

| counter | value |
|---|---:|
| `scheduler_fixed_prefix_route_descriptor_only_plan_hits` | 13 |
| `row_routed_mlp_fixed_block_descriptor_hits` | 320 |
| `row_routed_mlp_fixed_block_calls` | 320 |
| `row_routed_mlp_fixed_block_input_buffer_calls` | 320 |
| `row_routed_mlp_fixed_block_input_buffer_dense_rows` | 4896 |
| `row_routed_mlp_fixed_block_input_buffer_base_rows` | 9792 |

Quality gate:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_inputbuf_quality50_20260701
```

GSM8K-50 passes exactly: dense `0.7400`, SR24 `0.7400`, delta `0.0000pp`.

Clean Llama-3.1-8B math_reasoning, K=8, max tokens 128, 64 total requests:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_descriptor_inputbuf_clean_bs8_16_math128_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_descriptor_inputbuf_clean_bs32_64_cachefix_math128_20260701
```

| bs | dense total/full tok/s | input-buffer total/full tok/s | speedup total/full | accepted draft tokens/step dense/SR24 |
|---:|---:|---:|---:|---:|
| 8 | 859.605 / 1051.304 | 957.799 / 1055.993 | 1.114x / 1.004x | 1.418 / 1.394 |
| 16 | 1179.947 / 1663.707 | 1242.438 / 1592.869 | 1.053x / 0.957x | 1.437 / 1.426 |
| 32 | 1610.287 / 2209.855 | 1461.204 / 2657.114 | 0.907x / 1.202x | 1.413 / 2.381 |
| 64 | 1600.278 / 2906.912 | 1767.676 / 3400.508 | 1.105x / 1.170x | 1.429 / 2.333 |

Read: cache isolation is mandatory and now working; dense bs32/64 no longer
fails in the same run. The input-buffer path improves some allocator/assembly
overhead and can hit the 1.2x full-batch threshold at bs32, but it still misses
the requested total-throughput target and does not solve bs8/16. The remaining
problem is not that unimportant tokens are redundantly sent through dense; this
fixed-block path already avoids that. The next useful work is to make the mixed
MLP operator and scheduler fill real useful work to effective batch around 64,
or fall back to dense when that cannot be done.

Diagnostic note: `--sr24-breakdown` is incompatible with
`--sr24-default-vllm-compile` because the breakdown lock enters TorchDynamo's
compiled region. Use the graph-only diagnostic path for breakdown rows. The
first graph-only bs64 run with `gpu_memory_utilization=0.80` failed during KV
cache sizing after profile data was written, so I reran with 0.90:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_inputbuf_breakdown_graphonly_bs64_math64_gpu90_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/temp/sr24_inputbuf_breakdown_graphonly_bs64_math64_gpu90_20260701/speclink_t08/bs64/rep1/speclink_sr24_breakdown.json
```

The diagnostic run is not a clean throughput row; CUDA events slow it to
`555.910` tok/s. Its useful localization:

| item | value |
|---|---:|
| fixed-block calls | 1632 |
| dense rows per fixed-block call | 89.0 |
| base rows per fixed-block call | 178.0 |
| active requests per fixed-block call | 29.7 |
| base sparse Linear avg ms/call | 1.115 |
| gate_up base sparse avg ms/call | 1.191 |
| down base sparse avg ms/call | 1.040 |
| base sparse rows/call | 398.7 |
| base sparse total CUDA ms | 3926.0 |
| gate_up/down split total CUDA ms | 2095.6 / 1830.4 |

Read: the current breakdown confirms the remaining main cost is still the 2:4
sparse base branch, especially gate/up, even after fixed-block descriptor and
input-buffer changes. The next measurement should add dense-branch and
gather/scatter CUDA events inside the fixed-block MLP; without that, the
breakdown is too sparse-base-centric to close a PPoPP-style operator argument.

### 22. Python stream overlap is not the missing pipeline

I tested the fixed-block input-buffer path with the existing Python auxiliary
CUDA stream overlap path enabled for the dense-important branch.

Root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_inputbuf_overlap_clean_bs32_64_math128_20260701
```

Clean Llama-3.1-8B math_reasoning, K=8, max tokens 128:

| bs | dense total tok/s | overlap+input-buffer total tok/s | speedup |
|---:|---:|---:|---:|
| 32 | 1935.877 | 834.380 | 0.431x |
| 64 | 2209.130 | 1361.366 | 0.616x |

Read: Python-level stream overlap is strongly negative in the current vLLM hot
path. It breaks the graph-friendly fixed-block path and pays stream
synchronization/launch overhead without creating a useful pipeline. Do not use
this overlap path for throughput claims. If sparse and dense branches are to run
concurrently, it should be inside a graph-captured grouped/fused operator, not
Python-side stream orchestration.

### 23. Runtime base-only layer scope is not a valid current optimization

I added a diagnostic runtime scope that can force selected module leaves/layers
to skip dense residual correction, for example `down_proj=31`. This was meant to
test whether the tail of the verifier MLP could be made sparse-only while
keeping most correction elsewhere.

Default-compile probes failed badly:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_runtime_baseonly_gateup2631_quality50_rerun_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_runtime_baseonly_gateup31_quality50_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_runtime_baseonly_down31_quality50_20260701
```

Those rows produced SR24 GSM8K-50 around `0.12-0.14` against dense `0.74`, and
the vLLM logs showed TorchDynamo tracing the runtime branch helper. That points
to the same shared-graph hazard as earlier layer-heterogeneous formats.

I then reran a paired eager smoke to remove default compile from the explanation:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_runtime_baseonly_down31_paired_eager_smoke20_20260701
```

Even in eager mode, `down_proj=31` dropped GSM8K-20 from dense `0.7000` to SR24
`0.5500` (`-15pp`, paired regressions `4`, paired improvements `1`). This is
outside the allowed 8pp budget.

Read: do not use runtime base-only layer scope as the current speed path. The
runners now reject `--sr24-runtime-base-only-layer-ids-by-leaf` together with
`--sr24-default-vllm-compile`, because that combination can silently corrupt
quality. Eager-only runtime scope can remain a diagnostic, but it is not a
quality-passing optimization in the tested tail-layer forms.

## Current Decision

The bottleneck is not quality gating alone and not accepted draft length. The
quality-safe Python/PyTorch split designs are dominated by data format and
operator overhead:

- sparse-only can be fast when the whole MLP branch is base-only,
- quality-safe correction or channel-splitting adds enough work to erase the
  sparse win,
- layer-heterogeneous sparse/dense storage conflicts with default vLLM shared
  graph compilation,
- dummy padding or dense fill does not create enough useful work,
- Python-side sparse/dense stream overlap and runtime base-only layer scoping
  are both negative under current code,
- per-leaf dense fallback is a useful planner guardrail, but steady-state
  full-batch throughput still needs a better mixed operator,
- current `compressed_dense` residual correction is GPU-resident but still slow,
  and default compile can trace sparse weights incorrectly.

The next credible implementation should be operator-first:

1. Use one uniform graph-stable data format across layers.
2. Emit fixed-capacity route descriptors for verifier blocks instead of dynamic
   Python row lists.
3. Group real useful dense-important rows and sparse-only rows until the
   effective branch size is profitable; the current K=8/prefix2 microbench says
   that means effective batch around 64 for the MLP branch.
4. Use dense fallback when a branch is underfilled.
5. Implement dense-important and 2:4-sparse branches inside a captured grouped
   or fused operator; do not rely on Python stream overlap in the hot path.
6. If important rows are too few, coalesce useful rows across requests/layers or
   promote only enough low-priority rows to hit Tensor Core tile occupancy, with
   the promotion encoded in the route descriptor instead of Python-side
   scatter lists.
7. Treat the current 8pp budget as a quality constraint, not as a reason to run
   more scalar prefix sweeps. Any new candidate should first change the data
   layout or operator launch structure.

Do not spend more time on lossless gates, scalar threshold sweeps, channel-pair
fractions, or layer-heterogeneous default-compile candidates unless the graph
keying/data-format problem is fixed first.

## 2026-07-01 Current-Tree Correction

I reran the most aggressive noverify-sparse boundary candidate because it
looked attractive for total tokens/s but was too easy to misread:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_noverify_sparse_current_quality_gsm8k50_20260701/quality/lossy_prefix2_rowrouted_mlp_noverify_sparse_compile/report.md
```

Current-tree GSM8K-CoT `limit=50`, `max_new_tokens=512` result:

| candidate | dense acc | SR24 acc | delta | paired regressions | clipped SR24 |
|---|---:|---:|---:|---:|---:|
| `lossy_prefix2_rowrouted_mlp_noverify_sparse_compile` | 0.7800 | 0.1400 | -64pp | 35 | 11 |

This candidate is therefore rejected. The earlier same-name root that reported
`0.7400 -> 0.7400` should be treated as stale for the current tree or as not
actually exercising the current sparse-only noverify path. The fresh run shows
that broad noverify MLP sparsification changes the decoding trajectory badly:
output length increases, many samples clip at 512 tokens, and accuracy leaves
the allowed 8pp budget by a large margin.

I also attempted a small current-tree throughput rerun for the more conservative
`prefix2 + gate_up min_base=384 + descriptor/input-buffer` route:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup384_inputbuf_current_bs8_64_math512_20260701
```

That run is incomplete and should not be used as a comparison: it wrote dense
bs8/bs16 rows, then stopped after `dense_baseline bs32 server_start` with no
live vLLM process left. The valid current evidence for this branch remains the
earlier complete roots documented above: input-buffer/descriptor is a useful
data-format cleanup, but it does not solve low-batch steady-state throughput.

The actionable conclusion is unchanged but sharper:

1. The desired disjoint semantics are already present in the fixed-block path:
   important verifier rows go through dense, unimportant rows go through the
   2:4 branch, and unimportant rows are not followed by dense correction.
2. The broad noverify-sparse path is not quality-safe, even with an 8pp budget.
3. Prefix/threshold-only relaxation is not enough; the quality-safe live paths
   remain around parity except for tail effects.
4. The next useful implementation is an operator/scheduler change: a
   graph-stable fixed-capacity route descriptor, grouping of real useful rows to
   effective batch around 64, dense fallback when underfilled, and a grouped or
   fused dense-important plus 2:4-sparse-unimportant MLP operator. Python-side
   stream overlap remains a negative ablation and should not be used for claims.

## 2026-07-01 Scheduler Policy Artifact

I extended the packed-MLP planner so the microbench result is no longer only a
human-readable markdown table. The planner now also writes:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_packed_grouped_bucket_k8_bs8_64_prefix12_20260701/scheduler_policy.json
```

It is explicitly scoped as an operator-local policy, not a live serving claim.
The current K=8/prefix2 policy is:

| live bs | action | minimum grouped verifier blocks | target effective bs | mixed local speedup |
|---:|---|---:|---:|---:|
| 8 | dense fallback until grouped | 8 | 64 | 1.240x |
| 16 | dense fallback until grouped | 4 | 64 | 1.238x |
| 32 | dense fallback until grouped | 2 | 64 | 1.238x |
| 64 | use mixed single block | 1 | 64 | 1.240x |

This artifact is the concrete bridge to the next live implementation: the
scheduler should expose a queue of ready verifier MLP blocks keyed by module
weights and route descriptor. If the queue cannot satisfy
`min_grouped_verifier_blocks` within the latency budget, it should fall back to
dense for that block. If it can, the grouped operator should consume the fixed
descriptor and run one dense-important branch plus one 2:4-sparse branch over
the grouped rows. This keeps the user's required disjoint semantics while
avoiding the small-M sparse branch that has dominated the current live path.

## 2026-07-01 Live Scheduler-Policy Gate

I added the first live consumer for the planner artifact. It is deliberately
opt-in and conservative:

```text
SPECLINK_SR24_SCHEDULER_POLICY_PATH=/path/to/scheduler_policy.json
```

The GuideLLM matrix runner and the lm-eval runner expose this as:

```text
--sr24-scheduler-policy-path /path/to/scheduler_policy.json
```

When the fixed-block row-routed MLP path sees a compatible policy row, it checks
the current active request count, K, and protected prefix length. For the current
K=8/prefix2 planner, bs8/16/32 are classified as underfilled unless the future
scheduler has grouped enough ready verifier blocks. The current live gate cannot
perform that grouping yet, so it falls back to the dense MLP instead of running
the known-slow single-block mixed operator. The fallback is counted under:

```text
row_routed_mlp_fixed_block_scheduler_policy_dense_fallback
row_routed_mlp_full_dense_fallback_scheduler_policy_underfilled
```

The validation smoke was:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_scheduler_policy_gate_smoke_bs8_20260701_103939
```

It used bs8/K8/prefix2, loaded the policy JSON, and recorded 384 fixed-block
descriptor hits plus 384 scheduler-policy dense fallbacks in:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/temp/sr24_scheduler_policy_gate_smoke_bs8_20260701_103939/speclink_t08/bs8/rep1/speclink_sr24_breakdown.json
```

This smoke only proves that the opt-in policy gate is wired correctly. It should
not be used as a throughput claim: it ran with eager mode, linear breakdown, only
8 prompts, and 16 max tokens. The remaining optimization target is unchanged:
build the grouped verifier-MLP queue and grouped/fused operator so that the
policy can launch useful mixed work at effective bs64 instead of falling back to
dense at bs8/16/32.

The next check exposed a claim-safety issue in the first gate. The planner's
positive rows are for the offline `packed_parallel` packed verifier-block MLP,
but the current live fixed-block path is still the legacy split
dense/sparse/dense/sparse MLP. Therefore the live gate now treats a policy row
whose `mixed_operator` is not implemented by the serving path as dense fallback
unless this escape hatch is set explicitly:

```text
SPECLINK_SR24_SCHEDULER_POLICY_ALLOW_LEGACY_MIXED=1
```

A CPU directed test for bs64/K8/prefix2 produced exact dense output with:

```text
operator_unimplemented=1
dense_fallback=1
max_diff=0.0
```

The live smoke was:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_scheduler_policy_operator_gate_smoke_bs64_20260701_105426
```

Its breakdown shows the mixed path was reached, but every fixed-block MLP call
fell back because the required operator is still missing:

```text
mask_state_mixed = 41
row_routed_mlp_fixed_block_descriptor_hits = 1280
row_routed_mlp_fixed_block_scheduler_policy_operator_unimplemented = 1280
row_routed_mlp_fixed_block_scheduler_policy_dense_fallback = 1280
row_routed_mlp_full_dense_fallback_scheduler_policy_operator_unimplemented = 1280
```

This is not an optimization by itself. It prevents accidentally counting the
old split operator as the planner-backed packed operator. The next real
performance step is still to implement a live grouped/packed MLP operator or a
legal useful-row queue that can feed the existing packed-parallel operator
shape.

## 2026-07-01 Live Packed-Parallel Overlap Probe

I then connected the planner's `mixed_operator=packed_parallel` to the only
live path that currently resembles the offline microbench: fixed-prefix
row-routed MLP with CUDA stream overlap. This is still opt-in:

```text
SPECLINK_SR24_ROUTE_OVERLAP_STREAMS=1
```

For CUDA Graph replay, there is a second narrower opt-in:

```text
SPECLINK_SR24_ROUTE_OVERLAP_ALLOW_CUDAGRAPH=1
```

The guard remains conservative. Without overlap streams, a
`packed_parallel` policy row still falls back to dense. With overlap streams,
the policy can mark the fixed-block path as mixed-allowed. The first dispatch
smoke was:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_scheduler_policy_packed_parallel_smoke_bs64_20260701_110048
```

It proved the dispatch wiring:

```text
row_routed_mlp_fixed_block_scheduler_policy_hits = 992
row_routed_mlp_fixed_block_scheduler_policy_mixed_allowed = 768
row_routed_mlp_fixed_block_overlap_stream_calls = 768
row_routed_mlp_fixed_block_scheduler_policy_dense_fallback = 224
```

The 224 fallbacks are expected tail/underfilled cases. The 768 mixed calls used
the overlap-stream fixed-block path.

A graph-enabled smoke without linear breakdown was:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_scheduler_policy_packed_parallel_graph_smoke_nobreakdown_bs64_20260701_110858
```

It succeeded and showed partial graph replay:

```text
cudagraph modes = {"FULL": 7, "NONE": 24, "PIECEWISE": 1}
```

Running the same test with `--sr24-breakdown --sr24-breakdown-linear` failed at
server startup because Dynamo tried to compile `_breakdown_count`, hit the
Python lock context manager, and raised:

```text
torch._dynamo.exc.Unsupported: Unsupported context manager
```

So graph-enabled overlap probes must not use Python linear breakdown.

Finally, I ran a small bs64 speed sanity, still intermediate and not a final
matrix claim:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_packed_parallel_policy_speed_sanity_bs64_math128_20260701_111255
```

Setup: Llama-3.1-8B, EAGLE3 K=8, `math_reasoning`, bs64, 128 fixed requests,
max tokens 128, dense baseline versus `speclink_t08` with scheduler policy,
route overlap streams, and fixed-prefix graph opt-in.

| method | total tok/s | full-batch tok/s | accepted draft/step | GPU util |
|---|---:|---:|---:|---:|
| dense baseline | 2381.123 | 2762.610 | 1.407868 | 88.4% |
| speclink_t08 packed-parallel overlap | 1991.789 | 2603.799 | 1.407749 | 84.4% |

Speedup is only:

```text
total: 0.836x
full-batch: 0.943x
```

Acceptance length is essentially identical, so the slowdown is not caused by
lower draft acceptance. GPU utilization also drops. This means the current
Python/cuSPARSELt overlap-stream fixed-block path does not reproduce the
offline packed microbench's operator-level win in serving. The remaining work is
not another policy threshold: it is a lower-overhead grouped/fused operator or a
better small-M tensor-core sparse kernel that avoids the live split-path launch,
stream, and graph-coverage overheads.

### Negative Follow-Up Ablations

I ran three follow-ups to rule out cheaper explanations for the slowdown.

First, I disabled the no-verify dense MLP wrapper:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_noverify_mlp_fastpath_off_speed_sanity_bs64_math128_20260701_111912
```

| method | total tok/s | full-batch tok/s | accepted draft/step | GPU util |
|---|---:|---:|---:|---:|
| dense baseline | 2402.358 | 2774.843 | 1.409776 | 88.8% |
| SR24 noverify wrapper off | 1940.732 | 2516.314 | 1.426531 | 68.8% |

Speedup: `0.808x` total and `0.907x` full-batch. The wrapper is not the
slowdown source; disabling it lowers utilization.

Second, I enabled Triton fixed-block assembly:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_packed_parallel_triton_assembly_speed_sanity_bs64_math128_20260701_112222
```

| method | total tok/s | full-batch tok/s | accepted draft/step | GPU util |
|---|---:|---:|---:|---:|
| dense baseline | 2376.676 | 2766.415 | 1.408599 | 88.5% |
| SR24 + Triton assembly | 1909.590 | 2516.706 | 1.410455 | 69.0% |

Speedup: `0.803x` total and `0.910x` full-batch. Assembly is not the dominant
cost; the dense/sparse branch execution itself is the limiter.

Third, I rechecked the existing down-proj-only candidate under the same
current-tree bs64/max128 sanity:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down_only_current_speed_sanity_bs64_math128_20260701_112606
```

| method | total tok/s | full-batch tok/s | accepted draft/step | GPU util |
|---|---:|---:|---:|---:|
| dense baseline | 2376.276 | 2760.512 | 1.406547 | 88.5% |
| down-proj-only SR24 | 1872.761 | 2491.238 | 1.511968 | 62.9% |

Speedup: `0.788x` total and `0.902x` full-batch. This is important because
accepted draft length improved, yet throughput dropped. That directly answers
the active goal's first question for the current optimized candidates: slowdown
is not from lower acceptance length; it is from GPU underutilization and split
sparse execution overhead.

Current conclusion:

1. Base-only proves 2:4 can be fast, but fails quality.
2. All-corrected `compressed_dense` is already GPU-resident; its exact sparse
   residual path is too slow and compile-fragile.
3. `speclink_t08` quality-safe row routing preserves or improves accepted
   length, but live throughput loses because the current Python/cuSPARSELt
   split path underutilizes the GPU.
4. No-verify wrapper removal, Triton output assembly, stream overlap, and
   down-only policy tweaks do not fix the bottleneck.
5. The remaining credible path is a true grouped/fused sparse operator or
   better small-M 2:4 kernel with graph-stable fixed descriptors.

## 8pp Loss Budget Recheck

I re-ran a small quality-first gate after relaxing the target from lossless to
GSM8K accuracy loss within 8 percentage points. All runs used Llama-3.1-8B,
EAGLE3 K=8, GSM8K `limit=50`, and max new tokens 512. Throughput probes used
`math_reasoning`, max tokens 128, 96 fixed requests, one repeat, and
batch sizes 8/16/32/64. These are short operating-point probes, not final
median matrix results.

Quality roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lossy_speed_quality_sweep_20260701_113459
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lossy_speed_quality_sweep_20260701_114848
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lossy_speed_quality_sweep_20260701_121725
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lossy_speed_quality_sweep_20260701_122548
```

Throughput roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lossy_speed_quality_sweep_20260701_115959
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lossy_speed_quality_sweep_20260701_123325
```

Quality gate summary:

| candidate | dense acc | SR24 acc | delta pp | pass | interpretation |
|---|---:|---:|---:|---:|---|
| noverify sparse all MLP, prefix2 | 0.78 | 0.14 | -64 | no | too aggressive; model behavior breaks |
| noverify sparse all MLP, prefix1 | 0.78 | 0.16 | -62 | no | too aggressive |
| noverify sparse all MLP, dense-fill64 | 0.78 | 0.16 | -62 | no | dense-fill does not recover quality |
| criticalprefix4 bucket12 active-only | 0.78 | 0.78 | 0 | yes | quality-safe reference |
| gateup res16-25/base26-31 smallrow160 | 0.78 | 0.72 | -6 | yes | valid 8pp-loss candidate |
| gate_up-only noverify sparse | 0.78 | 0.40 | -38 | no | noverify sparse remains unsafe |
| front24 dense noverify, tail sparse | 0.78 | 0.66 | -12 | no | even layers 24-31 noverify sparse is too much |
| front16 dense noverify, tail sparse | 0.78 | 0.52 | -26 | no | worse tail-sparse quality |
| verifier-only row-route minbase64 | 0.78 | 0.76 | -2 | yes | only changing verifier rows is quality-feasible |
| verifier-only row-route minbase128 | 0.78 | 0.78 | 0 | yes | quality-safe verifier-only row route |

This isolates the quality boundary: verifier-only sparse routing can fit the
8pp budget, but sparse-only no-verify/no-mask MLP rows are not acceptable, even
when only the tail layers 24-31 are converted.

Short throughput summary for quality-passing candidates:

| candidate | bs8 total/full | bs16 total/full | bs32 total/full | bs64 total/full |
|---|---:|---:|---:|---:|
| criticalprefix4 bucket12 active-only | 0.901 / 0.893 | 0.945 / 0.957 | 0.939 / 0.912 | 1.195 / 1.104 |
| gateup res16-25/base26-31 smallrow160 | 0.918 / 0.954 | 0.790 / 0.920 | 0.848 / 0.938 | 1.040 / 0.969 |
| verifier-only row-route minbase64 | 0.854 / 0.952 | 0.819 / 0.954 | 0.781 / 0.914 | 1.006 / 0.994 |
| verifier-only row-route minbase128 | 0.853 / 0.955 | 0.811 / 0.936 | 0.815 / 0.946 | 0.831 / 0.900 |

The best short result is still only the conservative active-only candidate at
bs64 (`1.195x` total, `1.104x` full-batch). It does not meet the target of
`>=1.2x` on most batch sizes. The verifier-only row-route candidates answer the
"do not run dense after sparse for unimportant verifier tokens" question: the
semantics are quality-feasible, but the current implementation is slower at
bs8/16/32 and only breaks even at bs64. The limiting factor is therefore the
operator/data-format path, not the controller threshold.

Next implementation direction:

1. Keep no-verify/no-mask MLP dense by default. Do not spend time on broader
   noverify sparse policies until there is a better importance signal or a
   much narrower layer scope than 24-31.
2. Keep verifier-only disjoint routing as the semantic target: important rows
   run dense, unimportant rows run 2:4 sparse, with no dense recompute for
   unimportant rows.
3. Replace the current Python/cuSPARSELt split path with a graph-stable packed
   verifier-block data format. The scheduler should emit compact fixed
   descriptors rather than dynamic row-index tensors.
4. Fuse the MLP route into one grouped operator per layer: pack dense-important
   rows and sparse-unimportant rows into contiguous fixed blocks, run the dense
   and 2:4 branches from persistent workspaces, and assemble output in the same
   kernel or with a single fused epilogue.
5. For small important-row counts, use controlled dense-fill only inside the
   verifier block. Promoted rows are a tile-fill optimization and should be
   reported separately; they must not imply noverify sparse.
6. Treat overlap streams as a later operator-level optimization. The current
   live stream path does not help because launch/graph coverage and split
   sparse execution dominate.

## Packed Verifier-Block Coalescing Probe

I ran a focused packed-verifier MLP microbenchmark for the verifier-only
fixed-prefix2 route, K=8, Llama MLP shape, bf16, and capacity multiple 128:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_packed_verifier_mlp_prefix2_coalesce_probe_20260701_cont
```

The derived planner is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_packed_verifier_mlp_prefix2_coalesce_probe_20260701_cont/planner_1p2/operator_planner.md
```

Operator-level `packed_parallel` speedups versus dense were:

| bs | coalesce | effective bs | dense rows | base rows | packed_parallel / dense | local speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 8 | 24 | 48 | 1.662x slower | 0.602x |
| 8 | 4 | 32 | 96 | 192 | 1.063x slower | 0.941x |
| 16 | 1 | 16 | 48 | 96 | 1.405x slower | 0.712x |
| 16 | 4 | 64 | 192 | 384 | 0.811x of dense time | 1.234x |
| 32 | 1 | 32 | 96 | 192 | 1.059x slower | 0.944x |
| 32 | 2 | 64 | 192 | 384 | 0.810x of dense time | 1.235x |
| 64 | 1 | 64 | 192 | 384 | 0.808x of dense time | 1.238x |

The planner implication is:

| live batch | single-block action | grouping needed for 1.2x local speedup |
|---:|---|---:|
| 8 | dense fallback | none observed locally; serial upper bound needs 4 |
| 16 | dense fallback until grouped | 4 verifier blocks |
| 32 | dense fallback until grouped | 2 verifier blocks |
| 64 | use mixed single block | 1 verifier block |

This gives a concrete systems target for a PPoPP-style implementation:
small/medium batch cannot win with a single verifier block. It needs a
dependency-safe useful-row coalescer that batches same-layer verifier blocks
until effective bs is about 64, or it should fall back to dense.

I also ran a live bs64 serving check for the quality-feasible verifier-only
`minbase64` policy with `--sr24-route-overlap-streams` and graph opt-in:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_verifier_rowroute_minbase64_overlap_bs64_20260701_cont
```

| method | total tok/s | full-batch tok/s | accepted draft/step | GPU util | graph |
|---|---:|---:|---:|---:|---|
| dense baseline | 2192.026 | 2824.053 | 1.386 | 85.6% | |
| speclink_t08 + overlap | 1797.924 | 2603.314 | 1.408 | 70.7% | `{"FULL":44,"PIECEWISE":44}` |

Speedup was only `0.820x` total and `0.922x` full-batch. The accepted length
improved, but utilization dropped. This rules out the current Python-level
overlap streams as the solution: the operator microbench shows that useful-row
fill can be profitable, while live serving shows that the current split-path
implementation cannot realize that profit.

Updated engineering direction:

1. Add a live useful-row coalescing scheduler only if it can preserve decode
   dependencies and latency bounds. The coalescer should target effective
   verifier bs around 64 for prefix2/K8.
2. Move from stream-level overlap to a fused or grouped MLP operator interface:
   one graph-stable descriptor, persistent dense/base workspaces, no Python
   row-index construction in the hot path, and a fused final assembly.
3. Keep the current verifier-only row-route as a correctness/reference path,
   not as the performance path.

## 2026-07-01 Small-Scale Validation After the 8pp Relaxation

The user requested that future optimization stop requiring lossless accuracy,
allow up to about 8 percentage points of accuracy loss, and first validate on a
small scale before expanding to the full matrix.  I reran three focused checks.
All outputs are under `results.bak/` rather than `results/`.

### Row-routed operator guard throughput

Command output:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_operator_guard_math_bs8_64_20260701
```

Setup: Llama-3.1-8B, `math_reasoning`, EAGLE3 K=8,
`max_tokens=128`, fixed 128 total requests, client concurrency
`bs=8,16,32,64`, preset `lossy_prefix2_rowrouted_mlp_operator_guard`.

| bs | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1025.139 | 924.487 | 0.902x | 1070.027 | 1025.848 | 0.959x |
| 16 | 1637.179 | 1404.287 | 0.858x | 1718.162 | 1692.475 | 0.985x |
| 32 | 2053.162 | 1726.238 | 0.841x | 2271.186 | 2111.821 | 0.930x |
| 64 | 2373.936 | 2404.229 | 1.013x | 2774.867 | 2776.823 | 1.001x |

Interpretation: the current row-route path does avoid the core logical waste
for bs64: important rows go through dense MLP and base rows go through sparse
MLP, instead of running dense for every row.  However, the live split operator
only reaches parity.  For bs8/16/32 the operator guard mostly falls back to
dense because the sparse-base branch is underfilled, and the remaining SR24
hook/graph overhead makes it slower.

### Eager breakdown of the bs64 row-route path

Command output:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_operator_guard_bs64_breakdown_eager_20260701
```

This run used `--sr24-force-eager-after-preset --sr24-breakdown
--sr24-breakdown-linear --sr24-breakdown-exact-routing`, so the throughput is
diagnostic only.  Important breakdown values:

| component | value |
|---|---:|
| residual draft fraction | 0.25 |
| residual non-draft fraction | 1.00 |
| scheduler mask build CPU time | 7.77 ms/step |
| scheduler request routing loop | 7.75 ms/step |
| row-routed mixed calls | 480 |
| row-routed full-dense fallback calls | 736 |
| row-routed base gate/up sparse | 549.0 ms total |
| row-routed base down sparse | 474.4 ms total |
| row-routed dense gate/up | 111.3 ms total |
| row-routed dense down | 56.0 ms total |
| row-route build | 45.3 ms total |
| noverify dense gate/up | 529.2 ms total |
| noverify dense down | 275.0 ms total |

Interpretation: the row-routed verifier block is compute-heavy on the sparse
base branch, not on the dense important rows.  The current live path also pays
large scheduler Python time in eager diagnostics and still has many
underfilled/full-dense fallback calls.  This supports the systems direction:
the data format must become a graph-stable packed verifier-block descriptor,
and the operator must consume grouped blocks with persistent dense/base
workspaces.  It is not enough to tune the confidence threshold.

### GSM8K-50 quality check

Command output:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_operator_guard_gsm8k50_quality_20260701
```

Setup: Llama-3.1-8B, `gsm8k_cot`, `limit=50`, EAGLE3 K=8,
`max_new_tokens=512`, preset `lossy_prefix2_rowrouted_mlp_operator_guard`.

| mode | exact match | delta vs dense | paired regressions | paired improvements |
|---|---:|---:|---:|---:|
| dense_baseline | 0.7800 | 0.0000 pp | 0 | 0 |
| speclink_t08 | 0.7800 | 0.0000 pp | 0 | 0 |

Interpretation: the current row-route policy is inside the relaxed 8pp budget
on this GSM8K-50 gate.  Quality is not the bottleneck for this candidate; the
bottleneck is the operator/scheduler implementation.

### Static `accuracy_first` counterexample

Command output:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_first_math_bs8_64_20260701
```

Setup matches the row-route throughput run, but uses preset `accuracy_first`.

| bs | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1017.430 | 887.341 | 0.872x | 1056.651 | 924.690 | 0.875x |
| 16 | 1659.765 | 1434.648 | 0.864x | 1745.578 | 1567.125 | 0.898x |
| 32 | 2086.859 | 1875.305 | 0.899x | 2262.257 | 2172.957 | 0.961x |
| 64 | 2395.405 | 2224.725 | 0.929x | 2786.353 | 2700.762 | 0.969x |

Interpretation: a simple static sparse tail does not deliver the target on the
current tree.  It has good graph coverage, but the sparse MLP work is not
cheaper enough to overcome hook and sparse-dispatch cost.

### Updated design target

The current implementation already has the first required semantic
optimization for the verifier block: important rows and unimportant rows can
be disjoint, so an unimportant row that has already used sparse does not also
need dense correction.  The missing piece is a systems-quality operator and
scheduler around that semantic split.

Concrete next implementation target:

1. Data format: scheduler emits one compact `route_table[B, K+1]` or fixed
   block descriptor with row kind, valid width, dense prefix, sparse tail, and
   bonus row.  Avoid per-step `nonzero`, `index_select`, and Python list
   routing in the hot path.
2. Fill policy: if important rows are too few, group multiple verifier blocks
   until effective bs is about 64, or fall back to dense.  Do not promote
   unimportant rows to dense unless the promotion is explicitly counted as
   tile fill.
3. Operator: one grouped MLP entry point per layer, with persistent dense and
   sparse workspaces.  Dense-important rows and sparse-unimportant rows should
   be launched as one packed operator or as graph-captured sibling kernels with
   a fused assembly epilogue.
4. Concurrency: the dense branch and 2:4 branch can overlap only after launch
   overhead is amortized.  Current Python stream overlap is not enough; the
   grouped operator must own the overlap or use CUDA Graph-captured launches.
5. Quality gate: keep GSM8K-50 as the first fast gate with allowed drop <=8pp,
   then expand to the broader lm-eval and throughput matrices only after a
   candidate shows at least local bs64 operator speedup and non-regressing
   GSM8K-50 quality.

## 2026-07-01 follow-up: accepted length is not the `base_only_24` problem

Command output:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_current_math_bs8_64_20260701
```

Setup: Llama-3.1-8B, `math_reasoning`, EAGLE3 K=8, max tokens 128,
fixed 128 requests, `base_only_24` with graph-enabled direct cuSPARSELt.

| bs | dense total tok/s | base_only total tok/s | total speedup | dense full tok/s | base_only full tok/s | full speedup | dense accepted draft/step | base_only accepted draft/step | dense avg GPU util | base_only avg GPU util |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1021.023 | 607.731 | 0.595x | 1062.820 | 633.611 | 0.596x | 1.406 | 2.782 | 95.0% | 47.4% |
| 16 | 1626.972 | 852.059 | 0.524x | 1727.116 | 907.700 | 0.526x | 1.413 | 2.718 | 94.1% | 44.7% |
| 32 | 2079.753 | 1252.879 | 0.602x | 2264.923 | 1335.886 | 0.590x | 1.409 | 2.678 | 92.8% | 38.9% |
| 64 | 2369.916 | 2089.582 | 0.882x | 2758.078 | 2496.473 | 0.905x | 1.405 | 2.666 | 90.1% | 52.8% |

Interpretation: `base_only_24` is not slow because accepted draft length is
lower.  It accepts about 1.9x as many draft tokens per speculative step as the
dense verifier.  It is slow because the sparse verifier path underutilizes the
GPU and/or launches underfilled sparse work.  This points away from token
selection as the first bottleneck and toward operator occupancy, launch
amortization, and graph-stable sparse data layout.

## 2026-07-01 follow-up: true `all_corrected_24` is not a speed path yet

Dense-equivalent control output:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_densefastpath_control_math_bs32_64_20260701
```

True sparse-base plus sparse-residual MLP-only output:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_true_mlp_torchsparse_math_bs32_64_20260701
```

| variant | bs | dense full tok/s | SR24 full tok/s | full speedup | avg GPU util | backend | storage/dense |
|---|---:|---:|---:|---:|---:|---|---:|
| densefastpath control | 32 | 2278.282 | 2023.312 | 0.888x | 64.7% | torch_sparse/dense_fastpath | 1.000 |
| densefastpath control | 64 | 3024.969 | 3024.613 | 1.000x | 80.5% | torch_sparse/dense_fastpath | 1.000 |
| true MLP sparse+residual | 32 | 2270.307 | 1129.383 | 0.497x | 73.5% | torch_sparse/torch_sparse@cuda | 1.188 |
| true MLP sparse+residual | 64 | 3038.687 | 2290.386 | 0.754x | 90.5% | torch_sparse/torch_sparse@cuda | 1.188 |

Interpretation: an exact `all_corrected_24` decomposition currently costs too
much.  Even when CUDA Graph is active and GPU util is high at bs64, doing sparse
base plus sparse residual correction is slower than a single dense GEMM stack.
The densefastpath control also shows that some hook/compile overhead exists at
bs32, but the main issue is the true two-operator decomposition.

GPU-resident `compressed_dense` check output:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_compresseddense_gpu_gateup31_bs64_20260701
```

This narrow one-layer gate-up case confirmed the required residency path:
`sr24_compressed_residual_runtime_on_gpu=True`,
`sr24_residual_device_counts={"cuda:0": 1}`, and no CPU extraction fallback.
Throughput was still only 2755.764 full-batch tok/s vs 3047.348 dense
(0.904x).  So `compressed_dense` can be kept on GPU, but the current residual
materialization/correction shape is not enough to produce speedup.

## 2026-07-01 follow-up: more aggressive row routing did not help

`prefix=1` row-routed output:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix1_rowrouted_math_bs16_64_perf_20260701
```

This variant protects only one draft row plus the verifier bonus with dense
MLP; all other draft rows are sparse-only.  It is more aggressive than the
existing `prefix=2` operator-guard preset and is meant to use the relaxed 8pp
quality budget.

| bs | dense full tok/s | prefix=1 full tok/s | full speedup | dense accepted draft/step | prefix=1 accepted draft/step | prefix=1 avg GPU util |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1700.958 | 1514.808 | 0.891x | 1.402 | 1.388 | 67.8% |
| 32 | 2284.474 | 2076.037 | 0.909x | 1.392 | 1.395 | 58.2% |
| 64 | 3042.646 | 2439.353 | 0.802x | 1.426 | 1.401 | 69.5% |

Interpretation: reducing important/dense rows alone is not the missing
optimization.  It lowers GPU utilization and does not improve acceptance.
This again points to the current row-routed sparse branch being too fragmented
and underfilled.

Overlap-streams output:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_overlap_bs64_probe_20260701
```

At bs64, enabling `--sr24-route-overlap-streams` and
`--sr24-route-overlap-allow-cudagraph` reached only 2300.023 full-batch tok/s
vs 3045.648 dense (0.755x) with 52.0% average GPU util.  The current overlap
implementation is therefore not a useful final design; it adds separate stream
work without a packed high-occupancy schedule.

## Revised systems direction

The path that matches the relaxed accuracy target and a PPoPP-style systems
story is not another threshold sweep.  The next implementation should change
the verifier-block data format and the operator contract:

1. Scheduler data format: emit a compact fixed-shape verifier-block descriptor
   per active request, e.g. `{dense_prefix, sparse_tail, bonus_dense,
   valid_width}` plus packed row offsets.  Avoid per-step Python lists,
   `nonzero`, dynamic `index_select`, and ad hoc bucket row construction.
2. Operator format: consume grouped verifier blocks, not single small row
   lists.  The sparse branch should see enough rows to occupy the 2:4 kernel;
   otherwise the whole block should fall back to dense and be counted as such.
3. Disjoint compute: important rows use dense once; unimportant rows use sparse
   once.  There should be no dense correction for sparse-only rows in the lossy
   path.
4. Fill policy: if important rows are too few, fill at the block/operator level
   by grouping requests or consecutive verifier blocks.  Do not promote
   low-priority rows to dense unless that promotion is an explicit tile-fill
   ablation.
5. Concurrency: dense-important and sparse-unimportant work should be fused
   into one grouped MLP entrypoint, or captured as sibling kernels with
   persistent workspaces and a fused scatter/epilogue.  The current Python
   stream overlap is insufficient.
6. Evaluation gate: allow up to 8pp absolute accuracy loss, with GSM8K
   `limit=50` as the first quality gate.  Only run the larger bs8/16/32/64 and
   multi-dataset matrix after a candidate beats dense locally, especially at
   bs32/64.

## 2026-07-01 implementation update: fixed-prefix descriptor now reaches fixed-block MLP

Code change:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/vllm/vllm/speclink_sr24.py
```

The row-routed MLP path previously tried `_row_routed_mlp_fixed_block_output`
only when the scheduler did not also provide dense/base row tensors.  In the
normal non-exact serving path, the scheduler can provide a compact
fixed-prefix route descriptor and still leave row tensors available, so the
generic row-list path could run first.  The code now tries the fixed-block path
whenever `_current_fixed_prefix_route()` is present; the fixed-block function
still validates descriptor shape and falls back if incompatible.

Descriptor-hit diagnostic output:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedblock_patch_nonexact_breakdown_bs64_20260701
```

Important non-exact breakdown counters:

| counter | value |
|---|---:|
| `row_routed_mlp_fixed_block_dense_gate_up_cuda_ms` | 98.53 ms / 480 calls |
| `row_routed_mlp_fixed_block_base_gate_up_cuda_ms` | 438.62 ms / 480 calls |
| `row_routed_mlp_fixed_block_dense_down_cuda_ms` | 57.58 ms / 480 calls |
| `row_routed_mlp_fixed_block_base_down_sparse_cuda_ms` | 462.24 ms / 480 calls |
| `row_routed_mlp_fixed_block_assemble_cuda_ms` | 5.54 ms / 480 calls |
| `scheduler_fixed_prefix_route_descriptor_only_cpu_ms` | 0.61 ms / 39 calls |
| `scheduler_fixed_prefix_route_descriptor_only_cuda_ms` | 0.04 ms / 39 calls |

This confirms the fixed-block descriptor path is active in the real serving
route.  The earlier exact-routing breakdown intentionally disabled this path,
so it is only useful for old row-list diagnostics.

End-to-end throughput output after the code change:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedblock_patch_math_bs32_64_perf_20260701
```

| bs | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup | dense accepted draft/step | SR24 accepted draft/step | SR24 avg GPU util |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1971.420 | 2022.173 | 1.026x | 2279.155 | 2302.622 | 1.010x | 1.385 | 1.406 | 89.2% |
| 64 | 2224.192 | 2144.954 | 0.964x | 2857.991 | 2879.168 | 1.007x | 1.394 | 1.410 | 83.0% |

Compared with the previous operator-guard run, bs32 full-batch improved from
2111.821 to 2302.622 tok/s, and bs64 full-batch improved from 2776.823 to
2879.168 tok/s.  The fix removes unnecessary dynamic row-list routing overhead
and makes the compact descriptor path real, but it still only reaches parity
with dense, not the 1.2x target.

Follow-up backend probes after the fixed-block fix:

| probe | output | result |
|---|---|---|
| Triton fixed-block assembly | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedblock_patch_triton_assembly_bs32_64_20260701` | No stable gain; bs32/64 full-batch were essentially dense parity. |
| overlap streams | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedblock_patch_overlap_bs32_64_20260701` | No stable gain; bs64 total improved in one run, but full-batch fell below the default fixed-block path. |
| cuSPARSELt small-M `alg_id=1` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedblock_patch_alg1_bs64_probe_20260701` | Similar to default; not enough to justify making it default. |
| cuSPARSELt small-M `alg_id=2` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedblock_patch_alg2_bs64_probe_20260701` | Similar to `alg_id=1`; no clear win. |
| bs32 `route_min_base_rows=128` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedblock_patch_minbase128_bs32_probe_20260701` | Worse: bs32 full-batch 2071.212 vs dense 2268.528.  bs32 sparse tail remains underfilled, so the default 384-row dense fallback is appropriate. |

Current bottleneck after the fix: the fixed-block sparse base MLP itself.
Even with descriptor routing, bs64 spends most diagnostic CUDA time in
`base_gate_up` and `base_down_sparse`.  The next aligned optimization is a
real grouped/fused sparse MLP operator: keep the fixed descriptor interface,
but replace per-layer dense/base gather -> separate GEMM/2:4 -> activation ->
separate down -> assemble with one grouped operator or graph-captured sibling
kernels sharing persistent workspaces.  Without that operator work, threshold
or bucket tuning is unlikely to reach the requested 1.2x speedup.

## 2026-07-01 preset sweep after the fixed-block fix

The fixed-block route fix made `lossy_prefix2_rowrouted_mlp_operator_guard`
the current best local candidate, so the next check was whether older lossy
presets with smaller residual scope or tile-filled dense correction could do
better under the same current code.

All runs used Llama-3.1-8B, `math_reasoning`, EAGLE3 K=8, max tokens 128,
fixed 96 requests, and bs32/64.

| preset | output root | bs32 total | bs32 full | bs64 total | bs64 full | takeaway |
|---|---|---:|---:|---:|---:|---|
| `lossy_prefix2_rowrouted_mlp_operator_guard` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedblock_patch_math_bs32_64_perf_20260701` | 1.026x | 1.010x | 0.964x | 1.007x | Best current candidate; near dense parity, not 1.2x. |
| `mlpall_direct_prefix2` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_direct_prefix2_math_bs32_64_20260701` | 0.777x | 0.836x | 0.879x | 0.975x | Higher accepted length, but operator overhead dominates. |
| `lowresidual_gateup_riskcap2` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lowresidual_gateup_riskcap2_math_bs32_64_20260701` | 0.764x | 0.824x | 0.806x | 0.885x | Smaller scope lowers GPU utilization too much. |
| `gateup_res16_25_base26_31_critical4` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_res16_25_base26_31_critical4_math_bs32_64_20260701` | 0.764x | 0.899x | 1.032x | 0.978x | Some bs64 total-token gain, but full-batch still below dense. |
| `mlpall_tilefill_prefix2_bucket32_cublas` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_tilefill_prefix2_bucket32_cublas_math_bs32_64_20260701` | 0.845x | 0.925x | 0.771x | 0.893x | Tile-filled dense correction does not amortize the extra work. |

Quality gate for the current best candidate:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedblock_patch_gsm8k50_quality_retry_20260701
```

| mode | exact match | delta vs dense | paired regressions | paired improvements |
|---|---:|---:|---:|---:|
| dense_baseline | 0.7800 | 0.0000 pp | 0 | 0 |
| `lossy_prefix2_rowrouted_mlp_operator_guard` | 0.7800 | 0.0000 pp | 0 | 0 |

Conclusion from this sweep: the current algorithmic policy is inside the
relaxed 8pp GSM8K-50 budget, but all existing Python/Torch/cuSPARSELt routing
variants remain at or below dense parity.  Existing preset tuning is exhausted
for the 1.2x goal.  The next implementation step should be a systems operator:
one grouped fixed-block MLP kernel/entrypoint that consumes the descriptor
directly, keeps dense-prefix and sparse-tail work in persistent workspaces, and
avoids per-layer Python/Torch gather/copy/scatter.  This operator also needs an
explicit fill policy: if sparse-tail rows do not fill the 2:4 kernel enough,
fall back to dense or group multiple verifier blocks before launching sparse
work.

## 2026-07-01 lossy follow-up: higher K and layer-scoped MLP base-only

The next question was whether the current speed gap is mainly due to too few
useful sparse rows.  I tested three variants:

1. `speclink_t08` with all MLP rows sparse-only (`no_residual`) at K=8.
2. The same `speclink_t08` path at K=16, to increase sparse-tail fill.
3. `base_only_24` on only tail MLP layers, leaving earlier MLP layers dense.

### K=8 no-residual MLP `speclink_t08`

Roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_mlp_noresidual_graphonly_bs16_retry2_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_mlp_noresidual_graphonly_bs32_retry2_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_mlp_noresidual_graphonly_bs64_retry2_withdense_20260701
```

The bs64 rerun included an in-run dense baseline:

| bs | method | total tok/s | full tok/s | accepted draft/step | avg GPU util |
|---:|---|---:|---:|---:|---:|
| 16 | `speclink_t08` no-residual MLP | 1240.215 | 1290.337 | 2.499 | 63.8% |
| 32 | `speclink_t08` no-residual MLP | 1810.081 | 1949.251 | 2.418 | 62.9% |
| 64 | dense baseline | 2405.994 | 2779.230 | 1.410 | 89.4% |
| 64 | `speclink_t08` no-residual MLP | 2696.725 | 3206.573 | 2.492 | 70.9% |

At bs64 this reaches only `1.121x` total and `1.154x` full-batch speedup.
The higher accepted length is real, but GPU utilization drops sharply because
the MLP sparse branch is still an under-optimized separate branch.  At bs16 and
bs32 it is slower than the K=8 dense baseline.

### K=16 fill probe

Root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_mlp_noresidual_graphonly_k16_bs16_64_probe_20260701
```

| bs | dense total | SR24 total | total speedup vs K=16 dense | dense full | SR24 full | full speedup vs K=16 dense |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 972.015 | 1058.304 | 1.089x | 1057.746 | 1113.590 | 1.053x |
| 32 | 1152.767 | 1524.236 | 1.322x | 1303.506 | 1659.915 | 1.273x |
| 64 | 1214.615 | 1948.378 | 1.604x | 1506.310 | 2406.958 | 1.598x |

K=16 proves the fill hypothesis locally: the sparse path benefits when many
more rows are presented.  But the K=16 dense baseline is itself much slower
than K=8 dense EAGLE3, and K=16 SR24 still does not beat the K=8 dense baseline
at bs32/64.  Therefore "increase K" is not a serving solution; the correct
systems direction is to group useful verifier rows across requests/steps while
keeping the externally chosen K at the quality/performance optimum.

### Layer-scoped MLP base-only quality boundary

Full MLP base-only with dense attention was fast at bs64, but quality failed:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_attn_dense_mlp_sparse_gsm8k50_quality_20260701
```

GSM8K-50 dropped from `0.7200` to `0.2200` (`-50pp`), so it is only an upper
bound.

Tail16 MLP base-only was still too aggressive:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_mlp_tail16_gsm8k50_quality_20260701
```

It dropped from `0.7200` to `0.5200` (`-20pp`).

Tail8 MLP base-only passed the 8pp gate:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_mlp_tail8_gsm8k50_quality_20260701
```

It dropped from `0.7200` to `0.7000` (`-2pp`).

Throughput root for tail8:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_mlp_tail8_math_bs8_64_speed_20260701
```

| bs | dense total | tail8 total | total speedup | dense full | tail8 full | full speedup | tail8 accepted draft/step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1024.630 | 863.268 | 0.843x | 1066.000 | 900.654 | 0.845x | 1.510 |
| 16 | 1620.865 | 1459.149 | 0.900x | 1721.758 | 1562.547 | 0.908x | 1.533 |
| 32 | 2078.795 | 2022.582 | 0.973x | 2262.339 | 2277.153 | 1.007x | 1.544 |
| 64 | 2397.594 | 2412.325 | 1.006x | 2772.886 | 2867.055 | 1.034x | 1.541 |

Read: quality-safe static layer pruning does not cover enough compute to matter.
It slightly raises accepted length but has no meaningful throughput gain.

## Current conclusion

The experiments support a narrower systems claim:

- The 2:4 sparse math can be useful when it is fed enough rows.
- Accuracy-safe static layer scopes are too small to produce speed.
- Aggressive full-MLP sparse scopes are fast but fail GSM8K by far more than
  the 8pp budget.
- Raising K fills kernels but destroys the K=8 dense baseline, so it is not a
  fair or useful serving solution.

The next real implementation should be a grouped verifier-block path, not
another threshold or layer sweep.  The design should keep the current
fixed-prefix descriptor, but add a queue or packing buffer that groups useful
verifier blocks until the sparse-tail branch reaches an effective batch around
64.  If a block cannot be filled cheaply, the planner should use dense fallback.
The grouped operator should consume compact descriptors and persistent
workspaces directly, so the hot path avoids CPU-side row-list construction,
`index_select`, scattered small GEMMs, and separate Python-launched dense/sparse
branches.  Dense-important and sparse-unimportant work can then be launched as
graph-captured sibling kernels or a fused grouped MLP entrypoint; the current
Python stream-overlap experiment is not enough.

## 2026-07-01 grouping-opportunity live probe

I added a disabled-by-default live trace for the fixed-prefix row-routed SR24
path:

```text
SPECLINK_SR24_GROUPING_TRACE=1
SPECLINK_SR24_GROUPING_TRACE_PATH=/path/to/speclink_sr24_grouping_trace.jsonl
```

The trace records one row per verify-planning step: compact verifier-block
shape, active verifier blocks, fixed-prefix dense/base rows, the packed-MLP
planner's minimum grouping requirement, and whether the step could feed the
grouped operator if that operator existed.  The matrix runner exposes this as
`--sr24-grouping-trace` and resolves `--sr24-scheduler-policy-path` before
passing it into the vLLM server, because the server runs from
`examples/evaluate/eval-guidellm`.

Roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_grouping_opportunity_trace_2048_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_grouping_opportunity_trace_2048_fixed_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_grouping_overlap_streams_probe_2048_20260701
```

The first root is a negative setup run: SR24 failed at `max_model_len=4096`
because the SR24 dense-row residual storage left no KV-cache blocks.  The
comparable successful probe uses `max_model_len=2048`, `K=8`,
`max_tokens=256`, 64 total requests, and `math_reasoning`.

| bs | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup | dense accepted/step | SR24 accepted/step |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1145.145 | 1015.517 | 0.887x | 1235.207 | 1195.035 | 0.967x | 1.763 | 1.764 |
| 16 | 1630.353 | 1541.389 | 0.945x | 1933.201 | 1933.297 | 1.000x | 1.723 | 1.809 |
| 32 | 2221.958 | 1779.809 | 0.801x | 2714.802 | 2426.725 | 0.894x | 1.747 | 1.740 |
| 64 | 2674.521 | 2695.074 | 1.008x | 3525.982 | 3489.043 | 0.990x | 1.751 | 1.745 |

This again rules out accepted-length loss as the main slowdown.  The accepted
draft tokens per step are essentially unchanged, but the current live operator
does not outperform dense.

Grouping opportunity from the trace, after correcting the interpretation:

| bs | trace events | compact step % | avg active blocks | p50 active | p90 active | max active | raw count-ready % |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 823 | 93.2 | 7.30 | 8 | 8 | 8 | 72.7 |
| 16 | 419 | 89.3 | 14.13 | 16 | 16 | 16 | 81.9 |
| 32 | 262 | 90.8 | 23.15 | 30 | 32 | 32 | 71.0 |
| 64 | 132 | 99.2 | 45.82 | 62 | 64 | 64 | 86.4 |

The `raw count-ready %` column is intentionally marked raw: it only asks
whether the current live step has enough active requests to fill the row count.
It is not true packed-operator readiness.  The planner's
`min_grouped_verifier_blocks` means grouped verifier blocks or grouped steps,
not active requests inside one verifier block.  The current live scheduler
produces one compact verifier block per step, so bs8/16/32 still need a
cross-step or cross-block queue:

| live bs | planner action | verifier blocks needed | current live blocks | missing blocks |
|---:|---|---:|---:|---:|
| 8 | dense fallback until grouped | 8 | 1 | 7 |
| 16 | dense fallback until grouped | 4 | 1 | 3 |
| 32 | dense fallback until grouped | 2 | 1 | 1 |
| 64 | use mixed single block | 1 | 1 | 0 |

The trace now records this distinction explicitly with
`count_only_request_fill_ready`, `grouping_ready_now`,
`missing_grouped_verifier_blocks`, `block_group_fill_ratio`, and
`grouping_requires_cross_step_queue`.  For the `packed_parallel` planner target,
`operator_supported_live=0%` in the current code, so the bs8/16/32 rows must
fall back to dense unless a real grouped verifier-block queue and operator are
implemented.  bs64 is the only shape where a single verifier block can
theoretically feed the mixed path immediately; tail steps still need dense
fallback or queueing.

I also ran a direct `--sr24-route-overlap-streams` ablation:

| bs | path | total tok/s | full tok/s | note |
|---:|---|---:|---:|---|
| 32 | dense baseline | 2221.958 | 2714.802 | reference |
| 32 | policy dense fallback | 1779.809 | 2426.725 | `packed_parallel` policy treated as unsupported |
| 32 | legacy serial mixed | 2247.803 | 2682.788 | `--sr24-scheduler-policy-allow-legacy-mixed` |
| 32 | overlap streams | 1802.594 | 2537.591 | `--sr24-route-overlap-streams` |
| 32 | overlap streams + graph allow | 1816.541 | 2397.642 | also `--sr24-route-overlap-allow-cudagraph` |
| 64 | dense baseline | 2674.521 | 3525.982 | reference |
| 64 | policy dense fallback | 2695.074 | 3489.043 | near-dense hook fallback |
| 64 | legacy serial mixed | 2337.985 | 3476.980 | worse total throughput |
| 64 | overlap streams | 1996.504 | 3051.287 | large regression |
| 64 | overlap streams + graph allow | 2332.422 | 3455.214 | still worse than dense |

Simple Python-level stream overlap is not the right fix.  It gives a tiny bs32
improvement and a large bs64 regression, even when fixed-prefix graph replay is
allowed.  The legacy serial fixed-block mixed path is a useful diagnostic: it
slightly beats dense in bs32 total throughput, but misses full-batch speed and
falls behind at bs64.  Therefore the next implementation should be the real
systems path: a grouped verifier-block data format plus packed MLP operator,
not another threshold sweep, dense fallback tweak, or raw stream-overlap toggle.
The grouping unit is the verifier block/step descriptor, not the individual
active request.

Actionable next design:

1. Keep fixed-prefix K=8 and the 8pp accuracy budget.
2. Treat a verifier block as a compact descriptor:
   `(request_id, hidden_ptr, scheduled_width=9, prefix=2, dense_width=3,
   base_width=6)`.
3. Accumulate ready compact descriptors into per-layer grouped workspaces until
   the planner's effective batch target is met, e.g. grouped dense rows around
   192 and grouped base rows around 384 for the current K=8 prefix2 policy.
4. Use dense fallback when a group cannot be filled within the scheduler
   latency budget.  This is important for tail steps and small active counts.
5. Replace `index_select`/`index_copy` row assembly with persistent packed
   input/output workspaces addressed by descriptor offsets.
6. Launch dense-important rows and 2:4 sparse-unimportant rows from the grouped
   workspace as graph-captured sibling kernels, or fuse them behind one grouped
   MLP entrypoint.  The current stream-overlap branch should stay diagnostic.

## 2026-07-01 queue upper-bound and down-only sanity

I added an offline queue analyzer:

```text
examples/evaluate/eval-guidellm/scripts/analyze_sr24_grouping_queue_trace.py
```

It reads live `speclink_sr24_grouping_trace.jsonl` files and estimates how many
compact verifier blocks could feed the packed MLP planner if the scheduler were
allowed to buffer compatible verifier blocks.  This is an upper-bound
diagnostic only; it does not prove decode-step delay is latency-safe.

Output:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_grouping_queue_trace_analysis_20260701/queue_report.md
```

Key read:

| bs | max wait blocks | grouped block % | grouped row % | est MLP local speedup |
|---:|---:|---:|---:|---:|
| 8 | 7 | 54.237 | 59.080 | 1.117x |
| 16 | 3 | 60.963 | 69.091 | 1.133x |
| 16 | 7 | 90.909 | 97.367 | 1.212x |
| 32 | 1 | 39.496 | 55.984 | 1.082x |
| 32 | 7 | 76.891 | 95.961 | 1.174x |
| 64 | 0 | 30.534 | 42.349 | 1.063x |
| 64 | 7 | 84.733 | 97.866 | 1.196x |

This makes the queue direction more constrained than it first looked.  Even an
optimistic queue gets only about `1.17x` MLP-local speedup for bs32 and about
`1.20x` for bs64.  Since MLP is only part of end-to-end decoding, this is not
enough by itself to guarantee `1.2x` output tok/s across bs8/16/32/64.  A queue
may still help a future fused operator, but it should not be the only
optimization.

I also reran the best small-scale down-proj-only signal across bs8/16/32/64:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_downonly_prefix2_bs8_64_math512_20260701
```

Setup: Llama-3.1-8B EAGLE3 K=8, `math_reasoning`, max tokens 512, 64 fixed
requests, `down_proj` only, fixed prefix2+bonus dense, all other down rows
2:4-only, and dense fallback for base rows below 128.

| bs | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup | dense accepted/step | SR24 accepted/step |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1424.569 | 1088.485 | 0.764x | 1533.237 | 1297.993 | 0.847x | 2.426 | 2.357 |
| 16 | 2246.692 | 1636.563 | 0.728x | 2541.774 | 2165.027 | 0.852x | 2.443 | 2.332 |
| 32 | 2685.335 | 2298.337 | 0.856x | 3479.310 | 3215.848 | 0.924x | 2.461 | 2.471 |
| 64 | 2468.204 | 2754.729 | 1.116x | 4363.873 | 4067.233 | 0.932x | 2.329 | 2.377 |

This confirms the earlier bs32/64 sweep was mostly a tail/total-throughput
effect, not a stable full-batch operator win.  Down-only is not the main path.

The current optimization direction should therefore be:

1. Keep the disjoint semantics: dense-important rows and 2:4-only unimportant
   rows must not duplicate work.
2. Stop spending time on scalar prefix/threshold sweeps as the primary
   optimizer; they pass the 8pp GSM8K-50 gate but do not reduce the dominant
   sparse-base cost.
3. Focus on the operator/data-format layer: a graph-stable fixed-capacity MLP
   entry point that consumes dense-important and sparse-unimportant row buffers
   without Python row lists, `index_select`, tiny sparse GEMMs, or separate
   uncaptured launches.
4. Use queue/grouping only as a fill mechanism for that operator, with dense
   fallback when the grouped workspace cannot be filled cheaply.

## 2026-07-01 additional noverify-sparse ablation

I reran a focused PPoPP-style lossy ablation after the fixed-block descriptor
work to make sure the tempting broad noverify-sparse family is not accidentally
used again:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_ppopp_lossy_ablation_20260701
```

Setup: Llama-3.1-8B, EAGLE3 K=8, GSM8K-CoT `limit=50`,
`max_new_tokens=512`, quality budget `8pp`.  Throughput was configured for
`math_reasoning` bs8/16/32/64, but all semantically relevant candidates failed
the quality gate and were skipped.

| candidate | dense acc | SR24 acc | delta pp | paired regressions/improvements | result |
|---|---:|---:|---:|---:|---|
| descriptor input-buffer, all noverify sparse | 0.7800 | 0.1600 | -62 | 33/2 | reject |
| noverify sparse + dense-fill64 | 0.7800 | 0.1600 | -62 | 33/2 | reject |
| noverify sparse + dense-fill128 | 0.7800 | 0.1600 | -62 | 33/2 | reject |
| noverify sparse + cuSPARSELt small-M alg1<=160 | 0.7800 | 0.1400 | -64 | 34/2 | reject |
| noverify sparse + gate_up min_base=384 | 0.7800 | 0.1400 | -64 | 34/2 | reject |
| noverify sparse + Python overlap streams | 0.7800 | failed before valid SR24 score | | | reject |

Read: dense-fill, small-M cuSPARSELt algorithm selection, gate/up fill guards,
input-buffering, and Python stream overlap are operator-side changes; they do
not repair the large accuracy loss caused by broad noverify/no-mask sparse MLP.
These candidates should remain negative ablations.  The quality-feasible
semantic target is narrower: verifier-only disjoint routing where important
verification rows run dense once and unimportant verification rows run 2:4 once,
while noverify/no-mask MLP work remains dense until a much better importance
signal exists.

This reinforces the systems direction: the next useful implementation is not a
new prefix threshold.  It is a grouped/fused verifier MLP entrypoint with a
fixed descriptor data format, persistent branch workspaces, dense fallback for
underfilled groups, and optional grouping only to feed a profitable operator.

## 2026-07-01 fixed-block output buffer and layer-guard probes

I implemented an opt-in fixed-block output workspace:

- vLLM flag/env: `SPECLINK_SR24_FIXED_BLOCK_OUTPUT_BUFFER=1`
- runner flag: `--sr24-fixed-block-output-buffer`
- sweep candidate:
  `lossy_prefix2_rowrouted_mlp_verifier_only_outputbuf_compile`

The buffer is per MLP module (`output:{id(gate_up_module)}`) rather than global.
This avoids aliasing the previous layer's output when the next layer begins.
It is intentionally off by default.

Quality checks showed the output buffer itself does not cause the large
regression seen in noverify-sparse probes:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_outputbuf_quality_compare_20260701
```

Both `operator_guard_compile` and `operator_guard_outputbuf_compile` scored
`0.40` vs dense `0.80` on a GSM8K-5 smoke when `noverify` rows were also made
sparse.  The failure is therefore the semantic scope, not the output buffer.

The quality-feasible verifier-only scope passed GSM8K-50:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_verifier_only_outputbuf_sweep_20260701
```

| candidate | dense acc | SR24 acc | delta pp | pair reg/imp | result |
|---|---:|---:|---:|---:|---|
| verifier-only fixed-prefix2 + output buffer | 0.7800 | 0.7400 | -4 | 3/1 | pass |

But it did not speed up serving:

| bs | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 970.519 | 795.122 | 0.819x | 1050.667 | 997.038 | 0.949x |
| 16 | 1512.608 | 1128.670 | 0.746x | 1702.174 | 1522.580 | 0.894x |
| 32 | 1934.580 | 1486.093 | 0.768x | 2316.386 | 1973.101 | 0.852x |
| 64 | 2255.429 | 1885.356 | 0.836x | 3041.149 | 2985.160 | 0.982x |

The first breakdown explained why:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/temp/sr24_verifier_only_outputbuf_breakdown_eager_bs64_20260701_work
```

With `route_min_base_rows=384`, the mixed verifier MLP almost never ran.  The
run spent `1216` calls in `row_routed_mlp_full_dense_fallback_small_base`; the
quality-safe path was therefore mostly dense fallback, not useful 2:4 work.

Lowering the gate to `route_min_base_rows=64` did enable the mixed branch, but
made throughput worse:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_verifier_only_outputbuf_minbase64_bs64_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/temp/sr24_verifier_only_outputbuf_minbase64_breakdown_eager_bs64_20260701_work
```

| bs64 math | total tok/s | full tok/s | GPU util |
|---|---:|---:|---:|
| dense | 2240.983 | 3038.476 | 84.1% |
| minbase64 SR24 | 1504.768 | 2306.777 | 53.4% |

The eager breakdown showed the direct reason:

| component | total CUDA ms | calls | ms/call |
|---|---:|---:|---:|
| fixed-block base gate_up 2:4 | 1087.746 | 1088 | 1.000 |
| fixed-block base down 2:4 | 1135.491 | 1088 | 1.044 |
| fixed-block dense gate_up | 179.524 | 1088 | 0.165 |
| fixed-block dense down | 110.374 | 1088 | 0.101 |
| fixed-block assembly | 10.005 | 1088 | 0.009 |

So the current cuSPARSELt/Torch sparse path is not profitable for the small
verifier MLP branch.  The output assembly and scheduler mask are not the
dominant cost.

I then tested layer-guarded noverify sparse, keeping dense noverify MLPs only
in early layers:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_layerguard_noverify_sweep_20260701
```

| candidate | dense acc | SR24 acc | delta pp | pair reg/imp | result |
|---|---:|---:|---:|---:|---|
| dense noverify layers 0-23, sparse 24-31 | 0.7800 | 0.7000 | -8.000000000000007 | 5/1 | boundary/fail by strict gate |
| dense noverify layers 0-15, sparse 16-31 | 0.7800 | 0.5200 | -26 | 14/1 | reject |
| dense noverify layers 0-7, sparse 8-31 | 0.7800 | 0.2800 | -50 | 27/2 | reject |

Manual bs64 throughput for the boundary front24 case was also below dense:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_front24_noverify_bs64_manual_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_front24_noverify_compile_bs64_manual_20260701
```

| front24 variant | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup |
|---|---:|---:|---:|---:|---:|---:|
| eager/default | 2238.257 | 1626.421 | 0.727x | 3033.286 | 2825.421 | 0.931x |
| default compile | 2220.966 | 1814.520 | 0.817x | 3141.381 | 2755.356 | 0.877x |

Conclusion for the next implementation pass:

1. The semantic scope that stays within the 8pp GSM8K-50 budget is narrow:
   verifier-only definitely passes; noverify tail-only is at best on the
   boundary; earlier noverify sparse fails badly.
2. The current small-M verifier sparse branch is slower than dense, so forcing
   it on cannot produce a system speedup.
3. The current large-M noverify sparse branch still does not beat dense enough
   to offset hook/graph overhead, and its quality budget is already tight.
4. To reach the requested `1.2x` on bs8/16/32/64, the next useful work must be
   a real grouped/fused MLP operator/data format:
   fixed-capacity route descriptors, persistent dense/base workspaces,
   cuSPARSELt descriptor reuse or a custom grouped 2:4 kernel, dense fallback
   for underfilled groups, and optional queueing only to feed that profitable
   operator.  Further threshold-only sweeps are unlikely to close the gap.

## 2026-07-01 grouping trace and overlap-stream probes

I then used the packed-MLP microbench planner policy as a live tracing input:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_packed_grouped_bucket_k8_bs8_64_prefix12_20260701/scheduler_policy.json
```

The planner says the current fixed-prefix2 K=8 MLP shape is profitable only
when the effective verifier-block batch is around 64 rows of requests:

| bs | best single prefix | best single local speedup | required coalesce | required effective bs | action |
|---:|---:|---:|---:|---:|---|
| 8 | 1 | 0.606x | 8 | 64 | dense fallback until grouped |
| 16 | 1 | 0.719x | 4 | 64 | dense fallback until grouped |
| 32 | 1 | 0.982x | 2 | 64 | dense fallback until grouped |
| 64 | 2 | 1.240x | 1 | 64 | use mixed single block |

Live grouping trace:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_grouping_trace_bs8_64_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_grouping_trace_bs8_64_20260701/queue_analysis/queue_report.md
```

| bs | max wait blocks | grouped block % | grouped row % | avg blocks/group | estimated MLP local speedup |
|---:|---:|---:|---:|---:|---:|
| 8 | 0 | 0.000 | 0.000 | 0.000 | 1.000x |
| 8 | 7 | 42.593 | 47.545 | 8.000 | 1.090x |
| 16 | 3 | 49.770 | 60.021 | 4.000 | 1.106x |
| 16 | 7 | 88.018 | 96.943 | 4.548 | 1.204x |
| 32 | 1 | 40.984 | 57.082 | 2.000 | 1.086x |
| 32 | 7 | 77.869 | 95.041 | 2.500 | 1.176x |
| 64 | 0 | 30.263 | 42.704 | 1.000 | 1.062x |
| 64 | 7 | 89.474 | 98.462 | 1.659 | 1.209x |

Read: bs8 is unlikely to reach a 1.2x end-to-end target with this operator
shape because even an optimistic wait-7 queue reaches only 1.09x local MLP
speedup. bs16/32/64 need a small queue to feed enough compatible fixed-prefix
blocks, but this is only an operator-local upper bound; it does not include
attention, scheduler, or queueing latency.

I also tested whether the existing live auxiliary-stream path can recover the
planner's `packed_parallel` benefit at bs64:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_route_overlap_policy_bs64_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_route_overlap_graphallow_bs64_20260701
```

| variant | total tok/s | full-batch tok/s | vs dense full |
|---|---:|---:|---:|
| dense baseline bs64 | 2245.124 | 3041.780 | 1.000x |
| route-overlap, mixed eager | 1574.679 | 2683.719 | 0.882x |
| route-overlap, graph-allow | 1656.355 | 2304.672 | 0.758x |

The trace confirms that the bs64 full-batch steps were recognized as
`operator_supported_live=true` and `use_mixed_single_block`, so this is not a
policy bug. The current Python-level stream overlap still pays too much launch,
stream synchronization, and graph interaction overhead to expose the microbench
local win.

Updated direction:

1. Do not expand the noverify-sparse semantic scope unless a new accuracy guard
   is introduced; all-noverify sparse is far outside the 8pp budget and
   front24-only is a boundary case without speedup.
2. Do not chase Python auxiliary-stream overlap as the primary solution; it was
   slower than dense even at the best bs64 single-block point.
3. Do not wire the scalar Triton base24 prototype into serving.  The existing
   `sr24_triton_base24_mlp_actual_rows_k8_20260630` microbench shows it is
   `0.012x-0.033x` dense at the real K=8 row shapes; it is a useful data-format
   sketch, not a viable operator.
4. The next implementable system design should be a first-class grouped MLP
   route:
   fixed route descriptors, compact dense/base row-major buffers with stable
   capacity, one grouped sparse gate-up/down path, one grouped dense
   correction path, and a single fused assemble.
5. The scheduler should use the planner policy as a profitability guard:
   dense fallback for underfilled groups, short wait budgets for bs16/32/64,
   and no grouping for bs8 unless a new operator shape shows a better local
   speedup.

## 2026-07-01 PPoPP-Style Lossy Gate Refresh

I reran the current systems-oriented small gate after switching the target away
from lossless quality:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_ppopp_operator_probe_prefix12_bs8_64_20260701_live
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_ppopp_operator_guided_inputbuf_gate_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_ppopp_verifier_only_outputbuf_gate_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_ppopp_verifier_only_outputbuf_bs64_64req_20260701
```

The packed verifier-block MLP microbench still supports the same system
direction. For K=8, important rows are the fixed dense prefix plus verifier
bonus row, and the remaining draft rows are 2:4 sparse-only. The refreshed
planner says the mixed dense/sparse operator needs an effective request batch
around 64 before it reaches the local 1.2x target:

| bs | best single local speedup | required coalesce | required effective bs | read |
|---:|---:|---:|---:|---|
| 8 | 0.605x | 8 | 64 | dense fallback until grouped |
| 16 | 0.717x | 4 | 64 | dense fallback until grouped |
| 32 | 0.981x | 2 | 64 | dense fallback until grouped |
| 64 | 1.240x | 1 | 64 | single-block mixed is locally viable |

This answers the "too few important tokens" question: padding/fill alone is not
the main lever. The branch shapes have to be grouped/coalesced until both the
dense-important branch and 2:4 sparse branch are tensor-core friendly. Python
stream overlap is only a microbench sketch; live serving needs a lower-overhead
grouped/fused operator.

I also tested a more aggressive semantic scope where noverify/non-draft MLP rows
were sparse-only:

| candidate | GSM8K-50 dense | SR24 | delta | decision |
|---|---:|---:|---:|---|
| descriptor input-buffer with noverify sparse-only | 0.7800 | 0.1600 | -62pp | reject |

So the current accuracy boundary is not "make every unimportant row sparse".
For Llama math/GSM8K, broad noverify sparsification destroys quality far beyond
the 8pp budget. The viable scope is narrower: keep ordinary/non-verify TLM rows
dense, and apply SR24 only inside the speculative verifier block.

That verifier-only scope passed the quality gate:

| candidate | GSM8K-50 dense | SR24 | delta | paired regressions/improvements |
|---|---:|---:|---:|---:|
| verifier-only output buffer, fixed prefix2 | 0.7800 | 0.7400 | -4pp | 3 / 1 |

But the end-to-end throughput is still near parity, not the requested 1.2x:

| bs | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 866.461 | 862.445 | 0.995x | 1022.940 | 1019.077 | 0.996x |
| 16 | 1328.904 | 1337.025 | 1.006x | 1688.486 | 1689.537 | 1.001x |
| 32 | 1668.793 | 1682.074 | 1.008x | 2474.443 | 2459.988 | 0.994x |
| 64 | 2251.461 | 2231.793 | 0.991x | 3043.017 | 3029.842 | 0.996x |

The bs64 row above is the 64-request补测, because a 32-request run cannot form
a valid full-batch window at concurrency 64. Acceptance length is also not the
reason for the gap: the bs64 64-request run had dense accepted draft tokens per
step `1.426` versus SR24 `1.414`.

Updated implementation direction:

1. Keep the verifier-only fixed-prefix2 semantics as the quality-safe operating
   point for now: important verifier rows dense-only, other verifier draft rows
   2:4 sparse-only, ordinary TLM/noverify rows dense.
2. Do not broaden noverify sparse-only scope without a new accuracy mechanism;
   it failed GSM8K-50 by about 62pp in the current tree.
3. The next speed implementation must be a real grouped/fused MLP route: fixed
   capacity route descriptors, persistent dense/base workspaces, grouped
   compatible verifier blocks to effective batch 64, concurrent dense and 2:4
   branches inside the operator, and one fused assemble/writeback. More scalar
   threshold tuning, output-buffer tweaks, or Python stream overlap are not
   enough to reach 1.2x.

## 2026-07-01 Live Queue Feasibility Recheck

I extended `scripts/analyze_sr24_grouping_queue_trace.py` so it now writes
`queue_recommendations.{csv,json}` in addition to the per-wait summary. The
recommendation rows make the serving implication explicit: whether a single
block is enough, the minimum wait budget that reaches the target local MLP
speedup, and whether the row is only a long-wait upper bound.

The current analysis reused the live grouping trace from:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/temp/sr24_grouping_opportunity_trace_2048_fixed_20260701
```

and wrote:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_grouping_queue_analysis_2048_fixed_20260701_current
```

Recommendation table for target local MLP speedup `1.2x`:

| bs | single-block est | target reached | min wait for target | speedup at min wait | grouped block % | grouped row % | design read |
|---:|---:|:---:|---:|---:|---:|---:|---|
| 8 | 1.000x | yes | 15 | 1.216x | 91.786 | 97.213 | long_wait_queue_upper_bound |
| 16 | 1.000x | yes | 7 | 1.212x | 90.909 | 97.367 | long_wait_queue_upper_bound |
| 32 | 1.000x | no |  |  |  |  | queue_upper_bound_below_target |
| 64 | 1.063x | yes | 15 | 1.221x | 93.893 | 98.958 | long_wait_queue_upper_bound |

This narrows the next implementation choice. The same-weight packed microbench
is still useful for defining the target operator shape, but cross-step verifier
block grouping is not a free serving optimization: it requires delaying
verification, and bs32 does not reach the local 1.2x target even with a
max-wait-15 optimistic queue. Cross-layer grouping is also not equivalent to
the microbench because adjacent layers use different weights and have sequential
dependencies. Therefore the current main path should not be a live long-wait
queue first. It should be a better single-block small-M 2:4 sparse operator or
a true low-overhead CUDA/CUTLASS grouped/fused kernel for the current
fixed-prefix route. The queue can remain a fallback/fill mechanism after the
operator itself is faster.

## 2026-07-01 Small-M cuSPARSELt Alg Refresh

I reran the cuSPARSELt algorithm sweep on fixed-prefix2 sparse-tail row counts
instead of the older full-block row counts. For K=8, prefix2, the sparse branch
row counts are:

| bs | sparse-tail rows |
|---:|---:|
| 8 | 48 |
| 16 | 96 |
| 32 | 192 |
| 64 | 384 |

Output root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cslt_algos_fixedprefix2_baserows_20260701_current
```

Best alg per row count:

| rows | best alg | dense graph ms | sparse graph ms | local speedup |
|---:|---:|---:|---:|---:|
| 48 | 1 | 0.2387 | 0.1809 | 1.319x |
| 96 | 1 | 0.2719 | 0.2415 | 1.126x |
| 192 | 1 | 0.3319 | 0.3380 | 0.982x |
| 384 | 0 | 0.6425 | 0.4392 | 1.463x |

Only alg0 and alg1 are valid on this stack; alg2-7 fail cuSPARSELt attribute
validation. The actionable difference from the older small-M setting is that
rows=192 should use alg1, so I added:

```text
lossy_prefix2_verifier_only_outputbuf_smallm_alg1_t256_compile
```

This keeps the verifier-only fixed-prefix2/output-buffer policy but enables
`--sr24-cslt-small-m-alg-id-enable --sr24-cslt-small-m-threshold 256
--sr24-cslt-small-m-alg-id 1`.

Quality/throughput gate:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_verifier_only_outputbuf_smallm_alg1_t256_gate_20260701
```

GSM8K-50 stayed within the 8pp budget:

| dense | SR24 | delta | paired regressions/improvements |
|---:|---:|---:|---:|
| 0.7800 | 0.7400 | -4pp | 3 / 1 |

Throughput on math_reasoning, max128, 64 requests:

| bs | dense total | SR24 total | total speedup | dense full | SR24 full | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1889.355 | 1871.182 | 0.990x | 2266.570 | 2310.436 | 1.019x |
| 64 | 2236.985 | 2223.991 | 0.994x | 3033.493 | 3019.023 | 0.995x |

Read: the threshold-256 alg selector is worth keeping as a small
operator-selection knob because it improves the bs32 sparse-tail local shape and
slightly improves bs32 full-batch throughput. It still does not create an
end-to-end 1.2x speedup. The remaining bottleneck is not alg_id selection alone;
it is the current split dense/sparse branch structure and small-M sparse kernel
efficiency.

## 2026-07-01 Leaf-Aware Alg And Underfilled Sparse Branch

I extended the cuSPARSELt probe to compare CUTLASS versus cuSPARSELt and to
sweep gate_up/down alg pairs separately:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_sparse_backend_cutlass_vs_cslt_fixedprefix2_baserows_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cslt_alg_pairs_fixedprefix2_baserows_20260701
```

CUTLASS is not usable on this GPU/stack; it fails with:

```text
sparse_semi_structured_mad_op : Supported only on GPUs with compute capability 8.x
```

The alg-pair sweep shows the useful operator knob is projection-aware:

| rows | best gate_up alg | best down alg | local speedup |
|---:|---:|---:|---:|
| 48 | 1 | 1 | 1.330x |
| 96 | 0 | 1 | 1.265x |
| 192 | 0 | 1 | 1.189x |
| 384 | 0 | 0 | 1.461x |

I added leaf-aware env/runner plumbing:

```text
SPECLINK_SR24_CSLT_SMALL_M_THRESHOLD_BY_LEAF
SPECLINK_SR24_CSLT_SMALL_M_ALG_ID_BY_LEAF
--sr24-cslt-small-m-threshold-by-leaf
--sr24-cslt-small-m-alg-id-by-leaf
```

The new candidate is:

```text
lossy_prefix2_verifier_only_outputbuf_leafalg_pair_compile
```

It uses global alg1 for rows<=64 and `down_proj=1` for rows<=256. Quality
passes the relaxed gate:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_verifier_only_outputbuf_leafalg_pair_gate_20260701
```

| GSM8K-50 dense | SR24 | delta | paired regressions/improvements |
|---:|---:|---:|---:|
| 0.7800 | 0.7400 | -4pp | 3 / 1 |

But end-to-end throughput is still not a win:

| bs | dense total | SR24 total | total speedup | dense full | SR24 full | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1900.729 | 1880.831 | 0.990x | 2277.351 | 2309.239 | 1.014x |
| 64 | 2230.425 | 2174.497 | 0.975x | 3137.983 | 3108.818 | 0.991x |

The reason is underfilled sparse work. An eager diagnostic breakdown at:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_leafalg_pair_eager_linear_breakdown_bs64_20260701_summary
```

shows `row_routed_mlp_full_dense_fallback_small_base=2432`. The scheduler
generated only about 136 sparse-base rows/step in that probe, while the current
quality-safe fixed-prefix2 candidate keeps `route_min_base_rows=384`, so most
live decode steps fall back to dense.

I tested the obvious threshold/prefix relaxations:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_leafalg_pair_minbase96_probe_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_leafalg_pair_prefix1_minbase128_probe_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_leafalg_pair_overlap_bs64_probe_20260701
```

| candidate | bs32 total speedup | bs64 total speedup | bs32 full speedup | bs64 full speedup | read |
|---|---:|---:|---:|---:|---|
| prefix2 minbase384 leafalg | 0.990x | 0.975x | 1.014x | 0.991x | mostly dense fallback |
| prefix2 minbase96 leafalg | 0.734x | 0.766x | 0.911x | 0.806x | sparse branch runs but underutilizes GPU |
| prefix1 minbase128 leafalg | 0.715x | 0.967x | 0.858x | 0.971x | fewer dense rows not enough |
| prefix2 minbase384 leafalg + stream overlap | n/a | 0.672x | n/a | 0.733x | Python stream overlap lowers util |

Updated read:

1. The leaf-aware alg selector is useful plumbing, but it is only a small
   operator knob.
2. Lowering `route_min_base_rows` proves the current PyTorch/cuSPARSELt split
   MLP path is dominated by underfilled small kernels, not by threshold
   conservatism.
3. Relaxing prefix protection within the 8pp budget does not create speedup
   with the current operator shape.
4. The next credible path to 1.2x is a real grouped/fused fixed-route MLP
   operator or a faster small-M 2:4 kernel. The scheduler should feed it with
   useful sparse rows up to an effective batch around 64 and otherwise fall
   back to dense.

## 2026-07-01 Verifier-Only Prefix Boundary And Fast Dense Fallback

I made one low-risk runtime cleanup in `vllm/vllm/speclink_sr24.py`: the
fixed-block row-routed MLP path now handles the `base_count <
route_min_base_rows` guard locally. Instead of returning `None` and letting the
generic row-routed path rebuild route state before falling back, it directly
runs the MLP-level dense fallback with `reason=fixed_block_small_base`.

Small math_reasoning throughput probe:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixed_block_smallbase_densefastpath_probe_20260701
```

| bs | dense total | SR24 total | total speedup | dense full | SR24 full | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1923.211 | 1944.224 | 1.011x | 2281.038 | 2304.768 | 1.010x |
| 64 | 2231.072 | 2223.778 | 0.997x | 3142.766 | 3023.725 | 0.962x |

Read: this removes avoidable fallback overhead, but it does not change the
main bottleneck. The quality-safe prefix2 candidate still does not reach the
1.2x target.

I then separated two effects that were previously conflated:

- making ordinary no-mask decode rows sparse-only;
- reducing dense-protected rows only inside the speculative verifier block.

The broad noverify-sparse candidates are not usable. Prefix0 and prefix1 with
noverify sparse both collapse GSM8K-50 from `0.7800` to `0.1400`.

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lossy_prefix_gate_small_20260701
```

The new verifier-only candidates in
`examples/evaluate/eval-guidellm/scripts/run_sr24_lossy_speed_quality_sweep.py`
are:

```text
lossy_prefix1_verifier_only_outputbuf_leafalg_pair_compile
lossy_prefix0_verifier_only_outputbuf_leafalg_pair_compile
```

They keep ordinary no-mask decode dense and only change the speculative
verifier block. Quality passes the 8pp gate:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_verifier_only_prefix01_gate_20260701
```

| candidate | dense acc | SR24 acc | delta | pair reg/imp | best total speedup | best full speedup |
|---|---:|---:|---:|---:|---:|---:|
| prefix1 verifier-only | 0.7800 | 0.7600 | -2pp | 1 / 0 | 0.858x | 0.999x |
| prefix0 verifier-only | 0.7800 | 0.7400 | -4pp | 3 / 1 | 0.752x | 0.930x |

This is the clean boundary: fewer dense verifier tokens are accuracy-feasible
under an 8pp budget, but they are slower with the current split sparse path.
The failure is now clearly an operator/system issue, not an accuracy controller
issue.

I also rechecked Python-level dense/sparse stream overlap for the quality-pass
prefix1 verifier-only candidate:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix1_verifier_only_overlap_probe_20260701
```

| bs | dense total | SR24 total | total speedup | dense full | SR24 full | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1913.819 | 1329.544 | 0.695x | 2281.524 | 1958.200 | 0.858x |
| 64 | 2231.915 | 2215.107 | 0.992x | 3042.820 | 3016.899 | 0.991x |

Updated optimization direction:

1. Do not continue scalar prefix/threshold sweeps as the main path. Prefix0/1
   are quality-feasible verifier-only, yet slower.
2. Keep dense fallback for underfilled sparse rows. It is necessary to avoid
   small-kernel regressions.
3. The PPoPP-style system contribution needs a real data-format/operator
   change: fixed-route descriptors, stable workspaces, and either a grouped
   fused dense-important + 2:4-sparse-unimportant MLP operator or a better
   small-M sparse kernel.
4. Python-level stream overlap is not enough. Any concurrent dense/sparse path
   needs to be inside a lower-overhead CUDA/Triton/CUTLASS-style operator, not
   two PyTorch launches plus events per MLP.

## 2026-07-01 All-Corrected CompressedDense Breakdown

I also checked whether exact `all_corrected_24` can be made useful by keeping
compressed residual data on GPU. The answer is no for the current operator
shape.

Focused bs64/math/max64 eager run:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_compresseddense_gateup_eager_breakdown_bs64_20260701_cont
```

| method | total tok/s | full-batch tok/s | avg GPU util |
|---|---:|---:|---:|
| dense baseline | 1843.595 | 2796.692 | 78.8% |
| all_corrected_24 compressed_dense@cuda | 487.935 | 862.656 | 86.3% |

The breakdown file under the matching `temp/` work root shows the issue:

| component | total CUDA ms | calls | avg ms/call |
|---|---:|---:|---:|
| base sparse linear | 1319.383 | 1952 | 0.676 |
| compressed residual materialize | 3991.194 | 13664 | 0.292 |
| compressed residual GEMM | 1537.202 | 13664 | 0.113 |
| compressed residual add | 347.627 | 13664 | 0.025 |

This is not a CPU-copy problem. The residual is on CUDA, but `gate_up_proj`
has a large output dimension and the current path chunks the residual into
4096-output slices, so each sparse base call expands into seven residual
materialize + GEMM calls.

I then tried full residual-weight caching:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_compresseddense_gateup_cachefull_eager_breakdown_bs64_20260701_cont
```

For normal `max_model_len=4096`, server startup failed after profile-run:

```text
Available KV cache memory: 0.18 GiB
To serve at least one request with max seq len (4096), 0.52 GiB KV cache is needed.
```

The partial profile breakdown confirmed that caching removes most
materialization calls, but it trades the bottleneck for model-memory pressure.
A diagnostic `max_model_len=2048` run succeeded:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_compresseddense_gateup_cachefull_clean_bs64_maxlen2048_20260701_cont
```

| method | total tok/s | full-batch tok/s |
|---|---:|---:|
| dense baseline | 1809.559 | 2609.329 |
| all_corrected_24 cached compressed_dense@cuda | 951.518 | 1659.350 |

Read:

1. Exact all-corrected compressed residual is a good diagnostic/control, but
   it is the wrong acceleration target.
2. The PPoPP-style fast path must avoid dense residual work for low-importance
   tokens after their 2:4 sparse computation has already run.
3. The quality-safe candidate boundary is verifier-only lossy routing:
   ordinary no-mask decode remains dense, high-importance verifier rows use
   dense correction, and low-importance verifier rows use sparse-only output.
   Prefix0/1 verifier-only passes the relaxed GSM8K-50 8pp quality gate, but
   still does not speed up because the current sparse branch is underfilled.
4. The next implementation should be a real grouped/fused fixed-route MLP
   operator, or a faster small-M 2:4 kernel, with stable route descriptors,
   reusable input/output buffers, dense fallback for underfilled work, and
   optional in-kernel/consolidated concurrent dense-important plus sparse-base
   execution.

Runner hygiene fixed in the same pass: `--sr24-breakdown` now forces
eager/no-CUDA-graph in both the GuideLLM matrix runner and the lm-eval accuracy
runner. The breakdown counters use Python locks and CUDA events, so letting
Dynamo trace them can fail during vLLM startup profiling.

## 2026-07-01 Continuation: 8pp Lossy Boundary

I fixed one runner bug before continuing: in the GuideLLM matrix runner,
`--sr24-base-only-allow-compile` now implies `--sr24-allow-cudagraph`. Before
that, the command could still include `--enforce-eager`, so the base-only graph
ablation silently did not test the intended path.

With the graph path actually enabled, MLP-only `base_only_24` is a real speed
upper bound:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_continue_baseonly_mlp_only_allowcompile_fixed_bs64_math256_20260701
```

| method | total tok/s | full-batch tok/s | accepted draft/step | avg GPU util | CUDA graph |
|---|---:|---:|---:|---:|---|
| dense baseline | 2325.671 | 3444.110 | 1.7119 | 86.6% | - |
| MLP-only base_only_24 | 3905.823 | 6351.732 | 3.3445 | 88.1% | `{"FULL": 96, "NONE": 2}` |

But it is not a quality-safe solution under the requested 8pp budget:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_mlp_quality_gsm8k50_20260701
```

| method | dense acc | SR24 acc | delta | pair reg/imp | avg out tokens | clipped |
|---|---:|---:|---:|---:|---:|---:|
| MLP-only base_only_24 | 0.7200 | 0.2200 | -50pp | 28 / 3 | 160.98 | 10 |

I also tested the requested "do not run dense after sparse for unimportant
tokens" boundary on GSM8K-50:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lossy_prefix2_quality_speed_bs64_gsm8k50_20260701
```

| candidate | policy | dense acc | SR24 acc | delta | quality pass |
|---|---|---:|---:|---:|---:|
| `lossy_prefix2_rowrouted_mlp_noverify_sparse_compile` | ordinary/no-mask MLP rows sparse-only | 0.7800 | 0.1200 | -66pp | no |
| `lossy_prefix2_gateup_only_noverify_sparse_compile` | ordinary/no-mask gate_up rows sparse-only | 0.7800 | 0.4200 | -36pp | no |
| `lossy_prefix2_rowrouted_mlp_verifier_only_outputbuf_compile` | ordinary rows dense; only verifier rows split dense/sparse | 0.7800 | 0.7400 | -4pp | yes |

The quality-safe verifier-only candidate still has no speedup:

| method | total tok/s | full-batch tok/s | accepted draft/step | avg GPU util | CUDA graph |
|---|---:|---:|---:|---:|---|
| dense baseline | 2243.093 | 3035.591 | 1.4263 | 80.4% | - |
| verifier-only prefix2 | 2225.172 | 3021.742 | 1.4138 | 79.3% | `{"FULL": 44, "PIECEWISE": 44}` |

Read:

1. `base_only_24` is slow only when the graph path is accidentally disabled or
   when all leafs are sparsified. MLP-only graph base-only is fast, but its
   GSM8K accuracy loss is far outside the 8pp budget.
2. Sparse-only ordinary/no-mask decode rows are not acceptable for reasoning
   tasks, even when the verifier prefix rows keep dense protection.
3. The clean quality boundary is verifier-only routing: ordinary rows stay
   dense; only speculative verifier draft rows are split into dense-important
   and sparse-unimportant rows.
4. The remaining blocker is system/operator efficiency. The current PyTorch
   split path launches underfilled small sparse branches and does not beat
   dense. The next credible 1.2x path is a grouped/fused fixed-route MLP
   operator or a faster small-M 2:4 sparse kernel, with stable route
   descriptors, reusable input/output buffers, dense fallback for underfilled
   sparse rows, and in-kernel or low-overhead dense+sparse concurrency.

## 2026-07-01 Late Run: 8pp Tail-Sparse Controller Probe

I added fine-grained tail noverify-sparse candidates to
`scripts/run_sr24_lossy_speed_quality_sweep.py`:

- `lossy_prefix2_rowrouted_mlp_front28_dense_noverify_compile`
- `lossy_prefix2_rowrouted_mlp_front30_dense_noverify_compile`
- `lossy_prefix1_rowrouted_mlp_front28_dense_noverify_compile`
- `lossy_prefix1_rowrouted_mlp_front30_dense_noverify_compile`
- `lossy_prefix0_rowrouted_mlp_front30_dense_noverify_compile`

These are narrower than the earlier front24/front16 probes. Ordinary/no-verify
MLP rows stay dense in early layers; only the final 2 or 4 layers use the
2:4 sparse base. Verifier rows still use the fixed-prefix dense-important plus
sparse-unimportant split, so sparse-unimportant verifier tokens are not later
recomputed densely. The quality gate now uses a small epsilon so an exact
`-8pp` drop is not rejected by floating point roundoff.

Quality gate, GSM8K-50, max_new_tokens=512:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lossy_tail_sparse_prefix_quality_gsm8k50_20260701
```

| candidate | dense acc | SR24 acc | delta | result |
|---|---:|---:|---:|---|
| front28 / prefix2 | 0.7800 | 0.7400 | -4pp | pass |
| front30 / prefix2 | 0.7800 | 0.7000 | -8pp | boundary; recorded fail before epsilon fix |
| front28 / prefix1 | 0.7800 | 0.7400 | -4pp | pass |
| front30 / prefix1 | 0.7800 | 0.6600 | -12pp | fail |

I stopped the remaining prefix0/control rows after this because prefix1/front30
already exceeded the 8pp budget and the useful candidates were clear.

Short math_reasoning throughput, max_tokens=512, bs32/64:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lossy_tail_sparse_prefix_throughput_math_bs32_64_20260701
```

| candidate | bs | dense total | SR24 total | total speedup | dense full | SR24 full | full speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| front28 / prefix2 | 32 | 2937.9 | 2866.9 | 0.976x | 3316.0 | 3423.3 | 1.032x |
| front28 / prefix2 | 64 | 2949.7 | 3325.6 | 1.127x | 3967.2 | 3902.8 | 0.984x |
| front28 / prefix1 | 32 | 2731.8 | 2857.8 | 1.046x | 3350.1 | 3399.1 | 1.015x |
| front28 / prefix1 | 64 | 3337.2 | 3419.8 | 1.025x | 4012.2 | 3859.9 | 0.962x |

Interpretation:

1. Relaxing to an 8pp budget is useful for the controller: front28/prefix1 and
   front28/prefix2 both pass GSM8K-50 while creating more sparse-only rows than
   verifier-only prefix2.
2. The current implementation still does not deliver stable full-batch speed.
   The best total speedup was 1.127x at bs64, but full-batch speed was below
   dense. This is not enough for the requested 1.2x across bs8/16/32/64.
3. The next optimization should stop being a prefix/layer sweep. The main
   blocker is fixed-block MLP execution: too many PyTorch launches, underfilled
   small-M cuSPARSELt branches, and separate dense/sparse branch assembly.
4. The PPoPP-style path should be a data-format/operator change: fixed route
   descriptors, reusable dense/base workspaces, dense fallback when useful
   sparse rows cannot fill the operator, and a fused or packed fixed-block MLP
   that can run dense-important rows and 2:4 sparse-unimportant rows with
   low-overhead concurrency.

## 2026-07-02 Follow-up: reuse-base and front28/prefix1 overlap probes

I tested two system-side variants of the quality-passing front28/prefix1
controller.  Both use GSM8K-50, `max_new_tokens=512`, EAGLE3 K=8, and the
8pp accuracy budget.  Both kept quality at `0.74` versus dense `0.78`, so the
remaining blocker is execution efficiency rather than the controller budget.

Reuse-base packed-operator proxy:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_front28_prefix1_reusebase_quality_throughput_20260702
```

This runs the 2:4 sparse MLP on the full verifier block, then overwrites
dense-important rows.  It increases sparse branch row fill and removes the
separate base-row gather, but repeats sparse work on important rows.

| bs | dense total | SR24 total | total speedup | dense full | SR24 full | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 2942.024 | 2879.849 | 0.979x | 3400.917 | 3449.939 | 1.014x |
| 64 | 3277.806 | 3428.141 | 1.046x | 3965.682 | 3929.130 | 0.991x |

Front28/prefix1 Python CUDA-stream overlap:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_front28_prefix1_overlap_quality_throughput_20260702
```

This keeps the disjoint dense-important and sparse-unimportant MLP branches, but
launches them on separate CUDA streams before the final assembly.

| bs | dense total | SR24 total | total speedup | dense full | SR24 full | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 2953.171 | 2825.234 | 0.957x | 3345.868 | 3315.957 | 0.991x |
| 64 | 2923.648 | 3314.643 | 1.134x | 3926.893 | 3839.691 | 0.978x |

Read:

1. Allowing 4pp quality loss is enough to create sparse-only verifier/tail rows,
   but the current PyTorch/cuSPARSELt split execution does not turn those rows
   into stable full-batch speedup.
2. Reuse-base is not the main route.  It can make bs32 full-batch look slightly
   better, but it spends redundant sparse work on dense-important rows and does
   not improve bs64 full-batch.
3. Python-level stream overlap is not the missing pipeline.  It helps bs64
   total in this short run because request-drain timing moves, but the
   full-batch core window is still below dense and bs32 regresses.
4. The next implementation should be a fixed-block grouped MLP operator rather
   than another scalar policy sweep: one stable route descriptor per verifier
   block, no dense recompute for sparse-unimportant rows, preallocated
   dense/base/output workspaces, a fill-aware fallback for underfilled sparse
   branches, and dense/sparse concurrency owned inside the operator or captured
   as a small fixed CUDA Graph.

## 2026-07-02 Follow-up: 8pp boundary and component bottleneck

I rechecked the current implementation with a component-level GPU probe before
running another serving sweep:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_component_probe_bs32_bs64_20260702
```

The probe uses K=8 verifier-equivalent row counts: 288 rows for bs32 and 576
rows for bs64.  It measures Llama MLP `gate_up_proj` and `down_proj` shapes.
The relevant result is that the current mixed path is not limited mainly by
accepted length.  It is limited by the way we compose the operator:

| rows | linear | residual frac | dense graph ms | base 2:4 graph ms | current mixed graph ms | row-routed split graph ms | ideal prefix-concat graph ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 288 | gate_up | 0.125 | 0.3430 | 0.2277 | 0.4236 | 0.5017 | 0.4100 |
| 288 | gate_up | 0.500 | 0.3435 | 0.2277 | 0.5523 | 0.4904 | 0.4241 |
| 576 | gate_up | 0.125 | 0.5933 | 0.3986 | 0.6312 | 0.8469 | 0.6960 |
| 576 | gate_up | 0.500 | 0.5919 | 0.3984 | 0.9533 | 0.8039 | 0.6800 |
| 576 | down | 0.125 | 0.3309 | 0.1667 | 0.2811 | 0.3304 | 0.2909 |
| 576 | down | 0.500 | 0.3325 | 0.1670 | 0.3723 | 0.3806 | 0.3510 |

Interpretation:

1. Raw 2:4 base GEMM is faster than dense, especially in `down_proj`, so the
   sparse weight format itself is not useless.
2. The current serving-like mixed path is often slower than dense because it is
   `base sparse for many rows + dense correction for important rows + gather /
   scatter assembly`.  This matches the previous end-to-end behavior.
3. Avoiding dense work on sparse-unimportant rows is necessary but not
   sufficient.  For `gate_up_proj`, even an ideal prefix-concat split is still
   slower than dense at these row counts because two small branches and
   activation/assembly overhead dominate.
4. The direct compressed-residual Triton path is not a viable shortcut in the
   current form; it is much slower than dense/cuSPARSELt in this row regime.

I then tested broader 8pp-budget controller points on GSM8K-50 with
`max_new_tokens=512`:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_tail_layer_quality_boundary_gsm8k50_20260702
```

| candidate | dense acc | SR24 acc | delta | dense-correct -> wrong | wrong -> correct |
|---|---:|---:|---:|---:|---:|
| front24 / prefix2 | 0.78 | 0.70 | -8pp | 5 | 1 |
| front28 / prefix2 | 0.78 | 0.74 | -4pp | 3 | 1 |
| front30 / prefix2 | 0.78 | 0.70 | -8pp | 5 | 1 |
| front28 / prefix1 | 0.78 | 0.74 | -4pp | 3 | 1 |
| front30 / prefix1 | 0.78 | 0.66 | -12pp | 6 | 0 |

Two more aggressive variants were rejected earlier in the same pass:

| candidate | dense acc | SR24 acc | delta |
|---|---:|---:|---:|
| all-MLP noverify sparse / prefix1 | 0.78 | 0.14 | -64pp |
| gate_up-only noverify sparse / prefix1 or prefix2 | 0.78 | 0.40 | -38pp |

So the useful controller region is not "make all unimportant rows sparse".
For Llama3 GSM8K, full-layer noverify sparse destroys accuracy.  The viable
region is tail-layer noverify sparse plus dense protection for the verifier
prefix/bonus.

Short serving throughput, `math_reasoning`, K=8, `max_tokens=512`, 128 total
requests:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_tail_layer_throughput_front24_prefix1front28_math512_bs32_64_20260702
```

| candidate | bs | dense total | SR24 total | total speedup | dense full | SR24 full | full speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| front24 / prefix2 | 32 | 2890.236 | 3435.910 | 1.189x | 3337.555 | 3780.976 | 1.133x |
| front24 / prefix2 | 64 | 3289.494 | 3641.900 | 1.107x | 3975.250 | 4222.490 | 1.062x |
| front28 / prefix1 | 32 | 2979.526 | 3240.792 | 1.088x | 3385.735 | 3506.816 | 1.036x |
| front28 / prefix1 | 64 | 3271.121 | 3319.128 | 1.015x | 3941.853 | 3863.434 | 0.980x |

I then added and tested `lossy_prefix2_rowrouted_mlp_front24_dense_noverify_compile`,
the same front24/prefix2 8pp-boundary controller with vLLM default compile
explicitly enabled:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_front24_prefix2_compile_quality_throughput_math512_bs32_64_20260702
```

| bs | dense total | SR24 total | total speedup | dense full | SR24 full | full speedup | SR24 accepted draft/step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 2745.229 | 3426.624 | 1.248x | 3366.557 | 3788.743 | 1.125x | 2.903 |
| 64 | 3339.120 | 3642.768 | 1.091x | 3980.198 | 4225.161 | 1.062x | 2.871 |

The quality row remained at dense `0.78`, SR24 `0.70`, delta `-8pp`, so the
compile variant is currently the best small-scale point.  It clears 1.2x total
at bs32 but not bs64, and the full-batch speedup is still only 1.125x/1.062x.

I also tested two assembly/data-format ablations that do not change the
controller:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_front24_outputbuf_tritonassemble_throughput_math512_bs32_64_20260702
```

| candidate | bs | total speedup | full speedup |
|---|---:|---:|---:|
| front24 + output buffer | 32 | 1.104x | 1.118x |
| front24 + output buffer | 64 | 1.042x | 1.057x |
| front24 + Triton assembly | 32 | 1.049x | 1.072x |
| front24 + Triton assembly | 64 | 1.080x | 1.040x |

Both were worse than plain front24 compile.  The bottleneck is therefore not
mainly fresh output allocation or the final slice-copy assembly.  The remaining
gap is the two-branch MLP execution itself and the fact that the run still has
many `PIECEWISE` graph steps.

I then ran bs64-only targeted systems ablations to test three common
explanations for the remaining bs64 gap:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_front24_bs64_fill_overlap_reusebase_throughput_20260702
```

| candidate | total speedup | full speedup | read |
|---|---:|---:|---|
| front24 + reuse sparse base output | 0.932x | 0.965x | sparse full-block fill is not the missing piece |
| front24 + dense-fill 128 | 0.929x | 0.971x | small dense branch is not the missing piece |
| front24 + dense-fill 256 | 0.931x | 0.957x | promoting more exact rows hurts speed |
| front24 + Python stream overlap | 0.871x | 0.939x | Python-level dense/sparse overlap is not enough |

This rules out the easy fixes.  The useful point remains plain
front24/prefix2/compile.  The path to 1.2x at bs64 needs a real grouped/fused
MLP implementation or a narrower sparse target that avoids splitting the
expensive `gate_up_proj` path.

I then tested the narrower target suggested by the component probe: keep
`gate_up_proj` dense and route only `down_proj`.

Full-layer down-only noverify sparse was too aggressive:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down_only_quality_gsm8k50_20260702
```

| candidate | dense acc | SR24 acc | delta |
|---|---:|---:|---:|
| down-only all layers / prefix2 | 0.78 | 0.52 | -26pp |
| down-only all layers / prefix1 | 0.78 | 0.48 | -30pp |

Tail-layer down-only was viable:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down_tail_quality_gsm8k50_retry_20260702
```

| candidate | dense acc | SR24 acc | delta | reg/imp |
|---|---:|---:|---:|---:|
| down-front24 / prefix2 | 0.78 | 0.70 | -8pp | 4/0 |
| down-front16 / prefix2 | 0.78 | 0.74 | -4pp | 5/3 |

Short throughput:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down_tail_throughput_math512_bs32_64_20260702
```

| candidate | bs | total speedup | full speedup |
|---|---:|---:|---:|
| down-front16 / prefix2 | 32 | 1.117x | 1.043x |
| down-front16 / prefix2 | 64 | 1.196x | 1.067x |
| down-front24 / prefix2 | 32 | 1.022x | 1.008x |
| down-front24 / prefix2 | 64 | 1.045x | 1.038x |

This is the first candidate that improves bs64 without splitting `gate_up_proj`.
It is also more accurate than all-MLP front24.  The downside is bs32: all-MLP
front24 is still faster at bs32, so the next useful experiment is a full
bs8/16/32/64 comparison of all-MLP front24 and down-front16.  If the split
holds, the implementation direction should become shape-aware routing: use the
all-MLP tail-sparse path only where its extra gate_up sparsity pays off, and
prefer down-only sparsity at larger batches where two-branch gate_up splitting
does not reach 1.2x.

I ran that bs8/16/32/64 comparison on `math_reasoning`, K=8,
`max_tokens=512`, 128 total requests:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_front24_vs_downfront16_bs8_64_math512_20260702
```

All-MLP tail sparse, `front24/prefix2/compile`:

| bs | dense total | SR24 total | total speedup | dense full | SR24 full | full speedup | accepted draft/step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1437.4 | 1569.3 | 1.092x | 1514.4 | 1677.6 | 1.108x | 2.902 |
| 16 | 2312.2 | 2401.8 | 1.039x | 2526.9 | 2637.2 | 1.044x | 2.771 |
| 32 | 2724.3 | 3414.2 | 1.253x | 3322.6 | 3760.6 | 1.132x | 2.887 |
| 64 | 2918.0 | 3688.2 | 1.264x | 3921.1 | 4289.1 | 1.094x | 2.876 |

Down-only tail sparse, `down-front16/prefix2/compile`:

| bs | dense total | SR24 total | total speedup | dense full | SR24 full | full speedup | accepted draft/step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1473.4 | 1427.5 | 0.969x | 1552.5 | 1531.7 | 0.987x | 2.713 |
| 16 | 2328.1 | 2339.3 | 1.005x | 2535.8 | 2539.7 | 1.002x | 2.723 |
| 32 | 2716.3 | 3222.2 | 1.186x | 3321.6 | 3560.6 | 1.072x | 2.762 |
| 64 | 2940.6 | 3795.4 | 1.291x | 3952.4 | 4435.5 | 1.122x | 2.778 |

This confirms the split:

- All-MLP tail sparse is the better high-acceptance, medium-batch candidate.
  It clears 1.2x total at bs32/64, but low batch is only 1.09x/1.04x.
- Down-only tail sparse is the better bs64 candidate and has a smaller accuracy
  loss, but it does not help bs8/16 and is slightly weaker than all-MLP at bs32.
- The low-batch problem is not accepted length: accepted draft tokens are still
  about 2.7-2.9 per step.  The loss is system overhead from splitting the MLP
  into small dense/sparse row groups and losing efficient graph/GEMM shapes.

System-design read after these runs:

1. The controller should keep an 8pp budget instead of requiring lossless
   output.  GSM8K-50 shows `front24/prefix2` and `down-front16/prefix2` are
   both inside the useful accuracy region.
2. The data format must be row-routed, not correction-only: unimportant rows
   should run only 2:4 sparse; important rows should avoid redundant sparse
   work whenever the dense block is large enough.  The current Python/PyTorch
   row-routed path has the right semantics but not the right kernel shape.
3. Low-batch speedup needs a grouped/fused verifier MLP operator.  It should
   pack important and unimportant rows into stable block descriptors, reuse
   preallocated workspaces, capture a small set of graph shapes, and run the
   sparse-unimportant and dense-important branches inside the operator instead
   of through Python-level launches.
4. When the important-token count is too small, the operator should either
   fill the dense side with the next-most-important rows or fall back to dense;
   Python-level dense fill did not help, so this needs to be inside the grouped
   operator to avoid extra launch/gather/scatter overhead.
5. The short-term serving policy should be batch/shape-aware: use all-MLP
   tail sparse near bs32, use down-only tail sparse near bs64, and keep bs8/16
   dense until the fused/grouped operator exists.

I then encoded the end-to-end batch split as a serving policy:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/configs/sr24_scheduler_policy_front24_serving_bs8_64.json
```

The first attempt also forced `--sr24-fixed-prefix-route-descriptor-only`.
That was wrong for the current live path: it preserved bs8/16 behavior but hurt
the high-batch rows.

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_front24_serving_policy_bs8_64_math512_20260702
```

| bs | total speedup | full speedup | accepted draft/step |
|---:|---:|---:|---:|
| 8 | 1.074x | 1.092x | 2.908 |
| 16 | 1.042x | 1.064x | 2.897 |
| 32 | 1.158x | 1.129x | 2.902 |
| 64 | 1.046x | 1.057x | 2.798 |

After removing descriptor-only and leaving only the scheduler policy
dense-bypass gate, bs32 recovered to the plain front24 result and bs8/16
improved modestly:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_front24_serving_policy_nodescriptor_bs8_16_math512_20260702
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_front24_serving_policy_nodescriptor_bs32_64_math512_20260702
```

| bs | dense total | SR24 total | total speedup | dense full | SR24 full | full speedup | accepted draft/step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1421.7 | 1622.8 | 1.141x | 1491.8 | 1675.2 | 1.123x | 2.856 |
| 16 | 2280.3 | 2633.3 | 1.155x | 2490.0 | 2773.4 | 1.114x | 2.890 |
| 32 | 2736.7 | 3435.1 | 1.255x | 3357.7 | 3795.9 | 1.131x | 2.926 |
| 64 | 3336.7 | 3665.6 | 1.099x | 4022.1 | 4297.1 | 1.068x | 2.863 |

Read:

- The policy gate is useful as a safety valve, but it is not the missing
  1.2x low-batch solution.  It only moves bs8/16 to about 1.14-1.16x.
- Descriptor-only should not be used as a default live optimization; it is a
  planner/data-format diagnostic until the grouped operator consumes the
  descriptor directly.
- For bs64, down-front16 remains the better current live candidate.  The
  front24 policy run has similar absolute SR24 throughput to plain front24 but
  loses speedup when the dense baseline run is higher.
- The next implementation step is still a grouped/fused operator or scheduler
  grouping path.  Policy-only gating cannot create enough useful work at
  bs8/16.

I also checked whether the quality budget could be spent more aggressively by
moving the dense/sparse boundary earlier.

All-MLP tail sparse front20/front22 and down-only tail sparse front12/front14
were added to:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/scripts/run_sr24_lossy_speed_quality_sweep.py
```

Quality probes were run with GSM8K limit 50 and an 8pp accuracy-loss budget:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_boundary_front20_down12_quality_speed_probe_20260702
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_boundary_front22_down14_quality_gsm8k50_20260702
```

| candidate | dense acc | SR24 acc | delta | regressions / improvements | quality |
|---|---:|---:|---:|---:|---|
| all-MLP front20 | 0.78 | 0.66 | -12pp | 7 / 1 | fail |
| down-only front12 | 0.78 | 0.66 | -12pp | 7 / 1 | fail |
| all-MLP front22 | 0.78 | 0.64 | -14pp | 8 / 1 | fail |
| down-only front14 | 0.78 | 0.70 | -8pp | 6 / 2 | pass |

The only new passing point was down-only front14, so I measured throughput:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down_front14_throughput_math512_bs8_64_20260702
```

| bs | dense total | SR24 total | total speedup | dense full | SR24 full | full speedup | accepted draft/step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1432.2 | 1394.1 | 0.973x | 1498.7 | 1481.2 | 0.988x | 2.649 |
| 16 | 2328.1 | 2237.3 | 0.961x | 2527.8 | 2454.1 | 0.971x | 2.647 |
| 32 | 2931.8 | 2973.0 | 1.014x | 3368.4 | 3432.3 | 1.019x | 2.669 |
| 64 | 3298.8 | 3710.9 | 1.125x | 3989.8 | 4323.8 | 1.084x | 2.624 |

This is worse than down-front16 at every important batch size.  It spends the
quality budget exactly, but loses accepted draft tokens and does not create a
faster kernel shape.  So the current boundary conclusion is:

- all-MLP front24 remains the useful all-MLP boundary; front22/front20 exceed
  the 8pp quality budget.
- down-only front16 remains the useful down-only boundary; down-front14 passes
  quality but is slower and lowers accepted length.
- down-front12 exceeds the quality budget.
- More aggressive sparsification is no longer the promising path.  The next
  speedup must come from removing row-routing overhead with a grouped/fused
  MLP operator and from scheduler grouping that feeds it stable packed rows.

I checked the existing grouped/packed MLP code path while doing this pass.  The
repo currently has planner and microbenchmark scaffolding, but not a live
serving operator that consumes packed descriptors inside the verifier MLP.  The
live path still goes through Python/PyTorch row routing, which explains why
policy-only and boundary-only changes cannot reliably reach 1.2x at bs8/16.

Current conclusion:

1. Relaxing to an 8pp budget helps.  The best current points are
   policy-gated all-MLP `front24/prefix2/compile` for bs8/16/32 and down-only
   `down-front16/prefix2/compile` for bs64.
2. The current implementation still does not satisfy the requested systems
   target.  It reaches at least 1.2x total throughput at bs32 and with the
   down-front16 bs64 row, but bs8/16 are still about 1.14-1.16x.
3. The failure mode is not low accepted length.  It is the operator/data-format
   overhead: small row groups, extra gather/scatter, separate dense/sparse
   launches, and weaker graph capture.
4. The next optimization should be a real grouped/fused row-routed MLP.  The
   dense-important and sparse-unimportant branches should be scheduled inside
   one operator-level path, with stable packed row descriptors and captured
   shapes.  Python-level stream overlap and dense-fill knobs already failed.
5. Until that exists, the serving policy should be shape-aware and conservative:
   dense for bs8/16, all-MLP tail sparse for bs32, and down-only tail sparse for
   bs64.

## 2026-07-02 Disjoint Dense/Sparse Verifier Probe

I also tested the semantic change requested for the next systems path: if a
token is classified as unimportant and already runs the 2:4 sparse path, it does
not get a dense correction.  Only important verifier rows run dense; ordinary
non-verifier decode rows remain dense.  This isolates the "no duplicate dense
work for sparse rows" idea from the more aggressive all-MLP tail-sparse policy.

Run root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_ppopp_disjoint_probe_20260702
```

Work root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/temp/sr24_ppopp_disjoint_probe_20260702_work
```

Configuration:

- model: Llama-3.1-8B target with EAGLE3, K=8
- quality: GSM8K-CoT, limit 50, max new tokens 512
- quality budget: at most 8 percentage points absolute accuracy loss
- throughput: `math_reasoning`, batch sizes 8/32/64, max new tokens 128
- candidates: verifier-only fixed prefix 2/1/0, output buffer enabled,
  projection-aware cuSPARSELt small-M algorithm selection

Quality:

| candidate | dense acc | SR24 acc | delta | pair regressions / improvements | quality |
|---|---:|---:|---:|---:|---|
| verifier-only prefix2 | 0.78 | 0.74 | -4pp | 3 / 1 | pass |
| verifier-only prefix1 | 0.78 | 0.76 | -2pp | 1 / 0 | pass |
| verifier-only prefix0 | 0.78 | 0.74 | -4pp | 3 / 1 | pass |

Throughput, total tokens/s speedup:

| candidate | bs8 | bs32 | bs64 | best total |
|---|---:|---:|---:|---:|
| verifier-only prefix2 | 0.985x | 1.001x | 0.999x | 1.001x |
| verifier-only prefix1 | 0.862x | 0.818x | 0.985x | 0.985x |
| verifier-only prefix0 | 0.813x | 0.990x | 0.969x | 0.990x |

Throughput, full-batch steady-state speedup:

| candidate | bs8 | bs32 | bs64 | best full-batch |
|---|---:|---:|---:|---:|
| verifier-only prefix2 | 0.984x | 0.996x | 0.962x | 0.996x |
| verifier-only prefix1 | 0.944x | 0.883x | 0.997x | 0.997x |
| verifier-only prefix0 | 0.935x | 1.012x | 1.025x | 1.025x |

Read:

1. The disjoint semantics are quality-feasible under the 8pp budget.  The
   verifier-only rows can be made sparse-only without collapsing GSM8K accuracy.
2. The same change is not enough for speed.  It is near parity for prefix2 and
   worse for prefix1/prefix0.  Reducing the important dense rows makes the dense
   branch too small while leaving sparse launch, routing, gather/scatter, and
   graph-shape overheads in place.
3. This confirms that the main bottleneck is operator/data-format overhead, not
   the abstract choice of dense-vs-sparse rows.  The current Python/PyTorch
   row-routed path still creates small independent dense/sparse pieces.
4. The useful lossy operating point is still the more aggressive front24
   all-MLP tail-sparse policy, not verifier-only sparse.  The latter is a
   correctness/semantics probe for the eventual operator, not the final serving
   policy.

The next implementation should be phrased as a systems change:

- Build a compact per-step row descriptor that separates rows into
  `dense_important`, `sparse_unimportant`, and `dense_fallback` slabs.  Sparse
  rows must never be recomputed by dense correction.
- Feed that descriptor directly to a grouped/fused row-routed MLP path instead
  of materializing many small Python-side `index_select` / sparse / dense /
  `index_add_` operations.
- Run dense-important and sparse-unimportant work under one operator-level
  wrapper with preallocated output buffers and graph-capturable shapes.  Python
  stream overlap already failed to recover the overhead; the overlap has to be
  inside a lower-level grouped operator or a vLLM custom op boundary.
- For small important-token counts, use scheduler grouping or fallback dense.
  Filling the dense branch with unimportant tokens only helps if the fused
  operator can exploit the larger tile without violating the accuracy budget.
- Keep the quality target lossy and explicit: GSM8K limit 50 or larger, allow up
  to 8pp absolute accuracy loss, and report both accuracy and speed.  Requiring
  lossless behavior pushes the policy back toward dense and removes the speedup.

Near-term engineering target:

1. Keep `front24/prefix2/compile` as the main all-MLP lossy point for bs8/16/32.
2. Keep `down-front16/prefix2/compile` as the current bs64 fallback candidate.
3. Add a real packed grouped MLP prototype before running another full matrix.
   Without changing the data format/operator, the present code is unlikely to
   reach 1.2x on most of bs8/16/32/64.
