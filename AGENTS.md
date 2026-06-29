## SR24 Slowdown-First Rule

Current SR24/SpecLink tuning should start from a seven-part slowdown breakdown,
not from another threshold sweep. The current short note is:

```text
SR24_NEXT_SLOWNESS_BREAKDOWN.md
```

For each candidate, separate clean serving rows from diagnostic rows. Clean
serving rows are the only source for throughput, CUDA Graph coverage, accepted
length, and GPU utilization. Diagnostic rows with SR24 Linear timing or exact
routing counters are for localization only; their tok/s is distorted by CUDA
events and synchronization. The required parts are scheduler/mask build, sparse
base Linear, residual correction, gather/scatter, routing statistics, CUDA
Graph FULL/NONE counts, and GPU utilization.

Default entrypoint:

```bash
cd examples/evaluate/eval-guidellm
conda run -n spec python scripts/run_sr24_slowdown_breakdown.py
```

Keep these outputs in `examples/evaluate/eval-guidellm/results.bak/`. Treat the
current bottleneck as duplicated sparse-base plus correction work and graph
coverage until a fresh seven-part report shows otherwise.

Latest 2026-06-29 guardrail: do not promote coarse SR24 adaptive dense fallback
as the next speed path. On Llama-3.1-8B `math_reasoning` bs64/K8/max128,
`criticalprefix4_bucket16_directcslt` plus
`--sr24-adaptive-dense-fallback` dropped `speclink_t08` to `2880.598`
full-batch tok/s versus dense `3040.780`; the run kept CUDA Graph coverage and
GPU utilization, so the loss is an execution-planner/operator issue, not a
graph or idle-GPU issue. The adaptive planner was tightened afterward so capped
buckets use actual bucket-row counts and `SMALL_ROWS` defaults to `0`; this
recovers the pathological row but still does not beat the no-adaptive control
in a same-condition probe. Keep adaptive dense fallback and row-routed paths as
explicit ablations unless a new seven-part report proves a live serving win.

# Speculator Throughput Experiments

This repo is currently set up to benchmark Qwen3-8B with EAGLE3/P-EAGLE and
Llama-3.1-8B with FastDraft, Smurfs dynamic K, and EAGLE3 through vLLM and
GuideLLM.

## Environment

Use the existing conda environment:

```bash
conda activate spec
```

Known working environment:

- Python: `${CONDA_PREFIX}/bin/python`
- vLLM: `0.20.0`, installed editable from this repo's `vllm/`
- GuideLLM: `0.6.0`
- Torch: `2.11.0+cu130`
- Local package: this repo installed editable into `spec`

If recreating the environment, use Python 3.12, install GuideLLM and Hugging
Face Hub, then install this repo and the vendored vLLM editable. Keep vLLM's
CUDA build aligned with PyTorch's CUDA 13.0 stack; do not build it with a newer
system CUDA such as `${SYSTEM_CUDA_HOME}`, because that can produce PTX the
current driver cannot run.

```bash
conda create -n spec python=3.12
conda activate spec
pip install guidellm huggingface-hub
pip install -e . --no-deps
pip install -r ./vllm/requirements/build/cuda.txt
pip install \
  nvidia-cuda-nvcc==13.0.88 \
  nvidia-nvvm==13.0.88 \
  nvidia-cuda-crt==13.0.88 \
  nvidia-cuda-cccl==13.0.85
conda install -n spec -y -c conda-forge gcc_linux-64=13 gxx_linux-64=13

cd .
CU13=${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cu13
LIBCUDA_SO="${LIBCUDA_SO:?set LIBCUDA_SO to the system libcuda.so.1 path}"
ln -sfn lib "${CU13}/lib64"
ln -sfn libcudart.so.13 "${CU13}/lib/libcudart.so"
ln -sfn libnvJitLink.so.13 "${CU13}/lib/libnvJitLink.so"
ln -sfn libnvrtc.so.13 "${CU13}/lib/libnvrtc.so"
ln -sfn libnvvm.so.4 "${CU13}/lib/libnvvm.so"
mkdir -p "${CU13}/lib/stubs"
ln -sfn ${LIBCUDA_SO} "${CU13}/lib/stubs/libcuda.so"

CUDA_HOME="${CU13}" \
CUDACXX="${CU13}/bin/nvcc" \
CUDAHOSTCXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++ \
CC=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc \
CXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++ \
PATH="${CU13}/bin:${PATH}" \
LD_LIBRARY_PATH="${CU13}/lib:${LD_LIBRARY_PATH:-}" \
TORCH_CUDA_ARCH_LIST=12.0 \
MAX_JOBS=8 \
NVCC_THREADS=2 \
SETUPTOOLS_SCM_PRETEND_VERSION=0.20.0 \
  pip install -e ./vllm --no-deps --no-build-isolation --force-reinstall
```

GPU commands must run with real GPU access. In Codex sandboxed commands,
`torch.cuda.is_available()` may return false even when `nvidia-smi` sees the GPU;
run vLLM/GuideLLM commands with escalated GPU access.

## Models

Use local paths for speculator checkpoints. The Qwen3-8B base model defaults to
the Hugging Face ID and will use the local Hugging Face cache when available.

```bash
BASE_MODEL=Qwen/Qwen3-8B
EAGLE3_SPECULATOR_MODEL=../models/qwen3-8b-eagle3-speculator
PEAGLE_SPECULATOR_MODEL=../models/qwen3-8b-peagle-speculator
```

The current local paths are:

- EAGLE3: `../models/qwen3-8b-eagle3-speculator`
- P-EAGLE: `../models/qwen3-8b-peagle-speculator`
- Llama-3.1-8B base: `../models/llama-3.1-8b-instruct`
- Llama-3.1-8B FastDraft: `../models/llama-3.1-8b-fastdraft-150m-int8-hf`
- Llama-3.1-8B EAGLE3: `../models/llama-3.1-8b-eagle3-speculator`

To download the speculator checkpoints again:

```bash
cd examples/evaluate/eval-guidellm
./download_qwen3_8b_speculators.sh
./download_llama_3_1_8b_eagle3_speculator.sh
```

The base model is not copied into `../models` by default. Override it with
`QWEN3_8B_MODEL=../models/qwen3-8b` only if you want to force a specific
local base model directory.

## Vendored vLLM Source and Install

vLLM is vendored as a full source tree inside this repository:

```text
./vllm
```

This tree was imported from upstream vLLM `v0.20.0` and pushed to `main` in
commit `9def519`. Future vLLM edits for these experiments should be made in
`speculators/vllm`, not in `site-packages` and not in the older external
`../vllm` checkout.

Current local vLLM changes in `speculators/vllm` are Python-only:

- `peagle` support in
  `vllm/transformers_utils/configs/speculators/algos.py`
- conversion from `speculators_model_type=peagle` to `method=eagle3` with
  `parallel_drafting=true` in
  `vllm/transformers_utils/configs/speculators/base.py`
- motivation-breakdown instrumentation in
  `vllm/v1/worker/gpu_model_runner.py`, gated by
  `SPECLINK_BREAKDOWN=1`, `SPECLINK_BREAKDOWN_OUT`,
  `SPECLINK_BREAKDOWN_ALGO`, `SPECLINK_BREAKDOWN_BATCH_SIZE`, and
  `SPECLINK_BREAKDOWN_NUM_SPEC_TOKENS`
- Qwen3 verifier-detail instrumentation in
  `vllm/model_executor/models/qwen3.py` and
  `vllm/speclink_breakdown.py`, gated by
  `SPECLINK_BREAKDOWN_VERIFY_DETAIL=1`
- confidence/acceptance tracing for the SpecLink first validation experiment in
  `vllm/v1/spec_decode/llm_base_proposer.py`,
  `vllm/v1/sample/rejection_sampler.py`,
  `vllm/v1/worker/gpu_model_runner.py`, and
  `vllm/speclink_confidence_trace.py`, gated by
  `SPECLINK_TRACE_CONFIDENCE=1`. SR24 can also attach the verifier residual
  mask to these token records, so acceptance traces can report which accepted
  draft tokens were checked with base-only versus residual correction.
- SpecLink-CV live prefix/suffix verification slice in
  `vllm/speclink_cv.py`,
  `vllm/v1/core/sched/scheduler.py`,
  `vllm/v1/core/sched/output.py`, and
  `vllm/v1/worker/gpu_model_runner.py`, gated by
  `SPECLINK_CV_ENABLE=1`
- Smurfs dynamic K for FastDraft is implemented inside this repo's vendored
  vLLM in
  `vllm/smurfs_dynamic.py`,
  `vllm/v1/spec_decode/llm_base_proposer.py`, and
  `vllm/v1/core/sched/scheduler.py`, gated by
  `SPECLINK_SMURFS_DYNAMIC_ENABLE=1`
- SpecLink SR24 selective residual 2:4 experimental support in
  `vllm/speclink_sr24.py`,
  `vllm/v1/worker/gpu_model_runner.py`,
  `vllm/v1/spec_decode/llm_base_proposer.py`, and
  `vllm/model_executor/models/llama.py`, gated by
  `SPECLINK_SR24_ENABLE=1`. This is still experimental. The `dense_zero`
  backend, with legacy alias `prototype`, is correctness/backend-isolation
  only; the `torch_sparse` backend replaces the Llama target base weights with
  PyTorch
  `SparseSemiStructuredTensorCUSPARSELT`. By default residual correction uses
  compressed residual values materialized per corrected Linear call; setting
  `SPECLINK_SR24_RESIDUAL_BACKEND=torch_sparse` tries a second real sparse
  residual tensor but can exceed 32GB on full Llama-3.1-8B. With
  `SPECLINK_SR24_RESIDUAL_DEVICE=auto`, the torch-sparse/compressed-dense
  path keeps compressed residual values GPU-resident; use
  `SPECLINK_SR24_RESIDUAL_DEVICE=cpu` explicitly only as a memory fallback or
  CPU-transfer ablation. Current quality experiments should default to
  `SPECLINK_SR24_RESIDUAL_BACKEND=dense_rows`: it is memory-heavier, but it
  replaces corrected rows with the exact dense Linear output. On 2026-06-25,
  `compressed_dense` was confirmed GPU-resident but not dense-equivalent for
  bf16 all-corrected reconstruction, because the target Linear is split into a
  sparse-base GEMM plus a residual GEMM; use it only for storage/operator
  ablations until a fused numerically validated kernel exists. The same day,
  bs64/K8 `math_reasoning` dense-rows strategy runs showed `speclink_t08` at
  only about `0.92x` dense full-batch throughput despite slightly higher
  acceptance and about `95%` GPU utilization; policy-only variants
  (`all_if_any_low`, `critical_prefix` with min prefix `4` or `0`) did not close
  the gap. Route-all dense fallback improved only to about `0.94x` dense
  full-batch throughput, so simply falling back to dense on high-residual steps
  is not enough either. A later `low_confidence` batched-mask gate on
  bs64/math/max256 improved the cleaner selective path but still reached only
  `0.945x` dense full-batch throughput at threshold `0.8`; exact-routing
  diagnostics still showed `0.717` residual draft fraction. A follow-up
  `SPECLINK_SR24_SELECTIVE_MAX_RESIDUAL_DRAFT_ROWS=1` cap really lowered exact
  residual draft fraction to `0.124533`, but the clean no-sync serving gate was
  still only `0.950x` dense full-batch throughput (`3171.865` vs `3015.069`
  tok/s). Fixed-size bucket correction with
  `SPECLINK_SR24_RESIDUAL_BUCKET_SIZE=32` is the best current bucket direction,
  but the Triton override variant is negative. A follow-up microbench showed
  the default bucket delta/index-add graph path around `0.597ms` for bucket=32,
  while Triton override in-place was around `0.736ms`. The clean no-Triton
  serving rerun reached `3413.243` full-batch tok/s versus same-run dense
  `3391.668` on bs64/math/max256, only `1.006x` dense and still far from the
  `1.2x` target.
  Route-all and route-reuse dynamic row-routing variants were slower
  (`2840.040` and `2961.416` full-batch tok/s), so do not use those as the
  next optimization path. The same cap with mask-state synchronization reported
  `39.425ms/step` scheduler/mask time, so do not mix sync-heavy exact rows with
  clean serving throughput. A later token-level trace on the faster but
  inaccurate `down_proj=8-31` candidate showed that `low_confidence` leaves too
  many committed draft tokens uncorrected: accepted base-only fraction was
  `0.6983`, and most accepted base-only tokens were in the `0.8`/`0.9`
  DLM-score bins. Treat `low_confidence` as a speed diagnostic, not as the
  quality-safe default. The next runtime candidate to verify is
  `SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY=high_confidence`,
  `SPECLINK_SR24_THRESHOLD=0.9`, and
  `SPECLINK_SR24_SELECTIVE_MIN_PREFIX_RESIDUAL=2`; this comes from the offline
  projection in
  `examples/evaluate/eval-guidellm/results.bak/sr24_down8_31_trace_gate_20260625/analysis/report.md`
  but the clean GPU gates showed that changing only gate/up routing does not
  fix `down_proj=8-31`: GSM8K-50 remained `0.5600` versus dense `0.7200`.
  Correcting `down_proj=8-15` selectively while keeping `down_proj=16-31`
  base-only restored GSM8K-50 aggregate accuracy to `0.7200`, but full-batch
  throughput was only `3512.299` versus dense `3388.569` (`1.036x`), still far
  below the `1.2x` target. Do not rerun the gate/up-only `down8-31` candidate
  as a quality fix. The next speed path is therefore either a fused/graph-
  friendly mixed operator or a sharper quality gate for `down8-15` residual
  rows that preserves accuracy while correcting fewer rows. Use
  `SR24_SLOWDOWN_BREAKDOWN.md` as
  the current slowdown reference:
  it tracks scheduler/mask-build time, sparse base Linear time, dense-row
  correction time, gather/scatter, routing fractions, CUDA Graph coverage, and
  GPU utilization. A 2026-06-25 follow-up microbench under
  `results.bak/sr24_sparse_backend_probe_current_20260625_goal` and
  `results.bak/sr24_mixed_bucket_probe_current_20260625_goal` confirmed that
  `compressed_dense` tensors are GPU-resident, but current exact
  `all_corrected_24` graph paths are still slower than dense: gate/up
  rows=512 best exact graph is `0.757ms` versus dense `0.543ms`, and down
  rows=512 best exact graph is `0.338ms` versus dense `0.293ms`. The mixed
  bucket path is close to dense for gate/up (`0.534ms` versus `0.538ms`) and
  modestly faster for down (`0.263ms` versus `0.292ms`) at bucket=32, but this
  is not enough for a `1.2x` end-to-end gain. Treat this as evidence that the
  next `all_corrected_24` optimization requires a fused packed CUDA/Triton
  operator instead of another wrapper around two separate sparse/residual
  passes. A 2026-06-26 sparse backend probe in
  `results.bak/sr24_sparse_backend_alg0_probe_20260626` and
  `results.bak/sr24_sparse_backend_alg1_probe_20260626` made the backend
  conclusion stricter: the default cuSPARSELt alg0 is still the best supported
  PyTorch path; alg1 is slower for exact `all_corrected_24`, and forcing
  CUTLASS fails on the RTX 5090 with `sparse_semi_structured_mad_op :
  Supported only on GPUs with compute capability 8.x`. Do not spend further
  time trying CUTLASS or alg1 as the main `all_corrected_24` optimization on
  this machine. A 2026-06-26 goal-continuation probe in
  `results.bak/sr24_sparse_backend_residual_kernel_probe_20260626_goal`
  rechecked the compressed residual variants on the current environment. The
  device check showed mask/residual tensors on CUDA for the GPU paths, so
  `compressed_dense` is not slow because residual correction is happening on
  CPU. The exact all-corrected bound is still negative: for Llama gate/up
  rows=512, dense graph was `0.5390ms`, best exact two-sparse-GEMM graph was
  `0.7592ms`, cached compressed-dense graph was `1.0624ms`, and direct
  compressed Triton residual graph was about `35ms`. For down rows=512, dense
  graph was `0.2923ms`, best exact graph was `0.3347ms`, cached compressed was
  `0.4823ms`, and compressed Triton was about `18ms`. Treat the current
  compressed Triton residual kernel as rejected. Exact `all_corrected_24`
  needs a fused packed base+residual operator to become a speed path; otherwise
  use the dense fastpath as the exact-control ceiling and use no-fastpath rows
  only as operator diagnostics. A GSM8K-20 policy gate in
  `results.bak/sr24_lowconf_exact_gsm8k20_20260625_goal`,
  `results.bak/sr24_lowconf_t08_exact_gsm8k20_20260625_goal`, and
  `results.bak/sr24_prefixconf_vs_dense_gsm8k20_20260625_goal` showed the
  current controller tradeoff: `low_confidence@0.9` with min-prefix 2 still
  residual-corrects `91.7%` of draft rows, `low_confidence@0.8` still corrects
  `87.0%` and worsens accuracy, while `prefix_confidence@0.5` corrects only
  `16.1%` but drops GSM8K-20 score to `0.6000` versus dense `0.7000`.
  Therefore raw probability-threshold policies are not the main path to the
  `1.2x` target; either use a better importance signal or reduce the operator
  cost for mostly-corrected steps.
  For a direct "slow where" microbreakdown, run
  `examples/evaluate/eval-guidellm/scripts/profile_speclink_sr24_component_breakdown.py`
  under the `spec` environment. Current slowdown summary and microbreakdown
  outputs are
  `results.bak/sr24_requested_breakdown_summary_20260626` and
  `results.bak/sr24_requested_component_microbreakdown_20260626`; older focused
  outputs are
  `results.bak/speclink_sr24_component_breakdown_20260625_211745` for 512-row
  Llama MLP shapes and `results.bak/speclink_sr24_component_breakdown_20260625_211829`
  for 64-row shapes. Read: for older bucket/reuse-base rows, clean
  scheduler/mask build became sub-ms after the low-sync batched builder
  (`0.386ms/step` in that `speclink_t08` row), CUDA Graph coverage was not the
  first suspect, and GPU util remained high. The 2026-06-27 route-all refresh
  below supersedes this for graph-bucket route-all: it exposed a large
  row-index/bucket wall counter, so route-all work must be split before another
  controller sweep. Gate/up rows=512 is already slower
  than dense when residual rows reach 25% (`0.623ms` mixed graph versus
  `0.540ms` dense graph);
  down rows=512 only helps at low residual fractions and loses by 50%
  residual rows (`0.363ms` mixed versus `0.291ms` dense); at rows=128, gate/up
  mixed is `1.82-2.23x` dense across 12.5-87.5% residual rows and down sparse
  base is already `1.68x` dense before correction. The mask-build proxy is small
  after vectorization (`~0.003ms` vector mask and `0.008-0.010ms` bucket top-k),
  so the current first-order bottleneck is the mixed sparse-base plus dense-row
  correction operator, not global GPU idleness or the vectorized scheduler mask
  builder.
  A 2026-06-27 seven-part refresh updated this read with
  `results.bak/sr24_operator_ceiling_refresh_20260627/summary.md` and
  `results.bak/sr24_prefix4_thr07_trace_rejected_base_refresh_20260627/report.md`.
  Use this shape before further SR24 tuning: scheduler/mask build, base sparse
  Linear, residual correction, gather/scatter, routing row statistics, CUDA Graph
  FULL/NONE counts, and GPU util. The key result is still negative for the
  current two-pass mixed operator. Rows=512 gate/up dense graph is about
  `0.538ms`; serving-like mixed time is `1.03x` dense already at 12.5% residual
  rows and `2.26x` dense at 100%. Cached `compressed_dense` residual is
  CUDA-resident but still reaches `1.98x` dense at 100% residual because it does
  sparse base plus residual GEMM plus add/scatter. The prefix4/t0.7 trace also
  shows accepted base-only `0.0133` but rejected base-only `0.0697`; first
  rejected base-only tokens are quality risk because target logits select the
  recovered token. Constraining both risks below `0.05` needs residual fraction
  `0.6696`, and below `0.02` needs `0.8777`, which is already in the
  dense-time/slower regime. Do not continue with threshold-only sweeps as the
  main speed path; use them only as routing diagnostics unless a fused/packed
  base+residual operator or stronger low-risk row signal is added.
  A direct runtime check of the seemingly better offline candidate
  `critical_prefix@0.7`, `SPECLINK_SR24_SELECTIVE_MIN_PREFIX_RESIDUAL=4`,
  `extra_after_low=0`, `gate_up_proj=16-31` is also negative. Throughput under
  `results.bak/sr24_critical_prefix_t07_prefix4_throughput_bs64_math128_20260627`
  was only `2460.330` full-batch tok/s versus dense `3017.788` (`0.815x`),
  with acceptance slightly higher and GPU util higher, so this is still useful-
  work/operator overhead rather than acceptance collapse. The GSM8K-20 quality
  gate under
  `results.bak/sr24_critical_prefix_t07_prefix4_quality_gsm8k20_20260627`
  dropped from dense `0.7000` to `0.6000` with `2` paired regressions and no
  improvements. Its trace analysis still had accepted base-only `0.0351` and
  rejected base-only `0.0508`. Do not promote this critical-prefix candidate to
  a default quality fix; more conservative variants push residual fraction into
  the dense-time/slower operator regime.
  A 2026-06-27 prefix-confidence follow-up added Triton batched-mask support
  for `SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY=prefix_confidence` and verified
  slow-path/batched/GPU-count equivalence in
  `examples/evaluate/eval-guidellm/scripts/check_speclink_sr24_correctness.py`.
  Offline projection on the current trace picked
  `SPECLINK_SR24_PREFIX_THRESHOLD=0.05` as an accepted-prefix diagnostic
  (`accepted_base_only_fraction=0.0482`, projected residual draft fraction
  `0.5351`). Live bs64/math/max128 graph-enabled serving confirmed the quality
  side but rejected it as a speed path: `down_proj,gate_up_proj` reached
  `1.416` accepted draft tokens/step versus dense `1.395`, but only
  `2073.671` full-batch tok/s versus dense `3019.809` (`0.687x`); restricting
  to `gate_up_proj` reached `1.406` versus dense `1.396`, but only
  `2323.575` versus dense `3024.459` (`0.768x`). Both SR24 rows had high GPU
  util (`88-90%`) and server graph profile `{"FULL": 49}`, so the remaining
  slowdown is mixed sparse+residual useful-work efficiency, not accepted-length
  collapse or gross graph loss.
  The current best follow-up is `prefix_confidence@0.05`, `gate_up_proj` only,
  `SPECLINK_SR24_ROUTE_ALL_RESIDUAL_ROWS=1`, bucket512, low-sync stats off, and
  graph bucket enabled. On bs64/math/max128 it reached `2961.618` full-batch
  tok/s versus same-root dense `3018.108` (`0.981x`) with accepted draft/step
  `1.399` versus dense `1.395`. Disabling
  `SPECLINK_SR24_ROUTE_CONTIGUOUS_FASTPATH` was neutral (`2949.227`, `0.974x`),
  and the route-all CPU-sync ablation showed only the old sync-heavy path is
  clearly bad (`2195.085` full-batch tok/s); low-sync variants stayed in the
  `2806-2954` short-run band. The route-all component diagnostic now reports
  base sparse GEMM `0.676ms/call` for `125` base rows/call, dense correction
  `0.184ms/call` for `114` residual rows/call, and route-all gather/scatter
  `0.093ms/linear` (`0.023ms/event` is only event-averaged).
  A user-requested seven-part refresh under
  `results.bak/sr24_user_breakdown_routeall_bs64_math128_20260627` kept the
  same accepted-length story (`speclink_t08` accepted draft/step `1.404` versus
  dense `1.396`) and good graph coverage (`{"FULL":62,"NONE":2}`), but exposed
  clean `sr24_scheduler_mask_wall_cpu_ms_per_step=42.339`, almost all in
  `sr24_scheduler_row_index_bucket_wall_cpu_ms_per_step=42.083`. Diagnostic
  timing for the same route-all shape showed `scheduler_request_routing_loop`
  `7.019ms/step`, `scheduler_bucket_topk` `2.926ms`, base sparse
  `0.637ms/call`, dense correction `0.183ms/call`, and gather/scatter
  `0.093ms/linear`. A follow-up added
  `SPECLINK_SR24_ROUTE_ALL_SKIP_BUCKET=1` and split clean counters for bucket
  build versus mixed row indices. It was negative: bucket build fell to
  `0.001ms/step`, but `scheduler_mixed_row_indices_wall_cpu_ms_per_step` stayed
  at `42.113ms` and full-batch throughput fell to `2809.138` (`0.931x` dense).
  Forcing CUDA Graph NONE lowered mixed row-index wall time to `13.499ms/step`
  but was slower overall (`2611.693`, `0.863x`) because all steps became
  `NONE`. Therefore the next SR24 work should avoid dynamic
  `residual_mask.nonzero()` / `(~residual_mask).nonzero()` row-list
  materialization in `_compute_mixed_row_indices()` while keeping CUDA Graph
  coverage, then continue with fused/packed mixed Linear or a lower-overhead
  base sparse kernel.
  A follow-up on `down_proj=16-31` base-only plus gate/up cap1/bucket32 showed
  only a small speedup: no-adaptive full-batch `3234.353` versus same-run dense
  `3071.637` (`1.053x`), and gate/up adaptive dense fallback at fraction
  `0.05` full-batch `3199.531` versus same-run dense `3020.592` (`1.059x`).
  Both are far below `1.2x`. The small GSM8K-20 gate for the no-adaptive
  down16base route was aggregate-neutral (`0.7000` vs dense `0.7000`) but had
  `2` paired regressions and `2` paired improvements, so this is not a
  quality-safe final path. Results:
  `results.bak/sr24_noadaptive_down16base_bs64_math256_20260626`,
  `results.bak/sr24_adaptive_gate005_down16base_bs64_math256_20260626`, and
  `results.bak/sr24_noadaptive_down16base_accuracy_gsm8k20_20260626`.
  For lm-eval SR24 accuracy gates, prefer the direct Python runner rather than
  nesting the shell wrapper under another `conda run`; if `addr2line` resolves
  to the base conda prefix, set
  `ADDR2LINE=/ACALAB/stu1/miniconda3/envs/spec/bin/x86_64-conda-linux-gnu-addr2line`.
  `quality_safe_selective` now keeps mask-state synchronization and adaptive
  dense fallback off by default. A bs64/K8/math/max256 clean-serving ablation
  showed `speclink_t08` at `2541.967` steady output tok/s with adaptive
  fallback plus mask-state sync, `2655.928` with adaptive off but sync on, and
  `2680.803` with both adaptive and mask-state sync off, versus dense EAGLE3
  `2780.607`. Keep adaptive dense fallback as an explicit ablation, not the
  default speed path. When enabled, it is controlled by
  `SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK=1`,
  `SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_SMALL_ROWS=0`,
  `SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_GATE_UP_FRACTION=0.10`,
  `SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_DOWN_FRACTION=0.25`, and
  `SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_SMALL_DOWN_NO_RESIDUAL=1`.
  `SMALL_ROWS=0` intentionally disables the small-row rule. The old
  `SMALL_ROWS=128` behavior remains reproducible by explicit flag, but a
  2026-06-29 bs64/K8/math probe fired it `936/960` adaptive fallback calls and
  dropped `speclink_t08` below dense, so do not use it as the default planner.
  The lm-eval and GuideLLM runners include these values in the SR24 compile
  cache fingerprint, export them to the vLLM server when requested, and report
  the resulting fallback counters. The remaining likely speed work is still
  fused packed sparse/residual kernels or a much sharper quality-safe routing
  signal.
  A follow-up bucket-size sweep with the updated no-adaptive/no-sync preset did
  not open a speed path: bucket 32/16/8 gave `2671.077`/`2672.886`/`2661.978`
  steady output tok/s on the same bs64/K8/math/max256 setup, all still below
  dense EAGLE3 `2780.607`. Do not treat smaller residual buckets as the next
  main optimization unless a larger quality/speed run contradicts this.
  `SPECLINK_SR24_TARGET_LEAFS` narrows SR24 to selected
  Llama Linear leaves for attention-only or MLP-only ablations, and
  `SPECLINK_SR24_RESIDUAL_OUT_CHUNK` chunks runtime residual materialization to
  avoid large MLP temporary allocations.
  `SPECLINK_SR24_CACHE_COMPRESSED_RESIDUAL_WEIGHT=1` is a diagnostic
  `compressed_dense` optimization that caches the dense GPU residual tensor
  after first materialization; it trades memory for avoiding repeated residual
  unpacking and is not a storage-saving final path.
  The GuideLLM matrix runner now auto-enables this cache, prewarms it, and sets
  `SPECLINK_SR24_RESIDUAL_OUT_CHUNK=0` for the specific ablation
  `all_corrected_24 + torch_sparse/compressed_dense + no dense fastpath`, so
  the run measures the best current GPU-resident compressed path rather than
  repeated chunked materialization. Use
  `--no-sr24-auto-compressed-residual-fastpath` only to reproduce the older
  chunked-materialization diagnostic.
  `SPECLINK_SR24_COMPRESSED_RESIDUAL_TRITON=1` is an experimental
  diagnostic-only compressed residual matmul that computes directly from
  GPU-resident 2:4 complementary values with Triton. It is not a recommended
  serving path after the 2026-06-24 all-corrected ablation.
  `SPECLINK_SR24_DIRECT_CSLT_LINEAR=1` is also only a diagnostic switch for now.
  On the current `quality_safe_selective` bs64/K8 `math_reasoning` smoke it was
  effectively flat (`3397.543 -> 3399.042` full-batch tok/s), and the per-leaf
  microbench showed PyTorch sparse graph replay faster than direct cuSPARSELt
  alg0/alg1 for the main shapes. Do not make it the default unless a newer
  microbench reverses this result.
  `SPECLINK_SR24_TRITON_BUCKET_OVERRIDE=1` is also not the current serving
  recommendation for dense-row bucket correction; the 2026-06-25 bucket
  microbench and clean serving rerun favored the default bucket delta/index-add
  path. Keep the Triton override only as a negative/diagnostic ablation unless
  a newer microbench reverses this result. `SPECLINK_SR24_GATE_UP_SPLIT` supports
  `up_sparse`, `gate_sparse`, and experimental `channel_pair`; the
  `channel_pair` mode also reads
  `SPECLINK_SR24_GATE_UP_CHANNEL_DENSE_FRACTION`,
  `SPECLINK_SR24_GATE_UP_CHANNEL_STRATEGY`, and optional
  `SPECLINK_SR24_GATE_UP_CHANNEL_FUSED_ACT`, and should currently be treated as
  a negative/diagnostic ablation rather than the default `speclink_t08` path.
  `SPECLINK_SR24_ROW_ROUTED_MLP=1` is another experimental mixed-mask MLP-level
  routing path. It now defaults to
  `SPECLINK_SR24_ROW_ROUTED_MLP_MIN_DENSE_ROWS=128` and supports optional
  `SPECLINK_SR24_ROW_ROUTED_MLP_MAX_DENSE_ROWS=0` (`0` disables the upper
  dense-row guard) and `SPECLINK_SR24_ROW_ROUTED_MLP_MAX_BASE_ROWS=0`
  (`0` disables the base-row guard). Keep the lower guard unless a new run
  proves large corrected row groups are common. Use the base-row guard when
  testing row-routed MLP in serving, because large base-row groups currently
  split a large sparse MLP into dynamic gather/sparse/scatter work. The
  2026-06-25 live diagnostic showed the unguarded path firing on only about
  `3` dense rows out of `1251` total rows and costing `4.975ms/call`, so it is
  not the current route to a `1.2x` serving speedup.
  `SPECLINK_SR24_FORCE_CUDAGRAPH_NONE_FOR_MIXED=1` is the default correctness
  guard for mixed SR24 verify plans. The 2026-06-26 static-mixed graph ablation
  (`--sr24-static-mask-state mixed --no-sr24-force-cudagraph-none-for-mixed`)
  restored many FULL graph steps and improved the high-confidence/prefix3
  gate-up-only candidate from `2287.754` to `2487.325` total tok/s, but it
  remained below dense (`2785.079`) and still had `1` paired GSM8K regression
  on 20 samples. A later all-rows-residual graph-capture fix wires SR24 dummy
  CUDA Graph capture to the same static mask buffer used by runtime replay, but
  the GSM8K-64 gate still had `Pair reg=1` (`Pair imp=4`), while the eager
  low-sync control had `Pair reg=0`. A static-mask tail-fill follow-up still
  had `Pair reg=1`, so stale padded mask tails are not the whole explanation.
  A doc10 tokenized GSM8K replay under
  `results.bak/sr24_replay_graph_vs_eager_doc10_20260626/compare/report.md`
  found dense EAGLE3 and mixed CUDA Graph SR24 selected identical token ids for
  all 95 generated tokens, while SR24 eager diverged at generated position 33.
  This means mixed Graph recovery is not the only correctness or speed issue;
  eager numerical shape and the SR24 approximation still need paired gates.
  Keep mixed cudagraph as an explicit ablation, not a default speed path.
  Focused vLLM Llama Linear equivalence checks on 2026-06-26 showed
  `dense_rows` is exact (`max_abs=0`) for gate/up rows=512, down rows=512, and
  full MLP rows=128. `compressed_dense` is GPU-resident but not exact in bf16
  on the same checks: gate/up `max_abs=0.03125`, down `0.0625`, and full MLP
  `0.0625`. Use `dense_rows` for quality gates; keep `compressed_dense` as an
  operator/storage diagnostic until a numerically validated fused kernel exists.
  Before doing another SR24 controller or threshold sweep, first generate or
  refresh the seven-part slowdown breakdown with the single-entry protocol:
  `examples/evaluate/eval-guidellm/scripts/run_sr24_slowdown_breakdown.py`.
  It runs separate clean serving rows, instrumented serving rows, and a
  component microbench, then calls
  `examples/evaluate/eval-guidellm/scripts/summarize_sr24_breakdown.py` and
  `examples/evaluate/eval-guidellm/scripts/make_sr24_seven_part_breakdown.py`.
  Treat clean serving tok/s as performance evidence; treat
  `--sr24-breakdown-linear` rows only as localization evidence because CUDA
  event timing and exact routing counters perturb throughput. Example:
  `cd examples/evaluate/eval-guidellm && conda run -n spec python scripts/run_sr24_slowdown_breakdown.py`.
  This entrypoint now defaults `--sr24-allow-cudagraph` to on, so base-only
  clean rows measure the safe CUDA Graph path. Dynamic mixed `speclink_t08`
  still stays eager under `SPECLINK_SR24_FORCE_CUDAGRAPH_NONE_FOR_MIXED=1`
  unless an explicit graph-safety ablation disables that guard. The script also
  exposes `--sr24-reduce-cpu-sync/--no-sr24-reduce-cpu-sync` and
  `--sr24-sync-mask-state/--no-sr24-sync-mask-state`; use those to report CPU
  synchronization ablations instead of hand-editing the matrix command. Use
  `--include-cpu-sync-ablation` when you want the standardized five-row
  attribution: low-sync stats on, low-sync stats off, sync-mask-state,
  sync-heavy/no-reduce-sync, and low-sync GPU-count breakdown. It also exposes
  `--sr24-route-dense-fallback-fraction` for the conservative
  high-residual dense fallback ablation. It also forwards the current
  graph-capable bucketed-path switches: `--sr24-direct-cslt-linear`,
  `--sr24-bucket-dense-copy`, and the off-by-default
  `--sr24-sort-bucket-rows`. The entrypoint also forwards compressed-residual
  diagnostic switches: `--sr24-residual-out-chunk`,
  `--sr24-cache-compressed-residual-weight`,
  `--sr24-prewarm-compressed-residual-weight`,
  `--sr24-auto-compressed-residual-fastpath`,
  `--sr24-compressed-residual-triton`,
  `--sr24-compressed-residual-block-{m,n,g}`, and
  `--sr24-extract-chunk-rows`.
  The current slowdown references are
  `SR24_CURRENT_BREAKDOWN_20260628.md`,
  `SR24_CURRENT_SLOWDOWN_ANALYSIS.md`, and
  `examples/evaluate/eval-guidellm/results.bak/sr24_corrected_fast_candidate_breakdown_b12_bs64_math_k8_20260628/seven_part_report/report.md`.
  Read scheduler/mask build, base sparse Linear, residual correction,
  gather/scatter, routing counts, CUDA Graph FULL/NONE, and GPU utilization
  together before another controller or threshold sweep. Keep three row types
  separate: clean serving rows for tok/s, CUDA Graph, and GPU util;
  instrumented serving rows for CUDA-event localization; and component
  microbench rows for isolated Linear-shape ceilings. Do not use sync-heavy
  rows as throughput evidence.
  The 2026-06-28 clean read is: `base_only_24` is not accepted-length-limited
  and not globally GPU-idle. In the graph-safe bs64/math/K8/max256 reference
  run it reaches `3965.653` full-batch tok/s and `2785.660` total tok/s versus
  dense `3430.409` full-batch and `2317.632` total, with accepted draft/step
  `2.027`, GPU util `90.750%`, and CUDA Graph `{"FULL":126,"NONE":2}`. Do not
  force the latest mixed `speclink_t08` residual-by-leaf/default-compile
  options onto `base_only_24`: that combination can fail during vLLM profile-run
  compilation with a PyTorch lazy-allocation error. Use the standalone
  graph-safe base-only reference instead.
  The older default `speclink_t08` row with dynamic mixed Graph disabled is slow
  (`2929.340` full-batch tok/s, `0.853x`, CUDA Graph all `NONE`), but the
  current best graph-safe bucketed candidate is
  `SPECLINK_SR24_RESIDUAL_BUCKET_SIZE=16`,
  `SPECLINK_SR24_BUCKET_DENSE_COPY=1`, and
  `SPECLINK_SR24_DIRECT_CSLT_LINEAR=1`; it reaches `3930.796` full-batch tok/s
  versus same-root dense `3429.905` (`1.146x`) and total tok/s `2720.019`
  versus dense `2319.712` (`1.173x`), with CUDA Graph `{"FULL":94,"NONE":2}`.
  This is the best scoped serving result so far, but still below the `1.2x`
  target and only passed GSM8K-20 as a sanity check, not a full quality gate.
  Follow-up bucket-size checks with the same graph-safe direct-cslt/bucket-copy
  shape did not close the gap: bucket8 repeat reached only `1.148x` total and
  `1.156x` full-batch, bucket10 reached `1.163x` total and `1.154x`
  full-batch, bucket12 reached `1.173x` total and `1.156x` full-batch, and
  bucket16 reached `1.173x` total and `1.146x` full-batch. The earlier bucket8
  `1.190x` total row came from a weak dense baseline and should not be treated
  as stable evidence. Do not spend the next pass on another fixed bucket-size
  sweep unless the operator implementation changes. A follow-up off-by-default
  memory-locality ablation, `SPECLINK_SR24_SORT_BUCKET_ROWS=1` /
  `--sr24-sort-bucket-rows`, sorts capped bucket rows before dense gather and
  index-copy. It was negative on the same bucket12 shape: total speedup dropped
  from `1.173x` to `1.157x` and scheduler/mask wall rose from `0.336ms/step` to
  `0.913ms/step`, while CUDA Graph stayed `{"FULL":94,"NONE":2}`. Keep this
  flag off by default.
  The latest corrected seven-part breakdown uses the current graph-capable
  bucketed `speclink_t08` candidate without priority/direct-position routing:
  bs64/math/K8/max256,
  `critical_prefix@0.6,prefix4,extra1`, target leafs
  `gate_up_proj,down_proj`, residual layers
  `gate_up_proj=16-31;down_proj=8-15`, bucket12, direct cuSPARSELt, and bucket
  dense copy. Clean serving measured dense `3524.307` full-batch tok/s and
  `2599.815` total tok/s, `base_only_24` `4307.108` full-batch and `2955.659`
  total, and `speclink_t08` `3902.787` full-batch and `2625.737` total.
  `speclink_t08` is therefore `1.107x` full-batch but only `1.010x` total,
  below the `1.2x` target and far below the base-only upper bound. Accepted
  draft/step is not collapsed (`1.736` dense, `2.205` base-only, `2.103` SR24),
  GPU util is high (`91.154%` for SR24), and CUDA Graph coverage is healthy
  (`{"FULL":126,"NONE":2}`). Clean scheduler/mask work is sub-ms
  (`0.289ms/step`), gather/scatter is secondary (`0.014ms/call`), and the
  diagnostic mixed path localizes cost to sparse base plus correction: base
  sparse `1.041ms/call`, `gate_up_proj=16-31` sparse base `1.071ms/call`,
  dense correction `0.161ms/call`, draft residual/base rows `2928/2464`, draft
  residual fraction `0.543`, and bucket fill `0.989`. Therefore ordinary
  CPU-sync cleanup is now only an ablation/guardrail; the next speed work needs
  either a graph-safe fused/packed mixed operator or a routing signal that
  sharply lowers residual rows without paired accuracy loss. The older
  `sr24_user_requested_current_breakdown_bs64_math_k8_20260628` row enabled
  priority/direct-position routing and is superseded as the current speed
  reference. `SPECLINK_SR24_SELECTIVE_NON_DRAFT_POLICY=predicted_full_accept`
  is now supported by the batched GPU mask builder instead of falling back to
  the Python request loop, and the correctness check covers slow, uniform,
  indexed, and GPU-count paths. It is a negative speed ablation on the corrected
  bucket12 shape: `bonus` reached `2683.694` total / `3917.754` full-batch
  tok/s with accepted draft/step `2.171`, while `predicted_full_accept` reached
  `2638.930` total / `3903.531` full-batch tok/s with accepted draft/step
  `2.119`. Do not use bonus-row removal as the primary SR24 optimization path
  unless a later quality-aware policy changes acceptance behavior.
  A same-condition CPU-sync ablation for this corrected bucket12 path ran under
  `examples/evaluate/eval-guidellm/results.bak/sr24_fast_candidate_cpu_sync_ablation_b12_bs64_math256_20260628/`,
  with `low_sync_stats_on` at `3966.966` full-batch tok/s and `2705.051` total
  tok/s versus same-root dense `3435.870` full-batch and `2321.907` total.
  Scheduler-mask wall time is `0.338ms/step`, GPU util is `90.917%`, and CUDA
  Graph is healthy (`{"FULL":94,"NONE":2}`). `low_sync_stats_off` was not
  faster (`3861.572` full-batch), and `sync_heavy` is a bad path
  (`1960.373` full-batch, `{"NONE":128}`, `59.864%` GPU util). Treat CPU-sync
  cleanup as a guardrail rather than the remaining path to `1.2x`.
  A same-day `all_corrected_24` no-dense-fastpath graph launch check confirmed
  that the previous `data is not allocated yet` failure was vLLM compile-cache
  pollution rather than a fundamental SR24 graph blocker. The lm-eval and
  GuideLLM runners now assign an SR24-env-fingerprinted `VLLM_CACHE_ROOT`
  whenever SR24 CUDA Graph is allowed, not only under
  `--sr24-default-vllm-compile`. The check in
  `examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_no_densefastpath_graph_cachefix_gsm8k4_20260627/`
  ran with `enforce_eager=False`, captured mixed prefill/decode and FULL decode
  CUDA Graphs, and completed with status `ok`. Use this path for operator
  diagnostics; it does not change the core conclusion that exact
  `all_corrected_24` still needs a fused packed base+residual operator to be a
  speed path.
  A 2026-06-26 dense-fallback ablation with
  `SPECLINK_SR24_SYNC_MASK_STATE=1` and
  `SPECLINK_SR24_ROUTE_DENSE_FALLBACK_FRACTION=0.75` promoted the run to
  `mask_state=all_residual`. After fixing the matrix launcher so this path no
  longer forced `--enforce-eager`, CUDA Graph coverage recovered to
  `{"FULL":108,"NONE":20}`, but full-batch throughput stayed at only
  `3225.672` tok/s; a default-vLLM-compile variant reached `3301.611` tok/s
  with low GPU util and still remained below dense. Keep dense fallback as a
  correctness-conservative ablation, not the main `1.2x` speed path. Results:
  `examples/evaluate/eval-guidellm/results.bak/sr24_dense_fallback075_sync_bs64_math_k8_20260626/`,
  `examples/evaluate/eval-guidellm/results.bak/sr24_dense_fallback075_sync_graph_bs64_math_k8_20260626/`,
  and
  `examples/evaluate/eval-guidellm/results.bak/sr24_dense_fallback075_defaultcompile_bs64_math_k8_20260626/`.
  A follow-up guarded `--sr24-route-all-residual-rows` test added
  `--sr24-route-min-dense-rows`, `--sr24-route-min-base-rows`, and
  `--sr24-route-max-dense-fraction` to skip tiny split groups and fall back to
  full dense when almost all rows are residual. It was negative:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_guarded_routeall_bs64_k8_math256_20260626/report.md`
  reported dense `3415.473` full-batch tok/s versus guarded route-all
  `2880.807` (`0.843x`). Instrumented route counts in
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_guarded_routeall_instrumented_bs64_k8_math128_20260626/report.md`
  showed split-route sparse base `1.322ms/call`, dense GEMM `0.272ms/call`,
  route build `0.050ms/call`, plus `512` max-dense-fraction and `48`
  small-base full-dense fallbacks. Treat Python-level row splitting as a ruled
  out direction unless it is replaced by a true grouped/fused operator.
  The 2026-06-26 full protocol run is
  `examples/evaluate/eval-guidellm/results.bak/sr24_slowdown_breakdown_full_20260626_1331/`.
  The follow-up graph-on clean run is
  `examples/evaluate/eval-guidellm/results.bak/sr24_slowdown_breakdown_clean_graph_20260626_1338/`.
  In that clean graph run, `base_only_24` had higher accepted draft length than
  dense (`2.027` vs `1.698`), comparable GPU util (`89.167%` vs `88.429%`),
  CUDA Graph `{"FULL":126,"NONE":2}`, and full-batch `3837.702` vs dense
  `3434.714` (`1.117x`). So base-only is not acceptance-limited or globally
  GPU-idle; its remaining gap is operator/scope/capture details. A separate
  `all_corrected_24 + compressed_dense@cuda` probe at
  `examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_compressed_cuda_clean_bs64_math64_20260626_1345/`
  confirmed `residual_cpu_module_count=0`, `residual_cuda_module_count=16`, and
  `compressed_residual_runtime_on_gpu=true`, but full-batch was still only
  `3271.896` vs dense `4123.613` (`0.793x`, bs64/math/max64/K4). Treat this as
  evidence that all-corrected needs a fused/packed GPU operator, not CPU
  residual-transfer fixes. The GPU-count mask-builder ablation in
  `examples/evaluate/eval-guidellm/results.bak/sr24_speclink_gpu_count_mask_builder_bs64_math256_20260626_1350/`
  was also negative as a main speed path. A same-day K8 adaptive dense fallback
  ablation at
  `examples/evaluate/eval-guidellm/results.bak/sr24_speclink_adaptive_dense_fallback_bs64_k8_math256_20260626_1350/`
  improved the guarded mixed row only to `3272.762` full-batch tok/s versus
  same-run dense `3426.393` (`0.955x`) and still had CUDA Graph `{"NONE":192}`.
  Keep `SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK=1` as a conservative mitigation
  or ablation, not the main optimization path.
  A 2026-06-26 decision microbench at
  `examples/evaluate/eval-guidellm/results.bak/sr24_mixed_operator_decision_microbench_20260626_1136/`
  rules out the current routed-split implementation as the main speed path.
  For Llama gate/up `512x28672x4096`, residual fractions `0.0625/0.125`
  measured current mixed bucket-delta at `0.99x/1.03x` dense, while routed split
  was `1.52x/1.47x` dense and even the ideal prefix-concat upper bound was
  `1.25x/1.21x` dense. At residual fraction `1.0`, all-corrected current mixed
  was `2.17x` dense, while routed/prefix were only near dense; this is why
  `all_corrected_24` should use the dense fastpath unless a real fused
  sparse+residual kernel exists. Do not make `--sr24-route-all-residual-rows`
  the main optimization without a new kernel that avoids the small-GEMM and
  gather/scatter penalties.
  The 2026-06-26 exact-down microbench showed the best quality-conservative
  row-routed MLP point around `128` dense rows per `512` rows: `0.7490ms` vs
  dense `0.8660ms` (`1.156x`). Even the no-final-assemble lower bound was only
  `1.191x`, so use the max guard to keep row-routed MLP in measured favorable
  dense-row ranges rather than enabling it for all residual fractions. The
  2026-06-26 live same-layer gate/up+down diagnostic showed
  `SPECLINK_SR24_TRITON_ROUTE_ASSEMBLY=1` is a negative row-routed MLP
  ablation: Triton final assembly cost `1.079ms/call`, while the default
  `index_copy_` assembly cost `0.0385ms/call`. Keep Triton route assembly off
  for row-routed MLP until a new live diagnostic reverses this result.
  2026-06-25 follow-up: standalone grouped channel-pair MLP microbench reached
  up to `1.166x` dense MLP graph speed at dense fraction `0.0625`, but serving
  did not preserve the gain. bs64/math/max256 full-batch output tok/s was
  `3489.055` vs dense `3395.881` when applied only to `gate_up_proj=16-31`,
  `3472.193` vs dense `3392.952` when applied to all gate/up layers, and
  `3353.088` vs dense `3397.073` with fused activation enabled. Result roots:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_channel_split_microbench_current_20260625`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_channel_pair_baseonly_frac00625_speed_bs64_math256_20260625`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_channel_pair_baseonly_alllayers_frac00625_speed_bs64_math256_20260625`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_channel_pair_fusedact_alllayers_frac00625_speed_bs64_math256_20260625`.
  Keep channel-pair as a diagnostic/storage path unless a future fused MLP
  kernel turns the microbench speedup into an end-to-end serving gain.

The old standalone `Others/Smurfs` tree has been removed and is not used for
current runs. Do not route Smurfs benchmarks through a standalone Smurfs
checkout or a separate Smurfs conda environment. Use the `spec` environment and
the vendored `speculators/vllm` install.

Install or refresh vLLM from the repo root with editable mode. The current
machine uses PyTorch `2.11.0+cu130`, so point the vLLM build at the conda
environment's CUDA 13.0 compiler stack instead of the system CUDA:

```bash
cd .
conda run -n spec python -m pip install -r vllm/requirements/build/cuda.txt
conda run -n spec python -m pip install \
  nvidia-cuda-nvcc==13.0.88 \
  nvidia-nvvm==13.0.88 \
  nvidia-cuda-crt==13.0.88 \
  nvidia-cuda-cccl==13.0.85
conda install -n spec -y -c conda-forge gcc_linux-64=13 gxx_linux-64=13

CU13=${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cu13
LIBCUDA_SO="${LIBCUDA_SO:?set LIBCUDA_SO to the system libcuda.so.1 path}"
ln -sfn lib "${CU13}/lib64"
ln -sfn libcudart.so.13 "${CU13}/lib/libcudart.so"
ln -sfn libnvJitLink.so.13 "${CU13}/lib/libnvJitLink.so"
ln -sfn libnvrtc.so.13 "${CU13}/lib/libnvrtc.so"
ln -sfn libnvvm.so.4 "${CU13}/lib/libnvvm.so"
mkdir -p "${CU13}/lib/stubs"
ln -sfn ${LIBCUDA_SO} "${CU13}/lib/stubs/libcuda.so"

CUDA_HOME="${CU13}" \
CUDACXX="${CU13}/bin/nvcc" \
CUDAHOSTCXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++ \
CC=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc \
CXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++ \
PATH="${CU13}/bin:${PATH}" \
LD_LIBRARY_PATH="${CU13}/lib:${LD_LIBRARY_PATH:-}" \
TORCH_CUDA_ARCH_LIST=12.0 \
MAX_JOBS=8 \
NVCC_THREADS=2 \
SETUPTOOLS_SCM_PRETEND_VERSION=0.20.0 \
  conda run -n spec python -m pip install -e ./vllm --no-deps --no-build-isolation --force-reinstall
```

`pip install -e` is the modern replacement for `setup.py develop`. Python edits
under `speculators/vllm/vllm/` take effect on the next process start. C++/CUDA
extension edits still require rerunning the editable install, but CMake/Ninja
can reuse existing build artifacts if `vllm/.deps` and the build cache are kept.

Verify from a directory other than the speculators repo root, for example:

```bash
cd examples/evaluate/eval-guidellm
conda run -n spec python -c "import pathlib, vllm; print(pathlib.Path(vllm.__file__).resolve())"
conda run -n spec vllm --help
```

Expected import path prefix:

```text
./vllm/vllm/
```

Do not use `VLLM_USE_PRECOMPILED=1` for this workflow. The point of the local
source install is that future edits to `speculators/vllm` are the code being
run. Also avoid launching Python from the speculators repo root when checking
`import vllm`: the root contains a `vllm/` source directory and can shadow the
editable package as a namespace package. The motivation breakdown script changes
into `examples/evaluate/eval-guidellm` before validating the import path.

Older runs left direct-edit residue under:

```text
${CONDA_PREFIX}/lib/python3.12/site-packages/vllm/
```

That directory should not exist after the editable install. If it reappears and
`vllm.__file__` is `None` or imports resolve to `site-packages/vllm`, move the
stale directory out of the conda environment and rerun the verification above.

If the install fails after moving a partially built `vllm/.deps` directory and
CMake reports an old path in `CMakeCache.txt`, remove only generated build
subdirectories and retry:

```bash
find vllm/.deps -maxdepth 1 -type d \( -name '*-subbuild' -o -name '*-build' \) -exec rm -rf {} +
CU13=${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cu13
CUDA_HOME="${CU13}" CUDACXX="${CU13}/bin/nvcc" TORCH_CUDA_ARCH_LIST=12.0 \
MAX_JOBS=8 NVCC_THREADS=2 SETUPTOOLS_SCM_PRETEND_VERSION=0.20.0 \
  conda run -n spec python -m pip install -e ./vllm --no-deps --no-build-isolation --force-reinstall
```

Expected PEAGLE config parse:

```text
supported ['dflash', 'eagle3', 'peagle']
architectures ['Eagle3LlamaForCausalLM']
pard_token 151669
aux_layers [2, 18, 33]
spec_config {'method': 'eagle3', 'num_speculative_tokens': 4, 'parallel_drafting': True}
```

## Dataset

For the current smoke benchmark, use the repo-local `math_reasoning.jsonl` file:

```bash
cd examples/evaluate/eval-guidellm
mkdir -p data
hf download RedHatAI/speculator_benchmarks \
  --repo-type dataset \
  --include math_reasoning.jsonl \
  --local-dir data \
  --max-workers 8
```

Dataset path:

```text
examples/evaluate/eval-guidellm/data/math_reasoning.jsonl
```

The evaluation scripts also accept Hugging Face dataset syntax:

```text
RedHatAI/speculator_benchmarks:math_reasoning.jsonl
```

`scripts/run_guidellm.sh` strips the `path=` prefix emitted by this HF CLI
version before searching downloaded files.

## Structured 2:4 C4 Calibration

Structured 2:4 quality and layer-sensitivity experiments use a reusable C4
activation-RMS cache for activation-aware masking. Do not implicitly calibrate
from the evaluation datasets. The fixed prompt sample is:

```text
examples/evaluate/eval-guidellm/data/c4_calibration/c4_calibration_512_seed42.jsonl
```

Prepare or refresh the fixed C4 prompt sample with:

```bash
cd .
conda run -n spec python examples/evaluate/eval-guidellm/scripts/prepare_c4_calibration_dataset.py \
  --num-examples 512 \
  --seed 42 \
  --shuffle-buffer 10000 \
  --output examples/evaluate/eval-guidellm/data/c4_calibration/c4_calibration_512_seed42.jsonl \
  --force
```

Run model-specific calibration once, then reuse the generated RMS cache for
later experiments:

```bash
MPLCONFIGDIR=temp/matplotlib conda run -n spec python -u \
  examples/evaluate/eval-guidellm/scripts/residual_24_feasibility.py calibrate-24 \
  --models qwen3_8b,llama3_1_8b \
  --calibration-prompts examples/evaluate/eval-guidellm/data/c4_calibration/c4_calibration_512_seed42.jsonl \
  --calibration-max-seq-len 512 \
  --calibration-batch-size 1 \
  --dtype bf16 \
  --output-root examples/evaluate/eval-guidellm/data/c4_calibration/activation_rms/c4_512_seed42_bf16_max512
```

The default cache root is:

```text
examples/evaluate/eval-guidellm/data/c4_calibration/activation_rms/c4_512_seed42_bf16_max512
```

Run the current accuracy-first speculative serving comparison with token-dense
thresholds only:

```bash
cd .
MPLCONFIGDIR=temp/matplotlib conda run -n spec python -u \
  examples/evaluate/eval-guidellm/scripts/run_token_dense_accuracy.py \
  --models qwen3_8b,llama3_1_8b \
  --methods token_dense_t00,token_dense_t01,token_dense_t02,token_dense_t03,token_dense_t04,token_dense_t05,token_dense_t06,token_dense_t07,token_dense_t08,token_dense_t09,token_dense_t10 \
  --datasets gsm8k,math_reasoning \
  --gsm8k-num-examples 64 \
  --math-num-examples 64 \
  --accuracy-max-tokens 512 \
  --accuracy-concurrency 8 \
  --max-num-seqs 8 \
  --num-spec-tokens 8 \
  --output-root examples/evaluate/eval-guidellm/results/token_dense_accuracy_TIMESTAMP
```

`token_dense_tXX` is the current token-level routing experiment. It keeps the
TLM/base weights dense, attaches the same activation-aware 2:4 masks to Llama
target linears, records DLM `draft_selected_prob`, and during target
verification routes only low-confidence draft-token rows through the 2:4
masked weight. High-confidence draft rows, prefill rows, non-draft rows,
missing-score rows, and verifier bonus rows stay dense. The label threshold is
decimal shorthand: `token_dense_t07` means threshold 0.7, `token_dense_t05`
means 0.5, and `token_dense_t90` also means 0.90. This mode forces
`--enforce-eager` because the token mask changes per decoding step.

To test the low-overhead sensitive-layer preservation path, append
`_keep_first_N` to a method name. For example, the command below keeps the
first two transformer layers dense and applies the selected 2:4 method to all
remaining target-model layers:

```bash
MPLCONFIGDIR=temp/matplotlib conda run -n spec python -u \
  examples/evaluate/eval-guidellm/scripts/run_token_dense_accuracy.py \
  --methods activation_aware,activation_aware_keep_first_2,token_dense_t00,token_dense_t01,token_dense_t02,token_dense_t03,token_dense_t04,token_dense_t05,token_dense_t06,token_dense_t07,token_dense_t08,token_dense_t09,token_dense_t10 \
  --output-root examples/evaluate/eval-guidellm/results/token_dense_accuracy_TIMESTAMP \
  --resume
```

For LMeval accuracy, use the local wrapper that starts the same vLLM serving
path and then calls EleutherAI lm-evaluation-harness:

```bash
cd .
examples/evaluate/eval-guidellm/scripts/run_lm_eval_accuracy.sh \
  --mode all \
  --task smoke \
  --limit 4 \
  --output-dir examples/evaluate/eval-guidellm/results/token_dense_llama_math_gsm8k_thresholds16_20260617_145507/lm_eval
```

`--mode all` expands to `dense_ar`, `eagle3_dense`, `activation_aware`, and
`token_dense_t00` through `token_dense_t10`. For LogiQA, use
`agieval_logiqa_en` for the official multiple-choice slot and
`logiqa_generative` for the generate-until path; the older built-in `logiqa`
task depends on a dataset script that the current `datasets` package rejects.

## SpecLink SR24 Selective Residual 2:4

SR24 is a separate vLLM feature flag from both `SPECLINK_STRUCTURED_24_ENABLE`
and `SPECLINK_TOKEN_DENSE_ENABLE`. It splits selected Llama target-model Linear
weights into:

- `W_base`: 2:4 mask-selected weights kept in `module.weight`.
- `W_residual`: the complementary 2:4 values stored as a compressed value
  stream plus shared base-mask metadata.

Current SR24 performance work is slowdown-first. Before another controller or
threshold sweep, read `SR24_SLOWDOWN_BREAKDOWN.md` and report the seven required
fields: scheduler/mask build, base sparse linear, residual correction,
gather/scatter, routing statistics, CUDA Graph coverage, and GPU utilization.
The latest current-code seven-part read is
`results.bak/sr24_corrected_fast_candidate_breakdown_b12_bs64_math_k8_20260628/seven_part_report/report.md`.
For Llama-3.1-8B `math_reasoning`, EAGLE3 K=8, bs64, max new tokens 256,
the clean rows are dense `3524.307` full-batch tok/s, `base_only_24`
`4307.108` (`1.222x`), and `speclink_t08` `3902.787` (`1.107x`).
`speclink_t08` has healthy accepted draft length (`2.103` versus dense
`1.736`), high GPU util (`91.154%`), and good CUDA Graph coverage
(`{"FULL":126,"NONE":2}`); clean scheduler/mask work is only
`0.289ms/step`. The diagnostic row localizes the remaining cost to GPU-side
useful work: `gate_up_proj=16-31` sparse base is `1.071ms/call`, dense-row
correction is `0.180ms/call`, gather/scatter is only `0.014ms/call`, and draft
residual/base rows are `2928/2464`. Treat CPU-sync cleanup, fixed-bucket
mechanics, priority/direct-position routing, and gather/scatter-only changes as
ablations, not the main path, unless a fresh seven-part report reverses this.
With the current quality-safe scope, `base_only_24` shows enough sparse
headroom, but `speclink_t08` loses most of it to the two-pass mixed operator.
Reaching `1.2x` dense needs either fewer residual rows without paired accuracy
loss or a fused / packed mixed sparse-residual operator that avoids the current
two-pass cost.
Same-shape follow-ups: bucket dense-copy is only `1.026x`, Triton bucket dense
GEMM is negative (`0.962x`), adaptive dense fallback is negative (`0.952x`),
GPU-resident `compressed_dense` all-corrected is negative (`0.780x`), and direct
compressed Triton residual is rejected (`0.181x`).
The default-off `SPECLINK_SR24_ROW_ROUTED_DOWN_LINEAR=1` /
`--sr24-row-routed-down-linear` probe is also negative as a default speed path:
it is correctness-clean, but on the bs64/math/K8/max128 currentcandidate smoke
it reached `3159.199` full-batch tok/s versus the no-down-routing control
`3171.180`, with accepted draft/step `1.630` versus `1.645`. The linear
breakdown confirmed it actually ran (`row_routed_down_calls=160`) and spent
about `0.890ms/call` in down sparse base. Keep it as a diagnostic ablation,
not as the next recommended SR24 path.
The follow-up base-only scope speed-ceiling sweep is
`results.bak/sr24_baseonly_scope_sweep_bs64_math128_20260628_0203`; rerun with
`conda run -n spec python scripts/run_sr24_baseonly_scope_sweep.py` from
`examples/evaluate/eval-guidellm`. On bs64/math/max128, small/quality-related
scopes are below the `1.2x` target even before residual correction:
safe `gate_up=16-31,down=8-15` is `1.137x`, `gate_up=16-31` is `1.074x`,
`down=8-15` is `1.024x`, `down=16-31` is `0.964x`,
`gate_up=16-31,down=16-31` is `1.132x`, all `gate_up` is `1.127x`, and
tail `gate_up=31` with `up_sparse` is `0.973x`. Only all-MLP
`gate_up,down=0-31` has clear headroom (`1.627x`, accepted draft/step `2.434`).
Therefore the next `speclink_t08` optimization should either start from all-MLP
and solve quality with residual/dense protection, or implement a fused/packed
mixed sparse-residual operator; controller-only tuning on the smaller scopes
cannot reach `1.2x`.
The first all-MLP t08 serving point is
`results.bak/sr24_mlpall_t08_bs64_math128_20260628_0220`: dense full-batch
`3025.159`, all-MLP `base_only_24` `5399.310` (`1.785x`), and all-MLP
`speclink_t08` `3545.363` (`1.172x`) with accepted draft/step `2.345` and
CUDA Graph `{"FULL":55,"NONE":9}`. Treat it as the next speed candidate, not a
finished result: it still needs paired accuracy/token-level regression analysis
and another small speed gain before it satisfies the `1.2x` target.
The paired GSM8K-20 accuracy gate for that all-MLP candidate is
`results.bak/sr24_mlpall_t08_accuracy_gsm8k20_20260628_0225`: dense EAGLE3
accuracy is `0.7000`, all-MLP `base_only_24` drops to `0.1500`, and all-MLP
`speclink_t08` recovers aggregate accuracy to `0.7000` but has one paired
regression and one paired improvement. Do not treat all-MLP `speclink_t08` as
quality-safe until the regression row is traced.
The regression trace refresh with unreached suffix accounting is
`results.bak/sr24_mlpall_doc2_trace_analysis_unreached_20260628_0305`. For the
`doc_id:2` regression, `critical_prefix@0.4` has zero accepted/rejected/reached
effective-base rows but still has `0.5246` unreached effective-base suffix
rows; `critical_prefix@0.0` and `all_corrected_24` have zero base-only rows.
Future all-MLP controller work must report accepted, rejected, reached, and
unreached effective-base rows. Do not claim a selective policy is quality-safe
from accepted/rejected base-only risk alone; also check whether a selective
all-residual verify plan is control-flow equivalent to the all-corrected
no-fastpath control.
Before the next SR24/SpecLink tuning pass, produce a seven-part breakdown for
the candidate rather than inferring the cause from tok/s alone. The required
fields are scheduler/mask build, base sparse Linear time, residual correction
time, gather/scatter or bucket assembly time, routing statistics, CUDA Graph
FULL/NONE counts, and GPU utilization. The current quality-safe graph-on read
says scheduler/mask (`0.380ms/step`), gather/scatter (`0.012ms/call`), CUDA
Graph (`{"FULL":62,"NONE":2}`), and GPU util (`86.875%`) are not the primary
slow path; the remaining cost is GPU useful work, especially sparse base
(`gate_up_proj=16-31` about `1.023ms/call`) plus dense-row residual correction
(`0.148-0.171ms/call`) with many draft rows still corrected
(`14125/11395` residual/base draft rows). Treat threshold-only, CPU-sync-only,
and gather/scatter-only changes as ablations unless a fresh seven-part report
shows a different bottleneck.
The latest slowdown read is conditional by route. For the older bucket128
family, clean scheduler cost and CUDA Graph coverage were not the main
bottleneck; accepted draft length and mixed-operator useful work dominated. A
same-condition refresh showed `base_only_24` itself is not currently slow: it
gets higher accepted draft tokens/step than dense EAGLE3 with similar GPU
utilization and normal CUDA Graph coverage. For the current
`prefix_confidence@0.05 + gate_up_proj + route_all` shape, accepted length is
repaired, but the 2026-06-27 seven-part refresh exposed
`sr24_scheduler_row_index_bucket_wall_cpu_ms_per_step=42.083`; split
`_compute_residual_bucket()` and `_compute_mixed_row_indices()` before another
controller sweep. The first trace-backed `critical_prefix@0.4 + bucket512 +
reuse-base` candidate was a negative route: it lowered no useful overhead and
kept accepted draft tokens/step near `1.05`, so do not rerun it as the next
default candidate.
The fixed-shape bucket512 + Triton dense-correction control under
`results.bak/sr24_prefixconf_t005_bucket_triton_densegemm_bs64_math128_20260627`
removed the route-all dynamic row-list cost (`scheduler_mixed_row_indices` about
`0.001ms/step`, scheduler mask wall about `0.400ms/step`) while keeping CUDA
Graph coverage, but throughput fell to `2029.030` full-batch tok/s versus
same-root dense `3020.696` (`0.672x`). This means fixed bucket mechanics alone
are not the answer: after the row-list bottleneck is gone, the remaining slow
part is GPU useful work, especially sparse base plus 512-row dense correction.
The direct CPU row-list control under
`results.bak/sr24_routeall_directcpurows_bs64_math128_20260627` is also
negative: it removes GPU `nonzero` (`row_index_bucket` about `0.002ms/step`) but
turns into a draft-score GPU-to-CPU sync (`direct_cpu_rows` about
`43.200ms/step`) and reaches only `0.930x` dense. The score-free
`fixed_prefix` policy avoids that sync and is useful as a ceiling probe:
`fixed_prefix H=2 + route_reuse_base_output` reaches only `0.954x` dense, and
the looser H=0 speed ceiling reaches only `0.945x` dense. Therefore the current
next implementation should be a true mask-aware route-all/fused operator or
lower-overhead packed sparse/dense path, not another large fixed dense bucket,
not CPU score readback, and not a fixed-prefix policy-only sweep.
Relevant env switches are `SPECLINK_SR24_DIRECT_CPU_ROUTE_ROWS=1` and
`SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY=fixed_prefix`; both are experimental
ablation controls, not defaults.

At runtime every verifier row runs `W_base`; in the current `speclink_t08`
default, non-draft rows run `W_residual`, and draft-token rows use the
dependency-safe `critical_prefix` policy. That policy corrects the draft prefix
through the first low-confidence token (`probability <= threshold`) and leaves
the later suffix base-only. This avoids making an early verifier row base-only
and then trusting later rows whose KV/cache state depends on it. Set
`SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY=all_if_any_low` for a conservative
request-step ablation that corrects all draft rows whenever any draft token in
that speculative step is low-confidence or missing a score.
`SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY=low_confidence` marks only missing or
low-confidence draft rows residual and high-confidence draft rows base-only; it
now has a batched Triton mask-builder path for low-overhead speed gates, but it
is more aggressive and needs an accuracy gate. `high_confidence` remains a
row-independent diagnostic ablation. `prefix_confidence` corrects draft rows
while the cumulative product of selected-token probabilities remains above
`SPECLINK_SR24_PREFIX_THRESHOLD`; it is an accuracy probe for prefix-dependent
state, not a throughput default yet. Use `all_corrected_24` when every verifier
row should run
`W_base + W_residual` for dense-reconstruction checks.
The FFN `gate_up_proj` residual is added before SwiGLU. The current SR24
proposer path keeps selected-token probabilities as small CUDA tensors keyed by
request id and builds the verify residual mask on the target device. In
selective mode, forwards without an active verify mask, including prefill and
ordinary non-spec decode, default to residual correction when
`SPECLINK_SR24_SELECTIVE_CORRECT_NON_DRAFT=1`; otherwise base-only prefill can
pollute the KV cache even if later draft rows are corrected. Logging still
reads aggregate counts for diagnostics unless the CPU-sync ablation is enabled.

Feature flags:

```text
SPECLINK_SR24_ENABLE=1
SPECLINK_SR24_MODE=base_only|all_corrected|selective
SPECLINK_SR24_BACKEND=dense_zero|prototype|torch_sparse
SPECLINK_SR24_RESIDUAL_BACKEND=dense_rows|compressed_dense|torch_sparse
SPECLINK_SR24_RESIDUAL_DEVICE=auto|cpu|cuda
SPECLINK_SR24_THRESHOLD=0.8
SPECLINK_SR24_MASK_PATH=examples/evaluate/eval-guidellm/data/c4_calibration/sr24_masks/llama3_1_8b_activation_aware_24.pt
SPECLINK_SR24_LOG=.../speclink_sr24_events.jsonl
SPECLINK_SR24_STATS_PATH=.../speclink_sr24_stats.json
SPECLINK_SR24_STATS_INTERVAL=1
SPECLINK_SR24_REDUCE_CPU_SYNC=0
SPECLINK_SR24_SYNC_MASK_STATE=1
SPECLINK_SR24_STATIC_MASK_STATE=auto
SPECLINK_SR24_STATIC_ALL_RESIDUAL_DENSE_FASTPATH=0
SPECLINK_SR24_DIRECT_CSLT_LINEAR=0
SPECLINK_SR24_BASE_ONLY_DENSE_NONVERIFY=0
SPECLINK_SR24_ALL_CORRECTED_DENSE_FASTPATH=1
SPECLINK_SR24_SELECTIVE_CORRECT_NON_DRAFT=1
SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY=critical_prefix
SPECLINK_SR24_PREFIX_THRESHOLD=
SPECLINK_SR24_SELECTIVE_MIN_PREFIX_RESIDUAL=0
SPECLINK_SR24_SELECTIVE_MAX_RESIDUAL_DRAFT_ROWS=0
SPECLINK_SR24_TARGET_LEAFS=
SPECLINK_SR24_RESIDUAL_TARGET_LEAFS=
SPECLINK_SR24_BASE_ONLY_LAYER_IDS=
SPECLINK_SR24_BASE_ONLY_LAYER_IDS_BY_LEAF=
SPECLINK_SR24_RESIDUAL_LAYER_IDS_BY_LEAF=
SPECLINK_SR24_RESIDUAL_OUT_CHUNK=4096
SPECLINK_SR24_CACHE_COMPRESSED_RESIDUAL_WEIGHT=0
SPECLINK_SR24_COMPRESSED_RESIDUAL_TRITON=0
SPECLINK_SR24_EXTRACT_CHUNK_ROWS=128
SPECLINK_SR24_RESIDUAL_BUCKET_SIZE=0
SPECLINK_SR24_RESIDUAL_BUCKET_PRIORITY=0
SPECLINK_SR24_ROUTE_BUCKET_ROWS=0
SPECLINK_SR24_ROUTE_ALL_RESIDUAL_ROWS=0
SPECLINK_SR24_ROUTE_ALL_SKIP_BUCKET=0
SPECLINK_SR24_ROUTE_REUSE_BASE_OUTPUT=0
SPECLINK_SR24_ROUTE_DENSE_FALLBACK_FRACTION=1.1
SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK=0
SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_SMALL_ROWS=0
SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_GATE_UP_FRACTION=0.10
SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_DOWN_FRACTION=0.25
SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_SMALL_DOWN_NO_RESIDUAL=1
SPECLINK_SR24_TRITON_ROUTE_ASSEMBLY=0
SPECLINK_SR24_ROW_ROUTED_MLP=0
SPECLINK_SR24_ROW_ROUTED_MLP_MIN_DENSE_ROWS=128
SPECLINK_SR24_ROW_ROUTED_MLP_MAX_DENSE_ROWS=0
SPECLINK_SR24_ROW_ROUTED_MLP_MAX_BASE_ROWS=0
SPECLINK_SR24_FORCE_CUDAGRAPH_NONE_FOR_MIXED=1
SPECLINK_SR24_BREAKDOWN=0
SPECLINK_SR24_BREAKDOWN_LINEAR=0
SPECLINK_SR24_BREAKDOWN_EXACT_ROUTING=0
SPECLINK_SR24_BREAKDOWN_GPU_COUNTS=0
SPECLINK_SR24_BREAKDOWN_INTERVAL=2000
SPECLINK_CUDAGRAPH_STATS_PATH=.../cudagraph_stats.jsonl
SPECLINK_CUDAGRAPH_STATS_INTERVAL=1
```

`SPECLINK_CUDAGRAPH_STATS_PATH` is a generic, env-gated CUDA Graph runtime
counter used by the matrix runner when `--sr24-breakdown` is enabled. It records
`FULL`/`NONE`/`PIECEWISE` forward counts for dense and SR24 methods alike; it
does not enable SR24 by itself and should be treated as diagnostic-only.

`SPECLINK_SR24_BREAKDOWN=1` is also diagnostic-only. It now separates scheduler
mask-build time into request-count materialization, mask initialization, pending
score pop, per-request routing loop, score-policy CUDA fragments, mask writes,
mask-state synchronization, and bucket construction. With
`SPECLINK_SR24_BREAKDOWN_LINEAR=1`, it also records SR24 Linear CUDA events for
base sparse GEMM, residual dense/sparse correction, compressed residual
materialization, and gather/scatter. New SR24 Linear diagnostics also tag
events by leaf and coarse layer bucket, so
`scripts/summarize_sr24_breakdown.py` writes
`per_leaf_linear_breakdown.csv` when fresh profiling data includes
`gate_up_proj`, `gate_up_proj_layers_16_31`, `down_proj`,
`down_proj_layers_8_15`, or `down_proj_layers_16_31` rows. Do not use these
profiling runs as clean throughput numbers because CUDA events and exact
routing counters add overhead.
Both the GuideLLM matrix runner and
`examples/evaluate/eval-guidellm/scripts/run_lm_eval_accuracy.py` expose this
as `--sr24-breakdown`, `--sr24-breakdown-linear`,
`--sr24-breakdown-exact-routing`, `--sr24-breakdown-gpu-counts`, and
`--sr24-breakdown-interval`. Use `--sr24-breakdown-gpu-counts` when the goal is
to see residual/base row counts and bucket active rows with much less CPU-side
scalar synchronization than exact routing; it accumulates those counts on GPU
and reads them during breakdown snapshots.

`examples/evaluate/eval-guidellm/scripts/analyze_sr24_acceptance_trace.py`
should be used for routing-quality reads. It reports both accepted base-only
and rejected base-only fractions. Do not optimize only accepted base-only:
the first rejected draft token also uses target logits to choose the recovered
token, so a rejected base-only token can still cause accuracy drift. The script
also writes `sr24_critical_prefix_projection.csv`, which projects the runtime
`critical_prefix` policy before spending GPU time on a candidate.

Current all-corrected note: the default `all_corrected_24` dense fastpath is a
correctness/control path. With the dense fastpath disabled, current exact
all-corrected operator candidates are still slower than dense for the main
per-leaf shapes. The 2026-06-25 microbench under
`results.bak/sr24_sparse_backend_perleaf_shapes_20260625_2023/summary.md`
shows best exact graph times of `1.469ms` for `rows=900,out=28672,in=4096`
and `0.738ms` for `rows=900,out=4096,in=14336`, about `1.5x` dense graph time
in both cases. A 2026-06-27 large-row check under
`results.bak/sr24_allcorrected_large_rows_microbench_20260627/summary.md`
kept the same conclusion for rows `1024` and `2048`: the best exact graph path
was still `0.76x-0.85x` dense graph speed, and compressed cached residual was
also below dense despite all compressed tensors being CUDA-resident. Treat all
no-fastpath all-corrected variants as diagnostics until there is a fused packed
base+residual kernel. For large all-corrected diagnostics, use
`profile_speclink_sr24_sparse_backend.py --skip-triton-residual` to skip the
known-slow scalar Triton residual kernels and keep the run focused on dense,
PyTorch sparse, direct cuSPARSELt, and compressed cached paths.
The 2026-06-28 refresh tightened this rule. `all_corrected_24 +
compressed_dense@cuda + no dense fastpath` is GPU-resident (`24` CUDA residual
modules, `0` CPU modules) but only reached `0.780x` dense full-batch throughput
on bs64/math/max128; direct compressed Triton residual is rejected at `0.181x`.
Use the dense fastpath as the dense-equivalent control. The matrix runner now
automatically treats `all_corrected_24` with
`SPECLINK_SR24_ALL_CORRECTED_DENSE_FASTPATH=1` as an effective default-vLLM
compile path, so it no longer assigns an SR24 compile-cache profile unless the
real sparse/residual no-fastpath route is being tested. The validation root is
`results.bak/sr24_allcorrected_densefastpath_auto_defaultcompile_bs64_math128_20260628_0157`.

Current slowdown diagnosis rule: do a seven-part breakdown before another SR24
controller sweep. The 2026-06-27 no-fastpath all-corrected compressed-residual
bs64/math run shows the residual path is GPU-resident, not CPU-transfer bound:
`compressed_dense@cuda`, clean full-batch `2424.444` tok/s versus dense
`3021.430` (`0.802x`), GPU util `89.4%`, CUDA Graph `{"FULL":77,"NONE":2}`.
The instrumented row localizes the cost to sparse base `0.980ms/call` plus
compressed residual GEMM/add `0.557/0.135ms/call`; scheduler/mask is only about
`0.034ms/step`. Use
`scripts/run_sr24_slowdown_breakdown.py --no-sr24-all-corrected-dense-fastpath`
when the clean row must exercise the real sparse-base plus residual-correction
path instead of the dense-equivalent control path.

Do not generalize the all-corrected scheduler number to dynamic route-all
`speclink_t08`. The current route-all diagnostic root
`results.bak/sr24_user_breakdown_routeall_bs64_math128_20260627` shows
`speclink_t08` accepted length and CUDA Graph coverage are healthy, but clean
row-index/bucket wall time is large. The next optimization target is therefore
the dynamic row-index path first, then the remaining base sparse plus dense
correction Linear cost. The bucket-only skip ablation did not help; the
remaining scheduler-side wait is mixed row-list construction.

For SR24 accuracy debugging, do not treat a single
`dense_baseline`-vs-`speclink_t08` lm-eval sample difference as a proven SR24
precision regression. `dense_baseline` uses EAGLE3 speculative serving and has
shown run-to-run output variance on GSM8K reasoning prompts even with the same
prompt hash and `temperature=0`. The 2026-06-25 trace-joined check found the
older GSM8K regressions' first output divergence at `mask_pattern=11111111`
with zero reached base-only rows, and a fresh GSM8K-16 rerun had no
dense-correct to SR24-wrong regressions. Use `dense_ar`, token-level replay, or
repeated paired runs before declaring an SR24 accuracy loss.
The aggregator now reports both `AR ref` (pure target-model autoregressive
serving) and `Spec ref` (dense EAGLE3 speculative serving). Inspect both for
SR24: in the 2026-06-25 GSM8K-16 gate, `speclink_t08` was `-6.25pp` vs
`dense_ar` but `+12.50pp` vs `dense_baseline`.

`SPECLINK_SR24_RESIDUAL_DEVICE=auto` currently resolves to `cuda` for
compressed residual values. Use `cpu` only for an explicit memory fallback or
CPU-transfer ablation; it is not the performance path.

`SPECLINK_SR24_SELECTIVE_MIN_PREFIX_RESIDUAL=N` is a quality diagnostic for
selective SR24. It forces the first `N` draft-token rows in each speculative
verification step to run residual correction regardless of confidence score.
Default `0` preserves the normal policy. The Triton batched mask builder now
consumes this prefix override for the supported selective policies, including
the uniform-direct, indexed, and GPU-count paths. The GPU correctness script
checks `high_confidence + min_prefix=2` against the slow path. Use this setting
to test whether accepted base-only prefix rows are causing accuracy drift; keep
throughput checks separate because every forced-prefix row increases residual
correction work.

`SPECLINK_SR24_SELECTIVE_MAX_RESIDUAL_DRAFT_ROWS=N` is a
`low_confidence`-only speed/quality diagnostic. Default `0` leaves the policy
uncapped. When `N>0`, each request's verify step residual-corrects only the
first `N` missing or low-confidence draft rows; bonus/non-draft rows still
follow `SPECLINK_SR24_SELECTIVE_NON_DRAFT_POLICY`. The batched Triton mask
builder supports this cap. Use exact-routing runs to measure the true
residual/base draft-row split, and no-sync clean serving runs for throughput.

`SPECLINK_SR24_CACHE_COMPRESSED_RESIDUAL_WEIGHT=1` is an all-corrected /
compressed-dense diagnostic path. In the 2026-06-24 bs64 `all_corrected_24`
cache ablation, it reduced compressed residual materialization from about
`0.293 ms/call` to `0.010 ms/call` and improved total tok/s from about `558` to
`898`, with `5024/16` cached hit/miss events. The result lives at
`examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_compressed_cache2_ablation_summary_20260624/`.
It also showed that after residual materialization is removed, the remaining
cost shifts to the base sparse matmul and CUDA Graph / utilization issues; this
flag should not be treated as the final storage-efficient solution.
The 2026-06-28 code check confirmed that `compressed_dense` with
`SPECLINK_SR24_RESIDUAL_DEVICE=cuda` keeps mask bytes, compressed residual
values, and cached/prewarmed dense residual weights on GPU. The remaining
all-corrected slowdown is therefore duplicated GPU work
(`sparse base + residual GEMM/materialization`), not CPU-side residual
computation.

`SPECLINK_SR24_COMPRESSED_RESIDUAL_TRITON=1` is an all-corrected /
compressed-dense diagnostic path, not the current optimization path. It avoids
dense residual-weight materialization by launching a Triton kernel directly over
packed residual values, but the 2026-06-24 short bs64 run was slower than the
materialize+GEMM path (`84.5` vs `171.0` total tok/s, with Triton residual
about `6.47 ms/call`). A gate-up-shaped microbench
(`rows=512,out=28672,in=4096`) also showed the Triton residual variants at
about `34-55 ms`, far slower than dense residual GEMM at about `0.54 ms`.
Future all-corrected speed work should focus on CUDA Graph coverage and a real
fused sparse base+residual packed kernel, not this value-gather Triton path.

`SPECLINK_SR24_BACKEND=dense_zero` keeps the zeroed 2:4 base weight as a normal
dense-shaped tensor. It is the explicit name for the older `prototype` backend.
Use it for correctness and backend-isolation checks only: it can be fast because
it uses ordinary dense GEMM, but it stores dense-shaped zeroed weights and is
not a deployable structured-sparse speed path.

`SPECLINK_SR24_ALL_CORRECTED_DENSE_FASTPATH=1` is the default for
`all_corrected_24`. Because correcting every verifier row computes
`W_base + W_residual`, it is algebraically equivalent to the original dense
Linear. The fastpath keeps the original dense Linear instead of paying
`torch_sparse` base plus residual materialization. Disable it with
`--no-sr24-all-corrected-dense-fastpath` only for sparse/residual backend
ablation. The GuideLLM matrix and lm-eval runners do not force
`--enforce-eager` for `all_corrected_24` while this dense fastpath is enabled;
other SR24 modes still use eager conservatively. `all_corrected_24` must not
inherit selective/base-only layer filters: current code forces residual target
leafs to target leafs and ignores `SPECLINK_SR24_BASE_ONLY_LAYER_IDS*` plus
`SPECLINK_SR24_RESIDUAL_LAYER_IDS_BY_LEAF` in all-corrected mode. A 2026-06-25
GSM8K-5 eager smoke confirmed `dense_fastpath_noop=true`, `64/64`
`dense_fastpath` modules, identical dense-vs-all-corrected sample text, and no
paired regressions under
`examples/evaluate/eval-guidellm/results.bak/sr24_all_corrected_semantics_smoke_gsm8k5_20260625_1910/`.

`SPECLINK_SR24_REDUCE_CPU_SYNC=1` is a throughput ablation mode. It disables
exact hot-path residual/base-only token counts for selective rows and uses
masked full-row residual routing to reduce CPU scalar synchronization. Do not
use its draft residual fraction fields as exact metrics. With
`SPECLINK_SR24_SYNC_MASK_STATE=1` (default), SR24 still performs one
decode-step synchronization to classify the residual mask as
`all_residual`, `no_residual`, or `mixed`; this avoids the older per-Linear
`mask.any()/mask.all()` synchronizations while preserving all-residual
fastpaths. Use `SPECLINK_SR24_SYNC_MASK_STATE=0` or
`--no-sr24-sync-mask-state` for clean serving speed checks when you want to
avoid mask-state CPU synchronization. In the 2026-06-25 cap=1 breakdown,
leaving mask-state sync enabled added about `38.5ms/step` of CPU sync in the
diagnostic breakdown path; exact rows are useful for routing counts, but they
must not be used as clean throughput numbers.

2026-06-26 clean bs64/math/K=8/max256 CPU-sync refresh:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_current_lowsync_clean_bs64_math256_20260626/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_current_lowsync_no_batched_bs64_math256_20260626/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_current_syncstate_clean_bs64_math256_20260626/report.md
```

Dense EAGLE3 reached `3173.315` full-batch output tok/s. Current t08 with
`--no-sr24-sync-mask-state` and `SPECLINK_SR24_BATCHED_MASK_BUILDER=1` reached
`3101.591` full-batch tok/s (`0.977x` dense). Disabling the batched builder
dropped t08 to `2978.053` full-batch tok/s, so keep the indexed batched builder
for low-sync serving. Re-enabling mask-state sync reached `3085.741` full-batch
tok/s in this clean run, so the very large `scheduler_mask_build_cpu_ms` values
seen in breakdown mode should be read as sync-heavy diagnostic wait time rather
than as the dominant clean serving bottleneck. CPU-sync reduction remains useful
for avoiding misleading diagnostics, but the remaining t08 slowdown is mainly
the mixed sparse-base plus residual-correction operator path.

`SPECLINK_SR24_STATIC_MASK_STATE=auto|all_residual|no_residual|mixed` is an
opt-in CPU-sync ablation. `auto` preserves normal behavior. `all_residual` and
`no_residual` skip runtime mask-state reduction and force that verify state;
`mixed` skips the reduction but keeps mixed-mask routing. Pair
`SPECLINK_SR24_STATIC_MASK_STATE=all_residual` with
`SPECLINK_SR24_STATIC_ALL_RESIDUAL_DENSE_FASTPATH=1` only as a diagnostic
upper-bound: it keeps the original dense Linear weights and bypasses SR24
weight rewriting, so it is not a sparse speedup mode.

As of 2026-06-23, static `all_residual`, `no_residual`, and `mixed` are applied
before the one-step `residual_mask.sum().item()` mask-state sync. This matters
for CPU-sync ablations: static `all_residual` now really bypasses that CPU
synchronization. If `SPECLINK_SR24_STATIC_MASK_STATE=auto`, the code still
uses the sync path when `SPECLINK_SR24_SYNC_MASK_STATE=1`.

As of the later 2026-06-23 cleanup, static `all_residual` and `no_residual`
also disable SR24 proposer draft-score collection. In those modes
`build_verify_residual_mask()` does not inspect DLM confidence, so the proposer
must not spend time computing selected-token log-softmax scores just for SR24.
Expected static attach stats for `accuracy_gate_only` now include
`draft_scores_enabled=false`.

As of 2026-06-24, default-path bookkeeping is more tightly gated. When
`SPECLINK_SR24_EARLY_DENSE_TOKENS=0`, the proposer no longer appends
`_pending_generated_lens` entries and the verifier no longer pops that queue;
generated lengths are collected only when the early guard is actually enabled.
When runtime stats and SR24 breakdown are both disabled, selective routing also
skips scalar residual/base token counters that are not needed to build the
verify plan. These are small Python/lock cleanups for the default
`speclink_t08` path, not the main bottleneck. Short bs64/K=8/math smokes:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_generated_lens_gated_smoke_20260624/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bookkeeping_gated_smoke2_20260624/
```

The second run, after scalar counter gating, improved `speclink_t08` to
`2976.180` full-batch output tok/s and `90.36%` GPU util, but it still remained
below the same short-run dense EAGLE3 reference (`3186.870` full-batch tok/s,
`92.65%` GPU util). Keep focusing on sparse-base kernel utilization, CUDA Graph
coverage, and routing/mask setup.

`base_only_24` now has a stats-off scheduler fastpath: when
`SPECLINK_SR24_MODE=base_only` and runtime stats/breakdown are disabled,
`build_verify_residual_mask()` returns `state=no_residual` directly instead of
materializing per-request draft counts, allocating a residual mask, and scanning
the request loop. In the short bs64/K=8/math smoke:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_fastpath_gateup16_31_smoke_20260624/
```

`base_only_24` with `gate_up_proj=16-31` had higher acceptance than the dense
short-run reference (`24.78%`, `1.98` accepted draft tokens/step vs dense
`21.69%`, `1.73`) and higher full-batch output throughput (`3446.300` vs
`3186.870` tok/s). Its total fixed-request tok/s was still lower
(`2701.887` vs `2784.259`) because TTFT/tail behavior and lower measured GPU
util (`83.38%` vs `92.65%`) dominated the short run. So for this restricted
base-only setting, low acceptance is not the bottleneck; fixed-run slowness is
mostly utilization/tail rather than accepted length.

`SPECLINK_SR24_STATS_INTERVAL` controls how often verify summaries are flushed.
Keep it at `1` for exact windowed residual diagnostics. Larger values now write
only interval summaries instead of every verify step, reducing JSONL write
overhead for throughput ablations. The runner still marks windowed SR24
residual counters as non-exact because the latest summary can lag the
measurement boundary.

`SPECLINK_SR24_TARGET_LEAFS` accepts comma-separated Llama Linear leaf names.
Empty means all supported leaves. Useful grouped values are
`qkv_proj,o_proj` for attention-only and `gate_up_proj,down_proj` for MLP-only.

`SPECLINK_SR24_RESIDUAL_TARGET_LEAFS` is an optional subset of target leaves
that keep residual correction. Target leaves not listed there are base-only.
`SPECLINK_SR24_BASE_ONLY_LAYER_IDS` can restrict those base-only leaves to
selected transformer layers, for example `31` or `30-31`; unlisted layers stay
dense. This is the current way to test small MLP base-only additions without
allocating full-model residual tensors.
`SPECLINK_SR24_BASE_ONLY_LAYER_IDS_BY_LEAF` is the finer-grained form for these
base-only leaves, using semicolon-separated entries such as
`gate_up_proj=31;down_proj=30-31`. Listed leaves use their own layer ids;
unlisted base-only leaves are skipped when this option is non-empty.

`SPECLINK_SR24_RESIDUAL_LAYER_IDS_BY_LEAF` is the analogous finer-grained form
for dynamic token-level residual target leaves. Example:
`gate_up_proj=31;down_proj=31` makes only those MLP tail leaves use dynamic
row residual correction. When this option is non-empty, residual target leaves
not listed here can use per-module densefastpath, and unlisted layers of listed
leaves are left dense. This lets attention `qkv_proj,o_proj` remain exact
densefastpath while only selected MLP layers use token-level residual.

`SPECLINK_SR24_RESIDUAL_OUT_CHUNK=4096` materializes compressed residual
weights by output-channel chunks. This avoids the large full residual temporary
that OOMs MLP `gate_up_proj` on the 32GB RTX 5090. Set it to `0` only for the
old full-materialization ablation.

`SPECLINK_SR24_EXTRACT_CHUNK_ROWS=128` controls only model-load-time extraction
of compressed complementary residual values. Smaller values reduce transient
GPU memory and do not move runtime `compressed_dense` residual storage off GPU.

`SPECLINK_SR24_RESIDUAL_BACKEND=dense_rows` is diagnostic-only. It keeps a full
dense copy and uses dense rows for corrected tokens, so it is useful for
accuracy isolation but does not fit the full Llama-3.1-8B target model on the
32GB RTX 5090. `SPECLINK_SR24_RESIDUAL_BACKEND=torch_sparse` avoids runtime
compressed-residual materialization, but the current PyTorch conversion path
temporarily materializes a full residual dense tensor and also OOMs during
full-model loading on this GPU.

`SPECLINK_SR24_DIRECT_CSLT_LINEAR=1` bypasses `F.linear`/TorchDispatch and calls
`torch._cslt_sparse_mm` on the packed semi-structured tensor directly. Keep it
off by default. In the bs64 attention-only all-if-any-low ablation it reached
only `1535.035` steady output tok/s versus `1892.668` for the normal PyTorch
dispatch path, so it is a negative control rather than a performance path.

The throughput and lm-eval runners expose reusable SR24 presets so future runs
do not need to spell out every optimized flag:

```bash
--sr24-preset quality_safe_selective
--sr24-preset down8_15_residual_only
--sr24-preset quality_gateup_only
--sr24-preset gateup_cap0_dense_guard
--sr24-preset gateup_cap0_graph_probe
--sr24-preset lowresidual_gateup_riskcap2
--sr24-preset mlpall_lowconf_prefix5_tritonoverride
--sr24-preset speed_tradeoff_down16_base
--sr24-preset riskcap2_bucket16_directcslt
--sr24-preset fixedprefix4_bucket16_directcslt
--sr24-preset criticalprefix4_bucket16_directcslt
--sr24-preset accuracy_first
--sr24-preset accuracy_gate_only
--sr24-preset accuracy_down_only
--sr24-preset throughput_aggressive
```

`lowresidual_gateup_riskcap2` is the current measured gate-up-only speed probe:
`gate_up_proj=16-31`, `low_confidence@0.8`, prefix2, two risk-capped draft
rows, `bonus` non-draft correction, `dense_rows@cuda`, direct cuSPARSELt,
dynamic mixed CUDA Graph, and bucket8 by default. It is intentionally separate
from `riskcap2_bucket16_directcslt`, which also corrects `down_proj=8-15` and
was slower in the 2026-06-29 bucket follow-up. In the latest bs64/math/max256
gate-up-only sweep, bucket16/bucket8/bucket4 reached `1.068x`/`1.071x`/`1.054x`
full-batch speedup over dense. Read
`SR24_CURRENT_SLOWDOWN_BREAKDOWN_20260629.md` and the relative result roots
`examples/evaluate/eval-guidellm/results.bak/sr24_t08_lowresidual_gateup_riskcap2_clean_bs64_math256_20260629/`,
`examples/evaluate/eval-guidellm/results.bak/sr24_t08_lowresidual_gateup_riskcap2_bucket8_clean_bs64_math256_20260629/`,
and
`examples/evaluate/eval-guidellm/results.bak/sr24_t08_lowresidual_gateup_riskcap2_bucket4_clean_bs64_math256_20260629/`
before spending more time on bucket-size tuning.

`mlpall_lowconf_prefix5_tritonoverride` is the current all-MLP speed-target
probe: `gate_up_proj,down_proj` across all layers, `low_confidence@0.6`,
prefix5 residual correction, `bonus` non-draft correction, `dense_rows@cuda`,
bucket32, dynamic mixed CUDA Graph, and Triton bucket override. It is the only
measured SR24 route that has reached the requested `1.2x` dense full-batch
throughput target so far: in
`examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_lowconf_prefix5_nostats_bs64_math128_20260629/`
it reported dense/SR24 full-batch `3036.970/3650.240` tok/s (`1.202x`), with
accepted draft tokens/step `2.3339`, average GPU util `84.444%`, and CUDA Graph
`{"FULL":49}`. A current 3-repeat reproduction using this preset is under
`examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_lowconf_prefix5_preset_repro3_bs64_math128_20260629/`
and reports median dense/SR24 full-batch `3038.443/3648.217` tok/s (`1.201x`);
fixed-request total tok/s is still lower for SR24 (`2272.489/1925.203`), so
this is only a full-batch decode-window win. It is not yet a default
quality-safe path: paired GSM8K-50 under
`examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_lowconf_prefix5_paired_gsm8k50_20260629/`
reported dense/SR24 accuracy `0.7400/0.7200`, while the dense-repeat stability
gate under
`examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_lowconf_prefix5_stability_gate_20260629/`
reported zero stable regressions versus dense repeats. A current GSM8K-20
preset sanity gate under
`examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_lowconf_prefix5_preset_gsm8k20_current_20260629/`
reported dense/SR24 `0.6000/0.7000` with paired regressions/improvements `1/3`;
this is only a smoke quality check. Use this preset for focused speed/quality
debugging of the `1.2x` candidate, not as a proven safe baseline.

`quality_gateup_only` is the current safer paired-quality reference:
`gate_up_proj=16-31`, `all_if_any_low`, threshold `0.4`, prefix-4 residual,
`dense_rows@cuda`, no down-proj base-only tail, and CUDA-Graph-capable
low-sync serving flags. GSM8K-50 paired checks under
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_quality_gateup_only_preset_paired_gsm8k50_20260626/report.md`
reported dense `0.7200`, SR24 `0.7200`, `2` paired regressions, and `2`
improvements. Its bs64/math/max256 clean-serving speed check under
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_quality_gateup_only_clean_serving_bs64_math256_20260626/report.md`
was negative: dense/SR24 steady `2998.708/2775.333` tok/s and full-batch
`3034.666/2847.465` tok/s. It is a quality reference, not the final speed path.
A focused route-level dense fallback ablation under
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_quality_gateup_densefallback08_bs64_math256_20260626/report.md`
improved the same safe route to SR24 `2865.348` steady / `2926.322`
full-batch tok/s against same-root dense `2983.984` / `3031.380`
(`0.960x` steady, `0.965x` full-batch). Treat
`SPECLINK_SR24_ROUTE_DENSE_FALLBACK_FRACTION=0.8` as a useful ablation guard,
not a final speed solution; it mostly collapses high-residual mixed steps back
toward dense.

`gateup_cap0_dense_guard` is the 2026-06-27 guarded selective candidate:
`gate_up_proj=16-31`, `low_confidence@0.8`, cap0
(`SPECLINK_SR24_SELECTIVE_MAX_RESIDUAL_DRAFT_ROWS=0`), bucket32,
`dense_rows@cuda`, and adaptive dense fallback at gate/up residual fraction
`0.05`. An earlier GSM8K-50 smoke looked paired-safe under
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cap0_adaptive_dense005_quality_gsm8k50_20260627_0330/report.md`,
but the latest serving checks show it is not generally paired-safe. GSM8K-20
reruns after the bucket fallback guard still had dense `0.7000`, SR24 `0.7000`,
`Pair reg=2`, and `Pair imp=2` under
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bucket_guard_fix_quality_gsm8k20_20260627/report.md`.
`all_if_any_low@0.8` also forced `sr24_residual_draft_fraction=1.0` but still
had `Pair reg=2`, `Pair imp=2` under
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allifany08_quality_gsm8k20_20260627/report.md`.
The only current paired-safe small serving control is `all_corrected_24` with
the all-residual dense fastpath under
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_quality_gsm8k20_20260627/report.md`
(`Pair reg=0`, `Pair imp=0`). Treat selective presets as candidates that need a
fresh paired gate, not as precision-safe baselines.

`gateup_cap0_graph_probe` keeps the same config and additionally enables
dynamic-auto CUDA Graph with a stable bucket buffer
(`SPECLINK_SR24_CUDAGRAPH_BUCKET=1`,
`SPECLINK_SR24_FORCE_CUDAGRAPH_NONE_FOR_MIXED=0`, and
`SPECLINK_SR24_DYNAMIC_AUTO_CUDAGRAPH=1`). It is a graph precision probe, not a
quality-safe path: the GSM8K-10 smoke at
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_cap0_graph_guard_gsm8k10_20260627/report.md`
had dense `0.8000`, SR24 `0.7000`, and `Pair reg=1` even though the SR24
summary showed all draft rows residual-corrected. That summary field is a
routing-mask count, not proof that every requested row was actually corrected:
the bucketed `dense_rows` path caps actual correction at
`SPECLINK_SR24_RESIDUAL_BUCKET_SIZE` when scheduled rows exceed the bucket.
Newer summaries include `sr24_bucket_active_rows`,
`sr24_bucket_residual_requested_rows`, `sr24_bucket_active_rows_per_call`, and
`sr24_bucket_active_fraction_of_requested`; inspect those before drawing quality
conclusions from `SR24 draft residual`. Dynamic graph remains diagnostic until
the bucketed path either corrects every requested row or falls back to the
all-residual dense path.

`speed_tradeoff_down16_base` is retained only as a controlled negative/upper
bound; GSM8K-50 paired accuracy collapsed to `0.0200`, so do not use it as a
main solution. `down8_15_residual_only` is a down-only ablation without a
base-only tail. `quality_safe_selective` is an older broad gate/up+down
residual preset.

`accuracy_first` now maps to the safer `qkv_proj,o_proj` exact densefastpath
plus a `gate_up_proj=31` MLP tail with `--sr24-gate-up-split up_sparse`, so the
gate half remains dense and only the up half is converted to 2:4. Use
`accuracy_gate_only` for the older fully fused `gate_up_proj=31` sparse-tail
ablation. `accuracy_down_only` keeps only `down_proj=31` as a
negative/diagnostic ablation, because the down tail caused long repetitive
GSM8K outputs and large paired regressions in the 2026-06-26 hook-fix quality
gate. `throughput_aggressive` uses `gate_up_proj=31;down_proj=30-31`.
The older `accuracy_*` presets use residual `torch_sparse` and static
all-residual densefastpath. Newer selective presets use `dense_rows@cuda`.
Use `--sr24-preset manual` for historical/full-target ablations.

`SPECLINK_SR24_BASE_ONLY_DENSE_NONVERIFY=1` is a base-only diagnostic ablation.
It keeps a dense copy and uses it for non-speculative/non-verify forwards while
keeping speculative verifier rows on the sparse base. Keep it off by default:
full Llama-3.1-8B with all target Linear modules OOMs on the 32GB RTX 5090, and
attention-only bs64 fixed-request testing was slower than the normal base-only
path.

Generate or refresh the activation-aware mask cache after C4 activation RMS
calibration exists:

```bash
MPLCONFIGDIR=temp/matplotlib conda run -n spec python -u \
  examples/evaluate/eval-guidellm/scripts/generate_speclink_sr24_mask.py \
  --model-label llama3_1_8b \
  --model-path ../models/llama-3.1-8b-instruct \
  --calibration-cache-root examples/evaluate/eval-guidellm/data/c4_calibration/activation_rms/c4_512_seed42_bf16_max512 \
  --output-path examples/evaluate/eval-guidellm/data/c4_calibration/sr24_masks/llama3_1_8b_activation_aware_24.pt \
  --dtype bf16
```

Local semantic smoke, no GPU required:

```bash
conda run -n spec python \
  examples/evaluate/eval-guidellm/scripts/check_speclink_sr24_correctness.py
```

Profiler probe for the real PyTorch 2:4 sparse backend:

```bash
conda run --no-capture-output -n spec python -u \
  examples/evaluate/eval-guidellm/scripts/profile_speclink_sr24_sparse_backend.py \
  --warmup 3 \
  --repeats 10 \
  --output-root examples/evaluate/eval-guidellm/results.bak/speclink_sr24_sparse_backend_probe_TIMESTAMP
```

The current RTX 5090 probe confirms `SparseSemiStructuredTensorCUSPARSELT`
execution with profiler events such as `aten::_cslt_sparse_mm` and cuSPARSELt,
but the PyTorch sparse path is slower than dense for small verifier row counts.
This is sparse-kernel evidence only, not an end-to-end speedup claim.

Current row-count probe:

```text
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_sparse_backend_probe_rows_20260622_1555/
```

It tested rows `{64,128,256,512}` for representative Llama shapes. On RTX 5090,
PyTorch semi-structured base sparse took about `0.83ms` per Linear call; dense
BF16 for the same shapes was `0.019-0.132ms`. Thus base sparse was still
`6.3x-44.4x` slower and all-corrected sparse was `12.6x-88.1x` slower. This
means the current PyTorch backend is useful for correctness and kernel
existence checks, but not for the TODO's final throughput goal. A real
performance path needs a lower-overhead cuSPARSELt/CUTLASS integration or a
custom persistent/packed kernel instead of `F.linear` on PyTorch
`SparseSemiStructuredTensorCUSPARSELT`.

Focused dense-zero/sparse microbenchmark on 2026-06-23:

```text
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_sparse_backend_probe_densezero_20260623/
```

For rows `64` and Llama Linear shapes `(4096,4096)` and `(6144,4096)`,
dense-zero base GEMM was `1.0x` dense (`0.0187ms` vs `0.0191ms`, and
`0.0257ms` vs `0.0253ms`). PyTorch semi-structured sparse base was
`35.3-42.8x` slower than dense, and the current Triton residual prototype was
about `70x` dense. This confirms that the serving slowdown is the PyTorch
sparse base runtime, not zeroing/masking overhead.

Direct `aten::_cslt_sparse_mm` alg/split_k sweep:

```text
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_cslt_alg_sweep_20260622_1610/
```

The sweep called `_cslt_sparse_mm` directly over rows `{64,128,256,512}` and
Llama Linear shapes `(4096,4096)` and `(6144,4096)`. The best valid settings
were still `split_k=1` with `alg_id=0` or `1`, and the best direct sparse
latency stayed around `0.808-0.819ms`, only `0.92-0.97x` of PyTorch
`F.linear` sparse and still `6.2-41.9x` slower than dense BF16 for these row
counts. Do not add a separate direct-cslt backend unless a lower-overhead call
path or fused kernel is available; changing alg_id/split_k alone does not solve
the SR24 throughput bottleneck.

Direct-cslt and CUDA Graph replay probe on 2026-06-23:

```text
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_sparse_backend_probe_graph_directcslt_20260623/
```

For rows `64`, direct `_cslt_sparse_mm` without graph was still about
`0.81-0.91ms`, matching `F.linear` sparse. CUDA Graph replay of the same sparse
call dropped to about `0.03-0.05ms`, so the eager sparse path is dominated by
CPU/launch/descriptor overhead, not the actual sparse GEMM kernel. The
throughput and lm-eval runners have `--sr24-allow-cudagraph` and
`--vllm-compilation-config` for this ablation. If `--sr24-allow-cudagraph` is
used with `base_only_24`, `SPECLINK_SR24_BACKEND=torch_sparse`, and no explicit
`--vllm-compilation-config`, the runners automatically use
`{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":...}`
with a capture size large enough for `batch_size*(K+1)`. This matters for
bs64/K=8 because the verifier has up to `64*(8+1)=576` scheduled tokens, which
exceeds vLLM's default 512 capture ceiling.

Serving-level CPU-sync ablations on 2026-06-23:

```text
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_cpu_sync_attn_baseline_bs64_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_cpu_sync_attn_reduced_bs64_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_t08_attn_residual_sparse_staticmask_cg_smoke_bs64_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_maskstate_dense_baseline_bs64_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_maskstate_exact_attn_allifanylow_bs64_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_maskstate_sync_attn_allifanylow_bs64_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_maskstate_nosync_attn_allifanylow_bs64_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_directcslt_maskstate_sync_attn_allifanylow_bs64_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_baseonly_standard_bs64_fixed512_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_baseonly_dense_nonverify_bs64_fixed512_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_baseonly_attn_standard_bs64_fixed512_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_baseonly_attn_dense_nonverify_bs64_fixed512_20260623/
```

These used Llama-3.1-8B, EAGLE3 K=8, math_reasoning, bs64,
`max_tokens=64`, attention-only SR24 (`qkv_proj,o_proj`), and 20s measurement
windows. The exact-stats compressed residual baseline reached `839.332`
steady output tok/s. `--sr24-reduce-cpu-sync --sr24-stats-interval 32`
reached `876.525` steady output tok/s, about `1.04x`. This confirms CPU-side
scalar/stat synchronization is a real but small overhead. The static-mask
CUDA Graph ablation with `--sr24-residual-backend torch_sparse` captured some
full graph steps (`{"FULL": 85, "NONE": 139}`) but reached only `802.724`
steady output tok/s, slower than compressed residual eager. The current
`compressed_dense` residual path is deliberately kept eager for `speclink_t08`
because it rebuilds dense residual weights with advanced indexing during the
forward path, which CUDA Graph capture rejects.

A later fixed-64 request, `max_tokens=512` attention-only
`torch_sparse/dense_rows/all_if_any_low` ablation removed the remaining
per-Linear mask-state synchronizations by carrying an SR24 verify plan through
the forward context. Under the same math_reasoning bs64 setting, dense EAGLE3
baseline reached `2427.273` steady output tok/s. `speclink_t08` exact stats
reached `1253.379`; `--sr24-reduce-cpu-sync` with the default one-step
`--sr24-sync-mask-state` reached `1892.668` (`0.78x` dense); and
`--no-sr24-sync-mask-state` reached only `807.213` because it lost the
all-residual dense fastpath and had to run mixed-mask routing. Conclusion:
reducing CPU sync is a useful ablation and should stay enabled for throughput
tests, but it still does not make the current SR24 path faster than dense.
The direct packed `_cslt_sparse_mm` Python path is not useful: it reached
`1535.035` tok/s in the same setting, so the remaining issue is not simply the
TorchDispatch wrapper around `F.linear`.

Follow-up CPU-sync/compile-mode diagnostics for the same fixed-request
`max_tokens=512` setting:

| case | steady output tok/s | ratio to dense | note |
|---|---:|---:|---|
| dense EAGLE3 baseline, default vLLM compile | `2427.273` | `1.000x` | `VLLM_COMPILE`, normal vLLM graphs |
| auto mask-state sync, `FULL_DECODE_ONLY` mode none | `2121.059` | `0.874x` | `{"FULL":441,"NONE":7}` |
| static `all_residual`, `FULL_DECODE_ONLY` mode none | `2188.522` | `0.902x` | removes the one verify-step mask-state sync |
| static `all_residual`, gpu util `0.90`, log interval fix | `2203.041` | `0.908x` | KV cache grows to `37,040` tokens; event log drops from `455` to `14` lines but TPS is unchanged |
| static `all_residual` densefastpath, mode none | `2202.442` | `0.907x` | storage returns to `1.0x`, but compile mode still limits throughput |
| static `all_residual` densefastpath, default vLLM compile | `2387.788` | `0.984x` | confirms the remaining gap is mostly compile/graph mode, not CPU mask sync |

Conclusion: CPU-side mask synchronization is worth keeping as an ablation, but
the larger serving gap comes from forcing SR24 sparse paths into conservative
`mode=NONE/FULL_DECODE_ONLY` graph handling and from the PyTorch sparse backend.
Only the diagnostic densefastpath can use default vLLM compile and recover
dense-like speed; actual sparse selective SR24 still needs a lower-overhead
sparse/residual kernel path.

Base-only analysis under the same bs64 fixed-64 request, `max_tokens=512`
setting:

| case | steady output tok/s | ratio to dense | acceptance | note |
|---|---:|---:|---:|---|
| dense EAGLE3 baseline | `2427.273` | `1.00x` | `0.287` | normal dense target |
| full `base_only_24` | `1116.685` | `0.46x` | `0.544` | acceptance is higher, so low acceptance is not the bottleneck |
| attention-only `base_only_24` | `1355.730` | `0.56x` | `0.312` | sparse limited to `qkv_proj,o_proj` |
| attention-only dense-nonverify | `1097.712` | `0.45x` | `0.308` | full-model variant OOMs; attention-only is slower |

All base-only variants were forced eager, and SR24 logs reported only `NONE`
cudagraph modes. The result reinforces that the bottleneck is PyTorch
semi-structured sparse/cuSPARSELt eager overhead and graph coverage, not
accepted draft length.

Residual-backend follow-ups on 2026-06-23:

```text
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_residual_triton_probe_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_residual_tiled_probe_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_attn_t08_torchsparse_denserows_fastpath_bs64_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_t00_gpu_compressed_accuracy16_20260623/
```

The existing Triton residual24 prototype avoids materializing a dense residual
weight, but the original output-element kernel is not tiled like a real GEMM.
On rows `64`, out/in `4096`, it measured `1.1910ms` residual-only versus
`0.5261ms` for GPU compressed-dense materialization plus `F.linear`. A tiled
Triton prototype improved residual-only latency to `0.7705ms` for rows `64`,
but it was still slower than compressed-dense materialization and scaled poorly
(`5.2381ms` residual-only at rows `512`). Do not wire either Triton prototype
into serving. A
small runtime optimization now lets `torch_sparse/dense_rows` skip the sparse
base when every row in a Linear call needs dense correction. In the loadable
attention-only t08 smoke, that path reached `857.638` steady tok/s, not better
than compressed residual with reduced CPU sync (`876.525`), because most t08
steps are mixed rows rather than all-corrected rows.

For accuracy, a 16-sample GSM8K paired smoke with full target SR24,
`speclink_t08`, threshold `0.0`, `compressed_dense@cuda`, and all draft and
non-draft rows corrected reported dense `0.5625` and SR24 `0.6250`. Treat the
positive delta as sample noise; the important result is that full correction is
dense-level on this smoke. It also confirms the performance problem: the same
run generated only single-digit tokens/s while correcting every row.

All-corrected dense-fastpath serving check:

```text
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_allcorrected_densefastpath_vs_dense_bs64_20260623/
```

On Llama-3.1-8B, EAGLE3 K=8, math_reasoning, bs64, `max_tokens=64`, dense
baseline reached `2252.228` steady output tok/s and `all_corrected_24` with
the dense fastpath reached `2214.379` steady output tok/s (`0.98x`). This
confirms the all-corrected optimization restores dense-equivalent serving
speed, but it also confirms there is no 2:4 speedup in the all-corrected
semantics.

Focused SR24 component breakdown on 2026-06-24:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_clean_graph_compare_summary_20260624/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_backend_compare_component_20260624/
```

In the clean bs64/K=8/math_reasoning graph-on check with `max_tokens=128`,
dense EAGLE3 reached `2189.383` total output tok/s with graph counts
`{"FULL":36,"NONE":44,"PIECEWISE":1}`. `base_only_24` reached `1824.706`
total output tok/s with graph counts `{"FULL":28,"NONE":42,"PIECEWISE":1}`.
Its accepted draft tokens per step were higher than dense (`1.498` vs
`1.421`), so this slowdown is not caused by a lower accepted length. The likely
causes are lower average GPU utilization (`64.3%` vs `80.4%`) and less useful
CUDA Graph coverage for the sparse path in this short-output setting. In the
older `max_tokens=256` graph-on run, `base_only_24` was faster than dense, so
base-only should be interpreted as graph/kernel-utilization sensitive rather
than acceptance-limited.

For real sparse/residual `all_corrected_24` with the dense fastpath disabled,
GPU-resident `compressed_dense` residual storage is still slow because the
runtime rebuilds residual-weight slices on every forward. The timed bs64/K=8
component run measured `1065.526` total output tok/s. Its SR24 Linear timing
was: base sparse `729.930ms` total, compressed residual materialization
`2643.840ms`, compressed residual GEMM `1012.111ms`, and residual mask/add/copy
`362.592ms`. Switching the residual backend to prebuilt `torch_sparse` removed
the materialization cost and improved throughput to `1521.574` total output
tok/s, but it still ran two sparse GEMMs per corrected Linear call
(`0.822ms/call` base plus `0.910ms/call` residual) and remained slower than
dense. The optimization implication is that compressed residual should not be
optimized by more Python-side materialization; the useful next path is a fused
GPU kernel or graph-safe prebuilt residual representation that avoids both
per-forward materialization and two independent sparse launches.

Follow-up graph-safe `all_corrected_24` residual-sparse result:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_residual_sparse_graph_compare_20260624/
```

With `SPECLINK_SR24_RESIDUAL_BACKEND=torch_sparse`,
`--no-sr24-all-corrected-dense-fastpath`, and `--sr24-allow-cudagraph`, the
runner now automatically adds a `FULL_DECODE_ONLY` compilation config for
`all_corrected_24`. The explicit graph run reached `1855.218` total output
tok/s with graph counts `{"FULL":76,"NONE":2}`; the auto-config validation run
reached `1794.735` total output tok/s with graph counts
`{"FULL":81,"NONE":2}` and its `command.json` shows
`--compilation-config {"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":1024}`.
This is a real improvement over eager residual sparse (`1521.574`) and
compressed residual (`1065.526`), but still below dense EAGLE3 in the paired
bs64/K=8 short-output checks. If `all_corrected_24` is run with
`compressed_dense` residual and the dense fastpath disabled, keep it eager:
that path materializes residual slices during forward and is not treated as
graph-safe.

Selective `speclink_t08` routed-bucket ablation on 2026-06-24:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_t08_route_bucket_compare_20260624/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_t08_route_rows_max256_compare_20260624/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_t08_residual_sparse_graph_compare_20260624/
```

For bs64/K=8/math_reasoning with `max_tokens=128`, `speclink_t08` using
`dense_rows@cuda`, `critical_prefix+extra3`, non-draft `bonus`, and bucket size
32 reached `1474.339` total output tok/s. Adding
`--sr24-route-bucket-rows` reached `1510.786` (`+2.5%`) by avoiding sparse base
work on rows that are routed to dense correction. Adding
`--sr24-triton-route-assembly` did not help (`1502.204`). In the more relevant
`max_tokens=256` comparison, routed rows reached `2142.195` total output tok/s,
below the paired non-routed `speclink_t08` row from
`sr24_breakdown_current_lowsync_graphon_bs64_64req_20260624` (`2194.262`).
Treat route-bucket rows as a small/noisy ablation, not a default. The stable
bottlenecks remain scheduler mask construction and sparse/residual launch
overhead.

The same selective setup with `SPECLINK_SR24_RESIDUAL_BACKEND=torch_sparse`
can use the graph-safe path from `speclink_t08_allows_cudagraph()`. On the
short `max_tokens=128` check it improved over dense-rows selective
(`1789.836` vs `1474.339` total output tok/s) and reached graph counts
`{"FULL":73,"NONE":2}`. On the more relevant `max_tokens=256` check it reached
only `1857.506` total output tok/s with graph counts `{"FULL":191,"NONE":2}`,
below the paired dense-rows selective run (`2194.262`). Treat residual-sparse
selective as a graph-safe/low-memory variant, not the default speed path. Its
high graph coverage confirms that the remaining selective bottlenecks are the
per-step scheduler mask construction and the cost of sparse/residual work, not
only CUDA Graph eligibility.

Granular scheduler breakdown smoke after adding the finer timers:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_breakdown_granular_smoke_summary_20260624/
```

This was a tiny diagnostic run only (`bs=8`, 8 requests, `max_tokens=32`), not a
throughput claim. It verified that the new fields are written. The measured
`speclink_t08` scheduler mask build was `3.547ms/step`; request routing loop was
`3.131ms/call`, score-policy CUDA fragments `0.125ms/call`, mask writes
`0.017ms/call`, bucket build/topk `0.245/0.238ms/call`, and base sparse Linear
was `0.973ms/call`. The breakdown supports the current bottleneck hypothesis:
Python/request-level routing plus many small sparse/residual kernels dominate
before any end-to-end speedup is possible.

Granular bs64/K=8/max256 breakdown follow-up:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_granular_breakdown_main_bs64_k8_max256_summary_20260624/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_granular_allcorrected_backend_summary_20260624/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_granular_batchedmask_compare_20260624/
```

The main diagnostic run used `bs=64`, `K=8`, `max_tokens=256`,
`math_reasoning`, and 64 fixed requests. In that diagnostic setting,
`base_only_24` reached `2488.240` total output tok/s versus dense EAGLE3
`2292.725`, with higher accepted draft tokens/step (`2.024` vs `1.676`). Thus
base-only slowness is not caused by accepted length. `speclink_t08` reached only
`1884.714` total output tok/s; scheduler mask build was `11.150ms/step`, of
which the per-request routing loop was `10.876ms/call`. Linear timing still
showed base sparse `0.949ms/call`, residual dense GEMM `0.167ms/call`, and
gather/scatter `0.011ms/call`. The active bucket was nearly full
(`31.771/32`, fill `0.993`), so the dense-row correction bucket is not wasting
many slots.

For exact `all_corrected_24` with dense fastpath disabled, `compressed_dense` is
confirmed to run with GPU-resident residual values (`residual_device=cuda`), but
it is slow because it materializes residual weights during forward: total output
tok/s was `1069.383`, compressed residual materialization was `0.292ms/call`,
compressed GEMM `0.113ms/call`, and the base sparse GEMM `0.556ms/call`.
Prebuilt `torch_sparse` residual avoided materialization and improved to
`1876.446` total output tok/s with CUDA Graph `{"FULL":76,"NONE":2}`, but it
still paid base sparse `0.960ms/call` plus residual sparse `1.037ms/call`.
Therefore all-corrected cannot become a speed path without a fused CUDA/Triton
or lower-overhead sparse/residual kernel; computing base and residual as two
PyTorch sparse launches is structurally too expensive.

The newer clean bs64/K=8/max256 `speclink_t08` batched-mask check showed a small
end-to-end gain, not a full solution:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_clean_t08_default_bs64_k8_max256_20260624/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_clean_t08_batchedmask_bs64_k8_max256_20260624/
```

With the same graph coverage (`{"FULL":192,"NONE":2}`), the default mask builder
reached `2041.273` total output tok/s and batched mask builder reached
`2102.471` (`+3.0%`). Keep `--sr24-batched-mask-builder` opt-in: it reduces some
request-level routing overhead, but the remaining gap to dense EAGLE3 and the
1.2x target is dominated by sparse/residual operator cost.

The attempted route-row cache was removed after testing because it did not help:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_clean_t08_route_cached_rows_bs64_k8_max256_20260624/`
reached only `2033.059` total output tok/s with the same graph coverage
`{"FULL":192,"NONE":2}`. Moving dense/base route-row construction from every
Linear call to the scheduler did not recover the speed gap, so the useful route
work remains a lower-level fused operator rather than Python/Tensor list caching.

Conservative selective-policy serving checks:

```text
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_attn_allifanylow_denserows_bs64_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_attn_allifanylow_denserows_allresidual_fastpath_bs64_20260623/
```

For attention-only SR24 (`qkv_proj,o_proj`), `torch_sparse/dense_rows`, and
`all_if_any_low` at threshold `0.8`, residual draft fraction rose to about
`0.997` and steady throughput reached `1633.217` tok/s. After adding an
all-residual mask elision fastpath, the same setup reached `1640.123` tok/s.
This is much faster than critical-prefix dense_rows (`857.638` tok/s) because
it avoids most mixed sparse/dense row work, but it is still only about `0.73x`
of the paired dense baseline. In this conservative regime the method is mostly
dense correction plus eager-mode overhead, not a sparse speedup.

Larger-row follow-up:

```text
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_sparse_backend_probe_rows_large_20260622_1630/
```

Rows `{768,1024,1536,2048}` were tested for the same Llama shapes. PyTorch
2:4 base sparse was still slower than dense: `2.6-6.6x` for `(4096,4096)` and
`1.9-4.5x` for `(6144,4096)`. This rules out the simple explanation that only
very small verifier row counts caused the slowdown for the current bs=64/128
range.

Full SR24 correctness smoke through vLLM/EAGLE3, using only GSM8K first few
samples. This uses `--sr24-backend dense_zero` or the legacy alias
`--sr24-backend prototype`, so it validates routing and accuracy semantics but
must not be reported as 2:4 Sparse Tensor Core performance:

```bash
examples/evaluate/eval-guidellm/scripts/run_lm_eval_accuracy.sh \
  --mode sr24 \
  --task gsm8k_cot \
  --models llama3_1_8b \
  --use-task-manifests \
  --manifest-size 4 \
  --max-new-tokens 512 \
  --num-spec-tokens 8 \
  --batch-size 1 \
  --num-concurrent 1 \
  --max-num-seqs 1 \
  --sr24-backend dense_zero \
  --sr24-mask-path examples/evaluate/eval-guidellm/data/c4_calibration/sr24_masks/llama3_1_8b_activation_aware_24.pt \
  --output-dir examples/evaluate/eval-guidellm/results.bak/speclink_sr24_smoke_TIMESTAMP
```

Real base-sparse vLLM smoke. This runs only `base_only_24`, because the default
residual backend is a compressed-dense correction path. For all-corrected or
selective correctness with `torch_sparse`, `--sr24-residual-device auto` keeps
compressed residual values on GPU. If a memory-constrained smoke OOMs, rerun
with explicit `--sr24-residual-device cpu`; using
`--sr24-residual-backend torch_sparse` can still OOM on 32GB:

```bash
examples/evaluate/eval-guidellm/scripts/run_lm_eval_accuracy.sh \
  --mode base_only_24 \
  --task gsm8k_cot \
  --models llama3_1_8b \
  --use-task-manifests \
  --manifest-size 1 \
  --max-new-tokens 16 \
  --num-spec-tokens 8 \
  --batch-size 1 \
  --num-concurrent 1 \
  --max-num-seqs 1 \
  --sr24-backend torch_sparse \
  --sr24-residual-backend compressed_dense \
  --sr24-residual-device auto \
  --sr24-mask-path examples/evaluate/eval-guidellm/data/c4_calibration/sr24_masks/llama3_1_8b_activation_aware_24.pt \
  --output-dir examples/evaluate/eval-guidellm/results.bak/speclink_sr24_torch_sparse_baseonly_smoke_TIMESTAMP
```

Latest selective torch-sparse vLLM smoke after moving SR24 score deque popping
out of the GPU mask-construction critical section and fixing `selective` so
only high-confidence draft rows use residual correction:

```text
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_selective_draft_only_smoke_20260622_1715/
```

It ran `speclink_t08` on one GSM8K sample with `NUM_SPEC_TOKENS=8`,
`SPECLINK_SR24_BACKEND=torch_sparse`,
`SPECLINK_SR24_RESIDUAL_BACKEND=compressed_dense`, and
`SPECLINK_SR24_RESIDUAL_DEVICE=auto` under the older CPU-resident auto default.
The run completed successfully with `SR24 residual=0.1042` over draft tokens,
`missing_score_tokens=0`, no missing cached mask modules, and backend reported
as `torch_sparse/compressed_dense@cpu`. Current `auto` resolves to CUDA, so use
the newer GPU-resident follow-up results below for performance discussion.

Closed-loop throughput runner support:

```bash
conda run --no-capture-output -n spec python \
  examples/evaluate/eval-guidellm/scripts/run_llama31_vllm_fastdraft_smurfs_eagle3_matrix.py \
  --methods dense_baseline,base_only_24,all_corrected_24,speclink_t08 \
  --datasets gsm8k \
  --batch-sizes 8,16,32,64 \
  --repeats 3 \
  --prompt-limit 128 \
  --min-requests-per-run 128 \
  --max-tokens 512 \
  --eagle3-k 8 \
  --sr24-backend torch_sparse \
  --sr24-residual-backend compressed_dense \
  --sr24-residual-device auto \
  --sr24-mask-path examples/evaluate/eval-guidellm/data/c4_calibration/sr24_masks/llama3_1_8b_activation_aware_24.pt \
  --final-root examples/evaluate/eval-guidellm/results/speclink_sr24_TIMESTAMP \
  --work-root examples/evaluate/eval-guidellm/results.bak/speclink_sr24_work_TIMESTAMP
```

The runner now treats `dense_baseline` as vLLM+EAGLE3 with SR24 disabled and
the three SR24 methods as vLLM+EAGLE3 plus `SPECLINK_SR24_ENABLE=1`. SR24
servers normally add `--enforce-eager`; the explicit exceptions are the
SR24 CUDA Graph ablations, all-corrected dense fastpath, and dense-zero
isolation paths. `summary.csv` keeps one row per repeat, while
`median_summary.csv` and `report.md` report medians over successful repeats.
The outputs include steady-state output tokens/s plus SR24 residual draft
fraction, residual/base-only draft token counts, missing score count, backend,
storage-over-dense, and `peak_gpu_memory_mib` sampled with `nvidia-smi` during
each streaming case. The same output root also writes `iteration_log.csv` and
`iteration_log.md`; accuracy columns are left blank by this throughput runner
and should be filled or cross-referenced from the lm-eval accuracy run when
producing the final TODO report.
Use `--prompt-limit 128` for the TODO's first-128 GSM8K scope. The streaming
runner's request `dataset_index` is a global request counter, but prompt
selection uses `dataset_index % prompt_limit`, so the prompt pool is limited to
the first `N` JSONL rows.

Latest SR24 throughput-runner smoke:

```text
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_throughput_repeats_smoke_final_20260622_1810/
```

It ran `speclink_t08` only on GSM8K, bs=1, `--repeats 2`, `max_tokens=8`,
K=8, and wrote raw repeat rows plus `median_summary.csv`. This was a runner
smoke, not a final throughput number.

Current bs=8 end-to-end pilot evidence:

```text
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_endtoend_pilot_final_20260622_1825/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_all_corrected_pilot_final_20260622_1835/
```

These are pilot runs, not the final 128-sample x 3-repeat matrix. On GSM8K,
bs=8, K=8, short output windows:

- `dense_baseline`: steady-state `859.587` output tok/s (`max_tokens=64`).
- `base_only_24`: steady-state `153.983` output tok/s (`max_tokens=64`),
  `0.179x` of dense.
- `speclink_t08`: steady-state `37.185` output tok/s (`max_tokens=64`),
  `0.043x` of dense, residual draft fraction `0.222884`.
- `all_corrected_24`: steady-state `4.272` output tok/s (`max_tokens=16`),
  residual draft fraction `1.0`.

Together with the microbenchmarks above, this shows the original
PyTorch-sparse/CPU-resident compressed-residual implementation is a correctness
and instrumentation path, not a usable SR24 speedup path. The next performance
iteration should avoid PyTorch `SparseSemiStructuredTensorCUSPARSELT` for
small verifier row counts and avoid per-Linear residual materialization.

Formal SR24 TODO matrix completed on 2026-06-22:

```text
examples/evaluate/eval-guidellm/results/speclink_sr24_status_20260622/
examples/evaluate/eval-guidellm/results/speclink_sr24_status_20260622/formal_matrix_summary.csv
examples/evaluate/eval-guidellm/results/speclink_sr24_full_bs8_budget8192_20260622/
examples/evaluate/eval-guidellm/results/speclink_sr24_full_bs16_32_64_budget8192_20260622/
examples/evaluate/eval-guidellm/results/speclink_sr24_accuracy_20260622/
```

The completed run used the adjusted 32GB-safe budget
`--max-num-batched-tokens 8192 --gpu-memory-utilization 0.75`. At bs64,
steady-state output tok/s was: dense `3698.121`, base-only `1990.384`,
all-corrected `301.398`, and `speclink_t08` `436.278`. GSM8K first-128
accuracy at bs64 was: dense `0.796875`, base-only `0.03125`, all-corrected
`0.78125`, and `speclink_t08` `0.09375`.

CPU sync reduction ablation:

```text
examples/evaluate/eval-guidellm/results/speclink_sr24_reduce_cpu_sync_bs64_ablation_v2_20260622/
examples/evaluate/eval-guidellm/results/speclink_sr24_status_20260622/cpu_sync_ablation_summary.csv
```

With `--sr24-reduce-cpu-sync`, bs64 one-repeat steady tok/s was
`294.806` for all-corrected and `440.222` for `speclink_t08`, versus formal
baseline medians `301.398` and `436.278`. This does not show a meaningful
CPU-sync speedup; the dominant bottleneck remains the sparse/residual backend
and CPU-resident residual materialization.

Follow-up after switching the default selective policy to low-confidence
residual plus non-draft correction:

```text
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_lowsync_ablation_nosync_final_20260622/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_lowsync_ablation_reduce_final_20260622/
```

For `math_reasoning`, bs64, `max_tokens=64`, one repeat, exact-stats
`speclink_t08` reached `166.779` steady output tok/s and
`--sr24-reduce-cpu-sync` reached `173.472` steady output tok/s. This is about a
`1.04x` improvement, so CPU scalar synchronization is a useful ablation but not
the main bottleneck. The reduce-sync mode keeps total scheduled/draft/non-draft
counters accumulated, but draft residual/base-only counts are intentionally not
exact.

GPU-resident compressed residual follow-up on 2026-06-23:

```text
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_cuda_residual_bs64_after_chunkcheck_final_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_cuda_residual_reduce_sync_bs64_final_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_baseonly_prototype_bs64_retry_final_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_dense_matched_prototype_budget_bs64_final_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_allcorrected_dense_fastpath_bs64_final_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_allcorrected_dense_fastpath_accuracy16_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_dense_rows_t00_tiny_startup_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_residual_sparse_t00_tiny_startup_20260623/
```

The startup OOM from checking 2:4 masks with a full `keep.sum(dim=-1)` tensor
was fixed by chunking the check in `vllm/speclink_sr24.py`. Cached mask validity
is also checked on CPU before the final keep mask is moved to GPU, avoiding a
large load-time GPU temporary. After that, `--sr24-residual-device cuda`
succeeded for Llama-3.1-8B SR24 bs64 on the 32GB RTX 5090.

For `math_reasoning`, bs64, K=8, `max_tokens=64`, one repeat,
`torch_sparse/compressed_dense@cuda` produced:

| mode | steady tok/s | total tok/s | note |
|---|---:|---:|---|
| `dense_baseline` | 2272.017 | 2215.218 | same GPU-resident residual run config |
| `all_corrected_24` | 336.186 | 248.626 | exact stats, residual draft fraction `1.0` |
| `speclink_t08` | 333.073 | 246.821 | exact stats, residual draft fraction `0.809521` |
| `speclink_t08 --sr24-reduce-cpu-sync` | 337.768 | 261.039 | `stats_exact=false` |
| `all_corrected_24` dense fastpath | 2124.433 | 2055.314 | `torch_sparse/dense_fastpath@none`, storage/dense `1.0` |

Thus GPU-resident residual roughly fixes the severe CPU-transfer path, but it
is still only about `0.15x` of dense in this short-output bs64 setting. The
additional CPU-sync reduction is only about `1.014x` over the exact-stats
GPU-resident run, so keep it as an ablation rather than the main optimization
direction.

The prototype base-only isolation with the same short-output setting and a
matched dense run under `--max-num-batched-tokens 4096 --gpu-memory-utilization
0.92` produced dense `2250.397` steady tok/s and prototype base-only
`2891.561` steady tok/s. This proves the sparse backend overhead, not
speculative acceptance, is the main reason `torch_sparse` base-only is slow.
The prototype backend is dense-shaped and stores `1.5625x` dense weight bytes,
so it is not a deployable compression path.

After adding the explicit `dense_zero` backend name and removing forced eager
for `base_only_24` on dense-zero/prototype, a repeat wrote:

```text
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_densezero_baseonly_bs64_20260623/
```

Dense baseline reached `2231.086` steady tok/s and `base_only_24` with
`--sr24-backend dense_zero` reached `2997.845` steady tok/s. This is still an
isolation result rather than a valid final method, because the zeroed base
weight remains dense-shaped and base-only accuracy is not preserved.

Dense-zero selective follow-up on 2026-06-23:

```text
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_densezero_t00_accuracy16_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_densezero_t02_bs64_throughput_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_densezero_denserows_t02_bs64_throughput_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_densezero_denserows_attn_t02_bs64_throughput_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_densezero_denserows_attn_t08_bs64_throughput_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_densezero_denserows_rowsplit_attn_t08_bs64_throughput_20260623/
```

Key results:

- `speclink_t08 --sr24-backend dense_zero --sr24-threshold 0.0` matched the
  dense 16-sample GSM8K smoke (`0.6875` vs `0.6875`) because every draft row
  and every non-draft row was corrected.
- `dense_zero/compressed_dense`, all target leafs, threshold `0.2`, reached
  only `403.345` steady tok/s at bs64 because compressed residual
  materialization still dominates.
- `dense_zero/dense_rows`, all target leafs, threshold `0.2`, OOMed on 32GB:
  target SR24 storage was about `28.79GB` before loading the EAGLE3 drafter.
- `dense_zero/dense_rows`, attention-only (`qkv_proj,o_proj`), threshold `0.2`,
  reached `1707.915` steady tok/s (`0.77x` dense) with draft residual fraction
  `0.5784`.
- The same attention-only setup with threshold `0.8` reached `1784.192` steady
  tok/s (`0.80x` dense) with draft residual fraction `0.1947`.
- Disabling non-draft correction at threshold `0.8` reached only `1816.036`
  steady tok/s (`0.82x` dense) and is not accuracy-safe.
- A row-split prehook for `dense_zero/dense_rows` avoids computing base and
  dense correction for the same row, reducing peak memory, but attention-only
  threshold `0.8` still measured `1783.740` steady tok/s. Splitting into small
  row-indexed GEMMs does not provide the needed speedup.

Conclusion: dense-zero proves the current PyTorch sparse base is the problem,
but dense-zero selective correction is also not the final path. A correct
speedup needs the base-only rows to use a genuinely faster packed 2:4 kernel;
ordinary dense GEMM on zeroed weights cannot beat dense once corrected rows are
handled accurately.

The all-corrected dense fastpath is the current correct optimization for
`all_corrected_24`: it removes the sparse+residual double compute when every
row would be corrected anyway. A 16-sample GSM8K smoke wrote:

```text
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_allcorrected_dense_fastpath_accuracy16_20260623/
```

It reported dense `0.7500` and all-corrected fastpath `0.6875` exact-match on
only 16 samples. Treat this as a serving smoke, not a final accuracy claim; the
module-level SR24 correctness check still verifies dense reconstruction, and
larger paired accuracy should be rerun before making a final claim.

Selective accuracy diagnostics after the verifier-row mapping fix showed that
the draft token at position `i` is verified by row `start + i`; the bonus row is
`start + n`. The previous shifted mapping could place the last draft score on
the bonus row. With the corrected mapping, `critical_prefix` at threshold 0.8
was still too aggressive on GSM8K because it corrected only about 21.6% of
draft rows in the 32-sample smoke. A threshold-0.0 run corrected every draft row
but still differed from dense on the compressed residual path, which points to
the PyTorch sparse/residual accumulation path rather than the row selector
alone. Full dense-row correction is not currently loadable on the 32GB GPU.

Conservative selective residual ablations on 16 GSM8K samples wrote:

```text
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_all_if_any_low_accuracy16_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_attention_only_all_if_any_low_accuracy16_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_mlp_only_chunk4096_all_if_any_low_accuracy16_20260623/
```

The paired dense smoke was `0.7500`. Full SR24 with `all_if_any_low` corrected
all draft and non-draft rows but reached `0.5625`. Attention-only SR24
(`qkv_proj,o_proj`) reached `0.6250`; MLP-only SR24
(`gate_up_proj,down_proj`) with `SPECLINK_SR24_RESIDUAL_OUT_CHUNK=4096`
reached `0.6875`. This means the selective row policy alone is not the
remaining accuracy issue: both attention and MLP sparse/residual numeric paths
can move answers. The MLP-only chunked run also verified that chunked residual
materialization avoids the previous full-temporary OOM.

Follow-up on 2026-06-23 found that selective mode was still using base-only
weights for no-verify-mask forwards such as prefill. After fixing that path to
correct all rows when `SPECLINK_SR24_SELECTIVE_CORRECT_NON_DRAFT=1`, small
half-dtype GSM8K smokes wrote:

```text
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_half_critical_prefix_t02_nomaskfix_accuracy16_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_half_critical_prefix_t00_nomaskfix_accuracy16_20260623/
```

The paired dense half smoke was `0.6875`. `critical_prefix` threshold `0.2`
reached `0.6250` with draft residual fraction `0.8462`; threshold `0.0`
corrected every draft row and reached `0.7500`. This confirms the accuracy
failure was largely KV pollution from base-only prefill/non-verify rows plus
some remaining risk from accepted draft rows left base-only. Future speedups
must save residual work only on rows predicted not to enter the live KV state.

Post-fix bs64 `math_reasoning` throughput smokes with K=8, `max_tokens=64`,
and half dtype wrote:

```text
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_postfix_t02_bs64_throughput_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_postfix_t02_mlp_only_bs64_throughput_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_postfix_t02_fullrowfast_bs64_throughput_20260623/
```

Dense baseline reached `2227.610` steady output tok/s. `all_corrected_24`
dense fastpath reached `2118.252` (`0.951x`). `speclink_t08` with all target
leafs, threshold `0.2`, reached only `320.558` (`0.144x`) with draft residual
fraction `0.5949`. MLP-only target leafs improved to `430.290` (`0.193x`) but
remained far below dense. A compressed-residual full-row fastpath that removes
`index_select/index_add_` from full-row correction measured `319.858`, so
small indexing cleanup is not the bottleneck. The next speed path needs a
lower-overhead GPU sparse/residual kernel or a different live-row selection
strategy, not more CPU bookkeeping reductions.

Additional 2026-06-23 bs64 `math_reasoning` ablations, K=8, `max_tokens=64`,
half dtype:

```text
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_noeager_allcorrected_bs64_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_sync_exact_t08_bs64_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_reduce_cpu_sync_t08_bs64_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_stats_interval32_t08_bs64_20260623/
```

| case | steady output tok/s | note |
|---|---:|---|
| dense baseline | 2224.122 | same short-output bs64 setting |
| `all_corrected_24` dense fastpath without forced eager | 2185.261 | `0.982x` dense, better than the previous `0.951x` forced-eager run |
| `speclink_t08`, exact stats, interval 1 | 319.880 | current selective baseline |
| `speclink_t08`, `--sr24-reduce-cpu-sync`, interval 32 | 321.314 | only `1.004x` over exact stats; `stats_exact=false` |
| `speclink_t08`, exact stats, interval 32 | 320.335 | logging interval alone has no meaningful effect |

Conclusion: removing unnecessary eager helps `all_corrected_24`, but CPU-side
sync/logging reduction does not explain `speclink_t08` slowness. The remaining
bottleneck is still the PyTorch sparse/residual verifier path.

Follow-up base-only cudagraph ablations:

```text
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_baseonly_torchsparse_eager_bs64_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_baseonly_torchsparse_cudagraph_bs64_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_baseonly_torchsparse_cudagraph_fix_bs64_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_baseonly_torchsparse_cg_nocompile_bs64_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_baseonly_torchsparse_cg1024_modes_bs64_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_baseonly_fixed64_cg1024_bs64_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_baseonly_fixed64_eager_bs64_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_dense_fixed64_bs64_20260623/
examples/evaluate/eval-guidellm/results.bak/speclink_sr24_baseonly_torchsparse_cg1024_piecewise_bs64_20260623/
```

Default eager `base_only_24 torch_sparse` reached `1335.654` steady tok/s
against dense `2565.165`. Allowing cudagraph with default vLLM compile first
failed because deleting the Linear `weight` parameter breaks AOT assumptions;
SR24 now replaces the `_parameters["weight"]` entry with the sparse tensor
instead of deleting it. The next default-compile attempt reached
`_cslt_sparse_mm` but failed inside inductor/cuSPARSELt. With
`--vllm-compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY"}'`
the run succeeded but still measured `1335.299` steady tok/s, consistent with
falling back to eager because bs64/K=8 verifier batches can exceed the default
512 capture size. The runners now auto-generate a 1024-or-larger capture config
for this specific `base_only_24 torch_sparse --sr24-allow-cudagraph` ablation.
With explicit `max_cudagraph_capture_size=1024`, the continuous bs64/K=8 run
reached `1544.326` steady tok/s and reported `{"FULL": 150, "NONE": 426}`.
This is only about `1.16x` over eager because the streaming client continually
replenishes requests; many target forwards are mixed prefill/decode or otherwise
miss the `FULL_DECODE_ONLY` graph.

Fixed-request ablation, bs64 with exactly 64 requests and `max_tokens=512`,
isolates decode graph replay:

| case | steady output tok/s | cudagraph modes | note |
|---|---:|---|---|
| dense baseline | 2455.092 | normal vLLM graphs | exact model, EAGLE3 K=8 |
| `base_only_24 torch_sparse`, eager | 1088.190 | `{"NONE": 192}` | sparse eager/cuSPARSELt launch path |
| `base_only_24 torch_sparse`, `FULL_DECODE_ONLY`, capture 1024 | 4244.941 | `{"FULL": 190, "NONE": 2}` | proves CPU/launch/descriptor overhead is large when decode graph coverage is high |

`base_only_24` is not an accuracy-valid method, so the `4244.941` tok/s number
is only a backend/CPU-overhead diagnostic. A no-compile
`FULL_AND_PIECEWISE` attempt for the continuous workload was downgraded by vLLM
to `cudagraph_mode=NONE` and reached only `1346.286` steady tok/s. Mixed
prefill/decode graph replay would require the vLLM compile path, which currently
fails for the PyTorch sparse tensor path in inductor/cuSPARSELt.

First full correctness matrix requested in `TODO.md`:

```bash
for bs in 8 16 32 64; do
  examples/evaluate/eval-guidellm/scripts/run_lm_eval_accuracy.sh \
    --mode sr24 \
    --task gsm8k_cot \
    --models llama3_1_8b \
    --use-task-manifests \
    --manifest-size 128 \
    --max-new-tokens 512 \
    --num-spec-tokens 8 \
    --batch-size 1 \
    --num-concurrent "${bs}" \
    --max-num-seqs "${bs}" \
    --sr24-backend prototype \
    --sr24-mask-path examples/evaluate/eval-guidellm/data/c4_calibration/sr24_masks/llama3_1_8b_activation_aware_24.pt \
    --resume \
    --output-dir examples/evaluate/eval-guidellm/results/speclink_sr24_TIMESTAMP
done
```

`--mode sr24` expands to `dense_baseline`, `base_only_24`,
`all_corrected_24`, and `speclink_t08`. `dense_baseline` is original
vLLM+EAGLE3 with all SpecLink sparse/token routing flags disabled. The
aggregator writes SR24 fields into `summary.csv` and `report.md`, including
residual draft fraction, missing score count, attached module count, actual
weight storage bytes, sparse metadata bytes, mask metadata bytes, residual
backend, and storage-over-dense. Each result root should also keep
`iteration_log.csv` and `iteration_log.md`; append one row per mechanism
change, including the exact command, accuracy, TPS if available, residual
ratio, peak VRAM, profile note, and whether the change was kept or reverted.

For the current long-output lm-eval TODO, do not use the older LogiQA/MMLU/
HotpotQA task set. On a single RTX 5090, run only `qwen3_8b` and
`llama3_1_8b`; do not run 14B/32B/70B on this machine. The formal task set is:

```text
gsm8k_cot,minerva_math500,gpqa_diamond_cot_zeroshot,ifeval,humaneval_instruct,longbench_multi_news
```

Use fixed lm-eval sample manifests for reproducibility:

```bash
examples/evaluate/eval-guidellm/scripts/run_lm_eval_accuracy.sh \
  --mode all \
  --task gsm8k_cot,minerva_math500,gpqa_diamond_cot_zeroshot,ifeval,humaneval_instruct,longbench_multi_news \
  --models qwen3_8b,llama3_1_8b \
  --use-task-manifests \
  --manifest-size 200 \
  --max-new-tokens 512 \
  --num-spec-tokens 8 \
  --batch-size 1 \
  --num-concurrent 1 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.94 \
  --allow-unsafe-code \
  --humaneval-sandbox auto \
  --resume \
  --output-dir examples/evaluate/eval-guidellm/results/token_dense_lm_eval_long_output_5090x1_TIMESTAMP
```

The wrapper writes manifests to
`examples/evaluate/eval-guidellm/configs/task_manifests/`, uses repo-local
Hugging Face/evaluate caches under `examples/evaluate/eval-guidellm/temp/`, and
sets `SPECLINK_TOKEN_DENSE_STATS_INTERVAL=1` for token-dense modes so routing
fractions appear in `token_dense_stats.jsonl`. `run_meta.json` records
`started_at`, `ended_at`, and `elapsed_seconds` for each case. GPQA is a gated
Hugging Face dataset; the wrapper preflights
`Idavidrein/gpqa:gpqa_diamond` before launching vLLM and records a failed
`run_meta.json` if the available token lacks access. HumanEval must run with
`--allow-unsafe-code`; on this machine `bwrap` is available only with
non-sandboxed execution permissions, so GPU/lm-eval runs must be launched with
real execution permissions rather than Codex's default sandbox.

Smoke and sanity outputs for this path should go under `results.bak/`, for
example:

```bash
MPLCONFIGDIR=temp/matplotlib conda run -n spec python -u \
  examples/evaluate/eval-guidellm/scripts/run_token_dense_accuracy.py --smoke
```

`quality` and `layer-sensitivity` load this cache by default when they need
activation-aware RMS. If the cache is missing, generate it with `calibrate-24`
instead of falling back to evaluation prompts. A standard C4-calibrated layer
sensitivity run is:

```bash
MPLCONFIGDIR=temp/matplotlib conda run -n spec python -u \
  examples/evaluate/eval-guidellm/scripts/residual_24_feasibility.py layer-sensitivity \
  --models qwen3_8b,llama3_1_8b \
  --datasets mtbench,dolly,gsm8k,math_reasoning \
  --mtbench-num-examples 40 \
  --dolly-num-examples 64 \
  --gsm8k-num-examples 64 \
  --math-num-examples 64 \
  --generation-batch-size 4 \
  --dtype bf16 \
  --output-root examples/evaluate/eval-guidellm/results/structured_24_layer_sensitivity_qwen_llama_c4calib_TIMESTAMP
```

The command above is an offline Transformers quality check. For current
speculative-inference experiments, use the vLLM/EAGLE3 runner below instead.
It applies 2:4 only to the TLM/base large model at vLLM model-load time; the
EAGLE3 drafter/speculator remains dense. Accuracy is generated through vLLM
speculative decoding, while PPL is dense-vs-sparse TLM reference loss using the
same mask policy.

```bash
cd .
MPLCONFIGDIR=temp/matplotlib conda run -n spec python -u \
  examples/evaluate/eval-guidellm/scripts/run_structured_24_spec_quality.py \
  --models qwen3_8b,llama3_1_8b \
  --datasets mtbench,dolly,gsm8k,math_reasoning \
  --num-spec-tokens 8 \
  --mtbench-num-examples 40 \
  --dolly-num-examples 64 \
  --gsm8k-num-examples 64 \
  --math-num-examples 64 \
  --accuracy-concurrency 8 \
  --max-num-seqs 8 \
  --output-root examples/evaluate/eval-guidellm/results/structured_24_spec_tlm_eagle3_k8_TIMESTAMP
```

Useful variants:

```bash
# Smoke and sanity artifacts go under results.bak/.
MPLCONFIGDIR=temp/matplotlib conda run -n spec python -u \
  examples/evaluate/eval-guidellm/scripts/run_structured_24_spec_quality.py --smoke

# Resume an interrupted full run.
MPLCONFIGDIR=temp/matplotlib conda run -n spec python -u \
  examples/evaluate/eval-guidellm/scripts/run_structured_24_spec_quality.py \
  --output-root examples/evaluate/eval-guidellm/results/structured_24_spec_tlm_eagle3_k8_TIMESTAMP \
  --resume

# Re-run only the all-sparse and first/last dense-keep comparison.
MPLCONFIGDIR=temp/matplotlib conda run -n spec python -u \
  examples/evaluate/eval-guidellm/scripts/run_structured_24_spec_quality.py \
  --skip-layer-sensitivity \
  --output-root examples/evaluate/eval-guidellm/results/structured_24_spec_tlm_eagle3_k8_dense_keep_TIMESTAMP
```

The vLLM hook is disabled by default and is controlled by:

```text
SPECLINK_STRUCTURED_24_ENABLE=1
SPECLINK_STRUCTURED_24_MODEL_LABEL=qwen3_8b|llama3_1_8b
SPECLINK_STRUCTURED_24_CALIBRATION_CACHE_ROOT=examples/evaluate/eval-guidellm/data/c4_calibration/activation_rms/c4_512_seed42_bf16_max512
SPECLINK_STRUCTURED_24_POLICY=dense|single_layer|all_sparse|keep_first|keep_last|keep_first_last
SPECLINK_STRUCTURED_24_LAYER_INDEX=0
SPECLINK_STRUCTURED_24_KEEP_N=1
SPECLINK_STRUCTURED_24_STATS_PATH=temp/vllm_structured_24_stats.json
SPECLINK_TOKEN_DENSE_ENABLE=1
SPECLINK_TOKEN_DENSE_MODE=high_confidence_dense
SPECLINK_TOKEN_DENSE_THRESHOLD=0.7
SPECLINK_TOKEN_DENSE_STATS_PATH=temp/token_dense_stats.jsonl
```

Final files from the speculative runner:

- `structured_24_spec_quality.csv`: all dense-vs-sparse PPL and ACC rows.
- `layer_sensitivity.csv`: one-sparse-layer rows.
- `dense_keep_compare.csv`: all-sparse, keep-first, keep-last, and
  keep-first-last rows.
- `figures/layer_sensitivity_spec.png` and `figures/dense_keep_spec.png`.
- `runs/*/*/vllm_structured_24_stats.json`: vLLM-side proof that only TLM
  modules were masked.

## Running Benchmarks

Run commands from:

```bash
cd examples/evaluate/eval-guidellm
```

By default, experiment outputs go under:

```text
examples/evaluate/eval-guidellm/results/
```

`run_evaluation.sh` defaults to `results/eval_results_TIMESTAMP/`. Override the
root with either:

```bash
RESULTS_DIR=results/custom ./run_evaluation.sh -c ./configs/qwen3-8b-peagle.env
```

or:

```bash
./run_evaluation.sh -c ./configs/qwen3-8b-peagle.env --results-dir results/custom
```

EAGLE3 smoke:

```bash
GUIDELLM_RATE=1 REQUEST_TYPE=chat_completions conda run -n spec bash ./run_evaluation.sh \
  -c ./configs/qwen3-8b-eagle3.env \
  -o ./results/out_eagle3_smoke_localds_math \
  --port 8010
```

P-EAGLE smoke:

```bash
GUIDELLM_RATE=1 REQUEST_TYPE=chat_completions conda run -n spec bash ./run_evaluation.sh \
  -c ./configs/qwen3-8b-peagle.env \
  -o ./results/out_peagle_smoke_localds_math \
  --port 8011
```

Qwen3-8B EAGLE3 vs P-EAGLE comparison:

```bash
GUIDELLM_RATE=1 REQUEST_TYPE=chat_completions conda run -n spec bash ./run_qwen3_8b_eagle3_vs_peagle.sh
```

The comparison script writes to:

```text
results/qwen3_8b_eagle3_vs_peagle_TIMESTAMP/eagle3/
results/qwen3_8b_eagle3_vs_peagle_TIMESTAMP/peagle/
```

The config defaults are:

- EAGLE3 uses `NUM_SPEC_TOKENS=3`
- P-EAGLE uses `NUM_SPEC_TOKENS=4`
- P-EAGLE sets `PARALLEL_DRAFTING=true`
- Both use `BASE_MODEL=Qwen/Qwen3-8B` by default
- Both use `DATASET=data/math_reasoning.jsonl` by default

Expected output files in each output directory:

- `vllm_server.log`
- `guidellm_output.log`
- `guidellm_results.json`
- `acceptance_analysis.txt`

## Llama FastDraft/Smurfs Matrix

For the Llama-3.1-8B comparison of AR, FastDraft, Smurfs dynamic K, and EAGLE3,
use:

```bash
cd examples/evaluate/eval-guidellm
conda run -n spec python ./scripts/run_llama31_vllm_fastdraft_smurfs_eagle3_matrix.py
```

The matrix script compares:

- `vllm_ar`
- `vllm_fastdraft`
- `smurfs_fastdraft`
- `vllm_eagle3`

Defaults:

- datasets: `math_reasoning,mtbench,gsm8k,humaneval`
- client-side batch/concurrency: `8,16,32,64`
- `max_tokens=2048`
- FastDraft K=4
- EAGLE3 K=4
- Smurfs initial K=4, dynamic max K=12 below batch size 32 and 8 otherwise

Final outputs go to:

```text
examples/evaluate/eval-guidellm/results_final/llama31_vllm_fastdraft_smurfs_eagle3_matrix_2048_full_TIMESTAMP/
```

Intermediate server logs and per-run work files go to:

```text
examples/evaluate/eval-guidellm/temp/llama31_vllm_fastdraft_smurfs_eagle3_matrix_2048_full_TIMESTAMP/
```

For Smurfs runs, the server writes the live dynamic-K event log to each run
directory as `smurfs_dynamic_k.jsonl` via `SPECLINK_SMURFS_DYNAMIC_OUT`. The
WebUI Smurfs mode in `vllm_spec_webui.py` reads that log through
`--smurfs-k-log` and displays the current K.

## Motivation Breakdown

`motivation_breakdown.sh` runs the synthetic 1000-token prompt / 1000-token
output experiment requested for EAGLE3 and P-EAGLE:

- `batch_size=1 2 4 8 16`
- `NUM_SPEC_TOKENS=8 16 24`
- default `REQUESTS_PER_RUN=32`
- default `WARMUP_REQUESTS=4`
- default `MAX_NUM_BATCHED_TOKENS=8192`
- vLLM `--max-num-seqs` is set to the current batch/concurrency size
- default `SPECLINK_BREAKDOWN_VERIFY_DETAIL=1`, which also passes
  `--enforce-eager` to vLLM for this experiment

Run it from `examples/evaluate/eval-guidellm`:

```bash
cd examples/evaluate/eval-guidellm
conda run -n spec bash ./motivation_breakdown.sh
```

The script no longer accepts `--vllm-dir`, `--skip-vllm-setup`, or
`--setup-only`; vLLM is expected to already be installed editable from
`speculators/vllm`. At startup it verifies that `import vllm` resolves under:

```text
./vllm/vllm/
```

Outputs are written under `results/motivation_breakdown_TIMESTAMP/`, including:

- `status.tsv`
- per-run `vllm_server.log`, `guidellm_output.log`,
  `guidellm_results.json`, and `breakdown_events.jsonl`
- `concise_summary.csv`: the preferred compact result table. It keeps only
  model, batch size, `NUM_SPEC_TOKENS`, decode-stage verify/draft/other
  percentages, generated tokens per decode iteration, and end-to-end mean
  latency.
- `verify_detail_summary.csv`: Qwen3 verifier-only QKV projection, Attention,
  FFN, and verifier-other percentages and per-iteration times. Attention here
  includes q/k norm, RoPE, the attention kernel, and `o_proj`.
- `summary.csv`
- `raw_events.csv`
- `acceptance.csv`
- `motivation_breakdown.xlsx`, with `concise_summary` as the first sheet
- `motivation_breakdown.svg`
- `motivation_verify_breakdown.svg`

For P-EAGLE with `NUM_SPEC_TOKENS=16` or `24`, keep the scheduler budget large
enough. The default `MAX_NUM_BATCHED_TOKENS=8192` is intentional; vLLM otherwise
can fail during startup with `max_num_scheduled_tokens is set to ...`, because
parallel drafting reserves additional draft-token slots. The script also checks
that GuideLLM wrote a result JSON and that vLLM wrote breakdown events before it
marks a run as `ok`.

Set `SPECLINK_BREAKDOWN_VERIFY_DETAIL=0` to disable Qwen3 verify-detail
instrumentation. The detail mode uses CUDA events inside Qwen3 verifier layers
and is meant for breakdown analysis, not for clean throughput-only numbers.

## Confidence Acceptance Experiment

`run_speclink_confidence_acceptance.sh` tests whether DLM draft-token confidence
predicts TLM local acceptance. It does not implement chunked verification
scheduling.

The vLLM trace is off by default and is enabled only by:

```bash
SPECLINK_TRACE_CONFIDENCE=1
SPECLINK_TRACE_OUTPUT=temp/trace.jsonl
SPECLINK_TRACE_RUN_ID=qwen3_8b_eagle3_k4
SPECLINK_TRACE_DATASET_LABEL=math
SPECLINK_TRACE_MODEL_LABEL=qwen3_8b
SPECLINK_TRACE_METHOD=eagle3
SPECLINK_TRACE_NUM_SPEC_TOKENS=4
```

The hooks collect proposer logits in
`vllm/v1/spec_decode/llm_base_proposer.py`, buffer per-request draft records in
`vllm/speclink_confidence_trace.py`, and attach acceptance labels from
`vllm/v1/sample/rejection_sampler.py`. Records are aligned by vLLM request id
plus a per-request speculative-step counter. `dataset_label` separates math and
MTBench, and `model_label` separates Qwen and Llama traces in combined
analysis. `token_text` is left null to avoid tokenizer overhead in the hot path.

Run from `examples/evaluate/eval-guidellm`:

```bash
cd examples/evaluate/eval-guidellm
conda run -n spec bash ./run_speclink_confidence_acceptance.sh
```

The default command is the one-click reproduction for the final four-row report:
EAGLE3 `NUM_SPEC_TOKENS=8` on `{qwen3_8b,llama3_1_8b} x {math,mtbench}`. It
runs the four individual cases under:

```text
temp/speclink_confidence_acceptance_reproduce_TIMESTAMP/
```

and writes only the combined final report under:

```text
results/speclink_confidence_acceptance_datasets_TIMESTAMP/
```

Defaults:

- reproduction: EAGLE3 only, `REPRO_NUM_SPEC_TOKENS=8`,
  `REPRO_PROMPTS=80`, `REPRO_MAX_TOKENS=128`, `temperature=0`
- request concurrency defaults to `REPRO_REQUEST_CONCURRENCY=1`
- vLLM is launched with `--enforce-eager` for trace stability
- Qwen uses port `QWEN_PORT=8036`; Llama uses `LLAMA_PORT=8037`
- use `--single-case` for the older configurable single-model/single-dataset
  path

MTBench setup:

```bash
cd examples/evaluate/eval-guidellm
conda run -n spec python ./prepare_mt_bench_dataset.py --force
```

This downloads the official FastChat MTBench `question.jsonl` and writes:

```text
data/mt_bench_raw.jsonl
data/mt_bench.jsonl
```

The converted file has 80 rows. Multi-turn MTBench prompts are serialized as
`User turn N:` blocks followed by `Assistant:` so the completions endpoint can
be used consistently with the math dataset.

Useful variants:

```bash
# Preview the four final cases without launching vLLM
conda run -n spec bash ./run_speclink_confidence_acceptance.sh --dry-run

# Regenerate parsed CSV, calibration, figures, and report for an existing final run
conda run -n spec bash ./run_speclink_confidence_acceptance.sh \
  --analyze-only ./results/speclink_confidence_acceptance_TIMESTAMP

# Short main run for debugging
SPECLINK_SINGLE_CASE=1 MAIN_PROMPTS=128 \
conda run -n spec bash ./run_speclink_confidence_acceptance.sh --single-case --main-only

# Smoke only for one configurable case, written under temp/
conda run -n spec bash ./run_speclink_confidence_acceptance.sh --single-case --smoke-only

# Llama-3.1-8B EAGLE3 K=8 only
MODEL_LABEL=llama3_1_8b \
BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct \
EAGLE3_SPECULATOR_MODEL=../models/llama-3.1-8b-eagle3-speculator \
METHODS=eagle3 MAIN_NUM_SPEC_TOKENS=8 PORT=8035 \
conda run -n spec bash ./run_speclink_confidence_acceptance.sh --single-case --main-only

# Qwen3-8B MTBench, EAGLE3 K=8 only
DATASET_LABEL=mtbench \
DATASET=examples/evaluate/eval-guidellm/data/mt_bench.jsonl \
MODEL_LABEL=qwen3_8b \
BASE_MODEL=Qwen/Qwen3-8B \
EAGLE3_SPECULATOR_MODEL=../models/qwen3-8b-eagle3-speculator \
METHODS=eagle3 MAIN_NUM_SPEC_TOKENS=8 MAIN_PROMPTS=80 MAIN_MAX_TOKENS=128 \
REQUEST_CONCURRENCY=1 PORT=8036 \
conda run -n spec bash ./run_speclink_confidence_acceptance.sh --single-case --main-only

# Llama-3.1-8B MTBench, EAGLE3 K=8 only
DATASET_LABEL=mtbench \
DATASET=examples/evaluate/eval-guidellm/data/mt_bench.jsonl \
MODEL_LABEL=llama3_1_8b \
BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct \
EAGLE3_SPECULATOR_MODEL=../models/llama-3.1-8b-eagle3-speculator \
METHODS=eagle3 MAIN_NUM_SPEC_TOKENS=8 MAIN_PROMPTS=80 MAIN_MAX_TOKENS=128 \
REQUEST_CONCURRENCY=1 PORT=8037 \
conda run -n spec bash ./run_speclink_confidence_acceptance.sh --single-case --main-only

# Combine archived/intermediate case roots into one K=8 report
conda run -n spec python ./scripts/combine_speclink_confidence_results.py \
  --output-root ./results/speclink_confidence_acceptance_datasets_TIMESTAMP \
  --source qwen3_8b:math:./temp/speclink_confidence_acceptance_reproduce_TIMESTAMP/math_qwen3_8b_eagle3_k8 \
  --source llama3_1_8b:math:./temp/speclink_confidence_acceptance_reproduce_TIMESTAMP/math_llama3_1_8b_eagle3_k8 \
  --source qwen3_8b:mtbench:./temp/speclink_confidence_acceptance_reproduce_TIMESTAMP/mtbench_qwen3_8b_eagle3_k8 \
  --source llama3_1_8b:mtbench:./temp/speclink_confidence_acceptance_reproduce_TIMESTAMP/mtbench_llama3_1_8b_eagle3_k8 \
  --method eagle3 \
  --num-spec-tokens 8 \
  --analyze
```

Final outputs include:

- `commands.sh`
- `repro_report.md` for the default four-way combined report
- per-case `env_report.md` files under the corresponding temp work root
- `trace/DATASET_LABEL_MODEL_LABEL_METHOD_trace.jsonl`
- `parsed/*_token_level.csv`, `parsed/*_sanity.md`
- `calibration/*_calibrated.csv`, `calibration/*_summary.json`,
  `calibration/*_model_params.json`
- `figures/acceptance_by_position.png`, `confidence_bins.png`,
  `calibration_curve.png`, `confidence_fit_curve.png`, `reliability.png`,
  `reject_within_h.png`, `chunk_benefit.png`, plus the CSV data used to draw
  each figure. `confidence_fit_curve.png` uses dataset/model facets for combined
  K=8 reports so the actual-vs-fit curves remain readable without a legend.
- `summary.csv`, `summary.json`, `report.md`

The analysis script uses only installed lightweight dependencies:
`numpy`, `pandas`, and `PIL`. The current `spec` env does not include
`sklearn` or `matplotlib`, so logistic regression and calibration metrics are
implemented locally in
`scripts/analyze_speclink_confidence_acceptance.py`.

Current combined K=8 confidence/acceptance result:

```text
results/speclink_confidence_acceptance_datasets_20260525_200725/
```

It contains only EAGLE3 `NUM_SPEC_TOKENS=8` rows for
`{qwen3_8b,llama3_1_8b} x {math,mtbench}`. Key summary values:

```text
math/llama3_1_8b:    acceptance=0.5918, AUROC=0.8603, Spearman=0.6079
math/qwen3_8b:       acceptance=0.6201, AUROC=0.8208, Spearman=0.5296
mtbench/llama3_1_8b: acceptance=0.4787, AUROC=0.8218, Spearman=0.5573
mtbench/qwen3_8b:    acceptance=0.5599, AUROC=0.8135, Spearman=0.5403
```

Older confidence/acceptance case roots from the same development pass were
moved out of `results/` to:

```text
temp/moved_from_results_20260525_203200/
```

## Acceptance Jitter Experiment

`run_acceptance_jitter.sh` is the one-command reproduction for the accepted
draft-token count jitter figure. It uses normal vLLM speculative decoding with
`SPECLINK_TRACE_CONFIDENCE=1`; it does not use scheduler-level chunking or any
`SPECLINK_CHUNK_*` path.

Run from `examples/evaluate/eval-guidellm`:

```bash
cd examples/evaluate/eval-guidellm
conda run -n spec bash ./run_acceptance_jitter.sh
```

Default matrix:

- cases: `qwen3_8b:peagle qwen3_8b:eagle3 llama3_1_8b:eagle3`
- `NUM_SPEC_TOKENS_LIST="8 12 16"`
- workloads: `math mtbench synthetic_1000x1000`
- prompt counts: math=80, MTBench=80, synthetic=8
- real datasets use `max_tokens=128`
- synthetic uses exact token-id prompts with `SYNTHETIC_PROMPT_TOKENS=1000`
  and `SYNTHETIC_MAX_TOKENS=1000`

Final outputs are written under:

```text
results/accepted_count_jitter_TIMESTAMP/
```

Intermediate vLLM logs, responses, and raw trace JSONL files are written under:

```text
temp/accepted_count_jitter_work_TIMESTAMP/
```

Key final files:

- `step_level_acceptance.csv`: one row per decode step with
  `num_accepted`.
- `summary.csv`: mean/std of accepted tokens, `P(accepted < 2)`,
  `P(accepted < 4)`, full-prefix acceptance rate, and jitter metrics by
  workload/case/K.
- `accepted_count_distribution.csv`: empirical accepted-count distribution.
- `figures/math_accepted_count_jitter.png`
- `figures/mtbench_accepted_count_jitter.png`
- `figures/synthetic_1000x1000_accepted_count_jitter.png`
- `report.md`

Useful commands:

```bash
# Preview all 27 cases without launching vLLM
conda run -n spec bash ./run_acceptance_jitter.sh --dry-run

# Fast smoke into temp/
conda run -n spec bash ./run_acceptance_jitter.sh --smoke-only

# Re-analyze an existing intermediate work root into a final results dir
conda run -n spec bash ./run_acceptance_jitter.sh \
  --analyze-only ./temp/accepted_count_jitter_work_TIMESTAMP \
  --output-root ./results/accepted_count_jitter_TIMESTAMP
```

## Threshold Tradeoff Analysis

`scripts/run_threshold_tradeoff.sh` is a one-command experiment for confidence
threshold tradeoff:
1. run `run_acceptance_jitter.sh` (default workloads are `math,mtbench`, no
   synthetic case),
2. run `scripts/threshold_tradeoff.py` to compute the tradeoff.

`scripts/threshold_tradeoff.py` is an offline analysis over the confidence trace
and also does not launch vLLM.

The confidence rule is prefix sequence confidence:

```text
confidence(h) = product(draft_selected_prob[1:h])
```

For each threshold, the predicted token count is the largest prefix length whose
confidence stays above the threshold. If the first draft token is already below
the threshold, the prediction is 0.

Primary metrics:

- `error_probability`: `P(pred_tokens > actual_accept_tokens)` (overrun risk).
- `mismatch_probability`: `P(pred_tokens != actual_accept_tokens)`.
- `compute_efficiency`: `E(pred_tokens / K)`.

Pareto-optimal thresholds are those not dominated by another threshold with both
lower-or-equal `error_probability` and higher-or-equal `compute_efficiency`.

Run from `examples/evaluate/eval-guidellm`:

```bash
conda run -n spec bash ./scripts/run_threshold_tradeoff.sh
```

Analyze an existing trace root:

```bash
conda run -n spec python ./scripts/threshold_tradeoff.py \
  ./temp/accepted_count_jitter_work_TIMESTAMP \
  --output-root ./results/threshold_tradeoff_TIMESTAMP
```

Same offline command with custom cases and thresholds:

```bash
conda run -n spec python ./scripts/threshold_tradeoff.py \
  ./temp/accepted_count_jitter_work_TIMESTAMP \
  --output-root ./results/threshold_tradeoff_TIMESTAMP \
  --workloads math,mtbench \
  --models qwen3_8b,llama3_1_8b \
  --methods eagle3,peagle \
  --num-spec-tokens 8,12,16 \
  --thresholds 0.05,0.10,0.20,0.30,0.40,0.50
```

```bash
conda run -n spec bash ./scripts/run_threshold_tradeoff.sh \
  --workloads math,mtbench \
  --cases qwen3_8b:eagle3,llama3_1_8b:eagle3 \
  --num-spec-tokens 8,16 \
  --thresholds 0.05,0.10,0.15,0.20,0.30,0.40
```

Outputs:

- `threshold_tradeoff.csv`: all threshold points.
- `pareto_thresholds.csv`: non-dominated threshold points.
- `figures/threshold_tradeoff.png`
- `report.md`

## SpecLink-CV Trace Milestone

`tools/speclink_cv/` contains the current SpecLink-CV experiment scaffold:
chunk-size decision logic, request state machine, async verification queue,
roofline-packing policy, confidence calibration tools, unit tests, and a
trace-based experiment runner.

Run the current milestone from the repo root:

```bash
cd .
conda run -n spec python -m tools.speclink_cv.run_trace_experiment \
  --trace-root examples/evaluate/eval-guidellm/temp/accepted_count_jitter_work_TIMESTAMP \
  --output-root examples/evaluate/eval-guidellm/results/speclink_cv_TIMESTAMP \
  --workloads math,mtbench
```

The runner writes the requested result tree with:

- `00_env/env_report.{md,json}`
- `02_unit_tests/unit_test_summary.{md,csv,json}`
- `03_confidence_calibration/`
- `04_baselines/`
- `05_cv_ablation/cv_ablation_summary.csv`
- `06_scheduler_queue/`
- `07_roofline_packing/`
- `08_figures/cv_trace_tradeoff.png`
- `09_reports/SPECLINK_CV_REPORT.md`
- top-level `summary_metrics.{csv,json}`

Important limitation: this is an exact trace-level simulation over existing
one-shot EAGLE3 verification labels. The trace runner itself does not execute a
serving benchmark and must not be used to claim end-to-end throughput or latency
speedup. The separate live vLLM slice below changes scheduled speculative-token
shapes before target logits are computed, and the GuideLLM matrix runner records
real serving metrics for the live implementation.

## SpecLink-CV Live vLLM Slice

There is now a gated live vLLM implementation slice for fixed-half chunked
verification. It is intentionally narrow:

- enabled only with `SPECLINK_CV_ENABLE=1`
- requires the regular vLLM V1 scheduler path with vLLM's own
  `async_scheduling` disabled; the GuideLLM matrix runner automatically adds
  `--no-async-scheduling` for `cv_*` methods
- uses fixed half when confidence sizing is off, e.g. K=8 -> h=4
- when confidence sizing is on, carries proposal-time
  `draft_selected_prob` from the EAGLE3 drafter to the scheduler; without a
  calibration path it uses this as an uncalibrated local-acceptance proxy, and
  with `SPECLINK_CV_CALIBRATION_PATH` it applies the binning calibration model
  produced by `tools.speclink_cv.calibrate_acceptance`
- if the prefix rejects, it skips the suffix draft tokens
- if the prefix fully accepts, it masks the normal speculative bonus token,
  rolls scheduler progress back by that discarded bonus, and schedules the
  suffix for exact TLM verification
- when `SPECLINK_CV_ROOFLINE_PACKING=1`, estimates whether the current prefix
  chunk launch is underfilled using token/sequence budget utilization; if it is
  below `SPECLINK_CV_UTIL_THRESHOLD`, it falls back to exact one-shot
  verification for that step instead of running a small prefix chunk
- when `SPECLINK_CV_ASYNC_QUEUE=1`, prefix chunks enter a live scheduler queue
  before dispatch. The queue uses selected benefit, age, token budget, sequence
  budget, and roofline utilization to decide which queued prefix chunks run in
  the current scheduler step. Age timeout prevents starvation. This is a
  conservative first live queue, not the final full cross-request packing design.

Current environment variables:

```text
SPECLINK_CV_ENABLE=1
SPECLINK_CV_CONFIDENCE_SIZING=0
SPECLINK_CV_ASYNC_QUEUE=0
SPECLINK_CV_ROOFLINE_PACKING=0
SPECLINK_CV_CANDIDATE_CHUNKS=1,2,4,6,8,full
SPECLINK_CV_DEFAULT_HALF_POLICY=floor
SPECLINK_CV_MIN_BENEFIT=0.0
SPECLINK_CV_MAX_VERIFY_TOKENS_PER_STEP=0
SPECLINK_CV_MAX_VERIFY_SEQS_PER_STEP=0
SPECLINK_CV_MAX_QUEUE_WAIT_MS=2
SPECLINK_CV_UTIL_THRESHOLD=0.6
SPECLINK_CV_CALIBRATION_PATH=
SPECLINK_CV_LOG_JSONL=temp/events.jsonl
SPECLINK_CV_PROFILE_JSONL=temp/profile.jsonl
SPECLINK_CV_DEBUG_DUMP=0
```

`SPECLINK_CV_CONFIDENCE_SIZING=1` is wired into the live scheduler. If
`SPECLINK_CV_CALIBRATION_PATH` is empty, `draft_selected_prob` is used directly
as `a_hat` and events report `confidence_source=draft_selected_prob_uncalibrated`.
If the path points to a binning `calibration_model.json`, events report
`confidence_source=calibrated_binning`. `SPECLINK_CV_ROOFLINE_PACKING=1` is
wired as a live utilization gate and emits `roofline_fallback_one_shot` when an
underfilled prefix chunk is converted back to one-shot verification in sync
mode. With `SPECLINK_CV_ASYNC_QUEUE=1`, the scheduler emits
`verify_chunk_queued`, `async_queue_step`, and `verify_chunk_dequeued` profile
events and dispatches queued prefix chunks before exact verification.

`SPECLINK_CV_PROFILE_JSONL` is active in the live scheduler path. It records
newline-delimited JSON events for:

- `schedule_step`: scheduled seq/token counts, spec-token counts,
  prefix-chunk counts, remaining token budget, and config toggles.
- `verify_chunk_queued`, `async_queue_step`, `verify_chunk_dequeued`,
  `verify_chunk_waiting`: live async queue state, selected chunks, wait time,
  dispatch reason, predicted utilization, and budget use.
- `verify_chunk_scheduled`: prefix/suffix chunk length, suffix length,
  selected benefit, scheduled tokens, and budget context.
- `verify_chunk_result`: prefix accepted/rejected outcome, skipped suffix
  tokens, extra TLM forward count, and discarded speculative bonus count.
- `verify_chunk_decision`: roofline fallback decisions and predicted
  token/sequence utilization.

Focused checks from the repo root:

```bash
conda run -n spec python -m tools.speclink_cv.test_chunk_decision
conda run -n spec python -m tools.speclink_cv.test_state_machine
conda run -n spec python -m tools.speclink_cv.test_async_queue
conda run -n spec python -m tools.speclink_cv.test_roofline_packing
conda run -n spec python -m tools.speclink_cv.test_correctness_smoke
conda run -n spec python -m tools.speclink_cv.test_vllm_runtime_config
```

Live smoke status on 2026-05-26: Qwen/Qwen3-8B with the local EAGLE3
speculator, K=8, greedy `temperature=0`, and a 32-token fixed-half generation
matched baseline one-shot EAGLE3 token-for-token. A separate 16-token
confidence-sizing smoke also matched baseline. A calibrated confidence smoke
using the trace milestone `calibration_model.json` matched baseline and showed
`confidence_source=calibrated_binning` in the event JSONL. A roofline fallback
smoke with `--roofline-packing --util-threshold 0.99` also matched baseline and
emitted `roofline_fallback_one_shot`. A live async-queue smoke with
`--async-queue` matched baseline and emitted queue/dequeue/profile events. A
combined `--async-queue --roofline-packing --util-threshold 0.99` smoke also
matched baseline and dispatched via `no_other_ready_work` for the single-request
case. The smoke was run from
`examples/evaluate/eval-guidellm` to avoid repo-root `vllm/` import shadowing.
The event logs are temporary diagnostics under
`examples/evaluate/eval-guidellm/temp/speclink_cv_live_smoke_20260526_scheduler/`.

Reusable GPU smoke command:

```bash
conda run -n spec python tools/speclink_cv/live_correctness_smoke.py \
  --speculator-model ../models/qwen3-8b-eagle3-speculator \
  --max-tokens 32 \
  --output-json examples/evaluate/eval-guidellm/temp/speclink_cv_live_smoke.json \
  --event-jsonl examples/evaluate/eval-guidellm/temp/speclink_cv_live_smoke_events.jsonl \
  --profile-jsonl examples/evaluate/eval-guidellm/temp/speclink_cv_live_smoke_profile.jsonl
```

Add `--confidence-sizing` to exercise the live uncalibrated confidence path.
Add `--calibration-path path/to/calibration_model.json` with
`--confidence-sizing` to exercise live calibrated binning. Add
`--roofline-packing --util-threshold 0.99` to force the live underfilled-prefix
fallback path in a single-request smoke. Add `--async-queue` to exercise the
live prefix queue.

For batched correctness, run the same smoke over repo-local prompts:

```bash
conda run -n spec python tools/speclink_cv/live_correctness_smoke.py \
  --speculator-model ../models/qwen3-8b-eagle3-speculator \
  --prompts-jsonl examples/evaluate/eval-guidellm/data/math_reasoning.jsonl \
  --num-prompts 8 \
  --max-num-seqs 8 \
  --max-tokens 64 \
  --async-queue \
  --roofline-packing \
  --output-json examples/evaluate/eval-guidellm/temp/speclink_cv_live_batched_smoke.json
```

## SpecLink-CV GuideLLM Matrix Runner

`examples/evaluate/eval-guidellm/scripts/run_speclink_cv_guidellm_matrix.py`
starts `vllm serve`, waits for `/health` with a proxy-safe raw socket check,
runs GuideLLM, parses vLLM speculative metrics, and writes per-run logs plus
top-level `status.csv`, `summary_metrics.csv`, `summary_metrics.json`,
`report.md`, `scripts/run_commands.sh`, the TODO result tree
`00_env/` through `09_reports/`, raw run directories under `runs/`, and figure
source tables under `08_figures/`. For `cv_*` methods it adds
`--no-async-scheduling` because SpecLink-CV's live prefix/suffix scheduler
logic is implemented in the regular V1 scheduler; `SPECLINK_CV_ASYNC_QUEUE` is
the experiment's own verification queue and is independent from vLLM's
scheduler async mode.
The summary includes text-level `exact_match_vs_eagle3` by aligning GuideLLM
successful requests by `request_args` and comparing each `cv_*` output to the
matching `eagle3_oneshot` output for the same model/dataset/K/batch-size case.
Token-id exact-match evidence still comes from
`tools/speclink_cv/live_correctness_smoke.py`. Treat any `cv_*` row with
`exact_match_vs_eagle3 < 1.0` as a correctness warning, not as a valid speedup
claim.

The runner forces `NO_PROXY/no_proxy` to include local addresses and kills the
vLLM process group during cleanup. This matters in the current environment
because local HTTP proxy variables can otherwise trap `127.0.0.1` health checks,
and a failed API server can leave an EngineCore process holding GPU memory.

Smoke command:

```bash
cd .
conda run -n spec python -u examples/evaluate/eval-guidellm/scripts/run_speclink_cv_guidellm_matrix.py \
  --smoke \
  --max-requests 1 \
  --enforce-eager \
  --gpu-memory-utilization 0.75 \
  --port 8051 \
  --output-root examples/evaluate/eval-guidellm/temp/speclink_cv_guidellm_smoke_TIMESTAMP
```

Full TODO-shaped matrix command:

```bash
conda run -n spec python -u examples/evaluate/eval-guidellm/scripts/run_speclink_cv_guidellm_matrix.py \
  --models qwen3_8b,llama3_1_8b \
  --datasets math,mtbench \
  --ks 8,12 \
  --batch-sizes 8,16,32 \
  --methods pure_vllm,eagle3_oneshot,cv_half_sync_simple,cv_half_sync_roofline,cv_half_async_simple,cv_half_async_roofline,cv_conf_sync_simple,cv_conf_sync_roofline,cv_conf_async_simple,cv_conf_async_roofline \
  --max-requests 80 \
  --output-root examples/evaluate/eval-guidellm/results/speclink_cv_guidellm_TIMESTAMP
```

Equivalent one-command wrapper:

```bash
OUTPUT_ROOT=examples/evaluate/eval-guidellm/results/speclink_cv_guidellm_TIMESTAMP \
  examples/evaluate/eval-guidellm/scripts/run_speclink_cv_guidellm_full.sh
```

The wrapper defaults to `--resume`, `--enforce-eager`, and
`--disable-vllm-async-scheduling` for a conservative correctness/fairness run
where EAGLE3 one-shot and CV both use the regular scheduler path. Override with
`ENFORCE_EAGER=0` or `DISABLE_VLLM_ASYNC_SCHEDULING=0` only when you explicitly
want the default vLLM serving mode for the baselines. Use `CASE_OFFSET` and
`CASE_LIMIT` to run chunks of the full matrix.

Use `--dry-run` to write planned commands only, `--analyze-only` to rebuild
summary files from an existing output root, and `--resume` to reuse any run
directory that already contains `guidellm_results.json`. `--case-offset N` and
`--case-limit M` run or analyze a slice of the planned matrix, which is useful
for splitting the 240-case matrix across long GPU sessions. Reusing one output
root with `--resume` is the safest way to continue after interruption. The
runner passes GuideLLM `--random-seed 42` by default for reproducible dataset
sampling. Add `--disable-vllm-async-scheduling` when you need the EAGLE3
one-shot baseline to use the same regular V1 scheduler mode as the CV runs for
correctness/fairness diagnosis.

GuideLLM matrix smoke status on 2026-05-26: Qwen3 math, K=8, batch size 1,
`eagle3_oneshot` and `cv_half_async_roofline` both completed with
`status=ok`; rerun after the `--no-async-scheduling` fix confirmed that the CV
case emits live `verify_chunk_*` profile events. A separate
`cv_conf_async_roofline` smoke also completed with `status=ok` and emitted
`confidence_source=draft_selected_prob_uncalibrated`. A later full-path smoke
also generated `00_env/`, focused unit-test summaries, `08_figures/`, and
`09_reports/SPECLINK_CV_REPORT.md`. Output roots:

```text
examples/evaluate/eval-guidellm/temp/speclink_cv_guidellm_smoke_20260526_v6/
examples/evaluate/eval-guidellm/temp/speclink_cv_guidellm_conf_smoke_20260526/
examples/evaluate/eval-guidellm/temp/speclink_cv_guidellm_smoke_report_20260526_v7/
```

This smoke proves the runner, vLLM server startup, GuideLLM request path, and
live `verify_chunk_*` CV profile logging work. It is not a throughput claim
because it uses one request.

## SR24 CPU-Sync And Compile Cache Notes

Latest fixed-request SR24 diagnostics on 2026-06-23 used Llama-3.1-8B +
EAGLE3 K=8, `math_reasoning`, client concurrency 64, fixed total requests 64,
`max_tokens=512`, and outputs under:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/
```

Runner updates:

- SR24-only proposal context no longer computes generated lengths with
  `valid_sampled_tokens_count.detach().cpu().tolist()`. Length context is now
  computed only when confidence trace or token-dense tracing is enabled.
- SR24 runs that allow CUDA Graph, as well as `--sr24-default-vllm-compile`,
  assign an SR24-env-fingerprinted `VLLM_CACHE_ROOT` under
  `examples/evaluate/eval-guidellm/temp/vllm_compile_cache/` and record the
  cache root in command metadata/summary output. This avoids reusing a vLLM
  compile graph produced under a different SR24 env while still allowing
  same-config reruns to load cache.

Reliable same-code results:

| case | steady output tok/s | vs dense | result root |
| --- | ---: | ---: | --- |
| dense EAGLE3 baseline | `2920.812` | `1.000x` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_paired_dense_current2_bs64_20260623` |
| dynamic `auto`, dense_rows, cudagraph before lazy context | `2121.059` | `0.726x` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/speclink_sr24_t08_attn_denserows_cudagraph_bs64_fixed512_20260623` |
| dynamic `auto`, dense_rows, lazy proposal context | `2135.696` | `0.731x` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_auto_denserows_cudagraph_lazyctx_bs64_fixed512_20260623` |
| dynamic `auto`, lazy context, no mask-state CPU sync | `2149.227` | `0.736x` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_auto_denserows_cudagraph_lazyctx_nosyncstate_bs64_fixed512_20260623` |
| static `all_residual`, default compile, SR24-specific hot cache | `2337.726` | `0.800x` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_static_allres_denserows_defaultcompile_cachekey_hot_bs64_20260623` |

The older static `all_residual` result
`sr24_static_allres_denserows_defaultcompile_fastpathfix2_bs64_20260623`
reported `2959.932` tok/s, but it used the global vLLM compile cache. After a
dynamic `auto` compile failure, static runs could reuse incompatible cached
graphs. Treat that `2959.932` number as an unreliable cache-contamination
diagnostic, not a valid SR24 speedup.

Dynamic `auto` with default vLLM compile still does not run reliably:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_auto_denserows_defaultcompile_bs64_20260623
```

It fails during startup with Inductor tracing into SR24 sparse/dense_rows
linear state and raising `RuntimeError: The tensor has a non-zero number of
elements, but its data is not allocated yet.` The next real optimization target
is therefore the SR24 Linear/operator integration, not additional CPU stats
bookkeeping. CPU-sync reductions are useful ablations but only gave about
`1.3%` combined on the dynamic cudagraph path.

Additional fixed-request diagnostics on 2026-06-23 used the same Llama-3.1-8B +
EAGLE3 K=8, `math_reasoning`, batch/concurrency 64, fixed 64 requests, and
`max_tokens=512`. These runs added GPU utilization sampling and tested whether
the bottleneck is low acceptance, CPU synchronization, or graph coverage:

| case | steady output tok/s | avg GPU util | acceptance | CUDA graph modes | result root |
| --- | ---: | ---: | ---: | --- | --- |
| dense EAGLE3 baseline | `3252.145` | `91.55%` | `0.304` | n/a | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_current_util_matrix_bs64_20260623` |
| full `base_only_24`, eager sparse | `1114.241` | `24.00%` | `0.544` | `{"NONE":128}` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_current_util_matrix_bs64_20260623` |
| full `base_only_24`, `--sr24-allow-cudagraph` | `4361.172` | `91.67%` | `0.543` | `{"FULL":126,"NONE":2}` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_cudagraph_util_bs64_20260623` |
| full `speclink_t08`, `compressed_dense@cuda` | `249.654` | `69.09%` | `0.309` | `{"NONE":320}` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_t08_full_compressed_extractfix_bs64_20260623` |
| full `speclink_t08`, reduced CPU sync | `257.706` | `72.45%` | `0.318` | `{"NONE":320}` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_t08_reduce_sync_bs64_20260623` |
| attention-only `speclink_t08`, `compressed_dense@cuda` | `925.612` | `62.84%` | `0.308` | `{"NONE":256}` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_t08_attention_only_bs64_20260623` |
| attention-only `speclink_t08`, `torch_sparse` residual + CUDA Graph | `2683.003` | `90.00%` | `0.309` | `{"FULL":190,"NONE":2}` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_t08_attention_sparse_residual_cudagraph_bs64_20260623` |

Interpretation: `base_only_24` is not slow because of acceptance length; it has
higher acceptance than dense. The eager PyTorch semi-structured path underfills
the GPU. CUDA Graph coverage restores GPU utilization and makes the backend fast
in the diagnostic base-only mode. For real `speclink_t08`, reducing CPU sync
only improved full compressed-dense selective throughput by about `3.2%`
(`249.654 -> 257.706` tok/s). The dominant issue is that
`compressed_dense` selective residual stays eager and rebuilds residual work at
runtime. Attention-only graph-compatible `torch_sparse` residual reaches
`2683.003` tok/s, so the next optimization path is a graph-friendly residual
implementation and careful target-module selection, not more CPU bookkeeping
reduction.

Long-output confirmation on 2026-06-23 used Llama-3.1-8B + EAGLE3 K=8,
`math_reasoning`, batch/concurrency 64, fixed 64 requests, and
`max_tokens=2048`:

| case | steady output tok/s | vs dense | avg GPU util | acceptance | CUDA graph modes | result root |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| dense EAGLE3 baseline | `2382.632` | `1.000x` | `98.13%` | `0.479` | n/a | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_current_dense_base_allcorrected_bs64_max2048_20260623` |
| full `base_only_24`, graph-enabled | `4116.305` | `1.728x` | `96.97%` | `0.655` | `{"FULL":826,"NONE":6}` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_current_dense_base_allcorrected_bs64_max2048_20260623` |
| full `all_corrected_24`, dense fastpath | `2330.206` | `0.978x` | `97.68%` | `0.483` | `{"FULL":1749,"NONE":234,"PIECEWISE":1}` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_current_dense_base_allcorrected_bs64_max2048_20260623` |
| full `all_corrected_24`, static all-residual dense fastpath | `2335.061` | `0.983x` | `98.05%` | `0.481` | `{"FULL":1748,"NONE":235,"PIECEWISE":1}` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_static_allres_bs64_max2048_20260623` |

This confirms the base-only diagnostic path is not slow under graph capture:
accepted length is higher than dense and GPU utilization is full. The
`all_corrected_24` optimization is now the dense fastpath
(`torch_sparse/dense_fastpath@none`, storage/dense `1.0`), so it avoids the
old sparse-plus-residual double compute and avoids CPU `compressed_dense`
runtime materialization. The remaining `1.7-2.2%` gap to dense is SR24
wrapper/metrics/graph-fragmentation overhead rather than missing GPU-resident
compressed residual work.

Important fixed-request caveat: before 2026-06-24,
`--fixed-total-requests` still allowed workers to stop at the
warmup/measurement/cooldown deadline, so long-output rows could complete fewer
requests than requested. Use corrected fixed-request rows only when
`successful_requests == max_requests`.

Strict corrected bs64, fixed 128 request, `max_tokens=2048` results on
2026-06-24:

| case | total/steady output tok/s | full-batch output tok/s | vs strict dense total/full | successful requests | result root |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 baseline | `4080.093` | `4837.020` | `1.000x` / `1.000x` | 128 | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dense_vs_gateup_16_31_extra3_bucket32_fixed128_max2048_full_20260624` |
| `speclink_t08`, `gate_up_proj=16-31`, C4 mask, `critical_prefix+extra_after_low=3`, dense-row bucket32 | `3643.271` | `4591.778` | `0.893x` / `0.949x` | 128 | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dense_vs_gateup_16_31_extra3_bucket32_fixed128_max2048_full_20260624` |
| `base_only_24`, `gate_up_proj=16-31` upper bound | `4300.928` | `5401.893` | `1.054x` / `1.117x` | 128 | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_baseonly_fixed128_max2048_upperbound_20260624` |
| `base_only_24`, `gate_up_proj=8-31` upper bound | `4905.196` | `5823.714` | `1.202x` / `1.204x` | 128 | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_8_31_baseonly_fixed128_max2048_upperbound_20260624` |

The corrected strict measurement changes the conclusion for the current
dynamic candidate: it is below dense at long-output bs64. The quality-safer
static coverage `gate_up_proj=16-31` does not have enough base-only speed
headroom to reach `1.2x`, even with all dynamic residual overhead removed.
Wider `gate_up_proj=8-31` has enough speed headroom, but it is not quality-safe.
A paired GSM8K-50 run with the same C4 activation-aware mask gave dense
`0.7800` versus `base_only_24 gate_up_proj=8-31` `0.5200`
(`-26pp`) at:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_8_31_baseonly_quality_gsm8k50_20260624
```

Therefore CPU-side sync reduction is a useful ablation/default hygiene item,
but it cannot by itself solve the current target. The next viable path must
either recover quality for wider sparse coverage with very cheap residual
correction, or introduce a fused GPU sparse/residual operator that makes the
quality-preserving dynamic path cheaper.

Follow-up on 2026-06-24 tested the wider `gate_up_proj=8-31` selective path
with non-draft correction kept on. Be careful comparing lm-eval quality here:
`dense_baseline --enforce-eager` scores `0.7200` on the GSM8K-50 manifest,
while the default non-eager dense baseline scores `0.7800`. SR24 lm-eval
quality must therefore be compared against the serving mode it actually uses.

Quality, GSM8K-50, Llama-3.1-8B + EAGLE3 K=8:

| case | GSM8K-50 flexible | note | result root |
| --- | ---: | --- | --- |
| dense baseline, default compile | `0.7800` | non-eager default | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_8_31_extra3_bucket32_nondraft_graph_direct_quality_gsm8k50_20260624` |
| dense baseline, `--enforce-eager` | `0.7200` | eager-only reference | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dense_baseline_enforce_eager_quality_gsm8k50_20260624` |
| `base_only_24 gate_up_proj=8-31` | `0.5200` | speed upper bound but quality unusable | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_8_31_baseonly_quality_gsm8k50_20260624` |
| `speclink_t08 gate_up_proj=8-31`, `extra3`, no non-draft correction | `0.6600` | improves over base-only but still low | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_8_31_extra3_bucket32_quality_gsm8k50_20260624` |
| `speclink_t08 gate_up_proj=8-31`, `extra3`, non-draft correction | `0.7200` | matches dense eager but remains `-6pp` vs default dense | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_8_31_extra3_bucket32_nondraft_quality_gsm8k50_20260624` |
| same, graph-safe + direct cuSPARSELt | `0.7200` | actual throughput path; still `-6pp` vs default dense | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_8_31_extra3_bucket32_nondraft_graph_direct_quality_gsm8k50_20260624` |
| same, `extra8`, non-draft correction | `0.7200` | correcting all draft/non-draft rows did not recover default-dense quality | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_8_31_extra8_bucket32_nondraft_quality_gsm8k50_20260624` |
| static all-residual densefastpath, graph-safe | `0.7200` | shows this SR24/compile serving path follows the eager-quality branch | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_8_31_static_allres_densefastpath_graph_quality_gsm8k50_20260624` |

The actual graph-safe/direct throughput config is not a speed win either. A
same-run strict bs64, fixed 128 request, `max_tokens=2048` comparison completed
all 128 requests:

| case | total/steady output tok/s | full-batch output tok/s | speedup vs same-run dense total/full | result root |
| --- | ---: | ---: | ---: | --- |
| dense EAGLE3 baseline | `3432.907` | `4705.133` | `1.000x` / `1.000x` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dense_vs_gateup_8_31_extra3_bucket32_nondraft_fixed128_max2048_20260624` |
| `speclink_t08 gate_up_proj=8-31`, `extra3`, non-draft correction, bucket32, direct cuSPARSELt | `3055.420` | `4164.870` | `0.890x` / `0.885x` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dense_vs_gateup_8_31_extra3_bucket32_nondraft_fixed128_max2048_20260624` |

Trying to remove the SR24-specific compilation config with
`--sr24-default-vllm-compile --sr24-direct-cslt-linear` failed during vLLM
startup with the known Inductor error:

```text
RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet.
```

That failure was recorded at:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_8_31_extra3_bucket32_nondraft_defaultcompile_direct_quality_gsm8k50_20260624
```

So the current evidence rules out three simple fixes: reducing CPU-side sync
alone, increasing `extra_after_low` to correct every row, and switching this
dynamic dense-row path back to default vLLM compile. The remaining path is an
operator-level change that avoids the mixed sparse/dense-row overhead while
preserving the default-dense numerical/compile behavior.

Follow-up SR24 target-module selection on 2026-06-23 used Llama-3.1-8B +
EAGLE3 K=8, `math_reasoning`, batch/concurrency 64, 128 fixed requests, and
`max_tokens=512`. The older layer-level candidate was:

```bash
--sr24-target-leafs qkv_proj,o_proj,gate_up_proj,down_proj \
--sr24-residual-target-leafs qkv_proj,o_proj \
--sr24-base-only-layer-ids 30-31 \
--sr24-reduce-cpu-sync --no-sr24-sync-mask-state \
--sr24-static-mask-state mixed --sr24-static-mask-buffer --sr24-allow-cudagraph
```

| case | steady output tok/s | vs same-run dense | GSM8K 50 accuracy | note |
| --- | ---: | ---: | ---: | --- |
| same-run dense EAGLE3 baseline | `2279.398` | `1.000x` | `0.7000` to `0.7400` across paired 50-sample runs | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dense_eagle3_baseline_bs64_20260623` |
| attention-only residual, graph-safe masked | `2614.410` | `1.147x` | `0.7600` in one 50-sample run | accurate but below `1.2x` |
| all MLP base-only + attention residual | `4016.176` | `1.762x` | `0.1400` | speed upper bound; accuracy unusable |
| layer `31` MLP base-only + attention residual | `2578.067` | `1.131x` | not run | below `1.2x` |
| layers `30-31` MLP base-only + attention residual | `2915.628` | `1.279x` | `0.6800` vs dense `0.7000` | superseded by the per-leaf candidate below |

The strongest throughput SR24 point tested on 2026-06-23 is the per-leaf MLP
tail configuration:

```bash
--sr24-target-leafs qkv_proj,o_proj,gate_up_proj,down_proj \
--sr24-residual-target-leafs qkv_proj,o_proj \
--sr24-base-only-layer-ids-by-leaf 'gate_up_proj=31;down_proj=30-31' \
--sr24-reduce-cpu-sync \
--sr24-static-mask-state all_residual \
--sr24-static-all-residual-dense-fastpath \
--sr24-static-mask-buffer --sr24-allow-cudagraph
```

Same-code repeat evidence at `max_tokens=512`:

| case | output tok/s | vs dense | result root |
| --- | ---: | ---: | --- |
| dense EAGLE3 repeat median | `2479.295` | `1.000x` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dense_vs_last1_attn_densefastpath_repeat3_bs64_20260623` |
| per-leaf `speclink_t08` repeat median | `2975.665` | `1.200x` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_t08_leaf_gate31_down3031_repeat3_bs64_20260623` |

GSM8K-50 paired accuracy for the same per-leaf configuration matched dense
(`0.7400` vs `0.7400`, 4 dense-correct losses and 4 dense-wrong gains) at:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_leaf_gate31_down3031_gsm8k50_20260623
```

Additional long-output and accuracy evidence for this aggressive per-leaf
candidate:

| check | result | result root |
| --- | --- | --- |
| `math_reasoning`, bs64, fixed 64, `max_tokens=2048` | `3872.928` vs dense `2349.346` output tok/s (`1.649x`) | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_leaf_gate31_down3031_bs64_max2048_smoke_20260623` |
| GSM8K-100 | dense `0.7700`, SR24 `0.8100` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_leaf_gate31_down3031_gsm8k100_20260623` |
| Minerva Math500-100 | dense `0.3900`, SR24 `0.4000` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_leaf_gate31_down3031_minerva100_20260623` |
| IFEval-100, `max_new_tokens=1024` | prompt strict `-1pp`, prompt loose `-4pp`, instruction strict `+0.61pp`, instruction loose `-0.61pp`; still heavily clipped | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_leaf_gate31_down3031_ifeval100_max1024_20260623` |
| IFEval-50, `max_new_tokens=2048` | prompt strict `0pp`, prompt loose `-4pp`, instruction strict `+2.63pp`, instruction loose `0pp`; still 46/50 clipped | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_leaf_gate31_down3031_ifeval50_max2048_20260623` |

Because IFEval remains clipping-sensitive, the safer accuracy-first candidate is
a more conservative per-leaf MLP tail:

```bash
--sr24-target-leafs qkv_proj,o_proj,gate_up_proj,down_proj \
--sr24-residual-target-leafs qkv_proj,o_proj \
--sr24-base-only-layer-ids-by-leaf 'gate_up_proj=31;down_proj=31' \
--sr24-reduce-cpu-sync \
--sr24-static-mask-state all_residual \
--sr24-static-all-residual-dense-fastpath \
--sr24-static-mask-buffer --sr24-allow-cudagraph
```

Accuracy-first candidate evidence:

| check | result | result root |
| --- | --- | --- |
| `math_reasoning`, bs32, fixed 128, `max_tokens=2048`, repeat median | `3502.759` vs dense `1602.689` output tok/s (`2.185x`), CUDA graph `{"FULL":542,"NONE":2}` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_first_preset_bs32_repeat3_max2048_20260623` |
| same bs32 config, manual no-reduce-sync ablation | `3486.739` output tok/s (`0.995x` of reduce-sync median), CUDA graph `{"NONE":557}` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_first_manual_no_reduce_sync_bs32_max2048_20260623` |
| `math_reasoning`, bs64, fixed 64, `max_tokens=2048`, repeat median | `3314.733` vs dense `2374.322` output tok/s (`1.396x`), CUDA graph `{"FULL":906,"NONE":22}` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_leaf_gate31_down31_bs64_max2048_repeat3_20260623` |
| `math_reasoning`, bs64, fixed 64, `max_tokens=2048`, single smoke | `3320.182` vs dense `2376.086` output tok/s (`1.397x`), CUDA graph `{"FULL":907,"NONE":21}` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_leaf_gate31_down31_bs64_max2048_smoke_20260623` |
| GSM8K-20 lm-eval sanity | dense `0.7000`, SR24 `0.7000`; `sr24_preset=accuracy_first` persisted in `summary.csv` and `run_meta.json` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_first_preset_gsm8k20_20260623` |
| GSM8K-100 / Minerva-100 / IFEval-100 lm-eval, `max_new_tokens=2048` | GSM8K `0.7900 -> 0.7700` (`-2pp`, 4 losses/2 gains), Minerva exact `0.3900 -> 0.4000` (`+1pp`, 6 losses/7 gains), IFEval prompt strict `0.4600 -> 0.4700` but loose/inst dropped up to `-4.29pp`; IFEval was heavily clipped (`96/100`) | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_first_preset_gsm8k_minerva_ifeval100_2048_20260623` |
| GSM8K-100 | dense `0.7700`, SR24 `0.7700`; 4 losses and 4 gains | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_leaf_gate31_down31_gsm8k100_minerva100_20260623` |
| Minerva Math500-100 | dense `0.4200`, SR24 `0.4200`; 10 losses and 10 gains | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_leaf_gate31_down31_gsm8k100_minerva100_20260623` |
| IFEval-50, `max_new_tokens=2048` | prompt strict `+6pp`, prompt loose `+4pp`, instruction strict/loose `0pp`; still 46/50 clipped | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_leaf_gate31_down31_ifeval50_max2048_20260623` |

The 100-sample rerun is the current strongest quality signal. It shows that
`accuracy_first` is not yet quality-stable enough to call solved: the two
base-only MLP tail leaves (`gate_up_proj=31;down_proj=31`) can change reasoning
trajectories, including one GSM8K regression that repeats until clipping.
Next GPU ablations should split this tail into `gate_up_proj=31` only and
`down_proj=31` only before expanding throughput claims. If either leaf-only
variant keeps GSM8K and Minerva at dense while retaining at least `1.2x`
throughput, make it the safer preset and demote the current two-leaf preset to
an aggressive speed/quality tradeoff.

`--sr24-preset accuracy_first` was also checked at other batch/concurrency
values with `math_reasoning`, K=8, `max_tokens=2048`, and one full-batch
fixed-request run per point:

| batch/concurrency | dense output tok/s | speclink_t08 output tok/s | speedup | note |
| ---: | ---: | ---: | ---: | --- |
| 16 | `3091.351` | `2208.741` | `0.714x` | low batch does not benefit |
| 32 | `1602.689` | `3502.759` | `2.185x` | repeat median; dense low point was stable |
| 64 | `2374.322` | `3314.733` | `1.396x` | repeat median, strongest current evidence |
| 128 | `2885.190` | `3658.505` | `1.268x` | high-batch positive |

The bs16 result means the current recommendation is not a universal low-batch
speedup. Use the preset for high-concurrency/long-output serving; keep dense
EAGLE3 for low-concurrency latency-sensitive cases until a lower-overhead sparse
tail path is implemented.

CPU-side synchronization reductions remain useful ablations. For the current
`accuracy_first` static-all-residual path, a direct bs32/max2048 no-reduce-sync
ablation reached `3486.739` output tok/s versus the reduce-sync repeat median
`3502.759`, only about `0.5%` lower, even though CUDA graph coverage changed
from `{"FULL":542,"NONE":2}` to `{"NONE":557}`:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_first_manual_no_reduce_sync_bs32_max2048_20260623
```

For the older aggressive per-leaf candidate at `max_tokens=512`, turning off
`--sr24-reduce-cpu-sync` reduced median throughput from `2975.665` to
`2918.842` output tok/s and changed CUDA graph coverage from
`{"FULL":254,"NONE":2}` to `{"NONE":256}`:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_t08_leaf_gate31_down3031_no_reduce_sync_repeat3_bs64_20260623
```

The interpretation is that reduced CPU sync is a small measured effect in the
static `accuracy_first` path because it avoids the dynamic per-step mask-state
queries. Keep it as a clean ablation/default, but target-module selection and
operator integration are the main speed contributors. Disabling SR24 stats or
using direct cuSPARSELt did not materially improve the last1 candidate.

Latest static-mask no-sync patch check for `accuracy_first`,
`math_reasoning`, K=8, `max_tokens=2048`, fixed 128 requests:

| batch/concurrency | dense output tok/s | speclink_t08 output tok/s | speedup | result root |
| ---: | ---: | ---: | ---: | --- |
| 32 | `1607.833` | `3499.637` | `2.177x` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_first_static_nosync_bs32_64_max2048_20260623` |
| 64 | `2376.232` | `3872.426` | `1.630x` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_first_static_nosync_bs32_64_max2048_20260623` |

The strict `auto` mask-state A/B for the same bs64/fixed-128 setting reached
only `3083.268` output tok/s at:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_first_sync_ablation_auto_bs64_max2048_20260623
```

Treat this as a combined "auto/mixed residual path" cost, not a pure `.item()`
cost: `auto` classified the mask as `mixed`, so the run used
`torch_sparse/torch_sparse@cuda` instead of the static all-residual
`dense_fastpath@none` path.

Per-leaf/layer dynamic residual prototype:

```bash
--sr24-target-leafs qkv_proj,o_proj,gate_up_proj,down_proj \
--sr24-residual-target-leafs qkv_proj,o_proj,gate_up_proj,down_proj \
--sr24-residual-layer-ids-by-leaf 'gate_up_proj=31;down_proj=31' \
--sr24-static-mask-state mixed \
--sr24-static-all-residual-dense-fastpath
```

This attaches 32 `qkv_proj` and 32 `o_proj` modules as `dense_fastpath`, and
only layer-31 `gate_up_proj`/`down_proj` as `torch_sparse` dynamic residual.
The smoke result was `1420.413` output tok/s for bs32, `max_tokens=512`, fixed
64 requests, with storage/dense `1.0218`:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dynamic_mlp31_residual_smoke_bs32_max512_20260623
```

The GSM8K/Minerva 100-sample check at `max_new_tokens=2048` reached GSM8K
`0.78` and Minerva `0.38`:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dynamic_mlp31_residual_gsm8k_minerva100_2048_20260623
```

This `critical_prefix` version is not better than the current
`accuracy_first` tradeoff and is much slower, so keep it as an
implementation/quality ablation rather than the default path.

The more conservative dynamic MLP31 variant uses the same per-leaf/layer
residual filter with `SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY=all_if_any_low`.
It corrects all draft rows for a request step whenever any draft token in that
step is low-confidence. Current focused evidence:

| check | result | result root |
| --- | --- | --- |
| Paired GSM8K-100 / Minerva-100 / IFEval-100, `max_new_tokens=2048` | GSM8K `0.8000 -> 0.8000`; Minerva `0.4000 -> 0.4400`; IFEval prompt strict `0.4500 -> 0.4600`, prompt loose `0.5100 -> 0.4900`, inst strict `0.6135 -> 0.6258`, inst loose `0.6503 -> 0.6442` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dynamic_mlp31_allifanylow_paired_gsm8k_minerva_ifeval100_2048_20260623` |
| `math_reasoning`, bs64, fixed 128, `max_tokens=2048`, reduce-sync enabled | `3604.333` output tok/s, CUDA graph `{"FULL":679,"NONE":25}`, storage/dense `1.0218` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dynamic_mlp31_allifanylow_bs64_max2048_20260623` |
| paired throughput A/B, same shape | dense `2399.324`, SR24 `2102.039` output tok/s; SR24 CUDA graph `{"FULL":1961,"NONE":23}` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dynamic_mlp31_allifanylow_paired_throughput_bs64_max2048_20260623` |
| same throughput config, no reduce-sync | `1990.549` output tok/s, CUDA graph `{"NONE":1984}` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dynamic_mlp31_allifanylow_no_reduce_sync_bs64_max2048_20260623` |
| paired throughput with `full_batch_output_tokens_per_second` metric | dense total/full-batch `2400.716`/`5313.360`; SR24 total/full-batch `2104.592`/`4943.323` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dynamic_mlp31_allifanylow_static_mixed_fullbatch_bs64_max2048_20260623` |

For this dynamic MLP31 candidate, `--sr24-reduce-cpu-sync` is not merely a
small logging optimization. It switches the mixed-mask residual path away from
per-layer `nonzero()/numel()` host synchronizations and keeps the path graph
safe enough to get mostly full decode CUDA Graph coverage. The no-reduce-sync
ablation is only close to the slow paired throughput result and loses full
decode graph coverage. The earlier `3604.333` single-method run should not be
treated as a final speed claim: the paired A/B run exposed a long-tail request
effect and measured SR24 below the same-run dense baseline. Keep reduce-sync
enabled for future dynamic-residual throughput runs, keep no-reduce only as a
CPU-sync/graph-coverage ablation, and use repeat medians or a real full-batch
steady-state metric before making throughput claims for this quality-oriented
dynamic-residual candidate.

Follow-up on 2026-06-23 added `full_batch_output_tokens_per_second` to the
matrix runner. It estimates output tokens/s only over client-side full
concurrency generation windows, so fixed-total-request long-tail drain is
visible separately from full-batch behavior. In the paired bs64 run above,
SR24 is still slower than dense under the full-batch metric (`4943.323` vs
`5313.360`), so reducing CPU-side mask-state synchronization alone is not the
main remaining bottleneck.

A draft-only dynamic MLP31 ablation disables non-draft residual correction:

```bash
--no-sr24-selective-correct-non-draft \
--sr24-selective-residual-policy all_if_any_low
```

It improves fixed-total end-to-end elapsed time but not full-batch kernel
throughput:

| check | result | result root |
| --- | --- | --- |
| bs64, fixed 128, `math_reasoning`, static mixed | total/full-batch `3275.941`/`5074.367` output tok/s | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dynamic_mlp31_allifanylow_static_mixed_draftonly_fullbatch_bs64_max2048_20260623` |
| same, repeat=3 paired with dense | median total dense/SR24 `2379.161`/`3568.214` (`1.50x`); median full-batch dense/SR24 `5229.447`/`5050.936` (`0.97x`) | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_draftonly_static_mixed_repeat3_bs64_max2048_20260623` |
| same, `static_mask_state=auto` | total/full-batch `3240.405`/`5011.299`; mask remained `mixed`, so no residual kernel was skipped | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dynamic_mlp31_allifanylow_draftonly_auto_maskstate_fullbatch_bs64_max2048_20260623` |
| existing compact/eager low-confidence path | total/full-batch `1875.327`/`4688.304`, CUDA graph `{"NONE":1984}`, residual draft fraction `0.444478` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lowconfidence_draftonly_compact_eager_bs64_max2048_20260623` |
| draft-only GSM8K-100 | `0.79` flexible exact, previous same-sample dense reference `0.80` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dynamic_mlp31_allifanylow_draftonly_gsm8k100_quality_20260623` |
| draft-only Minerva/IFEval-100 | Minerva `0.39`; IFEval prompt strict/loose `0.47`/`0.49`, inst strict/loose `0.5890`/`0.6135` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dynamic_mlp31_allifanylow_draftonly_minerva_ifeval100_quality_20260623` |

Follow-up CPU-sync/logging and graph-friendly bucket ablations on 2026-06-23
used the same draft-only MLP31 config, bs64, fixed 128 requests,
`math_reasoning`, K=8, and `max_tokens=2048`:

| case | total tok/s | full-batch tok/s | note | result root |
| --- | ---: | ---: | --- | --- |
| bucket disabled, stats interval 32 | `3702.503` | `5105.517` | same-code current baseline | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_draftonly_bucket0_current_bs64_max2048_20260623` |
| `SPECLINK_SR24_RESIDUAL_BUCKET_SIZE=256` | `3292.338` | `5103.180` | fixed-size GPU `topk` bucket, approximate if true residual rows exceed 256 | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_draftonly_bucket256_bs64_max2048_20260623` |
| `SPECLINK_SR24_RESIDUAL_BUCKET_SIZE=512` | `3270.578` | `5052.655` | fixed-size GPU `topk` bucket | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_draftonly_bucket512_bs64_max2048_20260623` |
| bucket disabled, stats interval 0 | `3265.028` | `5065.574` | disables periodic SR24 JSONL flush; one earlier typo run under `sr24_draftonly_bucket0_stats0_bs64_max2048_20260623` is invalid and should be ignored | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_draftonly_bucket0_stats0_correct_bs64_max2048_20260623` |
| bucket disabled, runtime stats disabled | `3250.719` | `5019.858` | `SPECLINK_SR24_DISABLE_RUNTIME_STATS=1`, skips verify summary and CUDA graph counter updates | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_draftonly_disable_runtime_stats_bs64_max2048_20260623` |

`--sr24-residual-bucket-size` / `SPECLINK_SR24_RESIDUAL_BUCKET_SIZE` is kept as
an explicit ablation only. It computes residual correction for a fixed-size GPU
`topk` bucket from the mixed residual mask, which avoids CPU row-count sync but
adds `topk/index_select/index_add` overhead and can drop residual rows when the
true residual count exceeds the bucket. It did not improve full-batch
throughput in the tested bs64 run. Setting `--sr24-stats-interval 0` or
`--sr24-disable-runtime-stats` also did not improve full-batch throughput. Do
not treat these CPU/Python bookkeeping changes as the next primary optimization
path.

Additional CPU-sync/draft-score cleanup on 2026-06-23:

- `all_corrected_24` with `SPECLINK_SR24_ALL_CORRECTED_DENSE_FASTPATH=1` now
  bypasses SR24 Linear hooks, verify residual-mask construction, proposer
  draft-score/log-softmax recording, and CUDA graph mode counter updates. The
  static attach event reports `linear_hooks_enabled=false` and
  `draft_scores_enabled=false`.
- Same-shape fixed-request repeat=3, bs64, `math_reasoning`, K=8,
  `max_tokens=2048`, fixed 128 requests: dense median total/full-batch was
  `2402.678`/`5319.751`; `all_corrected_24` median total/full-batch was
  `2395.876`/`5316.881`. Treat this as dense-equivalent overhead after cleanup,
  not a sparse speedup. Result root:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_no_draft_scores_fixed128_repeat3_bs64_max2048_20260623`.
- The earlier no-mask repeat=3 before this cleanup had
  `all_corrected_24` median full-batch `5273.141` vs dense `5336.590`, with
  CUDA graph counter events still enabled. The cleanup mostly removed residual
  instrumentation overhead and made the densefastpath diagnostic behave like
  dense EAGLE3.

Gate-only `speclink_t08` screening on 2026-06-23:

| case | dense total tok/s | SR24 total tok/s | dense steady tok/s | SR24 steady tok/s | dense full-batch tok/s | SR24 full-batch tok/s | interpretation | result root |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `accuracy_gate_only`, fixed requests, repeat=3 | `2407.183` | `3530.215` | `2407.183` | `3530.215` | `5338.236` | `5154.459` | fixed-request completion is faster, but full-batch window is not faster | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_gate_only_repeat3_bs64_max2048_20260623` |
| `accuracy_gate_only`, continuous 60s | `4216.908` | `4217.232` | `4410.355` | `4287.157` | `4562.110` | `4445.357` | no stable serving throughput win under replenished load | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_gate_only_continuous_bs64_max2048_20260623` |
| `accuracy_down_only`, fixed-request screen | `2403.108` | `3358.359` | `2403.108` | `3358.359` | `5331.842` | `4959.818` | slower than gate-only and worse full-batch | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_down_only_bs64_max2048_screen_20260623` |

Latest short continuous closed-loop cleanup checks, Llama-3.1-8B + EAGLE3
K=8, `math_reasoning`, `max_tokens=2048`, `prompt_limit=128`:

| case | bs | dense steady/full-batch tok/s | SR24 steady/full-batch tok/s | speedup steady/full | note | result root |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `accuracy_gate_only`, runtime stats on | 64 | `4465.277` / `4410.875` | `4368.445` / `4329.105` | `0.978x` / `0.981x` | `draft_scores_enabled=true` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_t08_cpu_sync_stats_on_bs64_20260623` |
| `accuracy_gate_only`, runtime stats off | 64 | `4465.277` / `4410.875` | `4381.175` / `4345.061` | `0.981x` / `0.985x` | CPU runtime stats are not the main bottleneck | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_t08_cpu_sync_stats_off_bs64_20260623` |
| `accuracy_gate_only`, no unused draft scores | 64 | `4465.277` / `4410.875` | `4419.764` / `4385.362` | `0.990x` / `0.994x` | `draft_scores_enabled=false` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_t08_gate_only_nodraftscores_bs64_20260623` |
| `accuracy_gate_only`, no unused draft scores | 128 | `6205.910` / `6134.928` | `6318.281` / `6178.253` | `1.018x` / `1.007x` | small positive but not the `1.2x` target | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_t08_gate_only_nodraftscores_bs128_20260623` |
| `throughput_aggressive`, no unused draft scores | 128 | `6205.910` / `6134.928` | `6194.727` / `6063.840` | `0.998x` / `0.988x` | `down_proj=30,31;gate_up_proj=31`; more base-only work did not help | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_t08_throughput_aggressive_nodraftscores_bs128_20260623` |
| `accuracy_first`, no unused draft scores | 64 | `4465.277` / `4410.875` | `4381.301` / `4288.289` | `0.981x` / `0.972x` | `gate_up_proj=31;down_proj=31`, slower than gate-only | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_t08_accuracy_first_nodraftscores_bs64_20260623` |
| `accuracy_first`, no reduce-sync closed-loop ablation | 64 | `4361.892` / `4385.659` | `4403.228` / `4338.516` | `1.009x` / `0.989x` | manual no `--sr24-reduce-cpu-sync`; same conclusion, near parity | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_t08_accuracy_first_no_reduce_closedloop_bs64_rerun_20260623` |
| `throughput_aggressive`, closed-loop | 64 | `4360.658` / `4441.534` | `4381.533` / `4296.638` | `1.005x` / `0.968x` | `gate_up_proj=31;down_proj=30,31`; more sparse tail still does not overcome compile/path overhead | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_t08_throughput_aggressive_closedloop_bs64_20260623` |
| `accuracy_first`, no unused draft scores | 128 | `6205.910` / `6134.928` | `6184.425` / `6071.368` | `0.997x` / `0.990x` | no continuous win | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_t08_accuracy_first_nodraftscores_bs128_20260623` |
| `accuracy_gate_only`, direct cuSPARSELt | 128 | `6205.910` / `6134.928` | `6305.466` / `6169.429` | `1.016x` / `1.006x` | `SPECLINK_SR24_DIRECT_CSLT_LINEAR=1`, not better than default no-draft path | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_t08_gate_only_directcslt_bs128_20260623` |
| minimal `gate_up_proj=31` only | 128 | `6205.910` / `6134.928` | `6315.818` / `6171.566` | `1.018x` / `1.006x` | attaches only one SR24 module; cleaner storage, no runtime gain over preset | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_t08_minimal_gate_bs128_20260623` |

Static stats-off early return was also measured under
`sr24_speclink_t08_staticfast_nodraftscores_bs64_20260623` and
`sr24_speclink_t08_staticfast_nodraftscores_bs128_20260623`; it was
noise-level and is not kept as the recommended code path.

The current all-corrected sparse operator conclusion is from
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_operator_probe_current_20260623`,
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_operator_probe_allcorrected_graph_20260623`,
and
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_operator_probe_allcorrected_graph_large_rows_20260623`.
Graph capture removes most Python/CPU launch overhead: eager exact
`all_corrected_24` is around `1.7-2.7 ms`, while captured base+residual sparse
GEMMs fall to about `0.10-0.52 ms`. That is still not a stable win over dense:
for example, `rows=512,out=4096,in=4096` is `0.103 ms` captured all-corrected
versus `0.096 ms` dense, and `rows=2048,out=6144,in=4096` is `0.521 ms`
captured all-corrected versus `0.476 ms` dense. Direct cuSPARSELt two-GEMM
composition is slower. A follow-up probe using real Llama-3.1-8B MLP shapes is
at
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_operator_probe_llama_mlp_shapes_20260623`.
It shows base-only sparse can be faster for the large tail MLP Linears under
CUDA Graph replay: `gate_up_proj` shape `28672x4096`, rows `512`, is
`0.353 ms` sparse graph versus `0.541 ms` dense, and `down_proj` shape
`4096x14336`, rows `512`, is `0.168 ms` sparse graph versus `0.292 ms` dense.
However, one full MLP layer saves only about `0.31 ms` at rows `512`
(`0.833 ms` dense Linears versus `0.521 ms` sparse Linears), so changing only
the last one or two MLP leaves cannot produce a stable model-level `1.2x`
continuous full-batch speedup. Tail-sparse SR24 can now coexist with default
vLLM compile when `SPECLINK_SR24_DIRECT_CSLT_LINEAR=1`: the direct cuSPARSELt
path uses an opaque `torch.library.custom_op` boundary for the sparse Linear, so
Inductor no longer lowers it into an unsupported `_cslt_sparse_mm` dense-input
layout during CUDA graph profiling. This fixes the earlier
`cusparseLtDenseDescriptorInit(... CUSPARSE_ORDER_ROW)` startup failure, but the
opaque sparse op boundary still limits end-to-end speedup.

Default-compile opaque direct-cuSPARSELt checks on 2026-06-23, Llama-3.1-8B +
EAGLE3 K=8, `math_reasoning`, bs64:

| case | requests/max tokens | dense steady/full-batch tok/s | SR24 steady/full-batch tok/s | speedup steady/full | result root |
| --- | --- | ---: | ---: | ---: | --- |
| `accuracy_first`, short | 64 / 256 | `2348.371` / `3499.417` | `2486.760` / `3692.173` | `1.059x` / `1.055x` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_first_defaultcompile_opaque_vs_eagle3_bs64_20260623` |
| `accuracy_first`, no reduce-sync ablation | 64 / 256 | same baseline | `2472.964` / `3566.533` | `1.053x` / `1.019x` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_first_defaultcompile_opaque_no_reduce_ablation_bs64_20260623` |
| `throughput_aggressive` | 64 / 256 | same baseline | `2440.139` / `3482.563` | `1.039x` / `0.995x` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_throughput_aggressive_defaultcompile_opaque_bs64_20260623` |
| `gate_up_proj=31`, longer | 256 / 256 | `2844.650` / `3059.655` | `2953.753` / `3181.041` | `1.038x` / `1.040x` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_gate_only_defaultcompile_opaque_vs_eagle3_bs64_256req_20260623` |
| `gate_up_proj=30-31`, longer | 256 / 256 | same long baseline | `2987.876` / `3206.333` | `1.050x` / `1.048x` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_30_31_defaultcompile_opaque_bs64_256req_20260623` |
| `gate_up_proj=28-31`, longer | 256 / 256 | same long baseline | `2946.227` / `3210.610` | `1.036x` / `1.049x` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_28_31_defaultcompile_opaque_bs64_256req_20260623` |
| `gate_up_proj=30-31`, `transpose_result=True` | 256 / 256 | same long baseline | `2971.195` / `3211.911` | `1.045x` / `1.050x` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_30_31_defaultcompile_opaque_transpose_true_bs64_256req_20260623` |

The current best default-compile long run is `gate_up_proj=30-31` at about
`1.05x`, not `1.2x`. Reducing CPU synchronization is a useful ablation/default
for graph-safe static paths, but it changes the short `accuracy_first` result by
only a few percent and is not the main bottleneck. Keep
`SPECLINK_SR24_ALL_CORRECTED_DENSE_FASTPATH=1` as the diagnostic
dense-equivalent path until there is a fused packed sparse base+residual kernel;
composing existing sparse calls and opaque single sparse ops is not enough for
the target speedup. A CUTLASS semi-structured backend probe is not viable on the
RTX 5090 stack: PyTorch reports `sparse_semi_structured_mad_op` is supported
only on compute capability 8.x. A direct cuSPARSELt `transpose_result=True`
microprobe reduced one `gate_up_proj` sparse op from about `1.12ms` to
`0.97ms`, but the end-to-end long run stayed around `1.05x`.

Existing accuracy roots for the leaf-only split:

- `accuracy_gate_only`: GSM8K-100 `0.7700`, Minerva-100 `0.3800`;
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_gate_only_gsm8k_minerva100_2048_20260623`.
- `accuracy_gate_only`, paired after IFEval rerun: GSM8K-100 unchanged
  `0.7700 -> 0.7700`, Minerva-100 `0.3900 -> 0.3800`, IFEval prompt
  strict/loose `0.4000 -> 0.4100` and `0.4500 -> 0.4600`, but IFEval
  instruction-level loose/strict dropped `2.45pp` on a heavily clipped subset;
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_gate_only_paired_gsm8k_minerva_ifeval100_2048_20260623`.
- `accuracy_down_only`: paired dense/SR24 GSM8K-100 `0.7700 -> 0.8200`,
  Minerva-100 `0.4000 -> 0.3900`, but IFEval-100 prompt strict/loose dropped
  `0.4600 -> 0.4000` and `0.5100 -> 0.4200`;
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_down_only_paired_gsm8k_minerva_ifeval100_2048_20260623`.

Practical conclusion: reducing CPU synchronization is useful and should stay as
an ablation/default for the graph-safe static paths, but the current gate/down
leaf-only candidates do not produce a clear continuous full-batch throughput
win. Future SR24 speed work should target GPU-side sparse/residual operator
cost and request-level scheduling effects separately; do not claim stable
serving speedup from fixed-request total tok/s alone.

Interpretation: with `reduce_cpu_sync + static mixed +
torch_sparse` residual, the current mixed path still computes residual sparse
matmul for all rows and masks the output, so disabling non-draft correction
does not reduce full-batch residual kernel cost. The existing compact/eager
path really reduces residual rows, but loses CUDA Graph replay and is slower.
The next viable speed path is a lower-overhead custom GPU sparse base+residual
kernel or packed residual path; the fixed GPU `topk` bucket and CPU-side
logging/sync cleanup should be treated as ablations, not primary optimization
paths.

The layer `30-31` accuracy result root is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_mlp_last2_base_attn_residual_gsm8k50_20260623
```

Its paired GSM8K 50 result had 4 dense-correct losses and 3 dense-wrong gains
(`dense_correct_retention=0.8857`). Treat it as a promising candidate, not
final quality evidence; it still needs larger accuracy coverage and additional
datasets before publication. The all-MLP base-only variant must not be used as
a quality path despite its high throughput.

Full-model `SPECLINK_SR24_RESIDUAL_BACKEND=torch_sparse` still does not load on
the 32GB RTX 5090. Reordering base replacement before residual construction and
switching residual fill to `masked_scatter_` reduced the failing temporary
allocation from about `1.31GiB` to `896MiB`, but the full model still OOMs while
building the residual sparse tensor. Attention-only `torch_sparse` residual
loads and runs.

## Current Run Notes

EAGLE3 completed successfully with the local dataset:

```text
Output dir: examples/evaluate/eval-guidellm/out_eagle3_smoke_localds_math
Output throughput: 1086.1 generated tokens/s
Total throughput: 1120.7 tokens/s
Weighted acceptance rates: [0.707 0.476 0.287]
```

P-EAGLE successfully started after the compatibility patch and produced vLLM
`SpecDecoding metrics`, including four-position acceptance rates. The run was
interrupted before GuideLLM wrote `guidellm_results.json`; rerun the P-EAGLE
smoke command above for a complete result file.

## Troubleshooting

If PEAGLE startup fails with:

```text
Expected one of: {'eagle3': ..., 'dflash': ...}
```

then the vLLM PEAGLE compatibility patch is missing or vLLM was reinstalled.
With the current workflow, fix this in `speculators/vllm`, reinstall editable if
needed, and verify that `vllm.__file__` points to the vendored source tree.

If vLLM fails with:

```text
Failed to infer device type
```

check CUDA visibility:

```bash
conda run -n spec python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

For Codex-run GPU commands, use escalated execution; non-escalated sandbox
commands may not expose `/dev/nvidia*`.

If the Hugging Face dataset download path starts with `path=`, use the patched
`scripts/run_guidellm.sh` or pass a local `.jsonl` path directly.

## SR24 Gate-Up Follow-Up Notes, 2026-06-23

Current SR24 optimization status is tracked in:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results/speclink_sr24_status_20260622/SR24_CURRENT_ANALYSIS_20260623.md
```

Recent default-compile opaque cuSPARSELt runs show that aggressive base-only
`gate_up_proj` can hit the speed target but not the accuracy target:

- `gate_up_proj=8-31`: bs64/full-batch `3764.695` tok/s vs dense `3059.655`
  (`1.230x`), but GSM8K-50 drops `0.7400 -> 0.4400`.
- `gate_up_proj=16-31`: bs64/full-batch `3618.300` tok/s (`1.183x`), but
  GSM8K-50 still drops `0.7400 -> 0.6200`.
- `gate_up_proj=12-31`: bs64/full-batch `3614.654` tok/s (`1.181x`) and
  steady `3079.409` tok/s, so it did not improve over `16-31`; C4-mask
  eager accuracy also dropped to GSM8K-50 `0.5600` and Minerva-50 `0.1800`.
  The default-compile accuracy startup hit a cuSPARSELt
  `cusparseLtDenseDescriptorInit` shape failure during cudagraph profiling, so
  the recorded quality run is eager and should be used only for correctness.
- `--sr24-gate-up-split up_sparse` keeps the gate half dense and sparsifies
  only the up half. It runs correctly and is recorded in summaries as
  `sr24_gate_up_split=up_sparse`, but `gate_up_proj=8-31` only reached
  full-batch `3106.466` tok/s and GSM8K-50 `0.5800`.
- A sensitivity-selected 20-layer `gate_up_proj` set based on old whole-layer
  accuracy sensitivity reached GSM8K-50 `0.5200`; whole-layer sensitivity does
  not transfer cleanly to gate-up-only sparse selection.
- The current dynamic quality fix uses
  `--sr24-selective-residual-policy critical_prefix` plus
  `--sr24-selective-extra-after-low 4` with the C4 activation-aware mask and
  `gate_up_proj=16-31`. It keeps the prefix through the first low-confidence
  draft row and corrects four more draft rows after that. On 50-example
  manifests it reached GSM8K `0.7400` with draft residual fraction `0.7093`,
  and Minerva `0.3200` with draft residual fraction `0.6914`.
- `--sr24-selective-extra-after-low 3` is a better current quality/speed
  candidate on GSM8K-50: it also reached `0.7400` while reducing draft residual
  fraction to `0.5882`. The matching bs64 throughput with no mask-state sync
  and stats off was still only `2310.882` steady tok/s, so lower residual
  fraction alone did not fix the mixed-path overhead.
- CPU synchronization is a large penalty only in the exact-stats/eager dynamic
  path: the same dynamic policy at bs64, max tokens 256, reached `1537.430`
  steady tok/s with exact stats, `2295.671` with `--sr24-reduce-cpu-sync`, and
  `2338.383` with `--no-sr24-sync-mask-state --sr24-disable-runtime-stats`.
  This confirms reduced sync should stay as an ablation/default, but the
  dynamic mixed residual path is still slower than the dense EAGLE3 baseline.
- Later cleanup gates construction of per-step SR24 verify record dictionaries
  behind `runtime_stats_enabled()`. With `--sr24-disable-runtime-stats`, the
  hot path no longer builds the JSON/logging record only to discard it. This is
  CPU bookkeeping cleanup, not a new GPU kernel. Smoke:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_stats_off_record_gating_smoke_bs8_20260623`.
- Static stats-off `all_residual/no_residual` now returns before request-count
  scans in `build_verify_residual_mask()`. The aligned `gate_up_proj=16-31`
  default-compile ablation reached steady/full-batch `3148.485` / `3644.438`
  tok/s, versus the prior `3315.973` / `3618.300`; this is noise-level rather
  than a real speed win.
- Dynamic `--sr24-reduce-cpu-sync --sr24-disable-runtime-stats` now also skips
  exact per-step draft/residual token counting inside `build_verify_residual_mask()`.
  Non-`reduce_cpu_sync` runs still keep the old exact count path for mask-state
  decisions. This is a CPU bookkeeping/sync ablation and has passed the
  lightweight `check_speclink_sr24_correctness.py` smoke. The matched bs64
  bucket-128 throughput rerun reached steady/full-batch `2671.362` / `2889.659`
  tok/s versus the prior `2679.322` / `2916.491`, so exact-count skipping is
  noise-level and not a remaining speed solution.
- Dynamic SR24 proposer confidence now uses
  `selected_logit - logsumexp(logits)` instead of materializing full
  `log_softmax(logits)` before gathering the selected draft token probability.
  This preserves the same probability semantics while reducing proposer-side
  intermediate tensor pressure. In the matched `gate_up_proj=16-31`,
  `extra_after_low=3`, bs64 fixed-256-request ablation, steady/full-batch output
  tok/s changed from `2310.882` / `2503.430` to `2326.838` / `2519.845`
  (`1.007x`), so keep it as low-risk cleanup but do not treat it as the main
  bottleneck fix:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_dynamic_logsumexp_scores_bs64_256req_20260623`.
  A follow-up SR24-only streamed-score queue avoided retaining per-position
  logits lists but was slower, steady/full-batch `2304.840` / `2484.273`, and
  was reverted. Do not retry it without new profiler evidence:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_dynamic_streamed_scores_bs64_256req_20260623`.
- Existing capped residual-row ablations improve speed but still miss dense:
  `extra_after_low=3`, `SPECLINK_SR24_RESIDUAL_BACKEND=dense_rows`, and
  bucket sizes `256/128/64` reached steady `2581.900` / `2690.308` /
  `2661.109` tok/s respectively at bs64. Bucket `128` was best, but remains
  below the dense long baseline `2844.650` tok/s and is an approximate quality
  path because the current bucket chooses from a bool mask without confidence
  priority.
- A priority-aware bucket path is implemented behind
  `--sr24-residual-bucket-priority`. It ranks non-draft rows, missing-score
  rows, low-confidence draft rows, and early draft rows before applying the
  capped bucket. On GSM8K-50, `dense_rows` bucket `128` with priority reached
  `0.7600`, but its bs64 steady throughput was only `2547.097` tok/s before
  bucket caching and `2588.181` tok/s after caching bucket rows per verify
  step. This is still below the non-priority bucket `128` and dense baseline.
  Treat it as evidence that row priority can preserve quality, not as the
  current speed solution.
- Bucket rows are now computed once in `build_verify_residual_mask()` and reused
  by all SR24 Linear hooks in the same verify step. This removes repeated
  per-layer `topk`, but the non-priority `dense_rows` bucket `128` remained
  about the same (`2679.322` steady tok/s after caching), so repeated top-k was
  not the main mixed-path bottleneck.
- Widening dynamic sparse coverage from `gate_up_proj=16-31` to `8-31` with the
  same `extra_after_low=3`, `dense_rows` bucket `128` configuration reached only
  `2627.724` steady tok/s. Dynamic residual correction cost dominates enough
  that simply sparsifying more gate-up layers does not recover the static
  `8-31` speedup.
- `--sr24-route-bucket-rows` is implemented as a negative ablation for
  `torch_sparse + dense_rows` bucket paths. It routes bucket rows directly to
  dense Linear and only non-bucket rows to sparse base Linear, but splitting the
  work into two smaller GEMMs was slower: `2438.028` steady tok/s for the same
  bs64 bucket-128 setup. Keep it off by default.
- The focused mixed-bucket microbenchmark
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mixed_bucket_probe_gateup512_graph_20260623`
  shows why dynamic buckets miss the `1.2x` goal. For one Llama
  `gate_up_proj` shape (`rows=512,out=28672,in=4096`), dense graph is about
  `0.539ms` and base sparse graph is `0.356ms`, but serving-safe bucket128
  `base sparse all rows + dense selected rows + delta scatter` is `0.648ms`,
  about `1.20x` the dense cost. Routed bucket is also slower (`0.656ms`). This
  means the mixed residual dispatch removes the sparse GEMM gain before model
  effects are considered.
- The lower-bound rerun
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mixed_bucket_probe_gateup512_lowerbound_20260623`
  adds no-scatter timings. Serving-safe bucket128 is `0.6464ms`; its
  no-scatter lower bound is still `0.5767ms`, already `1.07x` dense. Bucket64
  no-scatter is about parity with dense (`0.5411ms` vs `0.5368ms`). Routed
  no-scatter is faster than dense for bucket128/256 (`0.4126ms` / `0.4732ms`),
  but output assembly adds `0.2427ms` / `0.1887ms`, making the full routed path
  slower than dense. This says a PyTorch serving-side rearrangement is unlikely
  to be enough; the useful direction is a fused GPU route/write operator or a
  different static-quality strategy.
- `--sr24-triton-route-assembly` / `SPECLINK_SR24_TRITON_ROUTE_ASSEMBLY=1` is
  implemented as an experimental routed-bucket assembly ablation only. It
  replaces the two PyTorch `index_copy_` output-assembly calls with a Triton
  scatter/copy kernel. Microbench for the same gate-up shape improved routed
  bucket128 from `0.6557ms` to about `0.543ms`, near dense `0.540ms`, and
  block sizes `4096/8192` did not materially improve over `1024`. However, the
  serving run was slower: steady/full-batch `2366.445` / `2536.269` tok/s,
  worse than both old routed bucket128 (`2438.028` / `2591.430`) and non-routed
  bucket128 (`2679.322` / `2916.491`). Keep it off by default and treat it as
  negative evidence that standalone output assembly replacement is not enough.
- A second microbenchmark-only Triton bucket override kernel was tested at
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mixed_bucket_triton_override_contig_gateup512_20260623`.
  It starts from `base_output.clone()` and overwrites bucket rows with dense
  outputs, avoiding `base_output.index_select`, delta materialization, and
  `index_add_`. After forcing `base_output.contiguous()`, correctness matched
  the existing paths (`max diff <= 2`), but graph timings were worse than the
  current bucket path: bucket64/128/256 were `0.6803` / `0.6932` / `0.8252ms`
  versus bucket-delta `0.5867` / `0.6476` / `0.8499ms` and bucket-replace
  `0.5878` / `0.6309` / `0.8247ms`. Do not wire this override kernel into
  serving.
- `batch_all_if_any_low` is implemented as a coarse all-or-none dynamic policy:
  if any draft token in the verify step is below threshold, every scheduled row
  is corrected; otherwise the whole step is base-only. It avoids row-level
  mixed residual and, with `dense_rows`, produced only all/no graph states
  (`{"FULL":313,"NONE":135}` in the stats-on run), but it did not beat the
  existing bucket path. Stats-on steady/full-batch was `2622.564` / `2848.031`
  tok/s; stats-off was `2671.353` / `2895.429` tok/s. The existing dense-row
  bucket128 path was `2690.308` / `2921.710`, and the same-shape dense EAGLE3
  baseline was `2844.650` / `3059.655`. Treat this as a negative speed
  ablation and do not spend accuracy runs on it without a new hypothesis:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_batch_all_if_any_low_denserows_bs64_256req_20260623`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_batch_all_if_any_low_denserows_nostats_bs64_256req_20260623`.
- Additional CPU-sync ablation on the current dynamic GSM8K quality candidate
  (`gate_up_proj=16-31`, `critical_prefix`, `extra_after_low=3`, C4 mask,
  runtime stats disabled) showed that keeping the once-per-step mask-state
  sync is slightly slower than the no-sync logsumexp path. The synchronized run
  reached steady/full-batch `2302.917` / `2479.214` tok/s, versus
  `2326.838` / `2519.845` for the no-mask-state-sync logsumexp run. Keep
  reduced CPU sync as a necessary throughput ablation and graph-safety hygiene,
  but do not treat CPU sync as the main remaining bottleneck:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_dynamic_syncmask_nostats_bs64_256req_20260624`.
- A GSM8K-task-aware activation-RMS SR24 mask was calibrated from 128 GSM8K
  prompts at
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gsm8k128_activation_rms_cache_20260624`.
  It does not fix the static layer-range quality problem: both static
  `gate_up_proj=8-31` and `gate_up_proj=16-31` measured GSM8K-50 exact match
  `0.6400`, with SR24 draft residual fraction about `0.2108`, in
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_8_31_gsm8k128mask_accuracy_gsm8k50_20260624`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_gsm8k128mask_accuracy_gsm8k50_20260624`.
  Do not expand this static task-aware layer-range path unless a new selection
  rule is added beyond layer range plus activation-RMS mask.
- `gate_sparse` split was attempted as the complementary gate/up split, but the
  current runner/SR24 environment semantics cannot express
  `SPECLINK_SR24_TARGET_LEAFS=gate_up_proj` with intentionally empty residual
  leafs. If the residual env var is absent, SR24 defaults residual leafs to the
  target leafs and correctly rejects gate/up split because it is only supported
  for base-only `gate_up_proj` targets. Do not modify the CLI for this
  low-priority branch unless gate/up split becomes a primary candidate again.
- Turning off residual correction for non-draft/bonus rows is a useful dynamic
  cost cut. With `gate_up_proj=16-31`, `critical_prefix + extra_after_low=3`,
  C4 mask, no mask-state sync, and runtime stats disabled, the uncapped
  torch-sparse residual path improved from `2326.838` / `2519.845` to
  `2686.284` / `2909.975` steady/full-batch tok/s. GSM8K-50 exact match was
  `0.7200`, and `residual_non_draft_fraction=0.0`:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_no_nondraft_accuracy_gsm8k50_20260624`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_no_nondraft_no_mask_sync_nostats_bs64_256req_20260624`.
- The best current live serving candidate is no-nondraft correction plus
  `dense_rows` capped residual rows. On Llama math bs64 fixed-256 requests:
  bucket32 reached steady/full-batch `2918.019` / `3240.510` tok/s and
  GSM8K-50 `0.7600`; bucket64 reached `2918.866` / `3180.026` tok/s; bucket128
  reached `2892.626` / `3161.221` tok/s and GSM8K-50 `0.7600`. A same-condition
  dense EAGLE3 baseline with bs64, 256 fixed requests, 256 max tokens, and the
  same prompt limit reached `2753.873` / `3076.812` tok/s. The current
  slice-write bucket32 candidate is therefore `1.064x` steady and `1.055x`
  full-batch, quality-plausible but still below the requested `1.2x`:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dense_eagle3_baseline_bs64_256req_20260624`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_no_nondraft_denserows_bucket32_bs64_256req_20260624`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_no_nondraft_denserows_bucket32_accuracy_gsm8k50_20260624`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_no_nondraft_denserows_bucket64_bs64_256req_20260624`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_no_nondraft_denserows_bucket128_bs64_256req_20260624`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_no_nondraft_denserows_bucket128_accuracy_gsm8k50_20260624`.
- A 2026-06-24 CPU-side synchronization follow-up replaced repeated SR24
  verify-context `ContextVar.get()` calls in the Linear hot path with a
  begin/end verify-plan cache. Correctness still passes. On the same bucket32
  candidate with direct cuSPARSELt enabled, steady/full-batch moved from
  `2929.027` / `3247.004` to `2958.197` / `3244.287` tok/s. This is only a
  small total-throughput improvement (`1.074x` steady and `1.054x` full-batch
  over the same-condition dense baseline), so CPU context lookup is not the
  main remaining bottleneck:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_bucket32_direct_fastctx_bs64_256req_20260624`.
- Direct cuSPARSELt plus `--sr24-default-vllm-compile` now starts for the
  bucket32 candidate, but it is not a win: `2886.903` / `3329.192` tok/s with
  lower average GPU utilization. Default compile without the direct custom op
  still fails during startup with a cuSPARSELt dense-descriptor unsupported
  error after the ContextVar blocker is removed. Keep default-compile sparse
  paths as diagnostics, not the current throughput setting:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_bucket32_direct_defaultcompile_fastctx_bs64_256req_20260624`.
- `extra_after_low=2` was tested as a lower-residual dynamic policy because it
  had an earlier GSM8K-50 score of `0.7000`, but serving throughput was worse:
  `2913.446` / `3181.611` tok/s. Keep `extra_after_low=3` as the current
  quality/speed candidate:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra2_bucket32_direct_fastctx_bs64_256req_20260624`.
- A route-bucket attempt to avoid computing the sparse base for dense bucket
  rows, including a temporary cached-base-rows implementation, was still
  negative: `2653.610` / `2853.457` tok/s. This confirms that the routed path's
  two-GEMM plus output-assembly/scatter structure dominates over the saved
  `base_rows` computation. The cached-base-rows code was removed; keep
  `--sr24-route-bucket-rows` off unless it is replaced by a fused operator:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_bucket32_route_cachedbaserows_bs64_256req_20260624`.
- Additional CPU-sync/draft-score ablations on this bucket32 candidate show
  the main gap is not CPU sync alone. The current no-sync/no-stats path is
  steady/full-batch `2918.019` / `3240.510` tok/s. Enabling the once-per-step
  mask-state sync drops to `2869.783` / `3190.488`; enabling runtime stats
  while keeping no-sync gives `2942.939` / `3211.712`, a single-run steady
  fluctuation with lower full-batch throughput; a temporary top1/top2
  margin-score draft confidence proxy reached only `2904.861` / `3200.656` and
  was removed from code. A `bucket_replace` ablation, which tried to overwrite
  full-residual bucket rows directly with dense output instead of the
  delta/index_add composition, reached only `2900.861` / `3178.182`; it was also
  removed from code because it added an all-residual bucket check without
  improving serving throughput. Keep `--sr24-reduce-cpu-sync`,
  `--no-sr24-sync-mask-state`, and `--sr24-disable-runtime-stats` for the
  current throughput candidate. Do not re-add the margin-score proxy or
  bucket-replace path without new profiler evidence:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_no_nondraft_denserows_bucket32_sync_mask_state_bs64_256req_20260624`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_no_nondraft_denserows_bucket32_runtime_stats_bs64_256req_20260624`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_no_nondraft_denserows_bucket32_margin_score_bs64_256req_20260624`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_no_nondraft_denserows_bucket32_replace_bs64_256req_20260624`.
- High-concurrency and CPU-sync follow-up on 2026-06-24 confirms the same
  conclusion. At bs128, the same-condition dense EAGLE3 baseline reached
  total/full-batch `3036.459` / `3439.548` tok/s, while the current bucket32
  `speclink_t08` candidate reached only `3164.055` / `3642.509` tok/s
  (`1.042x` / `1.059x`) despite higher acceptance (`0.2602` vs `0.2167`).
  The CPU-sync-heavy ablation at bs64 dropped to `1862.744` / `2026.776`
  tok/s and `71.54%` GPU utilization, versus the fast-sync candidate
  `2958.197` / `3244.287` and `95.51%` GPU utilization. Keep
  `--sr24-reduce-cpu-sync`, `--no-sr24-sync-mask-state`,
  `--sr24-static-mask-buffer`, `--sr24-direct-cslt-linear`, and
  `--sr24-disable-runtime-stats` for the current serving candidate, but treat
  remaining speed loss as mixed sparse/residual GPU work rather than CPU sync.
  A correctness-clean attempt to remove `base_output.clone()` from the bucket
  correction path reached only `2939.654` / `3232.731` and was reverted:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dense_eagle3_baseline_bs128_256req_20260624`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_bucket32_direct_fastctx_bs128_256req_20260624`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_bucket32_direct_sync_ablation_bs64_256req_20260624`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_bucket32_direct_fastctx_inplace_bs64_256req_20260624`.
- The same bucket32 candidate with `--sr24-residual-backend torch_sparse`
  loaded and reduced storage/dense from `1.6250` to `1.1875`, but it was
  slower than `dense_rows`: `2797.579` / `3073.836` tok/s with `91.20%` GPU
  utilization. This shows a second cuSPARSELt residual small-GEMM is not enough
  to fix the mixed residual path; keep `dense_rows` for the current candidate
  and treat a real fused sparse/residual GPU operator as the next operator-level
  direction:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_bucket32_torchres_direct_fastctx_bs64_256req_20260624`.
- The SR24 verify-mask writer now uses contiguous slice writes instead of
  creating a per-request `torch.arange` row-index tensor for each draft span.
  This is semantics-preserving because vLLM scheduled draft rows are contiguous
  within each request. On the same Llama math bs64 fixed-256 bucket32 candidate,
  it improved steady/full-batch throughput slightly to `2929.027` / `3247.004`
  tok/s. Keep this cleanup, but do not treat it as solving the main `1.2x`
  target:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_no_nondraft_denserows_bucket32_slice_bs64_256req_20260624`.
- A broader vectorized batch mask-build ablation was also tested and removed.
  It stacked request scores and used one `index_copy_` over the whole batch, but
  reached only `2896.822` / `3214.501` tok/s because stack/row-index/index_copy
  overhead outweighed fewer small kernels:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_no_nondraft_denserows_bucket32_vector_mask_bs64_256req_20260624`.
- Lowering the dynamic threshold or shrinking the capped residual bucket below
  32 is also a negative quality direction for the current candidate. On the
  same GSM8K-20 manifest, current threshold `0.8` + bucket32 scored `0.7000`;
  threshold `0.75` + bucket32 scored `0.6000`; threshold `0.6` and `0.7`
  exploratory runs also scored `0.6000`; threshold `0.8` + bucket16 scored
  `0.6000`. Do not run throughput for these lower-quality points unless a new
  row-priority mechanism is added. The lm-eval runner now treats
  `runtime_stats_enabled=false` SR24 runs as valid when
  `speclink_sr24_stats.json` shows attached modules, because verify events are
  intentionally disabled in no-stats runs:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_no_nondraft_denserows_bucket32_t075_accuracy_gsm8k20_20260624`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_no_nondraft_denserows_bucket16_accuracy_gsm8k20_20260624`.
- SR24 attach logic now supports mixed layer ranges within one leaf: residual
  layer ids take priority; layers outside the residual set but inside
  `SPECLINK_SR24_BASE_ONLY_LAYER_IDS_BY_LEAF` attach as base-only sparse
  modules; other layers are skipped. This enables ablations such as
  `gate_up_proj=8-15` base-only plus `16-31` dynamic residual without changing
  existing base-only-only or residual-only semantics.
- The contiguous mixed-layer sweep is a negative quality/speed result. Adding
  static base-only early gate-up layers improves speed only modestly and hurts
  GSM8K-50 quickly: `8-15` base-only plus `16-31` residual reached
  `3027.920` / `3305.098` tok/s but only `0.6200` GSM8K-50; `8-29` base-only
  plus `30-31` residual reached `3103.683` / `3382.444` tok/s but only
  `0.6000`. Intermediate speed points were `3062.828` / `3344.932` for
  `8-23` base-only and `3092.606` / `3379.512` for `8-27` base-only. The speed
  plateaus around `1.10x` full-batch and quality collapses before `1.2x`, so do
  not continue this contiguous early-base-only sweep without a new noncontiguous
  layer/channel selection rule:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_8_15_baseonly_16_31_extra3_no_nondraft_denserows_bucket32_bs64_256req_20260624`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_8_15_baseonly_16_31_extra3_no_nondraft_denserows_bucket32_accuracy_gsm8k50_20260624`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_8_23_baseonly_24_31_extra3_no_nondraft_denserows_bucket32_bs64_256req_20260624`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_8_27_baseonly_28_31_extra3_no_nondraft_denserows_bucket32_bs64_256req_20260624`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_8_29_baseonly_30_31_extra3_no_nondraft_denserows_bucket32_bs64_256req_20260624`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_8_29_baseonly_30_31_extra3_no_nondraft_denserows_bucket32_accuracy_gsm8k50_20260624`.
- Non-contiguous single-layer quick screens did not find a useful early
  gate-up layer. On the same GSM8K-20 manifest, the current `16-31` residual
  bucket32 candidate scored `0.7000`; adding only layer 8 or only layer 14 as
  base-only scored `0.6500`; adding only layer 15 scored `0.7500`, but its
  bs64 fixed-256 throughput was `2914.100` / `3172.260`, below the current
  bucket32 candidate `2918.019` / `3240.510`. Do not spend more time on
  single-layer early gate-up base-only unless a better selection signal is
  added:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_no_nondraft_denserows_bucket32_accuracy_gsm8k20_20260624`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_single8_baseonly_16_31_extra3_no_nondraft_denserows_bucket32_accuracy_gsm8k20_20260624`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_single14_baseonly_16_31_extra3_no_nondraft_denserows_bucket32_accuracy_gsm8k20_20260624`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_single15_baseonly_16_31_extra3_no_nondraft_denserows_bucket32_accuracy_gsm8k20_20260624`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_single15_baseonly_16_31_extra3_no_nondraft_denserows_bucket32_bs64_256req_20260624`.
- A follow-up single-layer screen filled in layers `9-13`: layer 9 scored
  `0.6000`, 10 `0.7500`, 11 `0.6500`, 12 `0.7000`, and 13 `0.7000` on the
  same GSM8K-20 manifest. The non-contiguous combination `10,15` scored
  `0.8000`, but its bs64 fixed-256 throughput was only `2898.002` / `3182.339`
  tok/s, below the current slice-write bucket32 candidate `2929.027` /
  `3247.004`. The wider `10,12,13,15` set scored only `0.6000`. Treat these
  as evidence that layer-level selection is too coarse: adding two early layers
  is not enough speed, and adding more layers hurts quality quickly:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_single9_13_sensitivity_gsm8k20_20260624`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_noncontig_sensitivity_gsm8k20_20260624`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_10_15_baseonly_16_31_extra3_no_nondraft_denserows_bucket32_bs64_256req_20260624`.
- Layer `0-7` single-layer and combination screens reached the same conclusion.
  On GSM8K-20, layer 1 scored `0.7500`, layers `0/6/7` scored `0.7000`,
  layers `2/3/4` scored `0.6500`, and layer 5 scored `0.6000`. Combining
  strong single layers did not compose: `1,10,15` scored `0.6500`,
  `0,1,10,15` scored `0.7000`, and `1,6,7,10,15` scored `0.5500`. The
  `0,1,10,15` bs64 throughput was only `2946.734` / `3147.387` tok/s,
  below the current bucket32 candidate `2958.197` / `3244.287`. Do not keep
  spending time on layer-level static gate-up combinations without a finer
  channel/block-level selection signal:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_single0_7_sensitivity_gsm8k20_20260624`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_combo_static_sensitivity_gsm8k20_20260624`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_0_1_10_15_baseonly_16_31_extra3_denserows_bucket32_bs64_256req_20260624`.
- `gate_sparse` split now runs if passed with
  `--sr24-residual-target-leafs none`; passing an empty residual leaf string
  leaves SR24 at its default residual=target behavior. The corrected
  `gate_sparse` GSM8K-20 quick screen scored `0.6500`, below the same-manifest
  candidate baseline, so the complementary gate/up split is also a negative
  quality direction:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_8_31_gate_sparse_accuracy_gsm8k20_20260624`.
- Paired-channel `gate_up_proj` split is implemented as an explicit ablation
  only, not the current main path. Use
  `--sr24-gate-up-split channel_pair`,
  `--sr24-gate-up-channel-dense-fraction`, and
  `--sr24-gate-up-channel-strategy`; optional
  `--sr24-gate-up-channel-fused-act` sets
  `SPECLINK_SR24_GATE_UP_CHANNEL_FUSED_ACT=1`. The split keeps selected
  intermediate gate/up channel pairs dense, sparsifies the remaining pairs, and
  requires dense `down_proj`; it bypasses runtime column scatter by keeping a
  grouped activation layout and pre-permuting `down_proj` columns. The
  standalone microbench shows the matmuls alone are promising but the full path
  is not enough: dense fraction `0.125/0.25` gives about `1.17-1.18x` grouped
  full-MLP speedup, while restoring original gate/up row order with
  `index_copy_` is a net loss. The serving bs64 fixed-256 run was negative:
  `2729.732` / `3018.499` tok/s versus the same dense baseline
  `2753.873` / `3076.812`, and below the current bucket32 candidate
  `2958.197` / `3244.287`. The fused-activation serving variant also ran
  correctly but stayed negative: `2744.606` / `3003.816` tok/s. Do not continue
  this as the main optimization path unless a fused MLP kernel removes more of
  the concat/activation/down-projection overhead and shows a real end-to-end
  win:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_channel_split_microbench_20260624`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_channel_pair_smoke_20260624`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_channel_pair_fused_act_smoke_20260624`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_channel_pair_16_31_bs64_256req_20260624`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_channel_pair_fused_act_16_31_bs64_256req_20260624`.
- `run_llama31_vllm_fastdraft_smurfs_eagle3_matrix.py` fixed-request mode was
  corrected on 2026-06-24. Previously, `--fixed-total-requests` workers still
  stopped at the `warmup+measurement+cooldown` deadline, so long-output fixed
  runs could report fewer successful requests than requested. The worker loop
  now ignores that deadline in fixed mode and waits until the requested count is
  exhausted. For speed claims from fixed-request rows, require
  `successful_requests == max_requests`; older rows with fewer successful
  requests are truncated diagnostics. A strict bs64, fixed128,
  `max_tokens=2048` rerun of dense EAGLE3 versus the current bucket32
  `speclink_t08` candidate completed all 128 requests for both methods:
  dense `4080.093` / `4837.020` tok/s, `speclink_t08` `3643.271` /
  `4591.778` tok/s, so the current candidate is `0.893x` total/steady and
  `0.949x` full-batch versus dense under corrected fixed-request accounting:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dense_vs_gateup_16_31_extra3_bucket32_fixed128_max2048_full_20260624`.

Keep reduced CPU synchronization and stats/logging cleanup as ablations and
graph-safe hygiene, but do not treat CPU sync as the main bottleneck. The next
credible speed path is a graph/compile-friendly fused GPU sparse/residual
operator that avoids the current two-GEMM-plus-scatter bucket composition, or a
different quality strategy that can stay on static base-only sparse layers.
The mixed-bucket microbenchmark now reports no-scatter lower-bound timings and
explicit assembly-overhead columns so future runs can separate GEMM cost from
output assembly/scatter cost.
Simple contiguous-tail expansion, including `gate_up_proj=12-31`, current
gate/up split, current top-k bucket variants, and old
layer-sensitivity-selected gate-up sets should not be repeated as the primary
optimization path without a new hypothesis.

2026-06-24 follow-up: the stricter bs64 fixed-128 `max_tokens=2048` reruns
confirm that reduced CPU synchronization is only a small ablation-level win.
For `gate_up_proj=8-31`, `extra_after_low=8`, no residual bucket, default vLLM
compile, and direct cuSPARSELt, same-run dense EAGLE3 reached total/full-batch
`3107.690` / `4772.493` tok/s while `speclink_t08` with reduced sync reached
`3039.717` / `3950.561`; disabling reduced sync reached `3117.455` /
`3840.767`, so the full-batch effect is only about `1.03x` and the main
bottleneck is still GPU-side mixed sparse/residual work and residual storage.
Restricting residual correction to `gate_up_proj=16-31` improved full-batch to
`4185.427` but stayed below same-run dense `4741.987`, with GSM8K-50 still
`0.7600` versus dense `0.7800`. Current result roots:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dense_vs_gateup_8_31_extra8_nobucket_defaultcompile_fixed128_max2048_20260624`,
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_8_31_extra8_nobucket_defaultcompile_noreduce_fixed128_max2048_20260624`, and
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dense_vs_gateup_16_31_extra8_nobucket_defaultcompile_fixed128_max2048_20260624`.

The `compressed_dense` residual backend remains the memory direction to test
next because it reports attached-module storage around `1.125x` dense instead
of `dense_rows` at `1.625x`. Its previous default-compile serving attempts
failed because runtime residual materialization lowered boolean masked
assignment to `aten.index_put_`, which is not CUDA-graph-capture safe. A small
CUDA graph probe showed `masked_scatter_` captures successfully, so runtime
compressed residual materialization in `vllm/vllm/speclink_sr24.py` now uses
`masked_scatter_`. Python compile and
`examples/evaluate/eval-guidellm/scripts/check_speclink_sr24_correctness.py`
pass; a vLLM serving smoke still needs to be rerun with GPU access because the
first escalated smoke request was blocked by the tool approval reviewer before
vLLM launched.

Follow-up GPU runs on 2026-06-24 resolved that smoke and measured the residual
backends. The corrected `compressed_dense` smoke must include
`--sr24-target-leafs gate_up_proj` and `--sr24-direct-cslt-linear`; omitting
them attaches too many modules and lets Inductor lower sparse `F.linear` into
unsupported `_cslt_sparse_mm`. The corrected smoke attached 16 modules,
reported `SPECLINK_SR24_DIRECT_CSLT_LINEAR=1`, completed default compile and
CUDA graph capture, and wrote:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_compressed_direct_maskedscatter_gateup16_31_smoke_gsm8k10_20260624`.

Strict bs64 fixed-128 `max_tokens=2048` results for
`gate_up_proj=16-31`, `extra_after_low=8`, Llama-3.1-8B + EAGLE3 K=8,
`math_reasoning`:

| path | total/steady tok/s | full-batch tok/s | tpot ms | storage/dense | root |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `3441.321` | `4724.508` | `12.756` | n/a | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dense_vs_gateup_16_31_extra8_compressed_direct_fixed128_max2048_20260624` |
| `compressed_dense`, chunk `4096` | `1621.859` | `2942.252` | `20.404` | `1.1250` | same root |
| `compressed_dense`, chunk `0` full materialization | `2270.345` | `2933.749` | `20.544` | `1.1250` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra8_compressed_fullmaterialize_direct_fixed128_max2048_20260624` |
| `torch_sparse` residual | `3214.659` | `3914.370` | `15.218` | `1.1875` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra8_torchres_direct_fixed128_max2048_20260624` |
| prior `dense_rows` diagnostic | `3349.400` | `4185.427` | `14.324` | `1.6250` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dense_vs_gateup_16_31_extra8_nobucket_defaultcompile_fixed128_max2048_20260624` |

Conclusion: `compressed_dense` is now GPU-resident and graph-safe, but runtime
residual materialization is too expensive. Full materialization avoids the
7-chunk loop but still has the same poor full-batch throughput. A second real
semi-structured sparse residual GEMM is the best lower-storage residual path so
far, but still trails `dense_rows` and dense. The next credible optimization is
an operator-level fused packed sparse/residual GPU path, not more CPU-sync
tuning or `compressed_dense` materialization variants.

Additional bs64 fixed-256-request, `max_tokens=512` CPU-sync and quality
follow-up on 2026-06-24:

| candidate | GSM8K-50 acc | total/steady tok/s | full-batch tok/s | note | root |
| --- | ---: | ---: | ---: | --- | --- |
| dense EAGLE3, same reduced-sync run | n/a | `3432.015` | `3782.590` | baseline for the no-nondraft run | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_ablation_followup_bs64_256req_20260624` |
| `speclink_t08`, `gate_up_proj=16-31`, `extra3`, bucket32, no non-draft correction, reduced CPU sync | `0.6600` vs dense `0.7800` | `3771.639` | `4263.633` | fast but not quality-safe; speedup comes with wrong outputs and higher acceptance | throughput: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_ablation_followup_bs64_256req_20260624`; quality: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_bucket32_no_nondraft_quality_gsm8k50_20260624` |
| same no-nondraft candidate, no reduced CPU sync | n/a | `3516.571` | `3938.496` | CPU-sync-heavy ablation; reduced sync gives about `1.07x` total and `1.08x` full-batch for this unsafe point | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_heavy_ablation_followup_bs64_256req_20260624` |
| `speclink_t08`, `gate_up_proj=16-31`, `extra3`, bucket32, non-draft correction, reduced CPU sync | `0.7800` vs dense `0.7800` | `3120.869` | `3563.766` | quality-safe on GSM8K-50, but below same-run dense EAGLE3 `3533.085` / `3820.385` | throughput: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_bucket32_nondraft_throughput_bs64_256req_20260624`; quality: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_bucket32_nondraft_quality_gsm8k50_20260624` |
| same quality-safe candidate with `extra_after_low=8` | not rerun; expected no worse than extra3 but not claimed | `3230.351` | `3584.162` | slightly faster than extra3, still below dense EAGLE3; not enough to change direction | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra8_bucket32_nondraft_throughput_bs64_256req_20260624` |

Interpretation: reducing CPU-side synchronization is useful and should stay as
an ablation/default hygiene item for graph-safe selective SR24, but the
quality-safe path still loses to dense EAGLE3. Disabling non-draft correction
creates an apparent speedup by changing generation quality and acceptance, so
do not use that configuration for final speed claims. The remaining bottleneck
is the GPU-side sparse/residual execution plan and the quality requirement that
bonus/non-draft rows remain corrected.

Additional quality/speed sweep on 2026-06-24 after the CPU-sync ablation:

| candidate | GSM8K-50 acc | total/steady tok/s | full-batch tok/s | note | root |
| --- | ---: | ---: | ---: | --- | --- |
| `predicted_full_accept` non-draft policy, `critical_prefix+extra3`, bucket32 | `0.7600` vs dense `0.7800` | not expanded | not expanded | GPU-side predicted bonus-row correction is not quality-safe | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_bucket32_predfull_quality_gsm8k50_20260624` |
| no-reduced-sync safe all-nondraft, bucket32 | n/a | `3178.363` | `3515.629` | no-reduce/static differences are small compared with GPU path cost | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_bucket32_nondraft_noreduce_throughput_bs64_256req_20260624` |
| no-reduced-sync + static buffer safe all-nondraft, bucket32 | n/a | `3111.152` | `3405.142` | fairer no-reduce/static-buffer point; reduced sync is only a small full-batch win | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_16_31_extra3_bucket32_nondraft_noreduce_staticbuf_throughput_bs64_256req_20260624` |
| `batch_all_if_any_low`, threshold `0.5` | `0.7400` | not expanded | not expanded | too aggressive | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batchall_t05_nondraft_quality_gsm8k50_20260624` |
| `batch_all_if_any_low`, threshold `0.7` | `0.7400` | not expanded | not expanded | too aggressive | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batchall_t07_nondraft_quality_gsm8k50_20260624` |
| `batch_all_if_any_low`, threshold `0.75` | `0.7800` | `3194.785` | `3663.734` | quality-safe, but stats show almost all steps are all-residual dense | quality: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batchall_t075_nondraft_quality_gsm8k50_20260624`; throughput: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batchall_t075_nondraft_throughput_bs64_256req_20260624` |
| `batch_all_if_any_low`, threshold `0.8` | `0.7800` | `3234.807` | `3625.850` | quality-safe but still below dense EAGLE3 `3576.558` / `3825.789` | quality: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batchall_t08_nondraft_quality_gsm8k50_20260624`; throughput: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batchall_t08_nondraft_throughput_bs64_256req_20260624` |
| `critical_prefix+extra3`, bucket16 | `0.7800` | `3226.599` | `3611.759` | quality-safe, slightly better than bucket32 total but not dense-beating | quality: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_extra3_bucket16_nondraft_quality_gsm8k50_20260624`; throughput: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_extra3_bucket16_nondraft_throughput_bs64_256req_20260624` |
| `critical_prefix+extra3`, bucket8 | `0.7800` | `3273.341` | `3564.305` | best quality-safe total tok/s in this sweep, still below same-run dense `3466.098` / `3815.992` | quality: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_extra3_bucket8_nondraft_quality_gsm8k50_20260624`; throughput: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_extra3_bucket8_nondraft_throughput_bs64_256req_20260624` |
| `critical_prefix+extra3`, bucket4 | `0.7600` | not expanded | not expanded | bucket too small | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_extra3_bucket4_nondraft_quality_gsm8k50_20260624` |
| `critical_prefix+extra1`, bucket8 | `0.7800` | `3106.048` | `3542.877` | less draft correction did not improve throughput | quality: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_extra1_bucket8_nondraft_quality_gsm8k50_20260624`; throughput: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_extra1_bucket8_nondraft_throughput_bs64_256req_20260624` |
| `up_sparse` split, all layers | `0.5000` | not expanded | not expanded | gate/up half-sparse is not quality-safe | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gate_dense_up_sparse_quality_gsm8k50_rerun_20260624` |
| `up_sparse` split, layers 16-31 | `0.6600` | not expanded | not expanded | still not quality-safe | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gate_dense_up_sparse_layers16_31_quality_gsm8k50_20260624` |
| `channel_pair`, dense fraction `0.5`, layers 16-31 | `0.6800` | not expanded | not expanded | coarse gate/up channel split is also not quality-safe | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_channel_pair_dense50_layers16_31_quality_gsm8k50_20260624` |

The `batch_all_if_any_low` stats probe explains why the quality-safe whole-step
strategy cannot produce a speedup: with threshold `0.75`, 227 of 229 logged
verify summaries were `all_residual` and only 2 were `no_residual`, so the run
mostly falls back to dense gate/up work:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batchall_t075_stats_step_probe_bs64_32req_20260624`.
The current quality-safe best is `critical_prefix+extra3`, bucket8, but it is
still below dense EAGLE3. The `1.2x` target is therefore not reachable by more
CPU sync reduction or current PyTorch-level residual bucket tuning; it needs a
fused GPU sparse/residual operator or a new quality policy that avoids
non-draft/bonus-row dense correction without changing answers.

2026-06-24 bucket implementation follow-up: the `dense_rows` residual bucket
path in `vllm/vllm/speclink_sr24.py` no longer clones the full `base_output`
before adding dense/residual bucket deltas. It now applies `index_add_` in
place on the temporary base output. The correctness script covers this
`reduce_cpu_sync=1`, bucket-size path. A CUDA microbenchmark shows the clone
removal saves about `0.034-0.035 ms` per Llama gate_up bucket call at
`rows=512,out=28672,in=4096,bucket=8/16/32`, moving the bucket-delta graph path
from about `1.05-1.07x` dense to `0.98-1.00x` dense:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mixed_bucket_inplace_probe_20260624`.

The serving validation for the current best quality-safe configuration
(`critical_prefix+extra3`, bucket8, non-draft correction, bs64 fixed 256
requests, `max_tokens=512`) improved full-batch throughput but not enough for
the target. Same-run dense EAGLE3 was `3453.500` total/steady and `3819.070`
full-batch tok/s; SR24 was `3250.800` total/steady and `3625.972` full-batch
tok/s. Compared with the previous bucket8 run (`3273.341` total and
`3564.305` full-batch), the clone removal mostly improves the full-batch
steady path; the dense-relative full-batch ratio is still only about `0.949x`:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_extra3_bucket8_inplace_throughput_bs64_256req_20260624`.

2026-06-24 operator follow-up: a predecoded-position Triton residual prototype
was added only to
`examples/evaluate/eval-guidellm/scripts/profile_speclink_sr24_sparse_backend.py`.
It keeps the complementary residual positions as GPU-resident `pos0/pos1`
tensors so the kernel does not decode packed masks or LUTs inside the inner
loop. This was a negative result and must not be wired into serving. On real
Llama `gate_up_proj` shape `out=28672,in=4096`, all-corrected CUDA Graph times
were:

| rows | dense ms | all sparse graph ms | all direct cslt0 graph ms | all Triton pos-tiled graph ms |
| ---: | ---: | ---: | ---: | ---: |
| 64 | `0.1541` | `0.2322` | `0.2372` | `4.5490` |
| 128 | `0.1658` | `0.2412` | `0.2507` | `8.7930` |
| 512 | `0.5412` | `0.7576` | `0.8007` | `34.8941` |

Result root:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_sparse_backend_pos_probe_20260624`.
Conclusion: removing mask decode/LUT work is not enough; this Triton residual
loop is dominated by its memory/computation organization. Continue keeping
CPU-sync reduction as an ablation/default hygiene item, but treat the main
remaining SR24 speed gap as GPU-side mixed sparse/residual execution, not
Python logging or mask-state synchronization.

2026-06-24 SR24 breakdown follow-up: SR24 breakdown instrumentation is now split
so profiling does not silently corrupt graph-on serving. `--sr24-breakdown` is
the low-sync serving breakdown and records scheduler CPU time, reduced routing
counters, bucket candidate rows, and CUDA Graph mode counts. It writes periodic
JSON snapshots without forcing a CUDA synchronization every verify step.
`--sr24-breakdown-exact-routing` intentionally synchronizes GPU scalars to
report exact residual/base routing counts and bucket fill ratio; use it only as
a CPU-sync ablation. `--sr24-breakdown-linear` records CUDA-event timings inside
SR24 Linear hooks and should be used only with eager/no-compile component
profiles. It is disabled by default because putting Python profiling locks
inside Linear broke `torch.compile` capture for `base_only_24`.

Latest breakdown report:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_breakdown_analysis_20260624/report.md`.
The quality-safe `gate_up_proj=16-31`, `critical_prefix+extra3`, bucket32,
`bonus` non-draft policy still trails dense EAGLE3 in clean serving: dense
EAGLE3 total/full-batch `3490.477` / `3827.897` tok/s versus `speclink_t08`
`3249.923` / `3579.665`. Low-sync graph-on breakdown for `speclink_t08`
reported `scheduler_mask_build_cpu_ms=31.812` per verify step, `FULL=69`,
`NONE=78`, `PIECEWISE=1`, and full-batch `3324.932` tok/s. Exact-routing
breakdown dropped total/full-batch to `1763.693` / `2936.259`, confirming that
GPU scalar synchronization is harmful and must stay out of clean throughput
runs. Eager Linear component profiling showed base sparse Linear dominates:
`1103.293ms` total (`1.0448ms/call`) versus residual dense GEMM `181.694ms`
(`0.1721ms/call`) and gather/scatter/delta about `34.6ms` total. The next
optimization target is a vectorized/graph-safe GPU mask and bucket builder plus
an operator-level fused sparse/residual path; more CPU-sync tuning alone is not
expected to make the quality-safe SR24 path faster than dense EAGLE3.

Reusable SR24 component summarizer:

```bash
conda run -n spec python examples/evaluate/eval-guidellm/scripts/summarize_sr24_breakdown.py \
  --roots examples/evaluate/eval-guidellm/results.bak/SERVING_GRAPH_ON_ROOT \
          examples/evaluate/eval-guidellm/results.bak/EAGER_LINEAR_PROFILE_ROOT \
  --output-root examples/evaluate/eval-guidellm/results.bak/SR24_COMPONENT_SUMMARY_ROOT
```

It joins `summary.csv` with each run's `speclink_sr24_breakdown.json` and
writes `breakdown_summary.csv` plus `report.md`. The report starts with a
`Bottleneck Diagnosis` table covering scheduler/mask build, request routing,
base sparse Linear, residual correction, gather/scatter, routing statistics,
CUDA Graph modes, and GPU utilization. Use graph-on low-sync rows for serving
behavior (`tok/s`, CUDA Graph modes, GPU util, scheduler mask/bucket time).
Use eager/no-compile `--sr24-breakdown-linear
--sr24-breakdown-exact-routing` rows only to localize Linear-hook component
time; do not report those rows as clean throughput.

Latest regenerated slowdown breakdown report:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_slowdown_breakdown_requested_20260624_191623/report.md`.

Current explicit breakdown requested on 2026-06-24:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_component_breakdown_current_20260624/report.md`.
The serving root is
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_breakdown_current_lowsync_graphon_bs64_64req_20260624`
and the component profile root is
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_breakdown_current_eager_linear_exact_bs64_64req_20260624`.
For Llama-3.1-8B + EAGLE3 K=8, bs64, `math_reasoning`, max_tokens 256 in
serving and 128 in eager component profiling, `speclink_t08` was slower than
dense EAGLE3 in service (`2194.262` vs `2296.851` total tok/s and
`3221.278` vs `3385.194` full-batch tok/s) while GPU util stayed similar
(`89.333%` vs `88.357%`). The service-state bottleneck is therefore not simple
GPU idleness. `speclink_t08` still showed many graph misses
(`FULL=114,NONE=77,PIECEWISE=1`) and `scheduler_mask_build_cpu_ms=25.120` per
verify step. In eager component profiling, base sparse Linear dominated
(`1.026ms/call`) over dense residual GEMM (`0.171ms/call`) and gather/scatter
(`0.008ms/call`), with the dense correction bucket fixed at 32 rows and filled
about `98.7%`. This confirms the remaining speed problem is mostly the
two-stage sparse-base plus dense-correction operator path plus graph/scheduler
overhead, not low accepted length or GPU scalar stats alone.

`--sr24-batched-mask-builder` / `SPECLINK_SR24_BATCHED_MASK_BUILDER=1` is an
opt-in scheduler/mask-build optimization ablation, not a quality default. The
first version rebuilt a temporary score matrix in the verifier; the indexed
version carries the proposer-side batched score tensor and row id on each score
row view so the Triton kernel can read scores without per-request score copies.
The batched kernel supports `critical_prefix` with `non_draft=bonus` and, after
the 2026-06-24 continuation, `non_draft=all`. Correctness smokes compare the
slow Python path and the batched path for both policies. Earlier bs64
`math_reasoning` serving did not improve enough:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batched_mask_ablation_20260624/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batchedmask_indexed_clean_bs64_128req_20260624/report.md
```

In low-sync breakdown, non-batched `speclink_t08` reported total/full-batch
`2194.262` / `3221.278` tok/s and `scheduler_mask_build_cpu_ms=25.120`. The
copy-style batched kernel was worse (`2213.751` / `3039.144`, mask CPU
`38.942`, kernel `0.957ms/step`). The indexed batched kernel was better than
the copy-style version but still not a speed solution (`2373.766` /
`3361.097`, mask CPU `39.743`, indexed kernel `1.318ms/step`). A clean paired
no-breakdown run with indexed batched mask builder still trailed dense EAGLE3:
dense `2782.674` / `3178.694` total/full-batch tok/s versus `speclink_t08`
`2396.808` / `3010.928`. Keep this path off by default; the evidence points
back to the mixed sparse/residual operator and graph coverage, not simply the
per-request threshold/mask expression.

2026-06-24 continuation: the batched builder now reuses static GPU int32
scratch buffers for starts, valid rows, score lens, bonus flags, and score-row
ids instead of allocating fresh `torch.tensor(..., device=cuda)` inputs every
decode step. With Llama3.1-8B, EAGLE3 K=8, bs64, fixed 64 requests, max_tokens
128, `gate_up_proj=16-31`, `dense_rows`, `critical_prefix+extra3`,
`non_draft=all`, and breakdown enabled:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allpolicy_batched_scratch_summary_20260624_1920/report.md
```

The non-batched row reported `scheduler_mask_build_cpu_ms=30.964` and total /
full-batch tok/s `1466.220` / `2374.379`. The pre-scratch batched row was
still dominated by setup (`scheduler_batched_mask_setup_cpu_ms=40.079`). The
scratch-buffer batched row reduced scheduler/mask build to `0.850ms/step`,
batched setup to `0.138ms/step`, and raised total / full-batch tok/s to
`2190.069` / `2980.109`, with GPU util up to `80.0%`. A paired clean
no-breakdown smoke still trailed dense EAGLE3:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allpolicy_batched_scratch_clean_pair_bs64_64req_20260624_1920/report.md
```

Dense EAGLE3 was `2193.637` total and `3030.025` full-batch tok/s; SR24 was
`1476.884` total and `2475.649` full-batch tok/s with similar acceptance
(`1.424` vs `1.421` accepted draft tokens/step) but lower GPU util (`60.36%`
vs `82.38%`). This means the CPU routing/mask bottleneck is largely fixed for
this path, but the active speed gap has moved to SR24 Linear/operator
efficiency and GPU underutilization. Do not claim a final speedup from this
alone.

2026-06-24 continuation: `run_lm_eval_accuracy.py` now exposes
`--sr24-batched-mask-builder` and passes
`SPECLINK_SR24_BATCHED_MASK_BUILDER=1`, records the flag in run metadata, and
includes it in the SR24 compile-cache fingerprint. Do not treat this as a
quality-safe default. Llama3 GSM8K-50 with the current gate_up_proj=16-31,
`critical_prefix+extra3`, `non_draft=bonus`, bucket32 candidate matched dense
(`0.7800` vs `0.7800`) when batched-mask was enabled:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batched_quality_gateup16_31_bonus_gsm8k50_20260624/report.md
```

The same low-overhead batched-mask path on Minerva-50 dropped exact_match from
the dense reference `0.3800` to `0.3200`. Enabling runtime stats happened to
produce `0.3800`, and a sync-mask probe plus the non-batched fallback both
landed at `0.3600`, so the previous Minerva `0.3800` should be treated as
unstable rather than proof that the policy is solved. `non_draft=all` also only
reached `0.3600`, so the remaining quality issue is not explained solely by
bonus/non-draft rows. Follow-up sample divergence showed that removing the
bucket cap recovered doc `23`, but docs `28` and `44` still regressed and doc
`32` improved. Re-running with `SPECLINK_SR24_EARLY_DENSE_TOKENS=128` after
fixing the batched builder to fall back when early-dense context is required
left the same `28/44` regressions and `32` improvement. Most importantly, the
dense-equivalent `all_corrected_24` no-op fastpath produced the same aggregate
shape on Minerva-50: dense correct `19/50`, all-corrected correct `18/50`, with
regressions on docs `28` and `44` and an improvement on doc `32`. Treat this
last `-2pp` as serving/speculative run-to-run or dense-equivalent control
variance unless it reproduces beyond the all-corrected control, not as proof of
SR24 sparsity error. Relevant diagnostic outputs:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batched_quality_gateup16_31_bonus_minerva50_20260624/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batched_minerva50_stats_debug_20260624/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_nondraft_all_minerva50_quality_20260624/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batched_bonus_syncmask_minerva50_probe_20260624/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fallback_bonus_minerva50_current_20260624/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_minerva50_batched_bonus_divergence_20260624_1935/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batched_bonus_minerva50_bucket0_diagnosis_20260624_1935/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batched_bonus_minerva50_bucket0_early128_fix_diagnosis_20260624_1935/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_minerva50_dense_equiv_20260624_1935/report.md
```

Keep `speclink_t08` quality status open, but compare it against both dense
EAGLE3 and the dense-equivalent `all_corrected_24` control. A sparse-path
quality bug should be claimed only when `speclink_t08` diverges beyond the
all-corrected control on the same prompt manifest or on repeated paired runs.

2026-06-24 active-goal follow-up: base-only and all-corrected diagnosis is in
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_progress_baseonly_allcorrected_20260624/report.md`.
For `base_only_24`, the slowdown question is not acceptance length: in the
bs64/max2048 same-run comparison, dense EAGLE3 accepted about `3.83` draft
tokens/step while `base_only_24` accepted about `5.24`, with GPU util about
`97%` and CUDA Graph `FULL=826,NONE=6`. If a base-only result looks slow, check
tail effects and full-batch throughput first. For exact `all_corrected_24`,
the dense fastpath is only a control path; true exact all-corrected sparse work
has no 2:4 FLOP reduction because base 2/4 plus residual 2/4 equals dense 4/4.
`compressed_dense` is already GPU-resident by default (`auto -> cuda`), but
runtime materialization is too slow. The runner now writes
`spec_acceptance_rate_pct`, `spec_estimated_steps`,
`spec_avg_selected_draft_tokens_per_step`, and
`spec_avg_accepted_draft_tokens_per_step`, and reports them in a
`Median Speculative Acceptance` table. Two small scheduler/proposer changes were
validated but are not a material speed fix: cached device arange buffers in the
SR24 mask builder and storing DLM draft-score row views instead of per-request
score clones. Clean bs64 fixed-256 validation after those changes reported
dense EAGLE3 full-batch `3873.968` tok/s and `speclink_t08` full-batch
`3594.001` tok/s, with accepted draft tokens/step `2.453` vs `2.417`.
Therefore the current `speclink_t08` speed gap is not accepted length; it is
lower GPU utilization and mixed sparse/residual execution overhead. Next work
should target vectorized/graph-safe GPU mask+bucket construction and then a
fused sparse/residual operator for the remaining corrected rows.

2026-06-24 latest SR24 slowdown report:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_current_progress_accuracy_speed_20260624/report.md`.
The current non-batched `gate_up_proj=16-31`, `dense_rows`,
`critical_prefix+extra3`, `non_draft=bonus` path matched dense on the
Minerva-50 manifest with no regressions or improvements. The batched mask
builder remains diagnostic-only: Minerva-50 graph/default compile had
regressions on doc ids `23,28,44`, and batched eager had regressions on
`23,44` plus one improvement on `6`. A fresh current-worktree Llama3.1-8B
EAGLE3 K=8 `math_reasoning` smoke at client-side concurrency bs64, max2048,
fixed 64 requests, and `--sr24-preset accuracy_first` reported dense
total/full-batch `2403.625` / `5292.333` tok/s and SpecLink total/full-batch
`3346.826` / `5181.223` tok/s. This is `1.392x` by total throughput but only
`0.979x` by full-batch throughput, so it is not yet a robust saturated-serving
speedup. The next run should be a continuous/full-batch-focused bs64 max2048
repeat with a larger prompt set, e.g. work under
`results.bak/sr24_current_accuracy_first_continuous_bs64_20260624`, and should
keep batch size interpreted as GuideLLM/client-side concurrency.

For the requested breakdown, current evidence is:

- scheduler / mask build: diagnostic rows still show about `10.5-12.0 ms/step`;
  batched mask CUDA kernel is small (`0.296 ms/step`) but its CPU-side setup is
  about `11.399 ms/step`, and the path is not quality safe.
- base sparse linear: `gate_up_proj=16-31` sparse base is the largest measured
  linear component at about `0.508-1.045 ms/call`.
- residual correction: dense-row correction GEMM is about `0.154-0.172
  ms/call`.
- gather/scatter: measured `index_select`/`index_add_` overhead is small, about
  `0.008-0.011 ms/call`.
- routing statistics: bucket fill is high (`0.987-0.993`), so the immediate
  issue is not empty buckets; accepted draft tokens/step are comparable or
  slightly higher than dense in the fresh smoke (`4.145` vs `3.918`).
- CUDA Graph: the fresh `accuracy_first` smoke still had good graph coverage
  (`FULL=905,NONE=23`), but full-batch throughput stayed below dense.
- GPU util: fresh dense and `accuracy_first` were both near full utilization
  (`98.07%` vs `97.97%`), while earlier `speclink_t08` dynamic-path rows showed
  lower utilization (`92.76%`) and worse TPOT; dynamic sparse/residual overhead,
  not accepted length, is the next target.

2026-06-24 `all_corrected_24` no-op dense fastpath cleanup: when
`SPECLINK_SR24_MODE=all_corrected`, backend is `torch_sparse`,
`SPECLINK_SR24_ALL_CORRECTED_DENSE_FASTPATH=1`, all selected target leafs are
also residual leafs, and no layer/split filters are active,
`apply_sr24_from_env()` now treats SR24 as a dense-equivalent no-op. It no
longer loads the mask cache or attaches per-module SR24 runtime attributes in
that case, and writes `dense_fastpath_noop=true`,
`residual_backend=dense_fastpath`, `residual_device=none`,
`linear_hooks_enabled=false`, and `draft_scores_enabled=false` to
`speclink_sr24_stats.json`. This is the intended optimized interpretation of
`all_corrected_24`: it is a correctness/control path equivalent to dense
EAGLE3, not a real sparse speedup path. The lm-eval runner's SR24 validity
check accepts this no-op stats file without requiring verify-mask events, and
records the actual no-op backend fields in `run_meta.json`.

Validation smoke:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_noop_smoke_20260624/report.md`.
Llama-3.1-8B + EAGLE3 K=8, `math_reasoning`, client-side concurrency bs64,
fixed 64 requests, max_new_tokens 256 reported dense total/full-batch
`2638.312` / `3361.764` tok/s and `all_corrected_24` total/full-batch
`2639.922` / `3365.722` tok/s. The all-corrected run recorded
`dense_fastpath_noop=true`, `storage/dense=1.0`, no SR24 linear hooks, and no
DLM draft-score collection. If `all_corrected_24` is slower in a future run,
first check whether `dense_fastpath_noop` is false or whether layer/split
filters disabled the full dense-equivalent fastpath.

2026-06-24 `speclink_t08` route-all residual-row ablation:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_route_all_densefallback_summary_20260624/report.md`.
This added `SPECLINK_SR24_ROUTE_ALL_RESIDUAL_ROWS=1` and
`SPECLINK_SR24_ROUTE_DENSE_FALLBACK_FRACTION` as explicit diagnostics for the
`torch_sparse + dense_rows` path. Route-all preserves the selective mask exactly
by sending every residual row to dense Linear and only non-residual rows to the
sparse base. It is correctness-covered in
`check_speclink_sr24_correctness.py`, including the dense-fallback branch.

On Llama-3.1-8B + EAGLE3 K=8, `math_reasoning`, client-side concurrency bs64,
fixed 64 requests, max_new_tokens 256, `gate_up_proj=16-31`,
`critical_prefix+extra3`, `non_draft=bonus`:

| variant | dense total tok/s | SR24 total tok/s | total ratio | dense full-batch tok/s | SR24 full-batch tok/s | full-batch ratio | SR24 GPU util |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| route all residual rows | `2305.372` | `1324.018` | `0.574x` | `3401.884` | `2232.526` | `0.656x` | `60.54%` |
| route all + dense fallback `0.5` | `2305.682` | `1983.607` | `0.860x` | `3401.405` | `3093.620` | `0.910x` | `88.94%` |
| route all + dense fallback `0.0` | `2650.589` | `2320.503` | `0.875x` | `3538.273` | `3115.587` | `0.881x` | `86.36%` |

The short linear breakdown showed the route-all split path doing about `55%`
dense rows and `45%` sparse rows, but still paying separate gather, sparse
GEMM, dense GEMM, and index-copy kernels. The largest component was routed base
sparse GEMM (`921.428ms` over `640` calls, about `1.440ms/call`), followed by
routed dense GEMM (`221.332ms`) and base index-copy (`87.600ms`). Dense fallback
recovers GPU utilization, but then the path approaches dense EAGLE3 without
beating it. Do not use dynamic per-row split routing as the main `speclink_t08`
optimization path; the next useful path is either coarse graph-safe
all-residual/no-residual scheduling, a fused sparse+dense residual operator, or
a narrower quality-safe base-only layer/module selection that avoids dynamic
residual correction altogether.

2026-06-24 base-only layer-range boundary:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_layer_range_boundary_20260624/report.md`.
This checked whether a wider pure base-only `gate_up_proj` range could replace
dynamic residual routing. The answer is no for contiguous tail ranges. Existing
throughput upper bounds show `gate_up_proj=8-31` base-only can reach about
`4905.196` total tok/s and `5823.714` full-batch tok/s, but GSM8K-50 drops from
`0.7800` to `0.5200`. The newly run `gate_up_proj=12-31` base-only quality
check also fails badly: GSM8K-50 `0.7800 -> 0.5600` and Minerva-50
`0.3800 -> 0.1800`. The narrower `gate_up_proj=16-31` upper bound reaches
`4300.928` total tok/s and `5401.893` full-batch tok/s, below the desired speed
 margin, and wider ranges already fail quality. Do not continue sweeping wide
contiguous base-only tail ranges as the main path; if base-only selection is
revisited, use a non-contiguous sensitivity-guided set of very few layers.

2026-06-24 strict leaf-only preset follow-up:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateonly_accuracyfirst_strict_bs64_max2048_summary_20260624/report.md`.
This compared `accuracy_gate_only` (`gate_up_proj=31`) against the current
`accuracy_first` (`gate_up_proj=31;down_proj=31`) under the same strict
Llama-3.1-8B + EAGLE3 K=8 `math_reasoning` setting: client-side concurrency
bs64, fixed 128 requests, and `max_tokens=2048`.

| preset | total tok/s dense | total tok/s SR24 | total speedup | full-batch tok/s dense | full-batch tok/s SR24 | full-batch speedup | SR24 accept % | SR24 GPU util |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `accuracy_gate_only` | `2927.049` | `3819.023` | `1.305x` | `4619.467` | `4513.180` | `0.977x` | `54.609` | `98.52%` |
| `accuracy_first` | `2846.918` | `3828.822` | `1.345x` | `4350.756` | `4415.143` | `1.015x` | `53.168` | `98.16%` |

This narrows the interpretation of the current speedups. Both presets have
near-full GPU utilization and good CUDA Graph coverage, so they are not slow
because the GPU is idle. The fixed-request total tok/s gain comes mostly from
higher acceptance and fewer decode iterations (`accuracy_gate_only`: `52201 ->
48824` steps; `accuracy_first`: `54163 -> 49896` steps), not from a faster
full-batch kernel path. `accuracy_gate_only` is the safer leaf-only direction
from the existing quality data, because `accuracy_down_only` has large IFEval
regressions, but it does not improve full-batch throughput. `accuracy_first`
keeps a small full-batch gain in this single run, but its existing 100-sample
quality evidence still has GSM8K/IFEval regressions. The requested robust
`1.2x` saturated-serving target is therefore still not solved by preset
selection alone; the next real optimization has to be either a fused
graph-safe sparse-tail operator or a different quality-safe target selection
that changes the full-batch kernel cost, not just accepted-length statistics.

2026-06-24 gate-up tail quality follow-up:
Two narrower static base-only `gate_up_proj` variants were checked before
running more throughput:

| candidate | result root | GSM8K-50 dense -> SR24 | Minerva-50 dense -> SR24 | decision |
| --- | --- | ---: | ---: | --- |
| `gate_up_proj=30-31` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup30_31_quality_gsm8k_minerva50_20260624` | `0.7200 -> 0.7000` | `0.3800 -> 0.3400` | reject; both tasks regress |
| `gate_up_proj=30` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup30_quality_gsm8k_minerva50_20260624` | `0.7400 -> 0.7200` | `0.3800 -> 0.4000` | not clean; GSM8K loses examples and reasoning paths still change |

These runs make the coarse layer-level target-selection story weaker. A single
late `gate_up_proj` layer can sometimes be close, but the paired loss/gain
pattern shows it is not a quality-safe optimization by itself. The next useful
selection direction is finer channel/block sensitivity inside a very small MLP
tail target, or a real fused operator that reduces kernel cost without changing
which rows are corrected.

2026-06-24 operator and accuracy follow-up:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_operator_accuracy_followup_20260624/report.md`.
The current Triton/compressed residual prototypes are not viable serving paths.
On representative Llama MLP bf16 shapes, base-only sparse graph replay is
faster than dense, but exact all-corrected base+residual is slower than dense:
rows=512 `gate_up` dense `0.5414ms`, base sparse graph `0.3550ms`, all sparse
graph `0.7586ms`; rows=512 `down` dense `0.2919ms`, base sparse graph
`0.1665ms`, all sparse graph `0.3338ms`. GPU `compressed_dense` is the right
storage/transfer direction compared with CPU materialization, but current
materialize-then-dense matmul is still far too slow (`4.8370ms` for rows=512
`gate_up`, `2.6175ms` for rows=512 `down`). The current Triton residual
prototype is orders of magnitude slower (`34.9456ms` and `18.6505ms` graph
all-corrected for the same two shapes), so do not wire it into vLLM serving.
Keep `all_corrected_24` as the dense no-op fastpath until there is a real fused
packed sparse/residual CUDA kernel.

2026-06-24 bucket override operator probe: `SPECLINK_SR24_TRITON_BUCKET_OVERRIDE=1`
adds a diagnostic in-place Triton overwrite kernel for the `torch_sparse +
dense_rows` residual bucket path. It keeps the normal sparse base output for
all rows, computes dense output on bucket rows, then overwrites only active
bucket rows instead of doing `base_output.index_select`, delta compute, and
`index_add_`. Correctness is covered by
`check_speclink_sr24_correctness.py`, but the microbenchmark is negative on the
representative Llama gate_up shape (`rows=512,out=28672,in=4096`): bucket32
graph time was `0.6628ms` for Triton override versus `0.5346ms` for the current
clone-free delta/index_add path. Result root:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bucket_override_inplace_probe_fix_20260624`.
Keep this flag off by default; it is useful as a measured negative ablation, not
as the path toward the requested 1.2x saturated-serving target.

2026-06-24 SR24 slowdown breakdown direction:
Use the component breakdown before trying more accuracy knobs. The current
summary script is:

```bash
cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm
conda run -n spec python ./scripts/summarize_sr24_breakdown.py \
  --roots RESULTS_OR_RESULTS_BAK_ROOT... \
  --output-root ./results.bak/sr24_component_breakdown_TIMESTAMP
```

The report should explicitly inspect:

| part | field family |
| --- | --- |
| scheduler / mask build | `scheduler_mask_build_cpu_ms_per_step`, `scheduler_request_routing_loop_cpu_ms_per_call`, mask init/write/topk fields |
| base sparse linear | `base_sparse_linear_cuda_ms_per_call` |
| residual correction | `residual_dense_gemm_cuda_ms_per_call`, `residual_sparse_gemm_cuda_ms_per_call`, or compressed residual fields |
| gather/scatter | `gather_input_index_select_cuda_ms`, `gather_base_index_select_cuda_ms`, `bucket_delta_compute_cuda_ms`, `scatter_index_add_cuda_ms` aggregated as `gather_scatter_cuda_ms_per_call` |
| routing statistics | draft/non-draft residual/base row counts, bucket active/candidate rows, `bucket_fill_ratio` |
| CUDA Graph | `sr24_cudagraph_mode_counts`, especially `FULL` vs `NONE` |
| GPU util | `avg_gpu_util_pct`, `peak_gpu_util_pct`, and the sample count |

Current focused slowdown report:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_requested_breakdown_current_20260624/report.md`.
The summary script now keeps clean serving rows separate from scheduler and
Linear diagnostic rows. In the clean serving row, dense EAGLE3 reaches
`3827.897` full-batch tok/s and `97.082%` average GPU util, while
`speclink_t08` reaches `3579.664` full-batch tok/s and `92.756%` average GPU
util. In the diagnostic rows, scheduler/mask build is high for selective SR24
(`11.150ms/step` versus `0.171ms/step` for `base_only_24`), and the exact
request routing loop accounts for most of it (`10.876ms/step`). The
Linear-localization row shows sparse base is larger than selected residual
correction (`0.508-1.026ms/call` base sparse versus `0.154-0.171ms/call`
residual dense GEMM), while gather/scatter remains tiny (`0.008-0.011ms/call`).
Bucket fill is high (`0.987-0.993`) in the routed rows, so the immediate slow
path is not wasted dense-row bucket capacity. Current evidence points to
CPU-side selective routing/mask setup plus the mixed sparse-base Linear path;
accepted length and gather/scatter are not the first bottlenecks.

2026-06-24 SR24 compressed residual storage guard:
`vllm/vllm/speclink_sr24.py` now records residual backend/device diagnostics in
`speclink_sr24_stats.json`: `residual_backend_counts`,
`residual_device_counts`, `residual_cpu_module_count`,
`residual_cuda_module_count`, `residual_extract_cpu_fallback_module_count`,
`compressed_residual_runtime_on_gpu`, and
`compressed_residual_non_gpu_modules`. The lm-eval and GuideLLM matrix
aggregators propagate these fields into `summary.csv`. Both runners expose
`--sr24-require-gpu-residual`, which sets
`SPECLINK_SR24_REQUIRE_GPU_RESIDUAL=1` and fails model attach if a
`compressed_dense` residual is not GPU-resident. Use this guard for
`all_corrected_24 --no-sr24-all-corrected-dense-fastpath` and `speclink_t08`
performance diagnostics so a CPU residual-storage fallback is not mistaken for
the intended GPU path. Keep `all_corrected_24` default dense fastpath separate:
it is an exact dense/no-op correctness control, not a measurement of the real
sparse-base plus residual-correction operator.

2026-06-24 all-corrected operator microbench:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_sparse_backend_microbench_continue_20260624/summary.md`
benchmarked representative Llama MLP shapes. Dense GEMM was `0.236-0.495ms`,
while the real `base sparse + compressed residual materialize/GEMM` path was
`2.581-5.145ms`. The experimental Triton residual path is not a serving
solution: the pos-tiled all-corrected path was `16.5-30.6ms`. Do not enable
`SPECLINK_SR24_COMPRESSED_RESIDUAL_TRITON` for throughput runs unless a newer
microbench proves it faster. Current evidence says `all_corrected_24` needs a
fused packed base+residual operator, or a different correction strategy, before
it can beat dense.

2026-06-24 batched mask-builder optimization attempt:
`vllm/vllm/speclink_sr24.py` now reuses small CPU int32 staging buffers for the
default batched mask builder instead of allocating `torch.tensor(list)` for
`starts`, `valid_rows`, `score_lens`, `has_bonus`, and score rows every verify
step. Correctness is covered by
`examples/evaluate/eval-guidellm/scripts/check_speclink_sr24_correctness.py`.
On the focused bs64/K8/max256 Llama math breakdown, this is only a small setup
improvement, not a full solution:

| variant | result root | setup CPU | full-batch tok/s | note |
| --- | --- | ---: | ---: | --- |
| old indexed batched builder | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_granular_batchedmask_bs64_k8_max256_20260624` | `11.399ms/step` | `3367.399` | old baseline |
| staging-buffer default | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batchedmask_stagingbuf_bs64_k8_max256_20260624` | `10.715ms/step` | `3400.958` | small improvement |
| default after GPU-count disabled | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batchedmask_default_after_gpucount_off_bs64_k8_max256_20260624` | `9.789ms/step` | `3362.829` | confirms default path does not use GPU-count kernel |

An attempted GPU-count mask builder was added behind
`SPECLINK_SR24_GPU_COUNT_MASK_BUILDER=1`, but it is deliberately off by default.
It reads vLLM's GPU-side scheduled/draft/cumsum buffers directly and reduces
setup CPU (`1.900ms/step`), but the new request-level Triton kernels are much
slower than the existing compact indexed kernel (`6.003ms` direct-request
kernel and `31.002ms` req-indexed kernel versus `0.266-0.296ms` for the old
indexed kernel). End-to-end full-batch throughput dropped to `2897.396 tok/s` in
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batchedmask_gpucount_bs64_k8_max256_20260624`.
Do not enable `SPECLINK_SR24_GPU_COUNT_MASK_BUILDER` for normal runs; it is a
measured negative diagnostic that shows the real path needs either a better
compact GPU kernel or to eliminate this mask build entirely, not simply move the
count arithmetic to a full-request Triton launch.

Follow-up on 2026-06-24: the known-worst GPU-count non-direct score-row branch
now falls back to the compact indexed builder instead of launching the old
request-indexed kernel. Correctness smoke covers this by recording scores in
one request order and building the mask in the reverse request order. Breakdown
smoke at
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/temp/sr24_gpu_count_fallback_smoke_20260624_v2/speclink_sr24_breakdown.json`
recorded `batched_mask_builder_gpu_count_indexed_fallback_steps=1` and no
`scheduler_batched_mask_req_indexed_kernel_cuda_ms`. This is a guardrail for the
diagnostic path; it does not make GPU-count the default path.

2026-06-24 uniform-direct batched mask builder diagnostic:
`vllm/vllm/speclink_sr24.py` now has
`SPECLINK_SR24_BATCHED_UNIFORM_DIRECT=1` as an explicit, off-by-default
diagnostic. It triggers only for uniform steady-state speculative verify steps:
all active requests present, identical draft length, `K+1` scheduled stride,
bonus rows present, and score rows aligned with request ids. In that shape it
launches `_critical_prefix_bonus_mask_uniform_direct_kernel` and avoids copying
`starts`, `valid_rows`, `score_lens`, `has_bonus`, and `score_rows` int32 arrays
from CPU to GPU. The default path still uses the compact indexed kernel, but
the int32 staging copy now uses `non_blocking=True` with the existing pinned
CPU buffers. Correctness smoke covers slow path, default indexed batched path,
explicit uniform-direct path, and the older GPU-count diagnostic. Breakdown
smoke:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/temp/sr24_uniform_direct_smoke_20260624/speclink_sr24_breakdown.json`.
It recorded `batched_mask_builder_uniform_direct_steps=1` and
`batched_mask_builder_indexed_steps=2`, but the unit-shape direct kernel was
slower (`1.488ms`) than the indexed kernel (`0.609ms`). Keep
`SPECLINK_SR24_BATCHED_UNIFORM_DIRECT` disabled unless a larger serving profile
proves otherwise; this is currently a measured negative diagnostic, not the
path toward the 1.2x target.

2026-06-25 follow-up with the current graph-on low-sync speed candidate:
`run_llama31_vllm_fastdraft_smurfs_eagle3_matrix.py` and
`run_lm_eval_accuracy.py` now expose these diagnostics as explicit CLI flags:
`--sr24-batched-uniform-direct` and `--sr24-gpu-count-mask-builder`. In the
uniform bs64/math/max256 run with low-confidence cap=1, bucket=32,
`--sr24-reduce-cpu-sync`, `--no-sr24-sync-mask-state`,
`--sr24-static-mask-buffer`, and `--sr24-batched-mask-builder`, the default
indexed path remained best:

| path | result root | full-batch tok/s | scheduler mask build | actual builder path | decision |
| --- | --- | ---: | ---: | --- | --- |
| indexed batched | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_ablation_indexed_bs64_math256_20260625` | `3343.639` | `0.388ms/step` | 181 indexed steps | keep as default |
| uniform-direct | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_ablation_uniform_direct_bs64_math256_20260625` | `3303.599` | `0.510ms/step` | 155 uniform + 38 indexed steps | negative |
| GPU-count | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_ablation_gpu_count_bs64_math256_20260625` | `3272.264` | `4.640ms/step` | 155 GPU-count + 38 indexed-fallback steps | negative |

Combined report:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_ablation_summary_bs64_math256_20260625_v2/report.md`.
This confirms CPU staging reduction as a valid ablation axis, but not a current
optimization path: uniform-direct reduces setup CPU but uses a slower uniform
kernel, and GPU-count is much slower in this shape. A fourth indexed run with
`--sr24-disable-runtime-stats` lowered scheduler/mask build accounting from
`0.388ms/step` to `0.297ms/step`, but full-batch throughput stayed flat
(`3343.639` vs `3340.451`), so runtime summary writing is not the end-to-end
bottleneck either.

2026-06-24 GPU-count copy gating cleanup:
`vllm/vllm/v1/worker/gpu_model_runner.py` now copies the auxiliary
`speclink_sr24_num_draft_tokens` buffer to GPU only when
`SPECLINK_SR24_GPU_COUNT_MASK_BUILDER=1`. The default selective SR24 path and
dense EAGLE3 baseline do not consume that buffer, so copying it every
speculative step was unnecessary instrumentation overhead and could pollute
baseline comparisons. Correctness still passes
`examples/evaluate/eval-guidellm/scripts/check_speclink_sr24_correctness.py`,
including the opt-in GPU-count builder. Focused Llama math bs64/K8/max256
serving check:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_sr24_gpu_count_copy_gated_bs64_k8_max256_20260624/report.md`.
It reported dense EAGLE3 `3387.348` full-batch tok/s and `speclink_t08`
`3273.787` full-batch tok/s, with
`sr24_gpu_count_mask_builder=False`. This cleanup is baseline hygiene, not the
main speed solution; SR24 still trails dense, so the remaining target is
selective routing/mask setup and sparse-base Linear execution.

2026-06-24 current operator-path follow-up:
Three focused Llama3.1-8B + EAGLE3 K=8 `math_reasoning` bs64 fixed-64,
max_tokens=256 runs compared the current quality candidate:

| variant | result root | dense full-batch tok/s | SR24 full-batch tok/s | SR24 total tok/s | SR24 GPU util | decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| default `critical_prefix+extra3`, bucket32 | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_sr24_gpu_count_copy_gated_bs64_k8_max256_20260624` | `3387.348` | `3273.787` | `2291.243` | `85.36%` | close but still slower |
| `--sr24-batched-mask-builder` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batchedmask_current_bs64_k8_max256_20260624` | `3394.584` | `3389.488` | `2237.239` | `82.93%` | fixes most full-batch gap but not total/tail; quality still needs a short gate |
| `--sr24-route-bucket-rows` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_route_bucket_rows_current_bs64_k8_max256_20260624` | `3397.700` | `2381.386` | `1708.509` | `68.53%` | negative; splitting sparse base rows and dense bucket rows underutilizes the GPU |

This confirms the current route-bucket operator path is not the fused operator
the goal needs: it saves sparse-base rows but pays separate gathers, sparse GEMM,
dense GEMM, and index-copy assembly, and drops GPU utilization. The batched mask
builder is the only current path that nearly closes the short-run full-batch
gap, but it still has unresolved quality status from earlier Minerva-50
diagnostics. Attempts to rerun Minerva-50/20 with `max_new_tokens=2048` in this
session were too slow for interactive iteration and left child
`VLLM::EngineCore` processes that had to be killed manually:
`sr24_batchedmask_minerva50_current_quality_20260624`,
`sr24_batchedmask_minerva20_current_quality_retry_20260624`, and
`sr24_batchedmask_minerva20_current_quality_retry2_20260624` are incomplete
quality attempts, not result roots to cite. Before another quality gate, check
`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader`
and clear stale `VLLM::EngineCore` children from interrupted runs. Use a much
smaller Minerva/GSM8K smoke first, then rerun Minerva-50 only if the short gate
does not regress beyond the dense-equivalent `all_corrected_24` control.

2026-06-24 continuation quality sanity gate for the batched-mask candidate:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batchedmask_gsm8k5_current_quality_20260624/report.md`.
It ran Llama3.1-8B EAGLE3 K=8 on 5 GSM8K samples with
`max_new_tokens=2048`, `--sr24-batched-mask-builder`,
`gate_up_proj=16-31`, `critical_prefix+extra3`, `non_draft=bonus`, and
`dense_rows@cuda`. Dense, `all_corrected_24`, and `speclink_t08` all reported
flexible exact_match `1.0000`. Treat this only as a smoke/sanity pass, not a
final accuracy result. Because it used `--sr24-reduce-cpu-sync`, draft
residual/base route counters are intentionally not exact in the summary; use
the dedicated breakdown roots for routing coverage and the larger
GSM8K/Minerva manifests for quality claims.

The matching Minerva-5 smoke is
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batchedmask_minerva5_current_quality_20260624/report.md`.
Dense, `all_corrected_24`, and `speclink_t08` all reported exact_match
`0.4000`, and there were zero paired regressions over 5 samples. Do not treat
this as quality solved: direct sample comparison shows `speclink_t08` diverged
from dense/all-corrected output text on doc ids `1` and `2`; those examples
kept the same correctness labels in the 5-sample smoke, but they show the
selective path still perturbs reasoning/output trajectories.

The larger Minerva-20 gate is
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batchedmask_minerva20_current_quality_final_20260624/report.md`.
Dense and `all_corrected_24` reported exact_match `0.2500`, while
`speclink_t08` reported `0.3000`; there were zero exact-match regressions and
one paired improvement on `doc_id=6`. Direct sample comparison still found
13/20 outputs diverging from dense text, so this is positive for the small
metric gate but not evidence of path equivalence. During this run, vLLM emitted
a benign shutdown `Traceback` (`Signal 15 ignored due to race condition`) after
an otherwise successful run. `aggregate_lm_eval_accuracy.py` now avoids
scanning logs for fallback errors when `run_meta.status=ok` and `meta.error` is
empty, so future summaries should not treat that shutdown race as a failed run.
The matching sample-level divergence report is
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batchedmask_minerva20_divergence_20260624/report.md`;
it records paired samples `20`, response text differs `13`,
same-correctness text differs `12`, regressions `0`, and improvements `1`.

The same follow-up compared dense and SR24 lm-eval samples for the safer
leaf-only presets. `accuracy_gate_only` had GSM8K-100 `5` regressions and `5`
improvements, and Minerva-100 `9` regressions and `8` improvements.
`accuracy_first` had GSM8K-100 `4` regressions and `2` improvements, and
Minerva-100 `6` regressions and `7` improvements. These are reasoning-path
changes, not just answer formatting: several regressions diverge in the first
few generated tokens, including Minerva cases where the first token switches to
a different solution strategy. This explains why adding sparse MLP leaves does
not compose cleanly. Future accuracy work should use these divergence samples
to validate a finer channel/block selection rule before running long throughput
jobs; another coarse layer-level sweep is unlikely to solve the target.

2026-06-24 `speclink_t08` early-token residual guard:
`vllm/vllm/speclink_sr24.py` now supports
`SPECLINK_SR24_EARLY_DENSE_TOKENS=N`. Default `0` preserves the previous path.
When `N>0`, SR24 asks the proposer for generated-length context and forces
draft rows whose `generated_len + draft_position < N` through the
residual/dense-corrected path. The speculative bonus row is also corrected when
it is still inside the same prefix. This is intentionally env-gated because it
adds generated-length context collection in the proposer hot path. The vLLM
worker only enables that length context when
`speclink_sr24.needs_length_context()` is true, so normal SR24 runs do not pick
up the extra CPU-side sampled-count sync.

The lm-eval and GuideLLM matrix runners expose this as
`--sr24-early-dense-tokens`, include it in SR24 compile-cache fingerprints, and
write `sr24_early_dense_tokens`,
`sr24_early_residual_draft_tokens`, and
`sr24_early_residual_non_draft_tokens` into aggregate summaries. The SR24
correctness smoke covers the mask semantics for a high-confidence draft step
where early prefix rows are corrected while later rows stay base-only.

Small GSM8K-20 smoke, Llama-3.1-8B + EAGLE3 K=8, `gate_up_proj=16-31`,
`dense_rows`, `critical_prefix+extra3`, `non_draft=bonus`,
`max_new_tokens=2048`:

| guard | dense score | `speclink_t08` score | early draft/non-draft rows | residual draft fraction | result |
| --- | ---: | ---: | ---: | ---: | --- |
| `--sr24-early-dense-tokens 0` | `0.6000` | `0.6000` | `0/0` | `0.5834` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_early_dense0_gateup16_31_gsm8k20_20260624/report.md` |
| `--sr24-early-dense-tokens 32` | `0.6000` | `0.6500` | `1592/158` | `0.6803` | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_early_dense32_gateup16_31_gsm8k20_20260624/report.md` |

This is only a small smoke, not a final quality claim. It confirms the guard is
active and does not immediately harm GSM8K-20.

The follow-up 50-sample quality gate did not validate early dense as the
solution:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_early_dense32_gateup16_31_gsm8k_minerva50_20260624/report.md
```

With `--sr24-early-dense-tokens 32`, Llama-3.1-8B + EAGLE3 K=8,
`gate_up_proj=16-31`, `dense_rows@cuda`, `critical_prefix+extra3`, and
`non_draft=bonus`, dense and `all_corrected_24` matched exactly, but
`speclink_t08` regressed:

| task | dense | `all_corrected_24` | `speclink_t08` | delta vs dense |
| --- | ---: | ---: | ---: | ---: |
| GSM8K-50 | `0.7800` | `0.7800` | `0.7200` | `-6.0 pp` |
| Minerva-50 | `0.3800` | `0.3800` | `0.3400` | `-4.0 pp` |

Do not keep increasing `SPECLINK_SR24_EARLY_DENSE_TOKENS` blindly. The next
quality work should identify first-divergence rows/tokens and use that evidence
to choose a finer channel/block or token policy. The performance work should
continue from the explicit slowdown breakdown instead of starting more
end-to-end sweeps.

2026-06-24/25 SR24 current-state update:

- The generated-length context for `SPECLINK_SR24_EARLY_DENSE_TOKENS` was fixed
  so it no longer adds sampled-token counts after
  `_update_states_after_model_execute()` has already appended them to
  `req_state.output_token_ids`. This is correct bookkeeping, but it did not
  solve the 50-sample quality gate. The rerun at
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_early_dense32_lenfix_gateup16_31_gsm8k_minerva50_20260624/report.md`
  still reported GSM8K-50 `0.7800 -> 0.7200` and Minerva-50
  `0.3800 -> 0.3400` for dense versus `speclink_t08`.
- Keep default `all_corrected_24` as the dense fastpath/no-op correctness
  control. The fastpath 20-sample GSM8K check at
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_fastpath_gateup16_31_gsm8k20_20260625/report.md`
  matched dense score, output-length stats, and spec-acceptance stats.
- The earlier no-fastpath `all_corrected_24` sample drift was caused by a
  mismatched eager/graph comparison, not by a bad dense reconstruction. The
  drift run at
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_real_gateup16_31_gsm8k20_20260624/report.md`
  compared eager SR24 against non-eager dense. The matched eager rerun at
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_real_gateup16_31_gsm8k20_both_eager_20260625/report.md`
  made dense and no-fastpath `all_corrected_24` sample-identical; the
  divergence report
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_real_gateup16_31_gsm8k20_both_eager_divergence_20260625/report.md`
  records response text differs `0`.
- A focused GPU-only diagnostic was added at
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/scripts/check_speclink_sr24_vllm_linear_equivalence.py`.
  It constructs a vLLM `MergedColumnParallelLinear` named like
  `model.layers.16.mlp.gate_up_proj`, applies real no-fastpath SR24, and
  compares against dense vLLM output. It needs real GPU access. The default
  small fp16 check and the Llama-shaped bf16 check
  `--llama-gate-up-shape --dtype bf16 --rows 9` both report `exact=True`
  (`shape=(9, 28672)` for the Llama-shaped run).
- The speed direction is still the explicit component breakdown: scheduler/mask
  build and request routing dominate (`~11 ms/step`, with `~10.9 ms/step` in
  exact request routing), while gather/scatter is tiny. CPU-sync reduction
  should be kept as a formal ablation, but its approximate routing counters are
  not exact row-coverage metrics.

2026-06-25 SR24 CPU-sync/mask-builder ablation:
Current slowdown tracking doc:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/SR24_SLOWDOWN_BREAKDOWN.md`.
Serving-style ablation summary:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_maskbuilder_breakdown_summary_20260625/report.md`.
The bs64/K8/math fixed-64-request rows show why the next step should be a
breakdown-first optimization, not another broad throughput sweep:

| variant | total tok/s | full-batch tok/s | avg GPU util | conclusion |
| --- | ---: | ---: | ---: | --- |
| exact stats/default mask `speclink_t08` | `1226.555` | `1986.705` | `63.038%` | badly CPU/sync-bound |
| `--sr24-reduce-cpu-sync` | `1615.788` | `2735.606` | `80.950%` | scalar-sync overhead is real |
| reduced sync + batched mask builder | `2044.510` | `2918.362` | `86.062%` | batching mask construction helps |
| reduced sync + batched mask + static buffer + graph + stats off | `2448.633` | `3345.052` | `91.308%` | near dense but still not a 1.2x full-batch win |

The paired optimized dense row was `2302.442` total / `3396.062` full-batch
tok/s, so the optimized `speclink_t08` row beats the short-run dense total
number but still trails full-batch by about `1.5%`. This means CPU sync and
Python-side mask setup explain the worst slowdown, but after they are reduced,
the remaining target is GPU-side mixed execution: sparse base Linear plus
residual correction needs a fused or lower-launch-overhead path. Keep profiling
rows (`--sr24-breakdown-linear`, exact routing) separate from serving rows; the
former localize slow components, while the latter are the throughput numbers to
optimize.

2026-06-25 CPU-sync follow-up with the current high-confidence candidate:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_ablation_summary_20260625/summary.md`.
This used Llama-3.1-8B EAGLE3 K=8, `math_reasoning`, bs/concurrency 64,
fixed 64 requests, max 256 tokens, runtime stats off, and
`gate_up_proj,down_proj` SR24 with `dense_rows` residual on GPU,
`high_confidence`, threshold `0.9`, `min_prefix_residual=2`, and
`non_draft_policy=bonus`.

| variant | dense full-batch tok/s | SR24 full-batch tok/s | SR24/dense |
| --- | ---: | ---: | ---: |
| sync mask-state on | `3367.714` | `3379.564` | `1.004x` |
| no mask-state sync | `3371.118` | `3303.268` | `0.980x` |
| no sync + indexed batched mask builder | `3374.227` | `3389.597` | `1.005x` |
| batched + GPU-count/direct attempt | `3371.054` | `3319.817` | `0.985x` |

Pathcheck confirmed the batched serving shape used indexed mask-builder steps
only: `batched_mask_builder_indexed_steps=21`,
`batched_mask_builder_gpu_count_steps=0`, and
`batched_mask_builder_uniform_direct_steps=0`. Keep
`SPECLINK_SR24_BATCHED_MASK_BUILDER=1` as a small off-by-default/explicit
speed ablation, but do not default to `SPECLINK_SR24_GPU_COUNT_MASK_BUILDER=1`
or `SPECLINK_SR24_BATCHED_UNIFORM_DIRECT=1` without a new positive serving
profile. The current 1.2x gap is not primarily the remaining CPU scalar sync;
it is the mixed GPU operator path: sparse-base Linear plus residual
`dense_rows` correction and bucket/topk assembly under high GPU utilization.

2026-06-25 mixed-operator follow-up:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_operator_ablation_summary_20260625/summary.md`.
Using the same bs64/math/max256 high-confidence candidate, the indexed batched
baseline was `3389.597` SR24 full-batch tok/s versus same-root dense
`3374.227` (`1.005x`). `SPECLINK_SR24_TRITON_BUCKET_OVERRIDE=1` was
`3376.515` versus dense `3368.424` (`1.002x`). `SPECLINK_SR24_ROUTE_BUCKET_ROWS=1`
 with `SPECLINK_SR24_TRITON_ROUTE_ASSEMBLY=1` was `3388.955` versus dense
`3369.937` (`1.006x`) but had lower total tok/s (`2573.735`). Do not treat
bucket override or route-bucket rows as the solution. They are flat-to-negative
in the current serving shape; a real fused sparse-base plus residual-correction
operator, or a quality-safe lower residual-row policy, is still needed.

2026-06-25 static all-residual down-tail probe:
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_static_all_residual_down_tail_summary_20260625/summary.md`.
The dynamic `batch_all_if_any_low` pathcheck showed
`mask_state_all_residual=41/41` on the short bs64/math run, so at this batch
shape the dynamic policy mostly degenerates to all-residual while still paying
mask/routing overhead. Making the state static and leaving only a `down_proj`
tail base-only gives:

| variant | dense full-batch tok/s | SR24 full-batch tok/s | SR24/dense | quality read |
| --- | ---: | ---: | ---: | --- |
| static `down_proj=16-31` base-only | `3411.714` | `3668.870` | `1.075x` | GSM8K aggregate tied but 4 paired regressions/4 improvements; Minerva `0.3800 -> 0.3400` |
| static `down_proj=8-31` base-only | `3371.258` | `3268.369` | `0.969x` | not worth quality gate |
| static `down_proj=24-31` base-only | `3359.004` | `3382.164` | `1.007x` | speed too small to pursue |

Do not use simple static down-tail base-only routing as the main SR24 solution:
the only meaningful speed point is not quality-safe, and the quality-safer
narrower tail has almost no speed gain. Keep it as an upper-bound diagnostic.

2026-06-25 SR24 follow-up diagnosis:

- Long-output low-sync serving run:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_optimized_lowsync_batched_graph_bs64_k8_math_max2048_20260625/report.md`.
  Dense EAGLE3 reached `4821.332` full-batch tok/s with `98.407%` GPU util;
  optimized `speclink_t08` reached `4286.066` full-batch tok/s with `97.964%`
  GPU util. `speclink_t08` accepted slightly more draft tokens per step
  (`4.066` vs `3.910`), so the remaining slowdown is not low acceptance or
  global GPU idle time. It is the mixed sparse-base plus residual-correction
  operator path.
- `compressed_dense` with `SPECLINK_SR24_RESIDUAL_DEVICE=auto` is already
  CUDA-resident. The cached full-weight/no-chunk diagnostic at
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_compressed_cache_fullweight_bs64_k8_math_max128_20260625/report.md`
  improves over repeated materialization, but still only reaches `2301.365`
  full-batch tok/s. Treat it as evidence that materialization/chunking is
  costly, not as the final storage-efficient implementation.
- A small quality smoke for `SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY=high_confidence`
  at threshold `0.8` is under
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_highconf_gateup16_31_quality_gsm8k_minerva20_20260625/report.md`.
  It reported GSM8K-20 dense `0.6000` vs `speclink_t08` `0.7000`, and
  Minerva-20 dense `0.2500` vs `speclink_t08` `0.2500`. This is only a
  20-sample direction check; use a larger paired manifest before treating it
  as the quality fix.

2026-06-25 SR24 high-confidence follow-up:

- `SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY=high_confidence` is now supported by
  the batched mask-builder Triton path. The GPU correctness script
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/scripts/check_speclink_sr24_correctness.py`
  passed and compares slow, indexed batched, uniform-direct batched, and
  GPU-count batched high-confidence masks.
- The 50-sample quality gate did not fix accuracy:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_highconf_batched_gateup16_31_gsm8k_minerva50_20260625/report.md`
  reports GSM8K-50 `0.7800 -> 0.7200` and Minerva-50 `0.3800 -> 0.3400`
  for dense versus high-confidence `speclink_t08`.
- Divergence reports under the same directory show GSM8K had `21/50` response
  divergences and `3` regressions; Minerva had `21/50` divergences and `2`
  regressions. First divergence often occurs mid-generation, so do not treat
  the issue as only an initial-token or early-prefix problem.
- `all_if_any_low`, `gate_up_proj=24-31`, and `gate_up_proj=16-23` 50-sample
  probes produced the same sample hashes as high-confidence on these tasks.
  A separate static `all_residual` versus `no_residual` GSM8K-10 sanity under
  `results.bak/sr24_static_*_residual_gsm8k10_sanity_20260625/` produced
  different sample hashes, so the mask state is not a no-op. Next quality work
  should instrument the regression doc ids at per-step/per-row or logit level,
  not continue broad threshold/layer sweeps.

2026-06-25 requested SR24 slowdown breakdown refresh:

- Latest combined breakdown report:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_requested_breakdown_combined_bs64_k8_20260625_0131/report.md`.
- It deliberately combines a graph-on serving run and an eager linear
  diagnostic run:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_requested_breakdown_graphon_bs64_k8_math_max512_20260625_0122/`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_requested_breakdown_linear_bs64_k8_math_max128_20260625_0130/`.
- Graph-on bs64/K8/math/max512 serving results: dense EAGLE3 `4357.297`
  full-batch tok/s, `base_only_24` `4993.135`, `speclink_t08` `4242.105`.
  GPU util is similar (`93.3%`, `93.3%`, `94.1%`) and `speclink_t08` has
  mostly `FULL` graph steps (`{"FULL":446,"NONE":2}`), so this run does not
  point to low acceptance, graph loss, or global GPU idle time.
- Eager linear diagnostic bs64/K8/math/max128 for `speclink_t08`: scheduler
  mask build `11.761 ms/step`, request routing loop `11.680 ms/step`, base
  sparse Linear `1.612 ms/call`, residual dense-row GEMM `0.358 ms/call`,
  gather/scatter/clone/index_copy `0.063 ms/call`, routed rows
  `16261/11579` draft residual/base and `3480/3787` non-draft residual/base.
- `vllm/vllm/speclink_sr24.py` now times the previously missing
  `dense_rows` residual correction path via `residual_dense_rows_*` CUDA event
  fields, and `scripts/summarize_sr24_breakdown.py` includes those fields.
  Treat `--sr24-breakdown-linear` rows as component diagnostics only.
- Current speed direction: optimize the CPU request-routing loop and the
  sparse-base plus residual-correction operator. Gather/scatter is measured
  small enough that it should not be the first target.

2026-06-25 focused SR24 replay and all-if-any-low follow-up:

- Offline `vllm.LLM.generate` replay artifacts for GSM8K `doc_id=11`:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/temp/sr24_doc11_replay_20260625_0140/`.
  Dense EAGLE3 and `all_corrected_24` both produce the correct `694` output.
  The high-confidence selective path diverges immediately after `donuts` and
  produces `494`; `critical_prefix` and `critical_prefix+extra_after_low=1`
  also diverge. `all_if_any_low` restores the dense output for this sample.
- The debug trace
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/temp/sr24_doc11_replay_20260625_0140/sr24_debug_trace.jsonl`
  shows the row-level reason: vLLM replacement tokens are sampled from target
  logits at the rejected draft row, and accepted rows in the same speculative
  step can write hidden/KV state used by later rows. High-confidence-only or
  short critical-prefix row routing is not quality-safe.
- Short bs64/K8/math/max512 serving follow-up:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_all_if_any_low_breakdown_followup_bs64_k8_math_max512_20260625/`.
  Dense EAGLE3 reached `3262.404` total / `4421.113` full-batch output tok/s
  with `93.150%` avg GPU util. `speclink_t08 all_if_any_low` reached
  `3020.225` total / `4196.164` full-batch output tok/s with `92.773%` avg GPU
  util and slightly higher accepted draft tokens per step (`2.553` vs `2.465`).
  Quality-safer routing still does not beat dense, so the optimization target
  remains a fused or lower-launch-overhead sparse-base plus dense-row residual
  operator. CPU-sync reduction should stay as an ablation, not the only main
  path.
- `all_if_any_low` is now supported by the Triton batched mask builder
  (`policy_id=2` in `vllm/vllm/speclink_sr24.py`). The GPU correctness script
  checks slow, indexed batched, and GPU-count batched masks for this policy and
  reports `speclink_sr24_correctness=ok`.
- Follow-up with the batched all-if-any-low path:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_all_if_any_low_batched_followup_bs64_k8_math_max512_20260625/`.
  Same-run dense EAGLE3 reached `2466.191` total / `4350.041` full-batch
  output tok/s; batched all-if-any-low reached `2205.030` total / `4260.277`
  full-batch output tok/s. The full-batch number improves over the previous
  non-batched all-if-any-low `4196.164`, but still trails dense by about `2.1%`.
  A short linear diagnostic at
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_all_if_any_low_batched_linear_diag_bs64_k8_math_max128_20260625/`
  reports `sr24_mask_state=all_residual`, so this quality-safe policy often
  collapses to dense verification and is not the final sparse speedup path.
- Exact-stats t=0.4 GSM8K-20 check:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allifany_threshold_gsm8k20_t04_exactstats_20260625/`.
  It used `all_if_any_low`, `threshold=0.4`, `gate_up_proj=16-31`, and
  `dense_rows@cuda`. It reported exact_match `0.6500`, spec acceptance
  `0.1902`, draft residual fraction `0.8432`, draft base-only fraction
  `0.1568`, and non-draft residual fraction `1.0000`. This confirms the policy
  is not always all-residual, but it corrects too many verifier rows to create
  a large sparse speedup. The next useful direction is either a prefix-safe
  routing policy that leaves far more rows base-only without changing verifier
  behavior, or a fused/lower-launch sparse-base plus residual-correction
  operator. Do not resume broad threshold sweeps until one of those mechanisms
  changes.

2026-06-25 SR24 route-reuse mixed correction follow-up:

- `SPECLINK_SR24_ROUTE_REUSE_BASE_OUTPUT=1` is a new narrow
  `torch_sparse + dense_rows` ablation. For mixed masks it keeps the full
  sparse `base_output` that has already been computed, runs dense Linear only
  for residual rows, and overwrites those rows. This is different from the old
  route-all split path, which also split/recomputed the base sparse GEMM and was
  strongly negative.
- `SPECLINK_SR24_ROUTE_DENSE_FALLBACK_FRACTION` is also applied at scheduler
  mask-state classification when `SPECLINK_SR24_SYNC_MASK_STATE=1`: if a mixed
  selective step has residual coverage above the threshold, SR24 can choose the
  all-residual dense fastpath for that step. Default `1.1` keeps this disabled.
  The fallback is conservative for accuracy but was negative in the current
  short throughput probe, so do not use it as the main path without retesting.
- Correctness smoke:
  `conda run --no-capture-output -n spec python examples/evaluate/eval-guidellm/scripts/check_speclink_sr24_correctness.py`
  reports `speclink_sr24_correctness=ok` and covers route-reuse mixed-mask
  output equivalence.
- Focused bs64/K8/math/max512 roots:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_route_reuse_bs64_k8_math_max512_20260625/`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_no_reuse_bs64_k8_math_max512_t04_20260625/`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_densefallback08_bs64_k8_math_max512_20260625/`.
  No-reuse t=0.4 reached `4293.186` full-batch output tok/s; route-reuse t=0.4
  reached `4301.408`; route-reuse plus dense fallback `0.8` reached
  `4250.507`. The same-run dense EAGLE3 row for route-reuse was `4430.813`.
  Route-reuse is a small positive change, but it is not remotely enough for the
  `1.2x` target. The next speed step still needs a fused/lower-launch operator
  or a quality-safe policy that leaves far more rows base-only.

2026-06-25 SR24 acceptance-trace follow-up:

- `vllm/vllm/speclink_confidence_trace.py` and `vllm/vllm/speclink_sr24.py`
  now let `SPECLINK_TRACE_CONFIDENCE=1` attach SR24 verifier routing fields to
  each token-level acceptance record: `sr24_uses_residual`, `sr24_score`,
  `sr24_policy`, `sr24_threshold`, `sr24_mask_state`, and
  `sr24_generated_len`.
- The parser is
  `examples/evaluate/eval-guidellm/scripts/analyze_sr24_acceptance_trace.py`.
  It writes token-weighted SR24 residual fractions, accepted base-only
  fractions, reached base-only fractions, and by-position summaries.
- No-prefix all-if-any-low t=0.4 GSM8K-5 trace:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_acceptance_trace_gsm8k5_20260625/acceptance_trace_analysis/report.md`.
  It reported `0.8250` residual fraction, `0.2533` accepted base-only
  fraction, `0.2208` reached base-only fraction, `0.1375` steps with accepted
  base-only tokens, and `1.4062` accepted tokens/step.
- Prefix-4 residual diagnostic:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_acceptance_trace_gsm8k5_prefix4_20260625/acceptance_trace_analysis/report.md`.
  With `SPECLINK_SR24_SELECTIVE_MIN_PREFIX_RESIDUAL=4`, accepted base-only
  fraction dropped to `0.0167` and steps with accepted base-only dropped to
  `0.0061`, but residual fraction rose to `0.9091`.
- Interpretation: accepted base-only prefix rows are a plausible source of
  accuracy drift, because later accepted/replacement logits can depend on the
  earlier row state. Prefix forcing reduces that risk, but it also leaves too
  little sparse work for a large throughput win. The next useful experiment is
  a larger GSM8K/Minerva quality gate for the prefix override, then a
  graph-safe batched/fused implementation if it helps quality.

2026-06-25 SR24 reoriented speed candidate:

- Current slowdown tracking doc:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/SR24_SLOWDOWN_BREAKDOWN.md`.
- The clean no-sync `low_confidence` candidate with
  `SPECLINK_SR24_SELECTIVE_MAX_RESIDUAL_DRAFT_ROWS=1` and bucket size `32`
  should currently run with `SPECLINK_SR24_TRITON_BUCKET_OVERRIDE=0`.
  The Triton override probe reached only `3108.103` full-batch tok/s versus
  dense `3171.865`; the no-Triton bucket delta/index-add rerun reached
  `3413.243` versus same-run dense `3391.668` in bs64/math/max256. This is a
  small `1.006x` clean-serving win, not the requested `1.2x` result.
- A GSM8K-20 accuracy gate for that same candidate is under
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lowconf_budget1_bucket32_accuracy_gsm8k20_20260625/report.md`.
  It reported dense EAGLE3 `0.6000` and `speclink_t08` `0.7000`, with no
  dense-correct/SR24-wrong paired samples. Treat this only as a small
  keep-going gate; it is not a final quality claim.
- 2026-06-26 paired retest: do not use
  `--sr24-preset speed_tradeoff_down16_base` as the main SR24 route. The
  GSM8K-50 paired gate under
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speed_tradeoff_paired_gsm8k20_20260626/report.md`
  reported dense EAGLE3 `0.7200` and `speclink_t08` `0.0200`, with `35`
  dense-correct/SR24-wrong samples. The divergence report under
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speed_tradeoff_divergence_gsm8k50_20260626/report.md`
  shows many first-token or first-few-token divergences and corrupted-looking
  tokens such as `HeaderCode`, `Intialized`, and `$LANG`; treat this as
  evidence that the `down_proj=16-31` base-only tail is not quality-safe.
- A conservative gate-up-only paired gate removed the down base-only tail:
  `target_leafs=gate_up_proj`, `residual_layer_ids_by_leaf=gate_up_proj=16-31`,
  `all_if_any_low`, `threshold=0.4`, and `min_prefix_residual=4`. Result:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_only_allifanylow_paired_gsm8k50_20260626/report.md`
  reported dense `0.7200`, SR24 `0.7200`, with `2` paired regressions and `2`
  improvements. Its divergence report under
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_only_allifanylow_divergence_gsm8k50_20260626/report.md`
  shows the remaining regressions diverging later, at token `40` and `24`,
  instead of immediate corrupted-token output. This is the current safer
  quality reference, but it corrects too many rows (`SR24 draft residual`
  `0.9161`, non-draft residual `1.0000`) to be the final speed path.
- Larger current-code quality gates failed for both the low-confidence cap=1
  bucket32 candidate and the more conservative
  `critical_prefix + extra_after_low=3 + non_draft=all` candidate. The current
  authoritative 50-sample retest for the latter is
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_extra3_nondraftall_gateup16_31_gsm8k_minerva50_20260625/report.md`,
  with GSM8K `0.7800 -> 0.7200` and Minerva `0.3800 -> 0.3400`. Do not use
  older optimistic small-sample results as a quality claim for
  `critical_prefix+extra3`.
- Keep clean serving rows separate from sync-heavy exact-routing and
  `--sr24-breakdown-linear` rows. Clean rows answer throughput, graph coverage,
  and GPU utilization; exact/debug rows answer routing fractions and component
  timing. Do not compare their tok/s directly.

2026-06-25 SR24 prefix-residual quality gate:

- Larger GSM8K/Minerva-50 prefix4 result:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix4_allifany_t04_gateup16_31_gsm8k_minerva50_20260625/report.md`.
- Configuration: `all_if_any_low`, threshold `0.4`,
  `SPECLINK_SR24_SELECTIVE_MIN_PREFIX_RESIDUAL=4`,
  `gate_up_proj=16-31`, `torch_sparse/dense_rows@cuda`.
- Result: GSM8K stayed `0.7800 -> 0.7200`; Minerva stayed
  `0.3800 -> 0.3400`. The regression doc ids are exactly the same as the
  earlier high-confidence run: GSM8K `11,13,15`; Minerva `28,44`.
- Offline divergence reports:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix4_allifany_t04_gateup16_31_gsm8k_minerva50_20260625/divergence_gsm8k/report.md`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix4_allifany_t04_gateup16_31_gsm8k_minerva50_20260625/divergence_minerva/report.md`.
- Do not treat prefix residual as the current quality fix. It reduced accepted
  base-only rows in the small trace, but it did not recover accuracy on the
  50-sample gate.
- 2026-06-25 follow-up: `compare_sr24_replay_logits.py` compares replay JSON
  token/logprob traces offline and writes `position_logprobs.csv` plus
  `report.md`. `replay_sr24_regression_sample.py` now has
  `--tokenized-requests` to mirror lm-eval local-completions token-id prompts;
  string-prompt `LLM.generate` is not equivalent to lm-eval for these samples.
  Current tokenized isolated replay for GSM8K doc `13` produces the same
  correct output for dense and selective SR24, with no selected-token
  divergence:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_replay_logit_compare_tokenized_prefix4_allifany_t04_doc13_20260625/report.md`.
  Older isolated replay JSONs for GSM8K doc ids `11`, `13`, and `15` also show
  no selected-token divergence between dense and selective SR24; doc `11` also
  shows no selected-token divergence across dense, real `all_corrected_24`, and
  selective SR24. Reports:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_replay_logit_compare_doc11_20260625/report.md`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_replay_logit_compare_regression_docs_20260625/`.
- `replay_sr24_serving_samples.py` was added to send lm-eval sample prompts to
  a running vLLM OpenAI completions server as tokenized requests, preserving
  original sample row order. A current-code serving replay of GSM8K rows `0-13`
  under the old failing `prefix4/all_if_any_low/t=0.4` config also produced the
  correct doc-13 answer `18`:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_serving_replay_prefix4_allifany_t04_doc13_20260625/rows0_13_target13.jsonl`.
  Therefore the old doc-13 `15` regression is not currently reproducible in
  isolated or ordered serving replay. Keep the quality status open and use
  fresh paired lm-eval gates for new policies instead of citing the old
  prefix4 regression as current.

2026-06-25 SR24 breakdown-first pivot:

- Current filtered slowdown doc:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/SR24_SLOWDOWN_BREAKDOWN.md`.
- Latest bucketed breakdown report:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_breakdown_user_pivot_bucket32_final_20260625/report.md`.
- Latest combined user-table report:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_breakdown_current_user_table_20260625/report.md`.
- Use three distinct row types when diagnosing SR24:
  clean serving without `--sr24-breakdown`, scheduler/bucket diagnostic with
  `--sr24-breakdown`, and Linear-component diagnostic with
  `--sr24-breakdown-linear`. Do not compare diagnostic tok/s directly against
  clean serving tok/s.
- The combined user-table report keeps the user's requested breakdown fields in
  one place: scheduler/mask build, base sparse Linear, residual correction,
  gather/scatter, routing statistics, CUDA Graph counts, and GPU utilization.
  Routing statistics such as draft residual/base rows should come from an
  exact-routing diagnostic row; clean serving may leave those draft counters
  empty to avoid reintroducing CPU/GPU sync overhead.
- The current bucketed candidate uses residual bucket size `32`. A no-bucket
  diagnostic measured the wrong full residual-GEMM path and showed residual
  correction near `3.215ms/call`; ignore that number for the current path.
  With `--sr24-residual-bucket-size 32`, the Linear diagnostic reported base
  sparse Linear around `0.869ms/call`, residual dense GEMM around
  `0.132ms/call`, and gather/scatter around `0.008ms/call`.
- In the fresh bs64/K8/math/max256 serving row, dense EAGLE3 reached
  `3423.714` full-batch tok/s and SR24 reached `3307.119` (`0.966x`).
  Accepted draft length was higher for SR24 (`1.846` vs `1.712`), GPU util was
  similar (`90.231%` vs `88.214%`), and SR24 CUDA Graph coverage was good
  (`{"FULL": 126, "NONE": 2}`). This points away from accepted length, global
  GPU idleness, or missing CUDA Graph as the main explanation.
- The current bottlenecks to attack are the expensive sparse-base Linear and
  the added residual correction on top of it. The old sync-heavy
  bucket-priority scheduler routing fallback is covered by the follow-up fix
  below; gather/scatter is not the first target unless future rows change.
- Current all-corrected microbench recheck with device audit:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_sparse_backend_devicecheck_gateup_20260625/summary.md`.
  The compressed residual CUDA tensors are on `cuda:0` (`mask`, residual
  values, cached dense residual, and decoded positions), so the intended
  compressed path is GPU-side. It is still slower than dense: for
  `rows=64/512`, dense graph is `0.155/0.544ms`, while the best exact
  all-corrected graph path is `0.191/0.761ms`. That is only `0.81x/0.71x`
  dense speed and still requires a fused packed CUDA/Triton operator to become
  useful.

2026-06-25 SR24 runtime-stat CPU-sync ablation:

- Summary:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_runtime_stats_cpu_sync_ablation_bs64_math128_20260625/summary.md`.
- Same bs64/K8/math/max128 SR24 config, with `high_confidence`,
  threshold `0.9`, `min_prefix_residual=2`, `non_draft=bonus`,
  `gate_up_proj,down_proj`, dense-row residual, bucket size `32`, and bucket
  priority.
- Eager mode was the wrong performance read: `speclink_t08` was only
  `0.724-0.729x` dense and showed `{"NONE":80}` graph counts when stats were
  on. Use graph-enabled rows for clean serving conclusions.
- With SR24 CUDA Graph coverage, runtime stats off changed `speclink_t08`
  full-batch throughput only from `2979.416` to `2984.317` tok/s (`+0.16%`).
  In eager mode the same toggle was `+0.82%`. Treat runtime-stat CPU sync as a
  hygiene/ablation item, not the main remaining bottleneck.

2026-06-25 SR24 priority-bucket mask-builder fix:

- `SPECLINK_SR24_RESIDUAL_BUCKET_PRIORITY=1` no longer forces the scheduler
  back to the Python per-request routing loop when exact stats and early dense
  tokens are off. The batched mask-builder kernels in
  `vllm/vllm/speclink_sr24.py` now optionally write residual priority scores,
  so priority-bucket selection can stay on the GPU batched path.
- Correctness coverage was added to
  `examples/evaluate/eval-guidellm/scripts/check_speclink_sr24_correctness.py`
  for priority tensors across the slow path, uniform/direct, indexed, and
  GPU-count batched plans. The GPU check reports
  `speclink_sr24_correctness=ok`.
- Short bs64/K8/math/max128 diagnostic:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_priority_batchedmask_scheduler_summary_bs64_math128_20260625/report.md`.
  In that run `speclink_t08` used `66` batched indexed-builder steps,
  `scheduler_request_routing_loop_cpu_ms` was absent, and scheduler mask build
  was `0.898ms/step`. Throughput was still only `3077.619` full-batch tok/s
  versus dense `3126.231` (`0.984x`), despite slightly higher accepted draft
  length (`1.475` vs `1.402`).
- Current read: the old `39-45ms/step` priority-bucket Python fallback is fixed
  for clean low-sync serving. Do not keep optimizing scheduler routing as the
  main path unless that field reappears. The remaining speed target is the
  mixed sparse-base plus dense-row residual-correction operator shape.

2026-06-25 SR24 quality-safe selective preset:

- `speclink_t08` has a new tested preset:
  `--sr24-preset quality_safe_selective`. It is available in both
  `examples/evaluate/eval-guidellm/scripts/run_lm_eval_accuracy.py` and
  `examples/evaluate/eval-guidellm/scripts/run_llama31_vllm_fastdraft_smurfs_eagle3_matrix.py`.
- The preset intentionally does not set any base-only layer filter. It uses
  `gate_up_proj,down_proj` as target and residual leafs, dynamic
  `dense_rows@cuda` residual on
  `gate_up_proj=16-31;down_proj=8-15`, `low_confidence`, threshold `0.9`,
  `min_prefix_residual=2`, `non_draft=bonus`, static mask buffer, batched mask
  builder, bucket size `32`, bucket priority, and reduced CPU sync.
- Do not use the old manual candidate with
  `--sr24-base-only-layer-ids-by-leaf down_proj=16-31` as the default
  `speclink_t08` path. In the current GSM8K-5 smoke it produced `0.0000`
  accuracy and EAGLE3 acceptance near `0.005`, because those down-proj rows
  were base-only for all verifier rows.
- Quality-safe smoke:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_quality_safe_preset_gsm8k5_20260625_1935/report.md`.
  The equivalent manual no-base-only run and paired diff are under
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_residual24_no_baseonly_gsm8k5_20260625_1931/`.
- CPU-sync ablation on bs64/K8/math/max512 fixed 128 requests:
  dense EAGLE3 full-batch `4039.702` tok/s; quality-safe SR24 with reduced CPU
  sync `3817.764`; reduced CPU sync plus runtime stats off `3871.534`; same
  policy with no reduced CPU sync `2106.502`. Reducing CPU sync is necessary,
  but it is not sufficient to beat dense; the remaining target is the mixed
  sparse-base plus dense-row residual operator. Result roots:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_ablation_dense_baseline_bs64_20260625_1940`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_ablation_quality_safe_reduce_bs64_20260625_1940`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_ablation_quality_safe_reduce_stats_off_bs64_20260625_1942`, and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_ablation_quality_safe_no_reduce_bs64_20260625_1940`.
- Two immediate operator toggles were negative on that same quality-safe,
  stats-off setting: `--sr24-route-all-residual-rows` reached only
  `3511.432` full-batch tok/s, and `--sr24-triton-bucket-override` reached
  `3855.249`, versus the no-toggle stats-off baseline `3871.534`. Do not spend
  more time on those toggles without a new kernel-level change. Result roots:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_operator_route_all_quality_safe_bs64_20260625_1946` and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_operator_triton_bucket_override_quality_safe_bs64_20260625_1948`.
- 2026-06-26 Triton dense-GEMM scatter block sweep:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_triton_densegemm_block_sweep_20260626/summary.md`.
  The serving prototype now has
  `SPECLINK_SR24_TRITON_BUCKET_DENSE_BLOCK_M`,
  `SPECLINK_SR24_TRITON_BUCKET_DENSE_BLOCK_N`, and
  `SPECLINK_SR24_TRITON_BUCKET_DENSE_BLOCK_K`, defaulting to `16/32/128`
  when `SPECLINK_SR24_TRITON_BUCKET_DENSE_GEMM=1`. This improves the gated
  prototype versus the old fixed `8/32/64`, but it is still not a main path:
  gate/up best was `0.6400ms`, slower than dense `0.5413ms` and current
  bucket-delta `0.5346ms`; down best was `0.2707ms`, below dense `0.2921ms`
  but still slower than current bucket-delta `0.2627ms`. Treat shallow
  dense-row-GEMM scatter fusion as ruled out for the gate/up bottleneck.
  A real serving A/B with the new default block is:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speedtradeoff_triton_densegemm_bestblock_bs64_math256_20260626/report.md`.
  For `speed_tradeoff_down16_base`, bs64/K8/math/max256, dense EAGLE3 was
  `2987.300` steady / `3037.168` full-batch tok/s, while the gated Triton
  dense-GEMM scatter row was `3018.570` steady / `3084.662` full-batch tok/s
  (`1.010x` steady, `1.016x` full-batch). This is only a small local gain and
  does not satisfy the SR24/Speclink target; the next useful work is still a
  true fused sparse-base plus correction operator or a routing change that
  makes gate/up residual rows rare while preserving accuracy.
- 2026-06-26 active-goal focused breakdown:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/SR24_SLOWDOWN_BREAKDOWN.md`
  now has a `Goal-Focused Breakdown, 2026-06-26` section. Key roots:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_clean_base_allcorrected_bs64_math256_20260626/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_allcorrected_compressed_chunked_bs64_math256_20260626/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_speclink_quality_gateup_bs64_math256_20260626/report.md`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_baseonly_no_chunked_prefill_bs64_math256_20260626/report.md`.
  `base_only_24` is not slow because accepted length drops: it accepted
  `3.542` draft tokens/step versus dense `1.722`, but full-batch throughput was
  only `2689.674` versus dense `3040.573`, with GPU util `52.23%` and graph
  `FULL=218,NONE=166`. Disabling chunked prefill did not help. Full-prewarm
  `all_corrected_24` with `compressed_dense@cuda` OOMed, while chunked
  GPU-resident compressed residual ran but only reached `625.530` full-batch
  tok/s with `residual_cuda_module_count=128` and
  `residual_cpu_module_count=0`. The quality-safe `speclink_t08` gate/up-only
  row reached `2815.981` full-batch tok/s with accepted draft tokens/step
  `1.808` and GPU util `97.53%`, so its current blocker is mixed gate/up
  sparse-base plus dense-row correction cost, not acceptance or idle GPU.
- 2026-06-26 leaf-subset check: do not use all supported Linear leafs as the
  default SR24 speed target. In bs64/K8/math/max256, dense EAGLE3 was
  `3040.573` full-batch tok/s. `base_only_24` all-leaf was only `2689.674`
  (`0.885x`) with GPU util `52.23%`, while `gate_up_proj` only was
  `3497.130` (`1.150x`), `down_proj` only was `3134.289` (`1.031x`),
  MLP-only `gate_up_proj,down_proj` was `3709.853` (`1.220x`), and
  attention-only `qkv_proj,o_proj` was `2609.039` (`0.858x`). Result roots:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_leaf_baseonly_gateup_bs64_math256_20260626/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_leaf_baseonly_down_bs64_math256_20260626/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_leaf_baseonly_mlp_bs64_math256_20260626/report.md`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_leaf_baseonly_attn_bs64_math256_20260626/report.md`.
  The derived `speclink_t08` candidate with only `gate_up_proj=16-31`
  residual and `low_confidence@0.8 cap1` improved over `quality_gateup_only`
  but still did not beat dense: `2989.012` full-batch tok/s (`0.983x`). Adding
  down base-only made it slower, `2923.097` (`0.961x`). Result roots:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_t08_gateup_lowconf_cap1_bs64_math256_20260626/report.md`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_t08_gateup_lowconf_cap1_downbase_bs64_math256_20260626/report.md`.
  Next SR24/Speclink optimization should avoid attention leafs and focus on
  reducing corrected gate/up row cost or count; leaf selection alone is not
  enough for `speclink_t08` to reach `1.2x`.
- 2026-06-26 row-routed MLP and base-only quality follow-up: the existing
  `--sr24-row-routed-mlp` path is negative for the current low-confidence cap1
  candidate. Clean bs64/K8/math/max256 full-batch tok/s was `2717.230`
  (`0.894x` dense) versus the matched linear-level candidate `2923.097`
  (`0.961x`). Breakdown root:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_t08_rowrouted_mlp_breakdown_bs64_math128_20260626/report.md`.
  The row-routed breakdown shows base sparse gate/up after row splitting costs
  `1.45154ms/call`, sparse down `0.89233ms/call`, and gate/up concat
  `0.18539ms/call`, so the current MLP row routing should remain an ablation,
  not a default. Quality checks also show the pure MLP speed upper bound is not
  usable: GSM8K-20 `base_only_24` with `gate_up_proj,down_proj` scored `0.0000`
  versus dense `0.7000` with `14` paired regressions, and gate_up-only scored
  `0.2000` versus dense `0.6500` with `10` regressions and `1` improvement.
  Reports:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_mlp_accuracy_gsm8k20_20260626/report.md`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_gateup_accuracy_gsm8k20_20260626/report.md`.
- 2026-06-26 evening SR24 direction update: continue with a
  breakdown-first path, not another broad controller/threshold sweep. The
  current checklist is scheduler/mask build, base sparse Linear, residual
  correction, gather/scatter, routing row statistics, CUDA Graph FULL/NONE
  counts, and GPU utilization. The first read should be
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/SR24_SLOWDOWN_BREAKDOWN.md`.
  The latest `accuracy_first/up_sparse` quality gate is
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_first_up_sparse_quality_gsm8k_minerva100_20260626/report.md`.
  GSM8K-100 is acceptable in aggregate, dense `0.7700` vs SR24 `0.7900`
  with `4` paired regressions and `6` improvements, but Minerva-100 still
  drops from dense `0.4200` to SR24 `0.3700` with `9` paired regressions and
  `4` improvements. Divergence reports under the same root show many Minerva
  regressions start within the first few generated tokens, so the issue is a
  reasoning-trajectory change. Performance-wise, the main bottleneck remains
  mixed `gate_up_proj` sparse-base plus residual-correction work, not low
  acceptance, global GPU idle, or CPU stats synchronization. CPU runtime-stats
  removal was only about a `3%` total-TPS cleanup and did not solve
  full-batch throughput. Future SR24 candidates should first prove they reduce
  corrected gate/up rows or fuse sparse base plus residual correction while
  preserving GSM8K/Minerva paired accuracy.
  A follow-up route-reuse cleanup now reuses scheduler-cached residual rows in
  the sparse Linear hook when `SPECLINK_SR24_ROUTE_REUSE_BASE_OUTPUT=1`,
  avoiding a second per-Linear `residual_mask.nonzero()` pass. The diagnostic
  root
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/temp/sr24_route_reuse_cached_rows_breakdown_bs64_math64_20260626_work/speclink_t08/bs64/rep1/speclink_sr24_breakdown.json`
  confirmed `route_reuse_base_output_cached_plan_hits=656` with only `42`
  scheduler row-index builds. This is cleanup, not a solved speed path: the
  clean route-reuse run
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_route_reuse_cached_rows_clean_bs64_math128_20260626/report.md`
  still had SR24 below dense, full-batch `2732.815` vs `3021.132` tok/s.
  Do not promote route-reuse as a default unless a later fused/grouped operator
  changes this conclusion.

2026-06-25 SR24 all-corrected and down-only follow-up:

- Current consolidated diagnosis is in
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/SR24_SLOWDOWN_BREAKDOWN.md`.
  Use that file before starting another threshold sweep.
- Latest requested bs64/K8/math/max256 breakdown is:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_serving_breakdown_combined_bs64_math256_20260625/report.md`.
  Treat the low-overhead rows as the throughput read: dense EAGLE3
  `3432.312` full-batch tok/s, `base_only_24` `4202.651`, and
  `speclink_t08` `3423.765` (`0.998x` dense). In those rows
  `speclink_t08` scheduler/mask build is only `0.386ms/step`, CUDA Graph is
  `{"FULL":94,"NONE":2}`, and sampled GPU util is `91.615%`; current
  `speclink_t08` is not slow because of global GPU idle or missing CUDA Graph.
  Use the exact/linear row only to localize components: it reports
  `48.232ms/step` scheduler/mask build because exact routing synchronizes
  heavily, sparse base `2.105ms/call`, residual dense correction
  `0.147ms/call`, gather/scatter `0.011ms/call`, draft residual/base rows
  `39779/6261`, non-draft residual/base rows `5755/3599`, and bucket fill
  `1.000`.
- Latest abs-position Triton residual probe:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_abspos_triton_probe_20260625/summary.md`.
  It did not fix exact `all_corrected_24`: rows=512 gate/up is dense graph
  `0.5384ms`, best exact all-sparse graph `0.7629ms`, abs-position Triton
  `36.1494ms`; rows=512 down is dense graph `0.2915ms`, best exact all-sparse
  graph `0.3349ms`, abs-position Triton `18.4823ms`. Do not pursue the current
  Triton residual path without a new fused/tensor-core-friendly kernel shape.
- 2026-06-26 clean serving strategy probes ruled out three simple policy-only
  shortcuts. Gate/up dense fallback plus down8-15 SR24 was total/full-batch
  `1.011x/0.968x` versus dense in
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_dense_down_sr24_graph_bs64_math256_20260626/report.md`.
  Down0-31 SR24-only was `1.004x/0.981x` in
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down_all_residualonly_bs64_math256_20260626/report.md`.
  The conservative `gate_up_proj=16-31` cap1/bucket32 residual plus
  `down_proj=16-31` base-only candidate was better but still only
  `1.116x/1.030x` in
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_cap1_down16_31_base_paired_bs64_math256_20260626/report.md`.
  Its existing GSM8K-50 gate is aggregate-neutral (`0.7200 -> 0.7200`) but not
  paired-stable (`4` dense-correct/SR24-wrong and `4` reverse changes):
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_cap1_down_base_accuracy_gsm8k50_20260625/report.md`.
  This exact tradeoff is now available in both SR24 runners as
  `--sr24-preset speed_tradeoff_down16_base`; it is only a controlled
  ablation/reproduction preset, not a final default.
  Do not spend more time on simple down-only expansion or gate/up dense
  fallback as the main path; neither reaches the requested `1.2x` target.
- `base_only_24` is not mainly slow because of low accepted length or idle GPU:
  clean serving rows show similar/high GPU utilization and good CUDA Graph
  coverage. The active bottleneck is the mixed sparse-base plus residual
  correction Linear path, especially `gate_up_proj`.
- `compressed_dense` is GPU-resident when `SPECLINK_SR24_RESIDUAL_DEVICE=cuda`:
  the real-shape backend probe reports CUDA mask, residual values, cached dense
  residual, and position tensors. The current
  `SPECLINK_SR24_COMPRESSED_RESIDUAL_TRITON=1` kernels are not viable for
  serving; they are tens of milliseconds on real Llama shapes and have large
  corrected diffs. Keep that flag off unless the kernel is replaced.
- `all_corrected_24` without the dense fastpath still needs a new fused packed
  sparse+residual operator. Best current exact graph path is two sparse GEMMs:
  rows=512 gate/up is `0.7586ms` versus dense `0.5384ms`; rows=900 gate/up is
  `1.4742ms` versus dense `0.9781ms`.
- 2026-06-28 goal-continuation all-corrected backend refresh:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_sparse_backend_probe_large_shapes_20260628_goal_continue/summary.md`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_triton_tile_sweep_large_shapes_20260628_goal_continue/summary.md`.
  The compressed-dense CUDA path again reports CUDA mask/residual/cached-weight
  tensors, so it is not CPU-side. It is still not the speed path: rows=512
  gate/up dense graph is `0.5401ms`, best exact all-sparse graph is
  `0.7606ms`, and compressed cached graph is `1.0607ms`; rows=512 down dense
  graph is `0.2928ms`, best exact all-sparse graph is `0.3375ms`, and
  compressed cached graph is `0.4835ms`. The residual Triton tile sweep did not
  rescue the current algorithm: best gate/up all-corrected Triton graph was
  `21.6977ms` and best down all-corrected Triton graph was `10.7931ms`.
  Keep `SPECLINK_SR24_COMPRESSED_RESIDUAL_TRITON=0` unless the kernel is
  replaced with a genuinely fused/tensor-core-friendly packed operator.
- Exact `dense_rows` all-row correction is the quality reference. With
  `threshold=1.0`, `non_draft=all`, bucket disabled, and exact routing,
  GSM8K-16 matched dense exactly with draft/non-draft residual fractions both
  `1.0`:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_t08_threshold1_allrows24_exact_gsm8k16_quality_20260625_2255/report.md`.
- A new preset is available in both SR24 runners:
  `--sr24-preset down8_15_residual_only`. It touches only
  `down_proj=8-15` with `dense_rows@cuda`, no base-only tail, reduced CPU sync,
  batched mask builder, bucket size `32`, `low_confidence@0.9`, and
  `non_draft=bonus`.
- Do not use `down_proj=16-31` base-only as a default path: GSM8K-64 was dense
  `0.7344` vs SR24 `0.6875`, paired regressions `5`.
- `down8_15_residual_only` is only an ablation, not the final speed path:
  GSM8K-64 was dense `0.7344` vs SR24 `0.7500` with paired regressions `2`,
  and bs64/math/max256 full-batch throughput was dense `3439.571` vs SR24
  `3487.446` (`1.014x`). It does not meet the 1.2x target.
- `run_lm_eval_accuracy.py` now matches the throughput runner for
  no-fastpath `all_corrected_24 + torch_sparse/compressed_dense`: with
  `--no-sr24-all-corrected-dense-fastpath` it automatically sets the effective
  compressed residual path to `residual_out_chunk=0`,
  `cache_compressed_residual_weight=1`, and
  `prewarm_compressed_residual_weight=1`, unless
  `--no-sr24-auto-compressed-residual-fastpath` or
  `--sr24-compressed-residual-triton` is used. Keep
  `--sr24-residual-device cuda --sr24-require-gpu-residual` for performance
  sanity runs; the run metadata records both CLI and effective values.
- Latest `speclink_t08` write-protection smoke:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_t08_protection_ablation_gsm8k16_20260626_1150/summary.md`.
  On GSM8K-16, `bonus_min4`, `all_min4`, and `all_min8` all reported dense
  `0.5625`, `all_corrected_24` `0.5625`, and `speclink_t08` `0.6875` with
  `0` paired regressions and `2` paired improvements. The useful signal is
  that `bonus_min4` reached the same smoke accuracy while keeping non-draft
  residual fraction at `0.000`, whereas `all_min4/all_min8` force non-draft
  residual fraction to `1.000`. Before adopting it, run a larger GSM8K/Minerva
  50 or GSM8K-64 quality gate.
- That larger `bonus_min4` quality gate failed:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_t08_bonus_min4_quality_gsm8k_minerva50_20260626_1205/summary.md`.
  Dense and `all_corrected_24` matched on GSM8K-50 (`0.7800`) and Minerva-50
  (`0.3800`), but `speclink_t08 bonus_min4` dropped to `0.7200` on GSM8K
  with `3` paired regressions and `0.3400` on Minerva with `2` paired
  regressions. Do not run throughput for this exact `bonus_min4` route; it is
  not quality-safe. The repeated GSM8K regression ids are `11`, `13`, and `15`.
- 2026-06-26 slowdown-breakdown refresh for the current SR24 direction:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_current_slowdown_seven_part_20260626_1240/report.md`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_current_slowdown_component_summary_20260626_1240/report.md`.
  The clean bs64/math/K8 row shows dense `3432.312` full-batch tok/s,
  `base_only_24` `4202.651` (`1.224x`), and `speclink_t08` `3423.765`
  (`0.998x`) with good GPU utilization and CUDA Graph coverage. The bottleneck
  is not scheduler/mask build or global idle GPU; it is mixed useful-work
  duplication in `gate_up_proj=16-31`: sparse base plus dense residual
  correction. Diagnostic per-leaf timing is in
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_current_slowdown_component_summary_20260626_1240/per_leaf_linear_breakdown.csv`.
  Prefer fused/packed mixed-operator work or residual-row reduction with a
  quality gate over another controller-only sweep.
- 2026-06-26 user-refresh slowdown entrypoint:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_slowdown_breakdown_user_refresh_20260626/report.md`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_component_breakdown_user_refresh_20260626/report.md`.
  Use this as the first read before further SR24 optimization. Clean serving
  rows answer end-to-end throughput, CUDA Graph, scheduler, and GPU util.
  Linear/component rows localize `gate_up_proj=16-31` sparse-base plus residual
  correction cost but should not be compared directly as serving throughput,
  because CUDA-event timing forces CUDA Graph `NONE`.
- 2026-06-26 static-tail hook fix: `linear_hooks_enabled()` must not disable
  SR24 hooks for `selective + static_mask_state=all_residual +
  static_all_residual_densefastpath + all_corrected_dense_fastpath` when
  `SPECLINK_SR24_BASE_ONLY_LAYER_IDS` or
  `SPECLINK_SR24_BASE_ONLY_LAYER_IDS_BY_LEAF` is set. Before this fix, static
  tail presets rewrote sparse/base weights but bypassed the SR24 Linear hook,
  causing severe acceptance collapse. Example old `accuracy_first`
  (`gate_up=31;down=31`) max2048 had accepted draft/step `0.0260` and
  full-batch `1010.388` tok/s:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_first_current_bs64_k8_math2048_20260626_1408/report.md`.
  After the hook fix, the same old gate+down tail recovered speed
  (`3324.941` total/steady tok/s and accepted draft/step `4.1660`) but failed
  GSM8K-20 quality (`0.6000 -> 0.1000`, with long repeated/clipped outputs):
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_first_hookfix_quality_gsm8k_minerva20_20260626_1428/report.md`.
  Therefore the old gate+down `accuracy_first` must remain a negative ablation.
- Current gate-up-only static-tail candidate:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_gate_only_hookfix_bs64_k8_math2048_20260626_1436/report.md`
  showed bs64/K8/math/max2048 dense `2400.146` versus SR24 `3650.540`
  total/steady tok/s (`1.52x`) and full-batch `5299.967` versus `5383.115`
  (`1.016x`), with accepted draft/step `3.8755 -> 4.4074`. The paired quality
  gate is not fully solved yet:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_gate_only_hookfix_quality_gsm8k_minerva20_20260626_1439/report.md`
  reported GSM8K-20 `0.6000 -> 0.5500` (`3` regressions, `2` improvements) and
  Minerva-20 `0.2500 -> 0.3500`. Because that still has GSM8K regressions,
  current `accuracy_first` is more conservative: `gate_up_proj=31` with
  `--sr24-gate-up-split up_sparse`. Keep `accuracy_gate_only` for reproducing
  the older fully fused gate-up-only result.
- `accuracy_first/up_sparse` sanity on 2026-06-26:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_first_upsparse_quality_gsm8k20_20260626/`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_first_upsparse_throughput_bs64_math256_20260626/report.md`.
  GSM8K-20 flexible exact match was dense `0.75`, SR24 `speclink_t08`
  `0.75` (`0.70` strict). The bs64/math/K8/max256 throughput sanity was dense
  `2319.275` total/steady tok/s and `3447.489` full-batch tok/s versus SR24
  `2595.208` total/steady (`1.119x`) and `3453.970` full-batch (`1.002x`).
  This is an accuracy-first candidate, not a completed `1.2x` full-batch speed
  result.
- Direct activation cleanup for `accuracy_first/up_sparse`:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_first_upsparse_directact_quality_gsm8k20_20260626/summary.md`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_first_upsparse_directact_throughput_bs64_math256_20260626/report.md`.
  The Llama MLP path now bypasses the intermediate `[gate, up]` concat for
  `up_sparse/gate_sparse` split modes and computes `silu(gate) * up` directly.
  GSM8K-20 stayed at `0.75` flexible exact match (`0.70` strict). A real-shape
  microbench showed only a small local non-fused cat/activation improvement
  (`1.3273ms -> 1.2966ms`), and the bs64/math/K8/max256 serving result stayed
  essentially unchanged at SR24 `2593.595` total/steady tok/s and `3451.744`
  full-batch tok/s. Treat this as cleanup, not the main `1.2x` route.
- Gate/up static-tail follow-ups on 2026-06-26:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_first_gatesparse_throughput_bs64_math256_20260626/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_first_upsparse_30_31_throughput_bs64_math256_20260626/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_upsparse_16_31_throughput_bs64_math256_20260626/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_gate_only_quality_gsm8k_minerva50_20260626/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_accuracy_gate_only_activationaware_quality_gsm8k50_20260626/report.md`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup31_selective_residual_quality_gsm8k50_20260626/report.md`.
  `gate_sparse` for `gate_up_proj=31` was slower than `up_sparse`
  (`3346.546` full-batch tok/s), `gate_up_proj=30-31/up_sparse` was still only
  full-batch parity (`3445.819` versus dense `3430.369`), and
  `gate_up_proj=16-31/up_sparse` regressed full-batch throughput
  (`3316.843` versus dense `3425.060`) plus acceptance. The older fully sparse
  `accuracy_gate_only` path has total-throughput headroom but fails GSM8K-50:
  dense `0.7800`, SR24 `0.6800`, `6` paired regressions and `1` paired
  improvement; Minerva-50 stayed aggregate-neutral at `0.3800` but had paired
  churn. A selective dense-residual attempt for `gate_up_proj=31`
  (`all_if_any_low@0.4`, prefix4, `dense_rows@cuda`) still failed GSM8K-50
  (`0.7800 -> 0.7000`, `4` regressions) and was visibly slow, with
  `1.625x` storage over dense. The `early_dense_tokens=32` repair attempt was
  also negative. With qkv/o also in the dynamic SR24 target set, GSM8K-50
  collapsed to `0.3200` with `23` paired regressions and `22/50` clipped
  outputs:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup31_earlydense32_quality_gsm8k50_20260626/report.md`.
  A cleaner gate-up-only rerun attached only `gate_up_proj=31` and still scored
  `0.6800` versus dense `0.7800`, with `5` regressions and no improvements:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup31_only_earlydense32_quality_gsm8k50_20260626/report.md`.
  An explicit C4 activation-aware mask rerun confirmed the cached mask was
  active (`mask_cache_method=activation_aware`, `cached_mask` on
  `gate_up_proj=31`) but made GSM8K-50 worse: dense `0.7800`, SR24 `0.6400`,
  `8` regressions and `1` improvement. This rules out "forgot the activation
  mask" as the explanation. `run_lm_eval_accuracy.py` still falls back to
  magnitude masks unless `--sr24-mask-path` is provided, so always record the
  mask path in SR24 quality runs. Do not rerun these as main paths, and do not
  implement a base-only early dense fallback as the next quality fix for fully
  sparse gate/up. The current quality-plausible static candidate remains
  `accuracy_first/up_sparse`, but it is parity-level full-batch speed; the next
  real route is a better importance signal for fully sparse gate/up or a fused
  sparse/residual operator.
- 2026-06-27 static leaf recheck at bs64/math/K8/max512 confirmed the same
  read on current code. `accuracy_gate_only` was graph-safe but only
  `1.013x` full-batch and `0.976x` total tok/s; `accuracy_first/up_sparse`
  was `0.993x` full-batch and `0.968x` total; `throughput_aggressive`
  (`gate_up_proj=31;down_proj=30-31`) showed `1.224x` total tok/s but only
  `1.027x` full-batch tok/s. The aggressive GSM8K-20 gate had dense `0.6000`
  and SR24 `0.7000` with `0` paired regressions and `2` paired improvements,
  but this is too small to prove quality. Treat the aggressive result as a
  drain/step-count upper-bound signal, not as a solved stable-kernel speedup:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_static_leaf_gate_only_throughput_bs64_math512_20260627/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_static_accuracy_first_throughput_bs64_math512_20260627/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_static_throughput_aggressive_bs64_math512_20260627/report.md`, and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_static_throughput_aggressive_gsm8k20_accuracy_20260627/report.md`.
  The route-contiguous ablation switch is now exposed in both the GuideLLM
  matrix runner and the lm-eval runner as
  `--sr24-route-contiguous-fastpath`, and is included in the SR24 compile-cache
  fingerprint and run metadata. Use it only with explicit routed row ablations.
- SR24 trace diagnostic fix on 2026-06-26:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup31_activationaware_trace_gsm8k50_20260626/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup31_activationaware_trace_divergence_gsm8k50_20260626_analyzerfix/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_trace_mask_smoke_gsm8k8_20260626/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup31_activationaware_tracefixed_gsm8k50_20260626/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup31_activationaware_tracefixed_divergence_gsm8k50_20260626/report.md`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup31_activationaware_tracefixed_risk_gsm8k50_20260626/report.md`.
  `SPECLINK_TRACE_CONFIDENCE` now receives SR24 verify-plan rows via
  `record_sr24_verify_mask`, and `analyze_sr24_sample_divergence.py` marks old
  traces without `sr24_uses_residual` as `missing` rather than rendering them
  as `00000000`. The GSM8K-8 smoke verified `2072/2072` trace rows with
  `sr24_uses_residual` and `sr24_mask_state=all_residual`. The full GSM8K-50
  fixed-trace rerun has `14800/14800` rows with SR24 mask fields, all
  `sr24_mask_state=all_residual`, and the 8 paired regressions all have
  divergence-step mask `11111111` plus `accepted_base_only_tokens=0`. For the
  static `gate_up_proj=31` ablation this shows the row-level verifier plan is
  fully residual; the quality loss comes from the module-level static base-only
  sparse gate/up layer, not a row-level residual-mask miss. The request-level
  risk analysis over the first 4 decode steps is also negative: regression and
  non-regression requests have similar early reject rate (`0.4715` vs
  `0.4581`) and accepted length (`1.1250` vs `1.2381`), and full regression
  recall requires flagging `43/50` requests. Do not use raw early
  DLM/acceptance features as the next quality fix.
- Route-all row-index caching follow-up on 2026-06-26:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_routeall_cached_rows_instrumented_bs64_math64_20260626/report.md`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_routeall_cached_rows_clean_bs64_math256_20260626/report.md`.
  Mixed verify plans now cache residual/base row indices once per step for
  `route_all_residual_rows` and `route_reuse_base_output`. The diagnostic run
  showed `route_all_residual_rows_cached_plan_hits=656`,
  `route_all_residual_rows_cached_base_rows_hits=400`, and no routed-call
  `route_build_cuda_ms`; scheduler row-index build averaged about
  `0.096ms/step`. Clean route-all improved from the earlier guarded full-batch
  `2880.807` tok/s to `3338.246` tok/s, but same-run dense was `3422.333`
  tok/s, so this cleanup should be kept but is not the final `1.2x` speed path.
- 2026-06-26 Triton bucket dense-GEMM parameter sweep:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_triton_bucket_param_sweep_gateup_20260626_1358/summary.md`.
  Best gate/up Triton graph was `0.7652ms`, slower than dense `0.5392ms` and
  the current bucket-delta path `0.5346ms`; do not use the current Triton
  bucket dense-GEMM prototype as the next speed path.
- 2026-06-26 gate/up pair-aware mask follow-up:
  `examples/evaluate/eval-guidellm/scripts/generate_speclink_sr24_mask.py`
  now supports `--gate-up-pair-aware`. It uses one shared 2:4 input pattern for
  each HF `gate_proj`/`up_proj` channel pair based on combined gate+up
  activation-aware importance, while keeping the normal unfused mask-cache
  format that vLLM reconstructs into fused `gate_up_proj`. Generated cache:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/data/c4_calibration/sr24_masks/llama3_1_8b_activation_aware_pair_gate_up_24.pt`.
  GSM8K-50 quality improved versus the previous activation-aware fully sparse
  `gate_up_proj=31` run: old `0.7800 -> 0.6400`, pair-aware rerun
  `0.7400 -> 0.7200` with `2` paired regressions and `1` improvement:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_pair_gateup_mask_quality_gsm8k50_20260626/report.md`.
  Throughput sanity remains below a robust `1.2x` full-batch result. At
  bs64/K8/math/max256, SR24 total/steady was `2829.610` versus dense
  `2750.760`, full-batch `3294.903` versus `3167.615`:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_pair_gateup_mask_speed_bs64_math_k8_20260626/final/report.md`.
  At max2048, SR24 total/steady was `3857.545` versus dense `3478.622`, but
  full-batch was `4643.182` versus dense `4859.273`:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_pair_gateup_mask_speed_bs64_math_k8_2048_20260626/final/report.md`.
  The no-runtime-stats ablation increased SR24 max2048 total/steady to
  `3984.691` and full-batch to `4659.001`:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_pair_gateup_mask_speed_bs64_math_k8_2048_nostats_20260626/final/report.md`.
  This means CPU-side SR24 stats synchronization costs about `3%` total TPS in
  this run but does not explain the remaining full-batch gap; the bottleneck is
  still sparse operator/graph/scheduling behavior.
  A narrower gate-up-only attach was then tested and promoted into the
  `accuracy_gate_only` preset for both lm-eval and GuideLLM runners. It attaches
  only `model.layers.31.mlp.gate_up_proj` with
  `SPECLINK_SR24_RESIDUAL_TARGET_LEAFS=none`,
  `SPECLINK_SR24_STATIC_MASK_STATE=no_residual`, and the pair-aware cache; it
  no longer attaches qkv/o densefastpath no-op modules. Static attach stats
  showed `module_count_attached=1`, `dense_fastpath_noop=false`, and storage
  over dense `0.625x`. Throughput on bs64/K8/math/max2048 was `3907.449`
  total/steady and `4685.937` full-batch tok/s:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_pair_gateup_only_noopless_speed_bs64_math_k8_2048_20260626/final/report.md`.
  GSM8K-50 stayed aggregate-equal to dense (`0.7200 -> 0.7200`) but still had
  paired churn (`2` regressions and `2` improvements):
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_pair_gateup_only_noopless_quality_gsm8k50_20260626/report.md`.
  This cleanup reduces storage/hook scope but is not the missing `1.2x`
  full-batch speedup; continue treating the bottleneck as sparse
  operator/graph/scheduling behavior plus remaining quality churn.
  A stronger GSM8K-100/Minerva-100 paired check shows that this one-module
  pair-aware `gate_up_proj=31` preset is still not quality-equivalent:
  GSM8K moved `0.7700 -> 0.7600` with `6` regressions and `5` improvements,
  and Minerva exact-match moved `0.4200 -> 0.4100` with `11` regressions and
  `10` improvements. The divergence reports show most regressions start at the
  first few generated tokens, especially Minerva first-token divergences, so
  this is a reasoning-trajectory change rather than a late answer-format issue:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_pair_gateup_only_noopless_quality_gsm8k_minerva100_20260626/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_pair_gateup_only_noopless_quality_gsm8k_minerva100_20260626/divergence_gsm8k/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_pair_gateup_only_noopless_quality_gsm8k_minerva100_20260626/divergence_minerva/report.md`.
  Do not treat `accuracy_gate_only` as a solved accuracy gate; it is a narrow
  speed/storage probe until a paired gate has near-zero regressions.
- 2026-06-26 exact `all_corrected_24` backend follow-up, bs64/K8/math/max256,
  scoped to `gate_up_proj=16-31` and
  `--no-sr24-all-corrected-dense-fastpath`: cached/prewarmed
  `compressed_dense` is GPU-resident (`compressed_dense@cuda`,
  `residual_out_chunk=0`, cache/prewarm true) but reached only `2812.986`
  full-batch tok/s versus same-run dense `3432.348` (`0.819x`);
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_all_corrected_compressed_gpu_clean_20260626_1904/report.md`.
  The current compressed Triton residual kernel is not a usable speed path:
  `836.243` full-batch tok/s versus dense `3429.336` (`0.244x`);
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_all_corrected_compressed_triton_clean_20260626_1906/report.md`.
  The best current exact backend is `torch_sparse` residual with direct
  cuSPARSELt: `3048.777` full-batch tok/s versus dense `3432.067`
  (`0.888x`);
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_all_corrected_torch_sparse_directcslt_clean_20260626_1909/report.md`.
  Conclusion: `compressed_dense` is already on GPU in the intended path; the
  remaining slowdown is duplicated sparse-base plus residual work, not CPU
  transfer. Keep exact `all_corrected_24` as a correctness/control path unless
  a new fused packed sparse-base plus residual operator is implemented; do not
  use the current Triton residual kernel as the next speed direction.
- 2026-06-26 user-requested seven-part slowdown breakdown refresh:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_slowdown_breakdown_user_table_bs64_math_k8_20260626/seven_part_report/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_slowdown_breakdown_user_table_bs64_math_k8_20260626/seven_part_report/seven_part_breakdown.csv`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_slowdown_breakdown_user_table_bs64_math_k8_20260626/component_summary/breakdown_summary.csv`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_slowdown_breakdown_user_table_bs64_math_k8_20260626/component_microbench/summary.md`.
  Clean serving bs64/K8/math/max256 shows dense full-batch `3417.553` tok/s,
  `base_only_24` `3837.893` (`1.123x`, CUDA Graph `{"FULL":126,"NONE":2}`),
  and `speclink_t08` `2938.588` (`0.860x`, CUDA Graph `{"NONE":128}`).
  Average GPU util is similar for all three clean rows, around `88%`, so the
  slowdown is not global GPU idleness or low accepted length. The diagnostic
  linear row localizes `speclink_t08` to `gate_up_proj=16-31`: sparse base
  `0.824ms/call`, dense correction `0.204ms/call`, gather/scatter only
  `0.021ms/call`, with draft residual/base rows `2827/1317` (`0.682`
  residual fraction). Treat diagnostic tok/s as invalid for throughput because
  `--sr24-breakdown-linear` forces synchronization and CUDA Graph `NONE`.
  The current direction is therefore breakdown-first: make dynamic mixed SR24
  graph-safe and avoid duplicated sparse-base plus dense-correction work, or
  find a quality-safe routing signal that greatly reduces corrected rows. Do
  not start another threshold-only controller sweep until one of those two
  bottlenecks changes.
- 2026-06-27 fresh slowdown-first SR24 check:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_math_bs64_k8_20260627/seven_part_report/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_math_bs64_k8_20260627/component_summary/report.md`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_math_bs64_k8_20260627/component_microbench/summary.md`.
  Clean bs64/K8/math/max128 shows dense full-batch `3013.096` tok/s,
  `base_only_24` `3339.718` (`1.108x`, CUDA Graph `{"FULL":62,"NONE":2}`),
  and `speclink_t08` `2424.316` (`0.805x`, CUDA Graph `{"NONE":64}`), with
  similar average GPU util around `83-85%`. Diagnostic linear timing localizes
  the mixed path to `gate_up_proj=16-31`: sparse base `0.851ms/call`, dense
  correction `0.208ms/call`, gather/scatter `0.022ms/call`, and draft
  residual fraction `0.684`. The microbench says the serving-like mixed
  gate/up path is only competitive below roughly `12.5%` residual rows, so the
  current bottleneck is dynamic mixed CUDA Graph loss plus too many corrected
  rows paying sparse-base first, not low accepted length or global GPU idle.
- 2026-06-27 route-all-contiguous follow-up is a negative speed path, not a
  mainline candidate:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_route_all_contiguous_clean_bs64_math256_20260627/report.md`.
  On bs64/K8/math/max256, dense reached `3428.251` full-batch tok/s and
  route-all-contiguous `speclink_t08` reached only `2364.868` (`0.690x`),
  with average GPU util `64.316%`, CUDA Graph `{"NONE":133}`,
  scheduler mask wall `5.791ms/step`, and row-index/bucket construction
  `5.381ms/step`. This confirms that the current route-all implementation's
  dynamic row compaction/index construction is itself a bottleneck; do not run
  accuracy gates on this route unless row routing is first made graph-safe and
  preallocated or fused on GPU. The stats/reporting path now records
  `route_contiguous_fastpath` in remaining SR24 summary branches, and
  `scripts/run_sr24_slowdown_breakdown.py` accepts
  `--sr24-route-contiguous-fastpath`.
- 2026-06-27 GPU-count mask-builder ablation for the guarded SR24 path:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gpu_count_builder_ablation_off_20260627/report.md`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gpu_count_builder_ablation_on_20260627/report.md`.
  Enabling `--sr24-gpu-count-mask-builder` triggered the direct GPU-count mask
  kernel on `25/40` steps, but short-run `speclink_t08` total tok/s dropped
  from `1108.395` to `1070.105`, and scheduler mask build rose from
  `19.915ms/step` to `22.597ms/step`. Do not make this switch the default for
  guarded SR24; it is a negative CPU-sync ablation unless a longer run
  contradicts it.
- 2026-06-27 guarded SR24 runtime-stats on/off clean ablation:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_guarded_stats_on_clean_bs64_math128_20260627/report.md`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_guarded_stats_off_clean_bs64_math128_20260627/report.md`.
  Disabling SR24 runtime stats gave only a small speed change for
  `speclink_t08`: total tok/s `1968.020 -> 1972.997`, full-batch tok/s
  `2868.569 -> 2874.058`. Use stats-off for clean speed-only runs, but do not
  treat runtime stats as the main bottleneck; graph loss and mixed useful-work
  duplication remain the limiting factors.
- 2026-06-27 guarded SR24 routing/correction ablations were also negative:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_guarded_routebucket_clean_bs64_math128_20260627/report.md`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_guarded_bucketcopy_clean_bs64_math128_20260627/report.md`.
  With bs64/math/K8/max128 and `gateup_cap0_dense_guard`,
  `--sr24-route-bucket-rows` reached `2870.514` full-batch tok/s and
  `--sr24-bucket-dense-copy` reached `2870.276`, versus the nearby stats-off
  reference `2874.058`. Both stayed CUDA Graph `{"NONE":64}`. Do not treat
  bucket-row routing or dense-copy correction as the main SR24 speed path; the
  bottleneck remains dynamic mixed graph loss plus sparse-base work on rows that
  also need dense correction.
- 2026-06-27 full-bucket graph and Triton bucket-GEMM probes:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fullbucket_graph_probe_bs64_math128_20260627/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_guarded_triton_bucketgemm_clean_bs64_math128_20260627/report.md`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_guarded_triton_bucketgemm_n64_bs64_math128_20260627/report.md`.
  A full-bucket graph-safe probe with `residual_bucket_size=1024` restored
  CUDA Graph coverage (`{"FULL":62,"NONE":2}`) but fell to `2375.573`
  full-batch tok/s versus dense `3026.491` (`0.785x`), because it effectively
  pays sparse base plus near-full dense correction. The current Triton bucket
  dense-GEMM prototype stayed at about `0.949x` dense full-batch ratio for both
  the default tile and a `BLOCK_N=64` single-point check. Do not spend more time
  on full-bucket graph or simple Triton bucket-GEMM tile sweeps unless a new
  fused/packed design changes the amount of repeated work.
- Static mixed CUDA Graph probe for the same bs64/K8/math/max256 shape:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_static_mixed_graph_probe_bs64_math256_20260626/report.md`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_static_mixed_graph_probe_bs64_math256_20260626/summary.csv`.
  Forcing a static `mixed` mask state and allowing CUDA Graph changed
  `speclink_t08` graph counts from the previous clean breakdown's
  `{"NONE":128}` to `{"FULL":190,"NONE":2}`. Full-batch throughput improved
  from about `2938.6` tok/s to `3353.4` tok/s, but same-run dense still reached
  `3522.0` tok/s. This confirms CUDA Graph loss was a major first-layer
  bottleneck, while the remaining gap is the mixed operator itself: sparse base
  plus dense-row correction is still paid for too many rows.
- Follow-up probes on the static mixed direction:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_static_mixed_graph_quality_gsm8k32_activationaware_20260626/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_static_mixed_graph_adaptive_probe_bs64_math256_20260626/report.md`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_probe_bs64_math256_20260626/report.md`.
  The activation-aware GSM8K-32 sanity had dense `0.7188` and
  `speclink_t08` `0.7500`, with `1` paired regression and `2` paired
  improvements; treat it only as a small sanity check, not as solved quality.
  A previous magnitude-mask quality sanity is not comparable to throughput
  rows because it omitted the activation-aware mask. Adaptive dense fallback
  was negative: full-batch `speclink_t08` `3282.1` tok/s versus same-run dense
  `3506.9`, worse than static mixed without fallback. Current MLP-level
  row-routing was also negative for full-batch throughput: `3112.0` tok/s
  versus dense `3431.7`, with accepted draft tokens/step dropping
  `1.697 -> 1.620` and twice as many attached modules. Do not promote adaptive
  dense fallback or the current row-routed MLP path as the main speed path.
- 2026-06-26 direct cuSPARSELt follow-up:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_direct_cslt_static_mixed_probe_bs64_math256_20260626/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_all_corrected_direct_cslt_probe_bs64_math256_20260626/report.md`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_auto_direct_cslt_baseonly_smoke_20260626/report.md`.
  Direct cuSPARSELt helps `base_only_24`: bs64/math/K8/max256 full-batch
  `4057.8` tok/s versus dense `3435.3` (`1.18x` full-batch, `1.22x`
  total/steady), with accepted draft tokens/step `2.025` versus dense `1.697`
  and similar GPU utilization. This reinforces that base-only is not slow due
  to acceptance collapse or idle GPU. The throughput and lm-eval runners now
  auto-enable direct cuSPARSELt only for `base_only_24` torch-sparse runs; use
  `--no-sr24-auto-direct-cslt-base-only` for the old base-only ablation. Do not
  apply this as a general default to `speclink_t08`: same probe had
  `speclink_t08` full-batch `3343.2` versus dense `3435.3`, and
  `all_corrected_24` direct cuSPARSELt was still `3067.2` versus dense
  `3508.0`. The remaining mixed/all-corrected bottleneck is residual work on
  top of sparse base, not sparse base dispatch alone.
- 2026-06-26 22:30 SR24 continuation:
  latest canonical slowdown diagnosis artifacts are
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_current_slowdown_diagnosis_20260626/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dense_fallback075_quality_gsm8k32_20260626/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_operator_upper_bound_microbench_20260626_2223/summary.md`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_all_corrected_densefastpath_clean_bs64_math_k8_20260626_2225/clean_serving/report.md`.
  In the current scoped bs64/math/K8/max256 path, `base_only_24` is not slow:
  it is faster than dense, has higher accepted draft tokens/step, similar GPU
  utilization, and mostly FULL CUDA Graph coverage. Do not explain future
  scoped `base_only_24` slowdowns as accepted-length collapse unless a fresh
  seven-part report proves that. `all_corrected_24` should remain the
  dense-equivalent correctness/control path: the dense fastpath is near dense
  (`3436.657` vs `3523.289` full-batch tok/s), while current no-fastpath exact
  sparse/residual backends are below dense. The `speclink_t08` bottleneck is
  still mixed useful-work duplication plus graph loss: diagnostic
  `gate_up_proj=16-31` shows sparse base `0.581ms/call`, dense correction
  `0.343ms/call`, and `75.6%` draft residual rows. A GSM8K-32 dense-fallback
  quality gate with fallback fraction `0.75` had dense `0.7188`,
  `speclink_t08` `0.7812`, `Pair reg=0`, `Pair imp=2`; treat this as a small
  correctness guard, not as solved quality or a speed path.
  The operator microbench says current serving-like mixed gate/up is already
  near dense at `6.25%` residual and is `1.86x` dense time at `75%` residual;
  even the ideal prefix-concat upper bound is only `1.05x` dense at `75%`.
  Therefore the next speed work should be either a controller that greatly
  reduces residual rows while passing paired quality gates, or a real fused
  packed mixed operator that avoids sparse-base work for rows later corrected
  by dense. Do not spend another iteration on threshold-only sweeps, current
  `compressed_dense` Triton kernels, or dense fallback tuning unless the run is
  explicitly a quality/control ablation.
- 2026-06-27 row-routed MLP ceiling refresh:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_refresh_20260627/summary.md`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_rows1024_refresh_20260627/summary.md`.
  At rows=512, quality-conservative exact-down row routing is at best
  `0.86x` dense graph time for bucket 128, which is only about `1.16x`
  operator speedup before serving overhead. At rows=1024, exact-down reaches
  `0.78x-0.82x` dense graph time for dense-row buckets 64-256, so there is
  theoretical headroom for `>1.2x` only if the runtime keeps corrected rows in
  a bounded low-to-moderate bucket and preserves CUDA Graph/packed execution.
  This does not change the negative conclusion for high residual fractions:
  current `all_corrected_24` no-fastpath and threshold-only `speclink_t08`
  remain too expensive because they pay sparse-base plus dense/residual work.
  A 2026-06-27 follow-up added
  `SPECLINK_SR24_ROW_ROUTED_MLP_REUSE_BASE_OUTPUT=1` and runner flag
  `--sr24-row-routed-mlp-reuse-base-output`. This path skips scheduler-side
  base-row complement construction for row-routed MLP and instead computes the
  sparse-base MLP for all rows before overwriting selected dense bucket rows.
  It fixed the measured scheduler row-index/bucket wall time in bs128/K8/math:
  bucket64 dropped from `75.841ms/step` to `0.118ms/step`. Throughput improved
  only modestly, though: reuse-base bucket64 reached `3568.978` full-batch
  tok/s versus the same comparison root's dense `3363.972` (`1.061x`), while
  bucket128 reached `3514.762`. Quality is still not solved. Llama GSM8K-20
  bucket64 dropped `0.6500 -> 0.5500` with `2` paired regressions; bucket128
  was aggregate-neutral `0.6500 -> 0.6500` but still had `1` paired regression
  and `1` paired improvement. Qwen GSM8K-20 bucket64 also had `1` paired
  regression. Results:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_reusebase_bs128_bucket64_probe_20260627/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_reusebase_bs128_bucket128_probe_20260627/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_reusebase_quality_gsm8k20_20260627/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_reusebase_bucket128_quality_gsm8k20_20260627/report.md`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_reusebase_microbench_20260627/summary.md`.
  Treat reuse-base as a useful scheduler-overhead fix and operator diagnostic,
  not as the final `speclink_t08` path.
- 2026-06-27 bs128 seven-part slowdown refresh:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_breakdown_reusebase_bs128_math_combined_20260627/seven_part_report/report.md`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_breakdown_reusebase_bs128_math_combined_20260627/component_summary/report.md`.
  Clean bs128/math/K8/max128 showed dense full-batch `3364.397` tok/s,
  `base_only_24` `3867.182` (`1.149x`), and `speclink_t08` `3585.037`
  (`1.066x`), with `speclink_t08` accepting `1.630` draft tokens/step versus
  dense `1.423`, avg GPU util `92.1%`, and CUDA Graph `{"FULL":62,"NONE":2}`.
  This means the current gap is not accepted-length collapse, not GPU idle, and
  not primarily graph loss in this row; the remaining issue is useful-work
  efficiency. The eager diagnostic row reported base sparse Linear
  `1.230ms/call`, residual dense correction `0.150ms/call`, gather/scatter
  `0.018ms/call`, clean scheduler wall `1.020ms/step`, and draft residual/base
  rows `30979/45`. Instrumented breakdown rows now force eager because CUDA
  events invalidate CUDA Graph capture; use clean rows for FULL/NONE graph
  counts.
- Row-routed MLP activation rule: `--sr24-row-routed-mlp` is only a requested
  config. The path actually runs only when `gate_up_proj` and `down_proj` are
  both SR24-attached in the same layer. The bs128 refresh enabled row-routed
  reuse-base but used `gate_up_proj=16-31` and `down_proj=8-15`, so there were
  no overlapping MLP layers and no `row_routed_mlp_*_calls`. Future row-routed
  reports must include `row_routed_mlp_calls` or
  `row_routed_mlp_reuse_base_output_calls`; do not infer execution from the
  boolean flag alone.
- 2026-06-27 true-overlap row-routed check:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_overlap16_31_bs128_math_20260627/seven_part_report/report.md`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_overlap16_31_eagerclean_bs128_math_20260627/clean_serving/report.md`.
  With overlapping `gate_up_proj=16-31;down_proj=16-31`, row-routed MLP
  actually ran (`row_routed_mlp_reuse_base_output_calls=16` in the diagnostic
  row). Graph-enabled clean serving failed during CUDA Graph capture because
  `row_routed_mlp_output()` calls dynamic `bucket_values.nonzero()` at
  `vllm/vllm/speclink_sr24.py:7872`. Eager clean serving was much slower than
  dense: full-batch `2159.013` vs dense `3365.482` tok/s, total `1486.537` vs
  `2624.085` tok/s, accepted draft tokens/step `0.331` vs `1.422`, with GPU
  util still `91.2%`. The diagnostic component split showed reuse-base MLP
  `6.445ms/call`, dominated by sparse-base MLP over all rows
  (`6.167ms/call`), while dense correction was only `0.278ms/call` and
  gather/scatter `0.014ms/call`. Treat all-row reuse-base MLP as a diagnostic,
  not the final `speclink_t08` route. The next speed path should be a
  graph-safe packed route that avoids capture-time `nonzero` and avoids
  computing sparse base for rows later overwritten by dense correction, while
  preserving paired accuracy gates.
- 2026-06-27 packed row-routed graph slice:
  `vllm/vllm/speclink_sr24.py` now gives CUDA Graph capture a persistent
  row-routed bucket plan (`residual_rows` plus `base_rows`) and adds a
  fixed-shape Triton bucket-complement kernel with startup prewarm. The
  `output_count` argument is runtime, not Triton `constexpr`, to avoid
  recompiling for every scheduled-token shape. Use this path with
  `SPECLINK_SR24_ROW_ROUTED_MLP=1`,
  `SPECLINK_SR24_ROW_ROUTED_MLP_REUSE_BASE_OUTPUT=0`,
  `SPECLINK_SR24_CUDAGRAPH_BUCKET=1`, and
  `SPECLINK_SR24_FORCE_CUDAGRAPH_NONE_FOR_MIXED=0`. Validation artifacts:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_packed_graph_triton_runtime_smoke_20260627/clean_serving/report.md`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_packed_graph_triton_runtime_bs128_math_gmem098_20260627/clean_serving/report.md`.
  The fix removes the original true-overlap Graph startup failure and reduces
  row-index/bucket wall time: bs32 dropped from `26.122ms/step` before the
  runtime-count/prewarm fix to `3.213ms/step`, and bs128 reports
  `0.140ms/step` with CUDA Graph `{"FULL":126,"NONE":2}`. However, this is not
  a speed win yet. On bs128/math/K8/max128, packed row-routed `speclink_t08`
  needs `--gpu-memory-utilization 0.98` to leave enough KV cache after Graph
  memory profiling; default `0.84` fails with no KV cache blocks. At gmem
  `0.98`, it reaches only `2180.612` full-batch tok/s and `1528.253` total
  tok/s versus dense `3360.792` and `2615.777`, with accepted draft tokens/step
  `0.330` versus dense `1.419` and GPU util `95.0%`. Current read: Graph and
  scheduler are no longer the first bottleneck for the packed path; the next
  work must fix routing/quality and recover accepted length before more
  scheduler tuning is useful.
- 2026-06-27 user-requested seven-part breakdown refresh:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_bs128_math_current_20260627_180905/seven_part_report/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_bs64_instrumented_current_20260627_180905/component_summary/report.md`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_bs128_math_packed_clean_20260627_180905/component_summary/report.md`.
  The `bs128_math_current` refresh accidentally used
  `--sr24-row-routed-mlp-max-base-rows 256`; use it only for fallback
  Linear/bucket attribution. Its diagnostic bs64 row shows base sparse Linear
  `1.427ms/call`, residual dense correction `0.138ms/call`, gather/scatter
  `0.014ms/call`, and sync-heavy exact routing `39.735ms/step`.
  The corrected packed clean row uses `--sr24-row-routed-mlp-max-base-rows 0`
  and shows the real current issue: full-batch `2113.161` tok/s, total
  `1493.588` tok/s, accepted draft tokens/step `0.295`, GPU util `94.4%`,
  scheduler wall `0.461ms/step`, row bucket `0.141ms/step`, and CUDA Graph
  `{"FULL":126,"NONE":2}`. Current read: scheduler/mask build, CUDA Graph, and
  GPU idle are not first; routing/quality collapses useful accepted draft
  length while the packed sparse/residual machinery still runs.
- 2026-06-27 bonus-priority/bucket sweep:
  `SPECLINK_SR24_BONUS_PRIORITY` is now wired through the vLLM SR24 path and
  exposed as `--sr24-bonus-priority` in both the main matrix runner and
  `run_sr24_slowdown_breakdown.py`. Default remains `4.0`; use `1.0` for the
  current diagnostic path where speculative bonus/non-draft rows should not
  dominate a capped residual bucket. Key artifacts:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bonus_priority1_bs128_math_packed_clean_20260627_VALID/clean_serving/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bonus_priority1_bucket128_bs128_math_packed_clean_20260627/clean_serving/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bonus_priority1_bucket256_bs128_math_packed_clean_20260627_rerun/clean_serving/report.md`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_exact_routing_bonus1_bucket64_bs64_math_stats1_20260627/seven_part_report/report.md`.
  On bs128/math/K8/max128, changing bonus priority from old/default behavior to
  `1.0` at bucket64 improves full-batch throughput from `2113.161` to
  `2810.116` tok/s and accepted draft tokens/step from `0.295` to `0.720`.
  Bucket128 reaches `2990.636` full-batch tok/s and `0.807` accepted
  draft/step, but row-bucket scheduling grows to `2.043ms/step`; bucket256 is
  worse at `2743.412` full-batch tok/s with `4.203ms/step` row-bucket time.
  The bs64 exact-routing diagnostic is sync-heavy and not a throughput row, but
  it shows draft residual/base `10472/0`, non-draft residual/base `1309/3788`,
  and bucket active/requested rows `9/11781`. Current read: the main bottleneck
  is the global capped residual bucket failing to preserve useful draft-row
  correction. Next SR24 optimization should try per-request fair bucket
  allocation or a stricter per-token confidence gate before more scheduler or
  larger-bucket tuning.
- 2026-06-27 draft-position priority sweep:
  `SPECLINK_SR24_DRAFT_POSITION_PRIORITY_SCALE` is now exposed as
  `--sr24-draft-position-priority-scale` in the main matrix runner and
  `run_sr24_slowdown_breakdown.py`; default `0.0` preserves old behavior.
  Positive values add a draft-position band to residual priority, so earlier
  draft positions dominate the capped residual bucket before later draft rows.
  Artifacts:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_position_priority2_bucket64_bs128_math_20260627/clean_serving/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_position_priority10_bucket64_bs128_math_20260627/clean_serving/report.md`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_position_priority10_bucket128_bs128_math_20260627/clean_serving/report.md`.
  On bs128/math/K8/max128 with bonus priority `1.0`, bucket64 scale `10`
  improves accepted draft tokens/step to `0.866` with low row-bucket cost
  (`0.148ms/step`), and bucket128 scale `10` reaches the current best
  `3101.722` full-batch tok/s, `2143.421` total tok/s, and `1.102` accepted
  draft tokens/step. This is still below dense, and bucket128 still spends
  about `2.040ms/step` in row-bucket scheduling. Next optimization should
  replace global top-k with a direct per-request/position bucket builder rather
  than only increasing bucket size or scalar priority.
- 2026-06-28 CPU-sync and bucket-budget refresh: the low-sync graph-safe
  bucket path is no longer primarily CPU-bound. In
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fast_candidate_cpu_sync_ablation_b12_bs64_math256_20260628/`,
  the clean low-sync row has scheduler/mask wall about `0.338ms/step`, CUDA
  Graph `{"FULL":94,"NONE":2}`, GPU util about `90.9%`, and full-batch
  `3967.0` tok/s. The sync-heavy row falls to `1960.4` full-batch tok/s and
  `{"NONE":128}`, so avoid sync-heavy diagnostics for throughput claims, but
  do not expect stats-sync cleanup alone to reach `1.2x`.
  On bs64/K8/math/max256, the fast bucket8 copy row reaches `3980.7`
  full-batch tok/s (`1.130x` same-run dense) but is not quality-safe: the
  matching Triton+copy GSM8K-50 probe is `0.60` versus dense `0.82`.
  Raising the global bucket budget restores quality only by losing the speedup:
  bucket64 priority copy is GSM8K-50 `0.76` versus dense `0.80` and
  full-batch `3339.8` versus dense `3432.8` (`0.973x`), while bucket256 copy is
  GSM8K-50 `0.78` versus dense `0.80` and full-batch `3331.1` versus dense
  `3433.3` (`0.970x`). The `quality_gateup_only` preset is also speed-negative
  here (`3288.3` versus dense `3436.3`, `0.957x`). Current read:
  `SPECLINK_SR24_RESIDUAL_BUCKET_SIZE` is a global per-step row budget, not a
  per-request budget. Small global buckets look fast because they leave many
  quality-relevant rows base-only; once enough rows are corrected, the current
  sparse-base plus dense-row overwrite operator is too slow. The next useful
  work is a better importance signal or a fused/packed mixed Linear/MLP
  operator, not another threshold-only sweep.
  Implementation updates from this pass: `run_lm_eval_accuracy.py` now exposes
  `--sr24-bonus-priority` and `--sr24-draft-position-priority-scale`;
  `run_sr24_slowdown_breakdown.py` now forwards `--sr24-preset`; and the
  Triton bucket-GEMM prototype now matches `--sr24-bucket-dense-copy` semantics
  when both flags are enabled. Keep Triton bucket GEMM diagnostic for now,
  because the bucket256 torch-copy probe (`0.78`) still beats Triton+copy
  (`0.74`) on GSM8K-50.
- 2026-06-27 breakdown-first SR24 pivot:
  latest combined report is
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_slowdown_diagnosis_user_table_20260627/report.md`.
  The current-parameter instrumented row is
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_slowdown_diagnosis_current_best_instrumented_bs64_20260627/component_summary/report.md`.
  Clean best bs128 has good CUDA Graph coverage (`{"FULL":94,"NONE":2}`) and
  high GPU utilization (`86.7%` average, `100%` peak), so do not treat the
  current slowdown as a simple graph miss or idle-GPU problem. The visible
  clean-serving scheduler cost is row-index/bucket construction
  (`2.040ms/step` out of `2.342ms/step`). The row-routed diagnostic shows
  base-side work dominated by `base_gate_up=1.330ms/call`, while dense
  correction is smaller (`0.290ms/call`) and gather/scatter is secondary
  (`0.144ms/call`). Future SR24 work should be breakdown-first: direct
  per-request/position bucket builder, then reduce row-routed base-side work,
  then re-check accepted length.
- 2026-06-27 direct bucket and reuse-base follow-up:
  direct-position bucket is wired as `--sr24-direct-position-bucket` /
  `SPECLINK_SR24_DIRECT_POSITION_BUCKET=1`, but it is not the current speed
  path by itself. The initial implementation created a CUDA tensor from a
  Python list each step and produced `56.090ms/step` batched-builder overhead;
  it now copies through a reused pinned CPU int64 buffer. Even after that fix,
  direct bucket scored only `2994.575` full-batch tok/s on bs128/math/K8/max128.
  The useful ablation is `--sr24-row-routed-mlp-reuse-base-output`:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_reuse_base_output_bucket128_bs128_math_20260627/clean_serving/report.md`
  reduces scheduler mask time to `0.465ms/step` and row bucket/index to
  `0.107ms/step`, but still reaches only `3102.340` full-batch tok/s. Its
  instrumented row
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_reuse_base_output_instrumented_bs64_math_20260627/component_summary/report.md`
  shows row-routed MLP reuse total `3.032ms/call`, sparse base side
  `2.709ms/call`, and dense correction side `0.323ms/call`. Current read:
  scheduler/complement can be made small, but end-to-end speed is still limited
  by row-routed sparse-base-side work versus accepted-token benefit. Next work
  should optimize/fuse the row-routed MLP path rather than only changing bucket
  selection.
- 2026-06-27 same-condition SR24 operator follow-up:
  isolated row-routed MLP microbench
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_probe_rows1024_buckets_20260627/summary.md`
  shows an operator-level path exists at rows=1024/bucket128: dense MLP
  `1.697ms`, exact-down Triton assemble `1.314ms`, no-final-assemble lower
  bound `1.274ms`, and reuse-base `1.322ms`. But same-condition serving with
  bs128/math/K8/max128 and `gpu_memory_utilization=0.95` is still negative:
  dense baseline
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dense_baseline_bs128_math_gpumem095_20260627/clean_serving/report.md`
  is `2620.402` total tok/s and `3294.699` full-batch tok/s with
  `1.423` accepted draft tokens/step. The best checked SR24 row in this group,
  reuse-base
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_reuse_base_output_bucket128_bs128_math_20260627/clean_serving/report.md`,
  is only `2221.430` total tok/s (`0.848x` dense), `3102.340` full-batch
  tok/s (`0.942x` dense), and `1.064` accepted draft tokens/step. Current read:
  clean scheduler/bucket cost is already low (`~0.44-0.47ms/step`), Triton
  final assembly is not enough, and GPU util is high. The next useful SR24
  optimization must recover accepted draft length or use a stronger row-value
  signal before spending more work on row-routed MLP assembly/bucket details.
- 2026-06-27 route-all/prefix-confidence follow-up:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/SR24_SLOWDOWN_BREAKDOWN.md`
  is the current source of truth. The best route-all diagnostic is
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefixconf_t005_gateup_routeall_graph_bs64_math128_20260627/clean_serving/report.md`:
  SR24 full-batch `2961.618` versus same-root dense `3018.108` (`0.981x`),
  accepted draft/step `1.399` versus dense `1.395`, graph profile
  `{"FULL":49}`. The corresponding CPU-sync ablation is under
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_ablation_prefixconf_routeall_bs64_math128_20260627/`.
  It shows `sync_heavy` is bad, but low-sync stats/mask variants are within
  short-run noise, so do not make CPU synchronization the main remaining
  explanation. The refreshed route-all component summary
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefixconf_t005_gateup_routeall_instrumented_bs64_math128_20260627/component_summary/report.md`
  localizes the cost to base sparse GEMM `0.676ms/call` plus dense correction
  `0.184ms/call`; route-all gather/scatter is `0.093ms/linear` and remains
  secondary. A graph-off dense-zero base ablation did not beat graph-off
  torch-sparse (`2615.838` vs `2619.522` full-batch tok/s). A small-base dense
  fallback with `SPECLINK_SR24_ROUTE_MIN_BASE_ROWS=160` was also negative:
  SR24 full-batch `2813.306` versus same-root dense `3020.687`. The next useful
  implementation direction is a fused/packed mixed Linear or better base sparse
  kernel under CUDA Graph, not dense-zero or small-base full-dense fallback.
- 2026-06-27 direct-position vector bucket follow-up:
  `_build_direct_position_bucket_from_active()` now has a device-vectorized
  fast path for full draft-position buckets. It builds position-major row ids on
  GPU when the bucket can be filled entirely by valid draft rows, and falls back
  to the selected-list path when padding or bonus rows are needed. Validation:
  `conda run -n spec python -m py_compile vllm/vllm/speclink_sr24.py
  examples/evaluate/eval-guidellm/scripts/check_speclink_sr24_correctness.py
  examples/evaluate/eval-guidellm/scripts/run_llama31_vllm_fastdraft_smurfs_eagle3_matrix.py`
  and the GPU correctness smoke
  `conda run -n spec python examples/evaluate/eval-guidellm/scripts/check_speclink_sr24_correctness.py`
  both passed. Throughput/routing smokes:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_direct_position_vector_smoke_bs16_20260627/clean_serving/report.md`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_direct_position_vector_bucket64_smoke_bs16_20260627/clean_serving/report.md`.
  These are negative/secondary ablations, not the current speed path:
  bucket32/bucket64 accepted only `0.443`/`0.536` draft tokens per step. The
  vector path can reduce part of builder overhead, but early position-major row
  allocation alone does not recover accepted-token value. Future SR24 work
  should be confidence/value-aware first, then scheduler-efficient; keep the
  seven-part breakdown fields in every candidate.
- 2026-06-27 breakdown-first refresh for the current non-route-all
  `speclink_t08` path:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_slow_parts_bs64_math128_20260627/seven_part_report/report.md`
  is the current seven-part evidence for Llama-3.1-8B, `math_reasoning`,
  EAGLE3 K=8, GuideLLM client-side concurrency/batch size 64, max new tokens
  128. Clean rows: dense EAGLE3 full-batch `3025.137` tok/s, base_only_24
  `3242.405` tok/s with `{"FULL":62,"NONE":2}`, and `speclink_t08`
  `2468.236` tok/s with `{"NONE":64}`. Clean scheduler/mask build for
  `speclink_t08` is only `0.674ms/step`, so scheduler/mask construction is not
  the main clean-path bottleneck in this configuration. The instrumented
  `speclink_t08` row localizes the operator-side cost to gate_up 16-31 sparse
  base `0.486ms/call`, dense residual correction `0.591ms/call`, and
  gather/scatter `0.078ms/call`; routing still sends many draft rows through
  residual correction (`17060/9484` draft residual/base). CPU-sync ablation:
  low-sync stats on/off and GPU-count rows stay near `1715-1768` tok/s, while
  sync-heavy drops to `1274.765` tok/s, so avoid diagnostic sync but do not
  treat ordinary stats overhead as the first-order explanation. Next SR24 work
  should make the mixed path graph-safe or reduce/fuse dense correction work;
  do not start with another threshold-only sweep.
- 2026-06-27 SR24 priority-signal and row-routed MLP check:
  capped residual-bucket priority in `vllm/vllm/speclink_sr24.py` now uses the
  policy-specific value signal: `high_confidence` ranks by DLM selected-token
  probability, `prefix_confidence` ranks by cumulative prefix probability, and
  `low_confidence` keeps risk severity. The GPU smoke
  `conda run -n spec python examples/evaluate/eval-guidellm/scripts/check_speclink_sr24_correctness.py`
  now covers priority direction plus row-routed MLP equivalence against the
  linear-level mixed MLP path. Serving evidence is negative/neutral:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefixconf_prioritysignal_bucket512_bs64_math128_20260627/clean_serving/report.md`
  stayed near the old prefix_conf bucket512 result (`2078.159` SR24 full-batch
  tok/s, accepted draft/step `1.416`, dense `3032.967`), while explicitly
  enabling `--sr24-row-routed-mlp` in
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefixconf_prioritysignal_rowrouted_bucket512_bs64_math128_20260627/clean_serving/report.md`
  was worse (`1951.156` SR24 full-batch tok/s, accepted draft/step `1.223`,
  avg GPU util `63.5%`). Do not promote row-routed MLP as the main
  `speclink_t08` path until a seven-part breakdown shows row selection,
  accepted length, and GPU util recover.
- 2026-06-28 corrected row-routed overlap breakdown:
  the follow-up constrained both residual leafs to late layers with
  `--sr24-residual-layer-ids-by-leaf 'gate_up_proj=16-31;down_proj=16-31'`
  and reran the user-requested seven-part breakdown. Artifacts:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_overlap16_31_breakdown_bs64_math128_20260628/clean_serving/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_overlap16_31_breakdown_bs64_math128_20260628/component_summary/report.md`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_overlap16_31_breakdown_bs64_math128_20260628/seven_part_report/report.md`.
  Clean serving is still below dense: dense full-batch `3026.088` tok/s,
  corrected row-routed `speclink_t08` `2597.598` tok/s (`0.858x`), with
  accepted draft/step `1.230` versus dense `1.403`, GPU util `86.2%`, and
  CUDA Graph `{"FULL":62,"NONE":2}`. Breakdown shows scheduler/mask clean cost
  only `1.310ms/step`; row-routed sparse base is `0.577ms/call`, dense
  correction is larger at `0.760ms/call`, and gather/scatter is secondary.
  Current diagnosis: the path is slow because it lowers useful speculative
  progress while paying both sparse base and large dense correction work. Future
  SR24 work should be breakdown-first and should not optimize row-routed MLP
  assembly alone; a candidate must first recover accepted draft length, then
  reduce/fuse correction work.
- 2026-06-28 base-only and all-corrected late-MLP refresh:
  same-scope Llama-3.1-8B `math_reasoning`, bs64, K=8, max new tokens 128 rows
  under `gate_up_proj=16-31;down_proj=16-31` show that `base_only_24` is not
  the current slow path. Artifacts:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_latemlp_bs64_math128_20260628/clean_serving/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_torchsparse_default_bs64_math128_20260628/clean_serving/report.md`,
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_torchsparse_directcslt_bs64_math128_20260628/clean_serving/report.md`,
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_operator_probe_20260628/summary.md`.
  `base_only_24` reaches `5389.838` full-batch tok/s versus same-root dense
  `3021.525`, with accepted draft/step `2.444`, GPU util `81.8%`, and CUDA
  Graph `{"FULL":69,"NONE":2}`. In contrast, `all_corrected_24` with
  torch-sparse residual is `2551.468` full-batch tok/s versus dense
  `3016.036`, while accepted draft/step remains normal (`1.400` versus dense
  `1.396`) and GPU util is high (`88.4%`). Direct cuSPARSELt is slightly worse
  (`2515.072` full-batch tok/s), so do not enable it for `all_corrected_24` by
  default. The operator probe confirms the reason: current exact graph paths
  are still slower than dense for representative Llama MLP shapes, and cached
  `compressed_dense` is GPU-resident but not faster. The next real
  all-corrected optimization needs a fused packed base+residual kernel; another
  Python dispatch/direct-cslt wrapper is not enough.
- 2026-06-28 current `speclink_t08` quality-safe candidate and slowdown
  breakdown:
  the current quality-safe selector is `critical_prefix` with
  `--sr24-threshold 0.6`, `--sr24-selective-min-prefix-residual 4`, and
  `--sr24-selective-extra-after-low 1`, using
  `gate_up_proj=16-31;down_proj=8-15` residual correction and
  `--sr24-selective-non-draft-policy bonus`. The GSM8K-50 quality gate is:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_gsm8k50_20260628/report.md`.
  It scores `0.7200`, matching the previous dense/spec-safe reference, while
  the acceptance trace reports accepted effective base-only fraction `0.0228`,
  rejected base-only fraction `0.0389`, and mean residual rows/step `4.455`.
  Same-condition CPU-sync serving ablation:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_bs64_math128_20260628/`.
  With `max_tokens=128`, no-sync full-batch SR24 is `2807.819` tok/s versus
  same-root dense `3014.282`; enabling mask-state sync lowers SR24 to
  `2691.279` and adds about `5.04ms/step` of mask-state wall time. The
  `--sr24-batched-uniform-direct` ablation is negative here (`2769.327`
  full-batch tok/s and larger batched-builder time), so keep the normal
  batched mask builder.
  Full seven-part breakdown:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_breakdown_bs64_math128_20260628/seven_part_report/report.md`.
  Clean serving is dense `3132.894` full-batch tok/s versus SR24 `2804.983`
  (`0.895x`), with SR24 GPU util `77.2%` and CUDA Graph `{"NONE":64}`.
  Clean scheduler/mask build is only `0.372ms/step`; instrumented timing
  localizes the mixed-path cost to sparse base `0.985ms/call`, residual
  correction `0.148ms/call`, and gather/scatter `0.012ms/call`. Conclusion:
  reducing CPU sync is useful and should stay enabled, but the remaining
  speed gap is mostly mixed-path graph loss plus inefficient sparse-base
  useful work, not Python stats or gather/scatter.
- 2026-06-28 mixed CUDA Graph positive SR24 turn:
  with the same `critical_prefix@0.6,prefix4,extra1` selector, enabling
  `--sr24-dynamic-auto-cudagraph --sr24-cudagraph-bucket
  --no-sr24-force-cudagraph-none-for-mixed` makes mixed `speclink_t08`
  graph-capable. Throughput artifact:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_graphon_bs64_math128_20260628/clean_serving/report.md`.
  Llama-3.1-8B, `math_reasoning`, bs64, K=8, max new tokens 128:
  dense full-batch `3026.614` tok/s, graph-on SR24 full-batch `3118.481`
  tok/s (`1.030x`), total tok/s `2003.767` versus dense `2184.300`, accepted
  draft/step `1.569`, avg GPU util `84.75%`, and CUDA Graph
  `{"FULL":78,"NONE":2}`. Paired GSM8K-50 quality artifact:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_graphon_paired_gsm8k50_20260628/report.md`.
  It reports dense `0.7200`, graph-on SR24 `0.7400`, pair reg `1`, pair imp
  `2`. This proves the graph guard was a major serving bottleneck and that the
  current graph-on path is quality-plausible, but it is still short of the
  requested `1.2x` dense target. Next experiments should keep graph-on and
  reduce/fuse residual correction work, starting with narrower residual leafs
  or fewer residual layers.
  Negative follow-ups under the same bs64/math128 graph-on setup:
  `gate_up_proj=16-31` only
  (`results.bak/sr24_critical_t06_prefix4_extra1_graphon_gateup16_31_bs64_math128_20260628/clean_serving/report.md`)
  reaches only `2968.512` full-batch tok/s with accepted draft/step `1.493`;
  `down_proj=8-15` only
  (`results.bak/sr24_critical_t06_prefix4_extra1_graphon_down8_15_bs64_math128_20260628/clean_serving/report.md`)
  reaches `2953.232` full-batch tok/s with accepted draft/step `1.443`;
  threshold `0.7`
  (`results.bak/sr24_critical_t07_prefix4_extra1_graphon_full_bs64_math128_20260628/clean_serving/report.md`)
  reaches `3088.045`, below threshold `0.6`; bucket16
  (`results.bak/sr24_critical_t06_prefix4_extra1_graphon_bucket16_bs64_math128_20260628/clean_serving/report.md`)
  reaches `3108.951`, also below bucket32. Do not rerun these as primary
  candidates unless a code change alters the operator cost model.
- 2026-06-28 older user-requested seven-part breakdown refresh:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_breakdown_bs64_math128_20260628_0322/seven_part_report/report.md`
  is a historical slowdown reference for the quality-safe
  `critical_prefix@0.6,prefix4,extra1` bucket32 path with
  `gate_up_proj=16-31;down_proj=8-15`. Clean serving rows on
  Llama-3.1-8B/math_reasoning/bs64/K8/max128 are: dense full-batch
  `3128.512` tok/s, `base_only_24` `3428.164` (`1.096x`), and
  `speclink_t08` `3182.699` (`1.017x`). `speclink_t08` still has healthy
  CUDA Graph coverage (`{"FULL":62,"NONE":2}`), avg GPU util `86.625%`, and
  accepted draft/step `1.606`, so do not diagnose this path as idle GPU,
  graph loss, or accepted-length collapse. The instrumented row localizes the
  useful-work cost to sparse base `1.007ms/call` plus dense-row correction
  `0.148ms/call`; gather/scatter is only `0.015ms/call`, and clean
  scheduler/mask build is `0.949ms/step`. Future candidates should first
  produce this seven-part table and then target a fused/packed mixed operator
  or much lower residual-row fraction. For multi-leaf SR24 layer specs, use
  semicolons, for example
  `--sr24-residual-layer-ids-by-leaf 'gate_up_proj=16-31;down_proj=8-15'`;
  comma between leaf specs is invalid because commas are reserved for layer
  lists inside one leaf.
- 2026-06-28 all-MLP SR24 CPU-sync/bucket/priority follow-up:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_bs64_math128_20260628_0220/report.md`
  is the current best all-MLP graph-capable `speclink_t08` row. It uses
  Llama-3.1-8B, `math_reasoning`, GuideLLM client-side concurrency 64, EAGLE3
  K=8, max new tokens 128, all MLP leafs (`gate_up_proj,down_proj`), bucket32,
  and the default bonus-priority policy. It reaches `3545.363` full-batch
  tok/s versus same-root dense `3025.159` (`1.172x`), accepted draft/step
  `2.345`, and CUDA Graph `{"FULL":55,"NONE":9}`, but total tok/s is still
  worse than dense (`1717.158` versus `2187.155`). Follow-up ablations were
  negative: stats-off/no-extra-sync
  (`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_stats_off_bs64_math128_20260628_0405/report.md`)
  reaches only `3462.887` full-batch tok/s, bucket16
  (`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_bucket16_bs64_math128_20260628_0415/report.md`)
  reaches `3394.140`, bucket64
  (`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_bucket64_bs64_math128_20260628_0418/report.md`)
  reaches `3284.260`, bonus priority 1
  (`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_bucket32_bonus1_bs64_math128_20260628_0425/report.md`)
  reaches `2746.878`, and draft-position priority scale 10
  (`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_bucket32_pos10_bs64_math128_20260628_0430/report.md`)
  reaches `2874.849`. Keep CPU-sync reductions enabled and keep sync-heavy
  routing diagnostics diagnostic-only, but do not spend more primary time on
  scalar bucket/bonus/position sweeps unless a new operator changes the cost
  model. The next plausible speed path is a fused or packed mixed Linear path
  that avoids paying sparse base for rows that are then corrected by dense
  residual work.
- 2026-06-28 all-MLP Triton bucket override gate:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_tritonoverride_bs64_math128_20260628_0500/report.md`
  reaches the speed target in the full-batch window: Llama-3.1-8B,
  `math_reasoning`, bs64, K=8, max128, all MLP residual leafs, Triton bucket
  override, `critical_prefix` prefix4 gives SR24 full-batch `3645.945` tok/s
  versus same-root dense `3025.805` (`1.205x`), total tok/s `1993.722`, and
  accepted draft/step `2.379`. It is not quality safe:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_tritonoverride_paired_gsm8k50_20260628_0510/report.md`
  reports GSM8K-50 dense `0.7200` versus SR24 `0.7000`, pair reg `4`, pair
  imp `3`. Prefix5 and prefix6 fall below the `1.2x` target, and
  `low_confidence` prefix4 is worse on quality (`0.6800`, pair reg `4`).
  Treat all-MLP Triton override as a speed upper bound, not the default SR24
  path. The current quality-safe graph-on scoped path remains the correctness
  candidate, but only reaches about `1.02-1.03x`; the next useful work is
  serving-shape regression tracing plus a fused/packed mixed Linear or a much
  sharper quality-aware selector, not another plain threshold/bucket sweep.
- 2026-06-28 graph-safe route-bucket split follow-up:
  `vllm/vllm/speclink_sr24.py` can now build persistent bucket/complement row
  plans for `--sr24-route-bucket-rows` under CUDA Graph, and the matrix runner
  no longer blocks dynamic-auto graph capture for that route when
  `--sr24-cudagraph-bucket` is enabled. Correctness smoke passed:
  `conda run -n spec python examples/evaluate/eval-guidellm/scripts/check_speclink_sr24_correctness.py`.
  The serving result is negative:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_routebucket_cached_graphon_allow_bs64_math256_20260628/report.md`
  has `speclink_t08` CUDA Graph `{"FULL":126,"NONE":2}`, but only
  `3671.220` full-batch tok/s and `2522.459` total tok/s versus same-root
  dense `3525.085` and `2603.175`. The normal scoped Triton bucket override
  remains better:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_qualitysafe_tritonoverride_graphon_bs64_math256_20260628/report.md`
  with `3855.145` full-batch and `2638.427` total tok/s. Conclusion:
  PyTorch split routing is not the fused operator; it skips sparse-base work on
  bucket rows but pays gather, small GEMMs, and output assembly. Keep it as a
  diagnostic ablation, not the default SR24 path.
- 2026-06-28 Triton bucket dense GEMM correction follow-up:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_triton_bucket_dense_gemm_graphon_bs64_math256_20260628/report.md`
  tests `--sr24-triton-bucket-dense-gemm` on the same scoped graph-on
  bs64/math/K8/max256 setup. It is negative: `speclink_t08` keeps CUDA Graph
  coverage (`{"FULL":94,"NONE":2}`) and accepted draft/step is `2.188`, but
  throughput is only `3581.018` full-batch and `2468.352` total tok/s. This is
  worse than normal Triton bucket override. Do not spend primary time tuning
  the correction-only Triton dense GEMM unless a new microbench shows a clear
  kernel-level win. The remaining speed path needs fused/packed sparse-base
  plus residual correction, not just removing the intermediate dense-output
  tensor.
- 2026-06-28 per-leaf residual backend follow-up:
  `SPECLINK_SR24_RESIDUAL_BACKEND_BY_LEAF` and runner flag
  `--sr24-residual-backend-by-leaf` allow narrow mixed-backend ablations such
  as `gate_up_proj=torch_sparse;down_proj=dense_rows`. This was motivated by
  the component microbench
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_component_bucket32_current_shape_20260628/summary.md`:
  with rows=512/bucket32, gate/up residual sparse delta is faster than the
  dense-row correction proxy at low residual fractions, while down residual
  sparse is slower. Serving result:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_leafbackend_gateup_sparse_down_dense_graphon_bs64_math256_20260628/report.md`
  reports `speclink_t08` `3885.001` full-batch tok/s and `2593.116` total
  tok/s with CUDA Graph `{"FULL":126,"NONE":2}`. This is a small full-batch
  improvement over scoped dense_rows+Triton override, but not enough for the
  `1.2x` target and total tok/s is lower. The all-corrected microbench
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_component_allcorrected_residual_sparse_20260628/summary.md`
  shows dual-sparse all-corrected is not a speed path: gate/up is `1.39x`
  dense and down is `1.19x` dense at residual fraction `1.0`.
- 2026-06-28 bucket-copy/direct-cuSPARSELt scoped candidate:
  the latest scoped graph-safe speed candidate keeps the quality-safe selector
  `critical_prefix@0.6,prefix4,extra1`, `bonus` non-draft correction,
  `gate_up_proj=16-31;down_proj=8-15`, `dense_rows@cuda`, low-sync stats,
  static mask buffer, batched mask builder, dynamic-auto CUDA Graph, and graph
  bucket capture, but changes the correction/operator switches to bucket16,
  `--sr24-bucket-dense-copy`, and `--sr24-direct-cslt-linear`. Throughput:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_qualitysafe_directcslt_bucketcopy_b16_graphon_bs64_math256_20260628/report.md`.
  On Llama-3.1-8B, `math_reasoning`, bs64, K=8, max new tokens 256,
  `speclink_t08` reaches total `2720.019` tok/s versus same-run dense
  `2319.712` (`1.173x`) and full-batch `3930.796` versus dense `3429.905`
  (`1.146x`), with accepted draft/step `2.198` and CUDA Graph
  `{"FULL":94,"NONE":2}`. A small GSM8K-20 paired sanity at
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_directcslt_bucket16_quality_gsm8k20_20260628/report.md`
  matches dense (`0.7000` vs `0.7000`, pair reg/imp `0/0`), but this is not a
  full quality proof. Bucket32 without direct-cslt is `2640.935` total,
  bucket32 with direct-cslt is `2680.897`, bucket8 with direct-cslt is
  `2704.736`, and bucket4 drops to `2643.398`. Use bucket16/direct-cslt as the
  current comparison point, but do not claim the objective is complete: it is
  still below the requested `1.2x` dense target.
- 2026-06-28 bucket-copy instrumented breakdown:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bucketcopy_instrumented_bs64_math128_20260628/seven_part_report/report.md`
  shows scheduler mask wall around `0.448ms/step`, diagnostic exact mask
  `0.474ms/step`, base sparse Linear `2.128ms/call`, residual dense GEMM
  `0.132ms/call`, and gather/scatter about `0.004ms/event`. This confirms
  bucket writeback is not the primary bottleneck; the remaining useful-work
  issue is the separate sparse-base pass plus dense correction. Direct
  cuSPARSELt helps the base sparse path but is insufficient by itself. The next
  real speed path remains a fused/packed mixed Linear/MLP operator or a
  quality-gated controller that sharply reduces corrected rows.
- 2026-06-29 all-MLP prefix5 slowdown refresh:
  `scripts/run_sr24_slowdown_breakdown.py` now expands
  `--sr24-preset mlpall_lowconf_prefix5_tritonoverride` locally before calling
  the matrix runner, so breakdown rows match the current all-MLP speed-target
  preset instead of the wrapper's older manual defaults. Fresh targeted output:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_lowconf_prefix5_breakdown_bs64_math128_20260629/seven_part_report/report.md`.
  On Llama-3.1-8B/math_reasoning/bs64/K8/max128, clean fixed-64 rows are:
  dense `3036.752` full-batch tok/s and `2267.509` total tok/s, `base_only_24`
  `5108.797` full-batch and `2700.166` total, and all-MLP prefix5
  `speclink_t08` `3661.049` full-batch (`1.206x`) but only `1953.180` total
  (`0.861x`). The component row reports base sparse `1.216ms/call`, residual
  correction `0.135ms/call`, gather/scatter `0.046ms/event`, draft
  residual/base rows `10953/1327`, non-draft residual/base rows `1535/3788`,
  and bucket fill `0.981`. Read: full-batch speed exists, but the path still
  corrects too many draft rows and loses fixed-request/continuous serving
  throughput. Continuous refill confirms this:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_lowconf_prefix5_streaming_bs64_math128_20260629/`
  gives dense/SR24 steady tok/s `2622.266/2288.145`, while the nonuniform dense
  fallback at
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_lowconf_prefix5_streaming_nonuniform_dense_bs64_math128_20260629/`
  is also negative (`2629.506/2125.526` steady tok/s) because it recovers GPU
  util/graph behavior but drops SR24 accepted draft tokens/step to `1.515`.
  Future SR24 work should start from a seven-part breakdown
  (scheduler/mask, sparse base, residual correction, gather/scatter, routing
  fractions, CUDA Graph modes, GPU util) and target residual-row reduction or a
  fused/packed mixed operator, not another plain bucket/threshold sweep.
- 2026-06-29 exact all-corrected slowdown follow-up:
  `run_llama31_vllm_fastdraft_smurfs_eagle3_matrix.py` has
  `--sr24-cudagraph-stats` for clean CUDA Graph FULL/NONE counts without heavy
  SR24 Linear timing, and `run_sr24_slowdown_breakdown.py` forwards it by
  default. The slowdown wrapper now preserves explicit
  `--sr24-residual-bucket-size` and `--no-sr24-residual-bucket-priority`
  overrides when expanding presets. Use this when checking exact
  `all_corrected_24`; otherwise the all-MLP preset can silently profile
  bucket32 instead of no-bucket exact correction. Latest no-bucket clean run:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_exact_nobucket_clean_graph_smoke_bs64_math64_20260629/`.
  On Llama-3.1-8B/math_reasoning/bs64/K8/max64, dense/all-corrected
  full-batch tok/s is `2628.996/1250.198`, accepted draft tokens/step is
  `1.148/1.129`, GPU util is `70.6%/60.7%`, and exact all-corrected uses CUDA
  Graph `{"NONE":32}`. The component profile at
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_exact_nobucket_direct_component_profile_bs64_math64_20260629/`
  reports scheduler mask build only `0.037ms/step`, but base sparse Linear
  `1.038ms/call` and residual dense full GEMM `0.673ms/call`. Current
  diagnosis: exact all-corrected is slow because it pays sparse base plus a
  second dense/full-residual pass and loses CUDA Graph, not because accepted
  length collapses. Future speed work should reduce corrected rows or fuse the
  mixed operator; do not use bucketed all-corrected as an exact control.
- 2026-06-29 current slowdown pivot:
  before another SR24 controller/threshold sweep, use the seven-part breakdown
  in `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/SR24_SLOWNESS_BREAKDOWN_CURRENT.md`
  and
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_current_user_pivot_breakdown_20260629/report.md`.
  The current read is: `base_only_24` has a real sparse-base upper bound and is
  not accepted-length limited; guarded `speclink_t08` is slow because it loses
  CUDA Graph coverage and still corrects many rows; exact no-bucket
  `all_corrected_24` is slow because it pays sparse base plus a second
  dense/full residual path; early-dense/default-compile `all_corrected_24`
  is near dense-equivalent (`2626.557` vs same-root dense `2840.707`
  full-batch tok/s), so SR24 bookkeeping itself is not the bottleneck.
  Every next speed candidate should report scheduler/mask, base sparse Linear,
  residual correction, gather/scatter, routing fractions, CUDA Graph modes, and
  GPU util before being treated as progress.
- 2026-06-29 follow-up for the active SR24 slowdown goal:
  exact `all_corrected_24` early-dense hook was rechecked at
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_earlydense_down015_bs64_math256_20260629_2/`.
  Llama-3.1-8B/math_reasoning/bs64/K8/max256 reported dense/all-corrected
  full-batch `3520.661/3479.719` tok/s and total `2345.053/2335.954`
  tok/s. This proves the hook path avoids the sparse-base plus dense-correction
  double compute and is dense-equivalent; it is not a sparse exact speedup.
  The current quality-safe `speclink_t08` speed candidate remains bucket16,
  direct cuSPARSELt, bucket dense copy, `critical_prefix@0.6`, prefix floor 4,
  extra-after-low 1, `non_draft=bonus`,
  `gate_up_proj=16-31;down_proj=8-15`. Its GSM8K-50 paired gate at
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_directcslt_bucket16_quality_gsm8k50_20260629/report.md`
  matched dense exactly (`0.7200/0.7200`, pair reg/imp `0/0`, avg output
  tokens `89.34`). The matching bs64/math/max256 throughput remains below the
  final target: total `2720.019` vs dense `2319.712` (`1.173x`) and full-batch
  `3930.796` vs dense `3429.905` (`1.146x`). Next work should target a fused
  mixed sparse-base+dense-correction operator or a selector that cuts corrected
  rows further without breaking the paired gate.
  A same-setup `predicted_full_accept` non-draft policy probe at
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_predfullaccept_bucket16_directcslt_bs64_math256_20260629/`
  is negative: it reduces non-draft residual fraction from `0.5719` to `0`,
  but SR24 total/full-batch throughput falls to `2665.145/3916.066` and
  accepted draft tokens/step drops to `2.176`. Keep `non_draft=bonus` for the
  current quality-safe candidate.
