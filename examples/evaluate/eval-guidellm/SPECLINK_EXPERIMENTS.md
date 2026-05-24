# SpecLink Experiment Organization

This file turns `speclink.md` into an executable experiment structure for the
current `eval-guidellm` checkout. It separates real end-to-end measurements from
trace simulation and microbenchmark-only evidence.

## Result Root

Use one timestamped root per full run:

```bash
export SPECLINK_TS="$(date +%Y%m%d_%H%M%S)"
export SPECLINK_ROOT="results/speclink_${SPECLINK_TS}"
mkdir -p "${SPECLINK_ROOT}"/{00_env,01_dataset,02_baselines,03_breakdown,04_sparse_challenge,05_speclink,tables,figures}
```

## Phase 0: Environment and Dataset

```bash
conda run -n spec python scripts/speclink_env_audit.py \
  --out "${SPECLINK_ROOT}/00_env/env.json"

conda run -n spec python scripts/speclink_probe_dataset.py \
  --dataset data/math_reasoning.jsonl \
  --out-dir "${SPECLINK_ROOT}/01_dataset"
```

Expected outputs:

- `00_env/env.json`
- `00_env/peagle_patch_check.txt`
- `01_dataset/dataset_profile.json`
- `01_dataset/dataset_examples.jsonl`

## Phase 1: Baseline Smoke

These are real vLLM + GuideLLM + accuracy runs. They require GPU access.

```bash
GUIDELLM_RATE=1 REQUEST_TYPE=chat_completions conda run -n spec bash scripts/speclink_run_method.sh \
  --method dense \
  --num-spec-tokens 0 \
  --port 8009 \
  --dataset data/math_reasoning.jsonl \
  --output-dir "${SPECLINK_ROOT}/02_baselines/dense_smoke" \
  --max-tokens 512 \
  --accuracy-limit 10 \
  --benchmark-limit 10 \
  --repeat-id 0

GUIDELLM_RATE=1 REQUEST_TYPE=chat_completions conda run -n spec bash scripts/speclink_run_method.sh \
  --method eagle3 \
  --num-spec-tokens 8 \
  --port 8010 \
  --dataset data/math_reasoning.jsonl \
  --output-dir "${SPECLINK_ROOT}/02_baselines/eagle3_smoke" \
  --max-tokens 512 \
  --accuracy-limit 10 \
  --benchmark-limit 10 \
  --dense-reference-jsonl "${SPECLINK_ROOT}/02_baselines/dense_smoke/accuracy_outputs.jsonl" \
  --repeat-id 0

GUIDELLM_RATE=1 REQUEST_TYPE=chat_completions conda run -n spec bash scripts/speclink_run_method.sh \
  --method peagle \
  --num-spec-tokens 8 \
  --port 8011 \
  --dataset data/math_reasoning.jsonl \
  --output-dir "${SPECLINK_ROOT}/02_baselines/peagle_smoke" \
  --max-tokens 512 \
  --accuracy-limit 10 \
  --benchmark-limit 10 \
  --dense-reference-jsonl "${SPECLINK_ROOT}/02_baselines/dense_smoke/accuracy_outputs.jsonl" \
  --repeat-id 0
```

## Phase 2: Formal Baseline Matrix

Run the following matrix after smoke succeeds:

- Methods: `dense`, `eagle3`, `peagle`
- K: dense `0`; EAGLE3/P-EAGLE `2,4,8`
- GuideLLM rate: `1,2,4`
- Repeats: `0,1,2`
- Accuracy: full `math_reasoning.jsonl` unless runtime is explicitly recorded
- Max tokens: `512`, increase to `1024` only if truncation is observed

After runs finish:

```bash
conda run -n spec python scripts/speclink_collect_baselines.py \
  --root "${SPECLINK_ROOT}/02_baselines" \
  --out-csv "${SPECLINK_ROOT}/tables/baseline_summary.csv" \
  --out-md "${SPECLINK_ROOT}/tables/baseline_summary.md"
```

To print the full baseline matrix commands without executing them:

```bash
bash scripts/speclink_run_matrix.sh \
  --root "${SPECLINK_ROOT}" \
  --matrix baseline
```

Add `--run` to execute the printed commands and `--skip-complete` to resume an
interrupted matrix without rerunning rows whose required files already exist and
whose `accuracy_summary.json` has `errors=0`. The resume check also requires
the existing row's recorded `accuracy_limit` and `benchmark_limit` to match the
requested run and the GuideLLM request count to match `accuracy_summary.n`, so
smoke rows are not reused for formal limited sweeps and partial GuideLLM rows
are rerun. For formal baseline rows, EAGLE3/P-EAGLE accuracy equivalence is
compared against the matching
`dense_rate{rate}_r{repeat}/accuracy_outputs.jsonl` reference. Use
`--accuracy-limit` and `--benchmark-limit` only for smoke/debug runs; omit them
for the formal matrix.

To resume just the current rate-1, repeat-0, K=8 comparison:

```bash
bash scripts/speclink_run_matrix.sh \
  --root "${SPECLINK_ROOT}" \
  --matrix baseline \
  --rates 1 \
  --repeats 0 \
  --k-values 8 \
  --skip-complete \
  --run
```

Audit the formal matrix after any resume:

```bash
conda run -n spec python scripts/speclink_matrix_status.py \
  --root "${SPECLINK_ROOT}" \
  --matrix baseline \
  --out-tsv "${SPECLINK_ROOT}/tables/formal_matrix_status.tsv" \
  --out-md "${SPECLINK_ROOT}/tables/formal_matrix_status.md"
```

## Phase 3: Breakdown

The required table is generated from `SPECLINK_PROFILE` JSONL events. The
current repo provides a patch helper for the installed vLLM package:

```bash
conda run -n spec python scripts/apply_vllm_speclink_patch.py
conda run -n spec python scripts/write_vllm_speclink_diff.py
```

The patch writes:

- `SPECLINK_TRACE_OUT`: plan-only sparse KV layout traces.
- `SPECLINK_PROFILE_OUT`: Speclink planner events plus coarse vLLM engine-step
  timing events.

The current timing fields are engine-level timings. `target_verify_forward_ms`
is the async batch-queue blocking model/sampling wait, not an isolated
verifier-only CUDA kernel time.

```bash
conda run -n spec python scripts/speclink_parse_profile.py \
  --profile-jsonl "${SPECLINK_ROOT}/03_breakdown" \
  --out-csv "${SPECLINK_ROOT}/tables/breakdown_summary.csv" \
  --out-md "${SPECLINK_ROOT}/tables/breakdown_summary.md" \
  --figure "${SPECLINK_ROOT}/figures/breakdown_stacked_bar.png" \
  --exclude-run-substring _worker_profile
```

To print the formal EAGLE3/P-EAGLE breakdown commands:

```bash
bash scripts/speclink_run_matrix.sh \
  --root "${SPECLINK_ROOT}" \
  --matrix breakdown
```

## Phase 4: Sparse Layout Challenge

Use real or explicitly marked proxy traces. Do not report simulator output as
real sparse-kernel speedup.

```bash
conda run -n spec python scripts/speclink_sparse_layout_sim.py \
  --traces "${SPECLINK_ROOT}/05_speclink/speclink_prob_profile_smoke/live_sparse_trace.jsonl" \
  --out-csv "${SPECLINK_ROOT}/tables/sparse_layout_summary.csv" \
  --out-md "${SPECLINK_ROOT}/tables/sparse_layout_summary.md" \
  --k-slices 2,4,8 \
  --topk-per-token 8 \
  --shared-budget 4 \
  --private-max 4 \
  --alpha 4 \
  --beta 4
```

Core comparisons:

- `independent_topk`
- `snapkv_static`
- `shared_only`
- `speclink_fixed`
- `speclink_prob`
- `speclink_prob_fallback`

Required context variants:

- `math_reasoning_orig.jsonl`
- `math_reasoning_pad2k.jsonl`
- `math_reasoning_pad4k.jsonl`
- `math_reasoning_pad8k.jsonl`

Generate the required variants with:

```bash
conda run -n spec python scripts/speclink_make_padded_math.py \
  --dataset data/math_reasoning.jsonl \
  --out-dir "${SPECLINK_ROOT}/04_sparse_challenge/padded_data" \
  --targets 0,2048,4096,8192
```

Normalize a live plan-only trace into the challenge trace schema with:

```bash
conda run -n spec python scripts/speclink_collect_sparse_traces.py \
  --live-trace "${SPECLINK_ROOT}/05_speclink/speclink_prob_profile_smoke/live_sparse_trace.jsonl" \
  --out "${SPECLINK_ROOT}/04_sparse_challenge/traces.jsonl"
```

Run a proxy sparse-attention microbenchmark with:

```bash
conda run -n spec python scripts/speclink_microbench_sparse_attention.py \
  --out-csv "${SPECLINK_ROOT}/tables/sparse_microbench.csv" \
  --out-md "${SPECLINK_ROOT}/tables/sparse_microbench.md" \
  --layouts dense_verifier,independent_topk,snapkv_static,shared_only,speclink_prob \
  --k-values 2,4,8,12 \
  --context-lens 512,2048,4096,8192 \
  --block-sizes 16,32,64 \
  --repeat 100 \
  --warmup 20 \
  --device cuda
```

The microbenchmark is labeled `torch_union_attention_proxy` unless replaced by
a real sparse verifier kernel path.

## Phase 5: Speclink Plan-Only Smoke

This is real serving only after the installed vLLM package has a speclink
plan-only patch that writes `SPECLINK_TRACE_OUT`.

```bash
SPECLINK_ENABLE=1 SPECLINK_MODE=plan_only SPECLINK_LAYOUT=speclink_prob \
GUIDELLM_RATE=1 REQUEST_TYPE=chat_completions conda run -n spec bash scripts/speclink_run_method.sh \
  --method speclink \
  --num-spec-tokens 8 \
  --port 8012 \
  --dataset data/math_reasoning.jsonl \
  --output-dir "${SPECLINK_ROOT}/05_speclink/speclink_prob_profile_smoke" \
  --max-tokens 512 \
  --accuracy-limit 10 \
  --benchmark-limit 10 \
  --dense-reference-jsonl "${SPECLINK_ROOT}/02_baselines/dense_smoke/accuracy_outputs.jsonl" \
  --repeat-id 1
```

To print the current plan-only layout/K matrix:

```bash
bash scripts/speclink_run_matrix.sh \
  --root "${SPECLINK_ROOT}" \
  --matrix speclink-plan
```

Collect the existing plan-only ablation rows into report tables:

```bash
conda run -n spec python scripts/speclink_collect_ablation.py \
  --root "${SPECLINK_ROOT}" \
  --out-csv "${SPECLINK_ROOT}/tables/speclink_ablation.csv" \
  --out-md "${SPECLINK_ROOT}/tables/speclink_ablation.md"
```

These plan-only ablation rows measure serving overhead, planner time, and
estimated sparse layout bytes while the dense verifier still runs. They are not
sparse-kernel speedup rows.

## Phase 5a: Acceptance Probability Calibration

The first calibrator uses the fields currently available in plan-only live
traces: moving-average `accept_probs`, `accepted_prefix_len`, position, rho,
risk, prompt length, and decode length. It does not yet use draft logprob or
entropy.

```bash
conda run -n spec python scripts/speclink_calibrate_acceptance.py \
  --traces "${SPECLINK_ROOT}/04_sparse_challenge/formal_plan_live_traces_sample500.jsonl" \
  --out "${SPECLINK_ROOT}/05_speclink/calibrator.pkl" \
  --summary-json "${SPECLINK_ROOT}/05_speclink/calibrator_summary.json" \
  --method logistic
```

The resulting `calibrator.pkl` is planner-probability calibration evidence only;
it is not sparse verifier quality evidence.

## Phase 5b: Formal G2 Plan-Only Ablation

The compact layout/K matrix above is not the full G2 sweep from `speclink.md`.
Use `speclink-g2` for the formal block-size, shared-budget, private-budget,
lambda-risk, and fallback matrix. Defaults encode the minimum formal sweep:

- draft families: `eagle3,peagle`
- K: `2,4,8`
- block size: `32`
- shared budget: `16,32,64`
- private max: `8,16`
- lambda risk: `0,1`
- fallback: `0,1`

Print the formal G2 commands:

```bash
bash scripts/speclink_run_matrix.sh \
  --root "${SPECLINK_ROOT}" \
  --matrix speclink-g2 \
  --accuracy-limit 30 \
  --benchmark-limit 30
```

Resume/run the matrix with GPU access:

```bash
bash scripts/speclink_run_matrix.sh \
  --root "${SPECLINK_ROOT}" \
  --matrix speclink-g2 \
  --accuracy-limit 30 \
  --benchmark-limit 30 \
  --skip-complete \
  --run
```

Audit G2 status:

```bash
conda run -n spec python scripts/speclink_matrix_status.py \
  --root "${SPECLINK_ROOT}" \
  --matrix speclink-g2 \
  --expected-benchmark-limit 30 \
  --expected-accuracy-limit 30 \
  --out-tsv "${SPECLINK_ROOT}/tables/speclink_g2_matrix_status.tsv" \
  --out-md "${SPECLINK_ROOT}/tables/speclink_g2_matrix_status.md"
```

The full default G2 matrix is deliberately large. For a quick plumbing check,
pin one row:

```bash
bash scripts/speclink_run_matrix.sh \
  --root "${SPECLINK_ROOT}" \
  --matrix speclink-g2 \
  --speclink-draft-methods eagle3 \
  --k-values 4 \
  --block-sizes 32 \
  --shared-budgets 16 \
  --private-max-values 8 \
  --lambda-risk-values 1 \
  --fallback-values 1 \
  --accuracy-limit 10 \
  --benchmark-limit 10 \
  --skip-complete \
  --run
```

## Phase 5c: Formal G4 Serving-Rate Comparison

This matrix compares the best current plan-only SpecLink serving setup against
the already-complete dense/EAGLE3/P-EAGLE baseline matrix. It is still
plan-only: the dense verifier path runs normally and sparse-kernel speedup is
not claimed.

Default `speclink-serving` settings:

- draft families: `eagle3,peagle`
- K: use `8`
- layout: `speclink_prob`
- block size: `32`
- shared budget: `16`
- private max: `16`
- lambda risk: `0`
- GuideLLM rates: `1,2,4`
- repeats: `0,1,2`
- benchmark/accuracy rows: `80`

Run or resume the G4 matrix:

```bash
bash scripts/speclink_run_matrix.sh \
  --root "${SPECLINK_ROOT}" \
  --matrix speclink-serving \
  --speclink-draft-methods eagle3,peagle \
  --k-values 8 \
  --rates 1,2,4 \
  --repeats 0,1,2 \
  --accuracy-limit 80 \
  --benchmark-limit 80 \
  --skip-complete \
  --run
```

Audit G4 status:

```bash
conda run -n spec python scripts/speclink_matrix_status.py \
  --root "${SPECLINK_ROOT}" \
  --matrix speclink-serving \
  --k-values 8 \
  --expected-benchmark-limit 80 \
  --expected-accuracy-limit 80 \
  --out-tsv "${SPECLINK_ROOT}/tables/speclink_serving_matrix_status.tsv" \
  --out-md "${SPECLINK_ROOT}/tables/speclink_serving_matrix_status.md"
```

## Phase 5d: Offline Sparse Verifier Quality

If `SPECLINK_MODE=sparse_kernel` is not integrated, do not report vLLM sparse
verifier accuracy. The following G5 workflow is narrower: it loads Qwen3-8B in
Transformers, builds `hidden_similarity_proxy` sparse candidates from dense
hidden states, applies dense causal attention and proxy sparse masks to padded
2k prompts, and compares verifier logits. It is offline/proxy evidence, not
vLLM sparse-kernel correctness.

```bash
conda run -n spec python scripts/speclink_sparse_quality_eval.py \
  --dataset-jsonl "${SPECLINK_ROOT}/04_sparse_challenge/padded_data/math_reasoning_pad2k.jsonl" \
  --context-label pad2k \
  --candidate-source hidden_similarity_proxy \
  --dense-reference-jsonl "${SPECLINK_ROOT}/02_baselines/dense_rate1_r0/accuracy_outputs.jsonl" \
  --out-jsonl "${SPECLINK_ROOT}/05_speclink/sparse_quality/offline_sparse_quality_pad2k_hidden.jsonl" \
  --out-traces-jsonl "${SPECLINK_ROOT}/04_sparse_challenge/hidden_sparse_traces_pad2k.jsonl" \
  --out-summary-csv "${SPECLINK_ROOT}/tables/sparse_quality_summary.csv" \
  --out-summary-md "${SPECLINK_ROOT}/tables/sparse_quality_summary.md" \
  --limit 4 \
  --max-positions 16 \
  --max-seq-len 2600 \
  --num-spec-tokens 8 \
  --block-size 32 \
  --shared-budget 16 \
  --private-max 16 \
  --lambda-risk 0 \
  --model Qwen/Qwen3-8B \
  --local-files-only
```

Generate layout/union/Jaccard/HBM metrics from those model-derived traces:

```bash
conda run -n spec python scripts/speclink_sparse_layout_sim.py \
  --traces "${SPECLINK_ROOT}/04_sparse_challenge/hidden_sparse_traces_pad2k.jsonl" \
  --out-csv "${SPECLINK_ROOT}/tables/sparse_layout_hidden_summary.csv" \
  --out-md "${SPECLINK_ROOT}/tables/sparse_layout_hidden_summary.md" \
  --k-slices 2,4,8 \
  --block-size 32 \
  --topk-per-token 32 \
  --shared-budget 16 \
  --private-max 16 \
  --lambda-risk 0
```

For a small pad4k extension, rerun the same quality command with:

```text
--dataset-jsonl "${SPECLINK_ROOT}/04_sparse_challenge/padded_data/math_reasoning_pad4k.jsonl"
--context-label pad4k
--out-jsonl "${SPECLINK_ROOT}/05_speclink/sparse_quality/offline_sparse_quality_pad4k_hidden.jsonl"
--out-traces-jsonl "${SPECLINK_ROOT}/04_sparse_challenge/hidden_sparse_traces_pad4k.jsonl"
--out-summary-csv "${SPECLINK_ROOT}/tables/sparse_quality_summary_pad4k.csv"
--out-summary-md "${SPECLINK_ROOT}/tables/sparse_quality_summary_pad4k.md"
--limit 2
--max-seq-len 4700
```

The combined report table is built from
`sparse_quality_summary_pad2k.csv` and `sparse_quality_summary_pad4k.csv`; the
combined hidden layout table uses
`04_sparse_challenge/hidden_sparse_traces_pad2k_pad4k.jsonl`.

## Phase 6: Report

```bash
conda run -n spec python scripts/speclink_write_figures.py \
  --results-root "${SPECLINK_ROOT}"

conda run -n spec python scripts/speclink_analyze_results.py \
  --results-root "${SPECLINK_ROOT}"
```

Final answer should point to:

- Result root
- `tables/baseline_summary.csv`
- `tables/breakdown_summary.csv`
- `tables/sparse_layout_summary.csv`
- `tables/sparse_quality_summary.csv`
- `speclink_experiment_report.md`
- `patches/vllm-speclink.diff`
- Which rows are real e2e, simulation, microbenchmark, or missing
