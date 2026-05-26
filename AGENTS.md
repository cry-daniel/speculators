# Speculator Throughput Experiments

This repo is currently set up to benchmark Qwen3-8B with EAGLE3 and P-EAGLE
speculators through vLLM and GuideLLM.

## Environment

Use the existing conda environment:

```bash
conda activate spec
```

Known working environment:

- Python: `/ACALAB/stu1/miniconda3/envs/spec/bin/python`
- vLLM: `0.20.0`, installed editable from this repo's `vllm/`
- GuideLLM: `0.6.0`
- Torch: `2.11.0+cu130`
- Local package: this repo installed editable into `spec`

If recreating the environment, use Python 3.12, install GuideLLM and Hugging
Face Hub, then install this repo and the vendored vLLM editable. Keep vLLM's
CUDA build aligned with PyTorch's CUDA 13.0 stack; do not build it with a newer
system CUDA such as `/usr/local/cuda-13.2`, because that can produce PTX the
current driver cannot run.

```bash
conda create -n spec python=3.12
conda activate spec
pip install guidellm huggingface-hub
pip install -e /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators --no-deps
pip install -r /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/vllm/requirements/build/cuda.txt
pip install \
  nvidia-cuda-nvcc==13.0.88 \
  nvidia-nvvm==13.0.88 \
  nvidia-cuda-crt==13.0.88 \
  nvidia-cuda-cccl==13.0.85
conda install -n spec -y -c conda-forge gcc_linux-64=13 gxx_linux-64=13

cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators
CU13=/ACALAB/stu1/miniconda3/envs/spec/lib/python3.12/site-packages/nvidia/cu13
ln -sfn lib "${CU13}/lib64"
ln -sfn libcudart.so.13 "${CU13}/lib/libcudart.so"
ln -sfn libnvJitLink.so.13 "${CU13}/lib/libnvJitLink.so"
ln -sfn libnvrtc.so.13 "${CU13}/lib/libnvrtc.so"
ln -sfn libnvvm.so.4 "${CU13}/lib/libnvvm.so"
mkdir -p "${CU13}/lib/stubs"
ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.1 "${CU13}/lib/stubs/libcuda.so"

CUDA_HOME="${CU13}" \
CUDACXX="${CU13}/bin/nvcc" \
CUDAHOSTCXX=/ACALAB/stu1/miniconda3/envs/spec/bin/x86_64-conda-linux-gnu-g++ \
CC=/ACALAB/stu1/miniconda3/envs/spec/bin/x86_64-conda-linux-gnu-gcc \
CXX=/ACALAB/stu1/miniconda3/envs/spec/bin/x86_64-conda-linux-gnu-g++ \
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
EAGLE3_SPECULATOR_MODEL=/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/qwen3-8b-eagle3-speculator
PEAGLE_SPECULATOR_MODEL=/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/qwen3-8b-peagle-speculator
```

The current local paths are:

- EAGLE3: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/qwen3-8b-eagle3-speculator`
- P-EAGLE: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/qwen3-8b-peagle-speculator`
- Llama-3.1-8B EAGLE3: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/llama-3.1-8b-eagle3-speculator`

To download the speculator checkpoints again:

```bash
cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm
./download_qwen3_8b_speculators.sh
./download_llama_3_1_8b_eagle3_speculator.sh
```

The base model is not copied into `../models` by default. Override it with
`QWEN3_8B_MODEL=/path/to/local/qwen3-8b` only if you want to force a specific
local base model directory.

## Vendored vLLM Source and Install

vLLM is vendored as a full source tree inside this repository:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/vllm
```

This tree was imported from upstream vLLM `v0.20.0` and pushed to `main` in
commit `9def519`. Future vLLM edits for these experiments should be made in
`speculators/vllm`, not in `site-packages` and not in the older external
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/vllm` checkout.

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
  `SPECLINK_TRACE_CONFIDENCE=1`

Install or refresh vLLM from the repo root with editable mode. The current
machine uses PyTorch `2.11.0+cu130`, so point the vLLM build at the conda
environment's CUDA 13.0 compiler stack instead of the system CUDA:

```bash
cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators
conda run -n spec python -m pip install -r vllm/requirements/build/cuda.txt
conda run -n spec python -m pip install \
  nvidia-cuda-nvcc==13.0.88 \
  nvidia-nvvm==13.0.88 \
  nvidia-cuda-crt==13.0.88 \
  nvidia-cuda-cccl==13.0.85
conda install -n spec -y -c conda-forge gcc_linux-64=13 gxx_linux-64=13

CU13=/ACALAB/stu1/miniconda3/envs/spec/lib/python3.12/site-packages/nvidia/cu13
ln -sfn lib "${CU13}/lib64"
ln -sfn libcudart.so.13 "${CU13}/lib/libcudart.so"
ln -sfn libnvJitLink.so.13 "${CU13}/lib/libnvJitLink.so"
ln -sfn libnvrtc.so.13 "${CU13}/lib/libnvrtc.so"
ln -sfn libnvvm.so.4 "${CU13}/lib/libnvvm.so"
mkdir -p "${CU13}/lib/stubs"
ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.1 "${CU13}/lib/stubs/libcuda.so"

CUDA_HOME="${CU13}" \
CUDACXX="${CU13}/bin/nvcc" \
CUDAHOSTCXX=/ACALAB/stu1/miniconda3/envs/spec/bin/x86_64-conda-linux-gnu-g++ \
CC=/ACALAB/stu1/miniconda3/envs/spec/bin/x86_64-conda-linux-gnu-gcc \
CXX=/ACALAB/stu1/miniconda3/envs/spec/bin/x86_64-conda-linux-gnu-g++ \
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
cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm
conda run -n spec python -c "import pathlib, vllm; print(pathlib.Path(vllm.__file__).resolve())"
conda run -n spec vllm --help
```

Expected import path prefix:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/vllm/vllm/
```

Do not use `VLLM_USE_PRECOMPILED=1` for this workflow. The point of the local
source install is that future edits to `speculators/vllm` are the code being
run. Also avoid launching Python from the speculators repo root when checking
`import vllm`: the root contains a `vllm/` source directory and can shadow the
editable package as a namespace package. The motivation breakdown script changes
into `examples/evaluate/eval-guidellm` before validating the import path.

Older runs left direct-edit residue under:

```text
/ACALAB/stu1/miniconda3/envs/spec/lib/python3.12/site-packages/vllm/
```

That directory should not exist after the editable install. If it reappears and
`vllm.__file__` is `None` or imports resolve to `site-packages/vllm`, move the
stale directory out of the conda environment and rerun the verification above.

If the install fails after moving a partially built `vllm/.deps` directory and
CMake reports an old path in `CMakeCache.txt`, remove only generated build
subdirectories and retry:

```bash
find vllm/.deps -maxdepth 1 -type d \( -name '*-subbuild' -o -name '*-build' \) -exec rm -rf {} +
CU13=/ACALAB/stu1/miniconda3/envs/spec/lib/python3.12/site-packages/nvidia/cu13
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
cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm
mkdir -p data
hf download RedHatAI/speculator_benchmarks \
  --repo-type dataset \
  --include math_reasoning.jsonl \
  --local-dir data \
  --max-workers 8
```

Dataset path:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/data/math_reasoning.jsonl
```

The evaluation scripts also accept Hugging Face dataset syntax:

```text
RedHatAI/speculator_benchmarks:math_reasoning.jsonl
```

`scripts/run_guidellm.sh` strips the `path=` prefix emitted by this HF CLI
version before searching downloaded files.

## Running Benchmarks

Run commands from:

```bash
cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm
```

By default, experiment outputs go under:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results/
```

`run_evaluation.sh` defaults to `results/eval_results_TIMESTAMP/`. Override the
root with either:

```bash
RESULTS_DIR=/path/to/results ./run_evaluation.sh -c ./configs/qwen3-8b-peagle.env
```

or:

```bash
./run_evaluation.sh -c ./configs/qwen3-8b-peagle.env --results-dir /path/to/results
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
cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm
conda run -n spec bash ./motivation_breakdown.sh
```

The script no longer accepts `--vllm-dir`, `--skip-vllm-setup`, or
`--setup-only`; vLLM is expected to already be installed editable from
`speculators/vllm`. At startup it verifies that `import vllm` resolves under:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/vllm/vllm/
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
SPECLINK_TRACE_OUTPUT=/path/to/trace.jsonl
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
cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm
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
cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm
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
EAGLE3_SPECULATOR_MODEL=/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/llama-3.1-8b-eagle3-speculator \
METHODS=eagle3 MAIN_NUM_SPEC_TOKENS=8 PORT=8035 \
conda run -n spec bash ./run_speclink_confidence_acceptance.sh --single-case --main-only

# Qwen3-8B MTBench, EAGLE3 K=8 only
DATASET_LABEL=mtbench \
DATASET=/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/data/mt_bench.jsonl \
MODEL_LABEL=qwen3_8b \
BASE_MODEL=Qwen/Qwen3-8B \
EAGLE3_SPECULATOR_MODEL=/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/qwen3-8b-eagle3-speculator \
METHODS=eagle3 MAIN_NUM_SPEC_TOKENS=8 MAIN_PROMPTS=80 MAIN_MAX_TOKENS=128 \
REQUEST_CONCURRENCY=1 PORT=8036 \
conda run -n spec bash ./run_speclink_confidence_acceptance.sh --single-case --main-only

# Llama-3.1-8B MTBench, EAGLE3 K=8 only
DATASET_LABEL=mtbench \
DATASET=/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/data/mt_bench.jsonl \
MODEL_LABEL=llama3_1_8b \
BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct \
EAGLE3_SPECULATOR_MODEL=/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/llama-3.1-8b-eagle3-speculator \
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
cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm
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
