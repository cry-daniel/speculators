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
- SpecLink-CV live prefix/suffix verification slice in
  `vllm/speclink_cv.py`,
  `vllm/v1/core/sched/scheduler.py`,
  `vllm/v1/core/sched/output.py`, and
  `vllm/v1/worker/gpu_model_runner.py`, gated by
  `SPECLINK_CV_ENABLE=1`

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

## Lightweight vLLM Speculative WebUI

`vllm_spec_webui.py` is a small Gradio visualization client for an already
running vLLM OpenAI-compatible server. It does not start vLLM, does not load
local model weights, and does not run a benchmark orchestrator.

Run from the repo root:

```bash
conda run -n spec python -m pip install gradio httpx
conda run -n spec python vllm_spec_webui.py \
  --server-url http://localhost:8000/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --method-name speclink \
  --num-spec-tokens 8 \
  --host 0.0.0.0 \
  --port 7860
```

The WebUI sends `batch_size=N` concurrent streaming Chat Completions requests to
`{server_url}/chat/completions`. It only displays request 0's generated text,
but the demo TPS counts non-empty streaming deltas from all N requests:

```text
demo TPS = current batch streaming deltas / elapsed seconds
```

This is only an interactive load visualizer for checking output and rough live
throughput under large batch load. It is not the paper throughput metric; use
the closed-loop fixed-window steady-state runner for final saturated output
tokens/s.

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

## SpecLink-CV Math Quality Experiment

`run_speclink_cv_math_quality.sh` is the current fast reproduction script for
math_reasoning quality and throughput after relaxing the gate from bit-for-bit
EAGLE3 equality to math answer EM. It is a SpecLink-CV diagnostic/follow-up
script, not the full TODO matrix runner.

Run from `examples/evaluate/eval-guidellm`:

```bash
cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm
bash ./run_speclink_cv_math_quality.sh
```

Defaults:

- model/dataset: `qwen3_8b` on `math`
- methods: `eagle3_oneshot,cv_half_async_simple`
- `K_LIST=16`
- `FORCE_PREFIX_LEN=8`, so the CV path verifies prefix `h=8` and treats the
  remaining 8 tokens as suffix
- `BATCH_SIZE_LIST=8,16,32`
- `MAX_REQUESTS=32`; keep this at least as large as the largest requested batch
  size, otherwise the measured actual batch can be far below the configured
  `BATCH_SIZE_LIST`
- `MAX_TOKENS=1024`, long enough for the current math EM check without letting
  Qwen3 run to the full model length
- `--allow-shape-drift-chunking`
- `--allow-batched-prefix-verification`
- `SPECLINK_CV_ALLOW_BATCHED_SUFFIX=1`
- `SPECLINK_CV_SUFFIX_REPLAY_ONE_SHOT_SHAPE=0`
- `BATCH_INVARIANT=0`; set `BATCH_INVARIANT=1` only for token-id exact
  debugging
- `DENSE_REALIGN_STEPS=0`, which disables the conservative dense TLM
  realignment guard after suffix rejection
- `ALLOW_CV_CUDAGRAPH=0`; set it to `1` only for an experimental run that does
  not force `--enforce-eager` for `cv_*` methods
- `PYTHON_BIN=/ACALAB/stu1/miniconda3/envs/spec/bin/python`; the script calls
  the spec-environment Python directly to avoid nested `conda run` failures

Outputs default to:

```text
temp/speclink_cv_math_quality_TIMESTAMP/
```

Set `OUTPUT_ROOT=results/...` only when intentionally producing a final result
bundle. The script writes the usual matrix artifacts and also extracts readable
math output comparisons:

- `09_reports/summary_metrics.csv`
- `09_reports/SPECLINK_CV_REPORT.md`
- `09_reports/performance_gap.md`
- `09_reports/performance_gap.csv`
- `09_reports/performance_gap.json`
- `09_reports/math_cv_wrong_outputs.md`
- `09_reports/math_cv_wrong_outputs.json`
- `09_reports/math_cv_drop_outputs.md`
- `09_reports/math_cv_drop_outputs.json`

`math_cv_wrong_outputs.*` contains every CV-wrong `question_id`. `cv-drop`
contains only regressions where EAGLE3 is correct and CV is wrong.
`performance_gap.*` compares each CV row with the matching EAGLE3 row and
parses vLLM rolling SpecDecoding metrics to explain whether skipped verifier
tokens became throughput. Regenerate it without launching vLLM:

```bash
conda run -n spec python \
  /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/tools/speclink_cv/analyze_performance_gap.py \
  ./temp/speclink_cv_math_quality_TIMESTAMP \
  --output-dir ./temp/speclink_cv_math_quality_TIMESTAMP/09_reports
```

To combine the current `{qwen3_8b,llama3_1_8b} x K={8,12} x bs={8,16,32}`
staged-CV math-quality slices without launching vLLM:

```bash
conda run -n spec python \
  /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/tools/speclink_cv/combine_math_quality_followups.py
```

The combined output defaults to:

```text
temp/speclink_cv_math_k8_k12_all_bs_staged_quality_combined_20260528/
```

It writes:

- `09_reports/math_quality_all_batch_summary.{md,csv,json}`
- `09_reports/math_quality_performance_diagnosis.{md,csv}`
- `08_figures/math_quality_speedup_curve.png`

If both the earlier 32-request bs=16 slice and the newer 64-request bs=16
slice are present, the combiner keeps the fuller 64-request row for each
scenario. This combined report is a relaxed math answer-EM performance view; it
does not close the original `TODO.md` strict greedy token-id correctness gate
for live h<K chunking.

Important terminology:

- `no realign` means `SPECLINK_CV_DENSE_REALIGN_STEPS=0`. Earlier conservative
  paths inserted dense TLM steps after suffix rejection to keep the EAGLE
  drafter state closer to exact one-shot behavior. That is expensive and was
  mainly for token-id exactness debugging. For math-quality runs, keep it as an
  explicit ablation and judge by math answer EM.
- `batched prefix` means SpecLink-CV's async prefix queue can dispatch several
  requests' prefix chunks in one vLLM scheduler step. This uses vLLM's normal
  continuous batching machinery with vLLM async scheduling disabled for state
  stability; it is not an external batcher and not `--async-scheduling`.
- vLLM already supports continuous batching. The current SpecLink-CV integration
  plugs prefix verification into that scheduler by allowing multiple queued
  prefix chunks to be selected together. Without
  `--allow-batched-prefix-verification`, the conservative default caps prefix
  verification to one sequence per TLM step and is much slower.
- `batched suffix` means accepted-prefix suffix chunks also stay in vLLM's
  batched scheduler path. Set `SPECLINK_CV_ALLOW_BATCHED_SUFFIX=1` for
  performance runs; otherwise each suffix request can be isolated and the saved
  verifier tokens are largely hidden by scheduler overhead.

Useful variants:

```bash
# Reproduce the previous K=8/h=4 configuration.
K_LIST=8 FORCE_PREFIX_LEN=4 \
bash ./run_speclink_cv_math_quality.sh

# Keep dense realignment enabled for a conservative comparison.
DENSE_REALIGN_STEPS=-1 \
bash ./run_speclink_cv_math_quality.sh

# Experimental: let cv_* run without forced eager/CUDAGraph disabled.
ALLOW_CV_CUDAGRAPH=1 \
bash ./run_speclink_cv_math_quality.sh

# Current performance/quality slice: compare bs=16 EAGLE3 against staged CV
# with K=16, h=8, no batch-invariant exactness guard, and no dense realign.
METHOD_LIST=eagle3_oneshot,cv_half_async_staged_simple \
K_LIST=16 BATCH_SIZE_LIST=16 FORCE_PREFIX_LEN=8 \
MAX_REQUESTS=32 MAX_TOKENS=0 \
ALLOW_CV_CUDAGRAPH=1 BATCH_INVARIANT=0 \
ALLOW_BATCHED_PREFIX_VERIFICATION=1 ALLOW_BATCHED_SUFFIX=1 \
DENSE_REALIGN_STEPS=0 \
bash ./run_speclink_cv_math_quality.sh

# Profile one run with Nsight Systems. Keep this in temp unless deliberately
# producing a final artifact.
NSYS_PROFILE=1 ALLOW_CV_CUDAGRAPH=1 PREFIX_FULL_CUDAGRAPH=1 \
METHOD_LIST=cv_half_async_staged_simple BATCH_SIZE_LIST=2 \
MAX_REQUESTS=2 MAX_TOKENS=8 \
bash ./run_speclink_cv_math_quality.sh

# Only regenerate qid-level output comparisons for an existing pair.
conda run -n spec python ./scripts/extract_math_outputs_by_qid.py \
  --mode cv-drop \
  --eagle3-dir ./temp/speclink_cv_math_quality_TIMESTAMP/runs/qwen3_8b_math_k16_bs16_eagle3_oneshot \
  --cv-dir ./temp/speclink_cv_math_quality_TIMESTAMP/runs/qwen3_8b_math_k16_bs16_cv_half_async_simple \
  --out-md ./temp/speclink_cv_math_quality_TIMESTAMP/09_reports/math_cv_drop_outputs.md
```

Nsight mode:

- `NSYS_PROFILE=1` wraps the vLLM server process in `nsys profile` and sets
  `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
- The server command uses `--trace=cuda,nvtx,osrt,cublas`,
  `--trace-fork-before-exec=true`, `--cuda-graph-trace=node`,
  `--gpu-metrics-devices=cuda-visible`,
  `--capture-range=cudaProfilerApi`, `--capture-range-end=repeat`, and passes
  `--profiler-config.profiler cuda` to vLLM.
- GuideLLM itself is unchanged. The runner calls
  `curl --noproxy '*' -f -sS -X POST http://127.0.0.1:PORT/start_profile`
  immediately before `python -m guidellm benchmark run`, then calls
  `/stop_profile` after the benchmark.
- After server shutdown, the runner runs `nsys stats` automatically. Each run
  directory gets `vllm_guidellm_profile*.nsys-rep`, `nsys_stats.txt`,
  `*_nsys_quick_summary.md`, `*_gpu_metrics_summary.csv`,
  `*_cuda_api_summary.csv`, and `*_kernel_gap_summary.json`.
- The quick summary is meant for the current utilization question: it exposes
  SM Active, Tensor Active, DRAM read/write activity, kernel-gap statistics,
  and CUDA API overhead. Use the `.nsys-rep` in the Nsight Systems GUI when the
  timeline itself matters.
- `nsys stats` exports a large intermediate SQLite file. The runner deletes it
  by default after writing the summaries; set `KEEP_NSYS_SQLITE=1` only when
  you need to inspect the raw SQLite tables.

Nsight smoke verified on 2026-05-28:

```text
temp/speclink_cv_nsys_smoke_fixed_20260528/
```

The smoke used Qwen3 math, K=16, h=8, bs=2,
`cv_half_async_staged_simple`, `ALLOW_CV_CUDAGRAPH=1`, and
`SPECLINK_CV_PREFIX_FULL_CUDAGRAPH=1`. The run produced
`vllm_guidellm_profile.1.nsys-rep` and `nsys_stats.txt`. The
`speclink_cv_profile.jsonl` evidence shows h+1 prefix verifier forwards entered
`CUDAGraphMode.FULL` with `uniform_decode_query_len=9`, so this mode can now
distinguish a true FULL graph prefix path from the older PIECEWISE mixed path.

Representative current K=16/h=8 result:

```text
temp/speclink_cv_math_quality_1024_qwen_k16_h8_batched_no_realign_bs8_req32_20260528/
temp/speclink_cv_math_quality_1024_qwen_k16_h8_batched_no_realign_bs16_32_req32_20260528/
```

These runs use `MAX_REQUESTS=32`, so the actual average batch is meaningful for
`bs=8/16/32`.

Summary:

```text
bs=8:  CV 293.0 tok/s vs EAGLE3 321.7 tok/s, math EM 22/32 vs 20/32
bs=16: CV 364.9 tok/s vs EAGLE3 449.6 tok/s, math EM 22/32 vs 21/32
bs=32: CV 522.3 tok/s vs EAGLE3 676.0 tok/s, math EM 16/32 vs 22/32
```

With enough requests to fill the serving pressure, this fixed K=16/h=8
configuration is not yet faster than EAGLE3. It is still useful evidence:
prefix rejection is high and suffix skipping works, but the current
prefix/suffix scheduler overhead and full-K DLM drafting cost still outweigh
the target-verifier savings at these batch sizes. The bs=32 row also fails the
math quality gate and should not be used as a quality-preserving speedup claim.

The performance-gap report gives the sharper diagnosis. For the K=16/h=8 rows,
CV skips 50% of suffix draft tokens and the estimated target verification token
ratio is about `9/17 = 0.529`, but rolling vLLM metrics show mean acceptance
length drops from roughly `2.85-2.90` on EAGLE3 to `2.45-2.47` on CV. Prefix
dispatch utilization is also low at smaller batches (`0.23` for bs=8 and
`0.42` for bs=16). Therefore the current bottleneck is not "suffix skip failed";
it is that DLM still drafts full K, h<K changes acceptance behavior, and small
prefix verifier batches do not use the GPU efficiently enough.

A pre-fix short K=16/h=4 bs=8 diagnostic at:

```text
temp/speclink_cv_math_quality_1024_qwen_k16_h4_batched_no_realign_bs8_req16_20260528/
```

exposed a real worker-side rollback bug: CV throughput was `167.9` tok/s versus
EAGLE3 `314.5` tok/s, and early prefix acceptance collapsed much more than it
should. The bug was that the worker added the skipped suffix length `K-h` to
EAGLE's `num_rejected_tokens_gpu`. That value is relative to the current target
forward; a prefix verifier forward contains only `h` draft tokens plus the
target bonus/recovered slot, so adding `K-h` over-shortened the drafter
`seq_lens`.

After the rollback fix, the bounded h=4 smoke at:

```text
temp/speclink_cv_h4_rollback_fix_smoke_20260528/
```

has much smaller early acceptance drift: prefix full-accept is `0.156`, close to
EAGLE3's position-4 acceptance `0.168`, and rolling pos1-4 deltas are only about
`-0.028/-0.031/-0.030/-0.022`. This is a correctness/acceptance sanity check,
not a speedup result: on the 8-prompt smoke CV was still slower (`120.9` tok/s
vs EAGLE3 `313.1` tok/s) because prefix dispatch is underfilled and every step
still pays full-K DLM drafting.

A TODO-shaped quick bundle after the same rollback fix is at:

```text
temp/speclink_cv_todo_live_after_h4_fix_20260528/
```

It records 7/7 unit tests passing and a live token-id gate for
Qwen3/math/K=8/bs=2/max16 where both `chunked` and `exactsafe` match EAGLE3
2/2. This is only a quick correctness pass; the full 48-case live gate and the
full GuideLLM matrix remain unrun.

The post-fix Qwen3/math K=16/h=8 bs=8/16/32 rerun is at:

```text
temp/speclink_cv_math_quality_1024_qwen_k16_h8_after_rollback_fix_bs8_16_32_20260528/
```

All six GuideLLM cases completed. Compared with the earlier pre-fix h=8 runs,
the early acceptance drift is now small, so the worker rollback bug is no longer
the dominant issue. End-to-end results are:

```text
bs=8:  CV 310.5 tok/s vs EAGLE3 314.7 tok/s, math EM 21/32 vs 21/32
bs=16: CV 381.2 tok/s vs EAGLE3 456.0 tok/s, math EM 19/32 vs 19/32
bs=32: CV 509.5 tok/s vs EAGLE3 623.5 tok/s, math EM 19/32 vs 21/32
```

Only bs=8 and bs=16 pass the math-quality gate; bs=32 is a quality drop and
must not be used as a speedup claim. The performance-gap report shows why h=8
still does not beat EAGLE3: suffix verification is skipped for about half of the
draft-token verification work, but fixed h=8 almost never survives to suffix
verification at bs=16/32 (`prefix full-accept = 0`) and DLM still drafts full K
before the prefix result is known. Prefix reject correctly skips suffix
verification; the remaining waste is draft-side work and scheduler/lookahead
bookkeeping that has already been paid for. The current next optimization
target is reducing draft-side full-K work and making skipped verifier tokens
visible to the scheduler's token/slot budget earlier, not more rollback fixes.

An experimental `ALLOW_CV_CUDAGRAPH=1` bs=8 rerun improved CV throughput from
`293.0` to `315.3` tok/s but still trailed EAGLE3 at `332.9` tok/s and failed
the math quality gate (`15/32` vs EAGLE3 `18/32`). Keep eager mode as the
default until this path is debugged.

### Steady-State Contribution Ablation

Use `run_speclink_cv_contribution_ablation.sh` for the performance-source
ablation after a promising staged-CV configuration is found. It keeps
skip-suffix enabled in every CV row and separates:

- non-staged CV: TLM suffix verification skip only;
- staged CV: TLM suffix skip plus delayed DLM suffix drafting;
- singleton-live staged CV: same h<K logic, but verifier work is restricted to
  nearly one request per scheduler step to expose batching/scheduling cost.

The script now defaults to `BENCHMARK_MODE=steady_state`, so reported
`throughput` is saturated output tokens/s at closed-loop concurrency
`BATCH_SIZE_LIST`, not finite-request drain makespan. In steady-state mode,
`MAX_TOKENS=0` is automatically replaced by `STEADY_STATE_MAX_TOKENS` or
`1024`, because the fixed-window client requires a positive `max_tokens`.

Representative paired runs:

```text
temp/speclink_cv_contribution_ablation_steady_qwen_llama_math_k12_h6_bs16_20260528/
temp/speclink_cv_contribution_ablation_steady_qwen_llama_math_k16_h8_bs16_20260528/
temp/speclink_cv_contribution_ablation_steady_k12_k16_compare_20260528/
```

K=16 configuration:

```bash
BENCHMARK_MODE=steady_state MODEL_LIST=qwen3_8b,llama3_1_8b \
DATASET_LIST=math K_LIST=16 BATCH_SIZE_LIST=16 FORCE_PREFIX_LEN=8 \
MAX_REQUESTS=64 STEADY_STATE_MAX_PROMPTS=64 \
STEADY_STATE_WARMUP_S=20 STEADY_STATE_MEASUREMENT_S=60 \
STEADY_STATE_COOLDOWN_S=10 MAX_TOKENS=1024 \
bash ./run_speclink_cv_contribution_ablation.sh
```

For the TODO K=12 slice, use `K_LIST=12 FORCE_PREFIX_LEN=6` with the same
steady-state settings.

To include an existing contribution run in a TODO-level report without
rerunning vLLM:

```bash
conda run -n spec python tools/speclink_cv/run_todo_experiment.py \
  --finalize-only \
  --output-root examples/evaluate/eval-guidellm/temp/todo_runner_TIMESTAMP \
  --contribution-import-root examples/evaluate/eval-guidellm/temp/speclink_cv_contribution_ablation_steady_k12_k16_compare_20260528
```

The imported rows are copied into `09_reports/summary_metrics.csv` with
`measurement_type=contribution_ablation` and an absolute `output_root`, so the
TODO report can trace each number back to the standalone raw run.

Current TODO finalize-only aggregate using the latest imported evidence:

```text
temp/todo_runner_current_imports_suffix_replay_default_v3_20260528/
```

This bundle imports the bs=8/bs=16 live correctness slices, the strict math
chunked/exactsafe slices, the all-batch math-quality summary, and the
steady-state contribution ablation. It records `48/48` full-live correctness
rows with 19 strict failures, `2/2` serving-smoke rows, `240/240` planned full-matrix
steady-state rows, `4` extra staged/best-candidate steady-state rows, `12/12`
relaxed math-quality rows, and 4 contribution rows.
Qwen3 math K=8 bs=8 is now complete for all 10 TODO methods:
`pure_vllm=718.73`, `eagle3_oneshot=1248.03`,
`cv_half_sync_simple=1080.87`, `cv_half_sync_roofline=1008.53`,
`cv_half_async_simple=1075.44`, `cv_half_async_roofline=1064.93`,
`cv_conf_sync_simple=963.89`, `cv_conf_sync_roofline=859.25`,
`cv_conf_async_simple=967.65`, and `cv_conf_async_roofline=941.80` saturated
output tok/s. All CV rows are below EAGLE3 (`0.688x-0.866x`). Confidence sizing
uses shorter prefixes (`selected_h_avg` roughly `3.15-3.35`) and often reduces
estimated TLM token ratio, but it is slower than fixed half in this small K/bs
case. These rows are not valid quality-preserving claims because this
full-matrix slice uses `max_tokens=128` and the math quality gate marks them
`quality_unreliable_short_outputs`. It is a reporting checkpoint, not a
completed TODO run.

Qwen3 math K=8 bs=16 is now complete for all 10 TODO methods:
`pure_vllm=1402.53`, `eagle3_oneshot=2021.13`,
`cv_half_sync_simple=1902.98`, `cv_half_sync_roofline=1903.34`,
`cv_half_async_simple=1894.27`, `cv_half_async_roofline=1913.83`,
`cv_conf_sync_simple=1710.89`, `cv_conf_sync_roofline=1702.94`,
`cv_conf_async_simple=1701.73`, and `cv_conf_async_roofline=1684.24`
saturated output tok/s, plus extra staged `cv_half_async_staged_simple=2310.87`.
The planned non-staged CV rows are all below EAGLE3 (`0.833x-0.947x`), and
confidence sizing is again slower than fixed half. Staged is faster but outside
the original 240-case TODO method set.
`run_todo_experiment.py` now recomputes speedup and short-output quality labels
after merging local and imported matrix rows, because local rows may store
`K/batch_size` as integers while imported CSV rows store them as strings.

Qwen3 math K=8 bs=32 is now complete for all 10 TODO methods:
`pure_vllm=2673.96`, `eagle3_oneshot=2627.63`,
`cv_half_sync_simple=2917.46`, `cv_half_sync_roofline=2900.42`,
`cv_half_async_simple=2920.26`, `cv_half_async_roofline=2910.60`,
`cv_conf_sync_simple=2656.06`, `cv_conf_sync_roofline=2649.80`,
`cv_conf_async_simple=2662.46`, and `cv_conf_async_roofline=2656.12`
saturated output tok/s. The fixed-half CV rows are about `1.10x-1.11x` over
EAGLE3 at this concurrency, while confidence-sizing rows are only about
`1.01x`. These rows are still
`quality_unreliable_short_outputs` because this full-matrix slice uses
`max_tokens=128`; use the separate math-quality follow-up for relaxed EM
claims.

Qwen3 math K=12 bs=8 is now complete for all 10 TODO methods:
`pure_vllm=718.40`, `eagle3_oneshot=1102.61`,
`cv_half_sync_simple=1012.31`, `cv_half_sync_roofline=1011.57`,
`cv_half_async_simple=1012.26`, `cv_half_async_roofline=1009.31`,
`cv_conf_sync_simple=876.09`, `cv_conf_sync_roofline=877.78`,
`cv_conf_async_simple=878.59`, and `cv_conf_async_roofline=866.88`
saturated output tok/s. Fixed-half CV is about `0.915x-0.918x` of EAGLE3,
and confidence-sizing is about `0.786x-0.797x`. In this small-batch K=12 row,
the shorter confidence-selected prefixes reduce estimated verifier tokens but
do not convert to throughput. These rows also use `max_tokens=128` and are
`quality_unreliable_short_outputs`.

Qwen3 math K=12 bs=16 is now complete for all 10 TODO methods:
`pure_vllm=1404.19`, `eagle3_oneshot=1515.10`,
`cv_half_sync_simple=1826.19`, `cv_half_sync_roofline=1828.46`,
`cv_half_async_simple=1839.43`, `cv_half_async_roofline=1887.70`,
`cv_conf_sync_simple=1550.28`, `cv_conf_sync_roofline=1558.62`,
`cv_conf_async_simple=1553.80`, and `cv_conf_async_roofline=1548.18`
saturated output tok/s. Fixed-half CV is `1.205x-1.246x` over EAGLE3 at this
concurrency; confidence-sizing is only `1.022x-1.029x`. The imported
`cv_half_async_roofline` row has a common-prompt math quality gate and is
`valid_quality_preserving_chunked`; the shorter `max_tokens=128` rows are
still `quality_unreliable_short_outputs`.

Qwen3 math K=12 bs=32 is now complete for all 10 TODO methods:
`pure_vllm=2670.53`, `eagle3_oneshot=1984.73`,
`cv_half_sync_simple=2480.67`, `cv_half_sync_roofline=2468.09`,
`cv_half_async_simple=2469.89`, `cv_half_async_roofline=2481.38`,
`cv_conf_sync_simple=2329.16`, `cv_conf_sync_roofline=2338.40`,
`cv_conf_async_simple=2338.73`, and `cv_conf_async_roofline=2329.86`
saturated output tok/s. Fixed-half CV is `1.244x-1.250x` over EAGLE3, and
confidence-sizing is `1.174x-1.178x`. However, pure vLLM is faster than both
EAGLE3 and CV in this short-output bs=32 row (`1.346x` versus EAGLE3), so this
scenario is useful for diagnosing speculative overhead under saturated short
outputs. These rows are still `quality_unreliable_short_outputs`.

Qwen3 MTBench K=8 bs=8 is now complete for all 10 TODO methods:
`pure_vllm=716.00`, `eagle3_oneshot=1039.40`,
`cv_half_sync_simple=965.39`, `cv_half_sync_roofline=964.77`,
`cv_half_async_simple=984.48`, `cv_half_async_roofline=962.55`,
`cv_conf_sync_simple=832.38`, `cv_conf_sync_roofline=818.86`,
`cv_conf_async_simple=817.94`, and `cv_conf_async_roofline=801.84`
saturated output tok/s. Fixed-half CV remains below EAGLE3 (`0.926x-0.947x`),
and confidence-sizing is slower (`0.771x-0.801x`). This mirrors the Qwen3 math
K=8 bs=8 result: at small batch and small K, prefix/suffix overhead dominates
the skipped verifier work. MTBench rows use the diagnostic local quality proxy,
not a real MTBench judge score, so they are currently
`quality_unreliable_short_outputs`.

Qwen3 MTBench K=8 bs=16 is now complete for all 10 TODO methods:
`pure_vllm=1394.40`, `eagle3_oneshot=1655.21`,
`cv_half_sync_simple=1684.46`, `cv_half_sync_roofline=1691.47`,
`cv_half_async_simple=1687.66`, `cv_half_async_roofline=1688.67`,
`cv_conf_sync_simple=1423.07`, `cv_conf_sync_roofline=1414.22`,
`cv_conf_async_simple=1434.86`, and `cv_conf_async_roofline=1428.16`
saturated output tok/s. Fixed-half CV is slightly above EAGLE3
(`1.018x-1.022x`), but confidence-sizing remains below EAGLE3
(`0.854x-0.867x`). Compared with bs=8, increasing concurrency makes the
fixed-half prefix/suffix overhead mostly disappear, but the confidence-selected
shorter prefixes still do not pay for their extra scheduling cost.

Qwen3 MTBench K=8 bs=32 is now complete for all 10 TODO methods:
`pure_vllm=2633.50`, `eagle3_oneshot=2181.79`,
`cv_half_sync_simple=2552.86`, `cv_half_sync_roofline=2491.35`,
`cv_half_async_simple=2527.53`, `cv_half_async_roofline=2555.25`,
`cv_conf_sync_simple=2250.76`, `cv_conf_sync_roofline=2230.72`,
`cv_conf_async_simple=2207.58`, and `cv_conf_async_roofline=2185.29`
saturated output tok/s. Fixed-half CV is `1.142x-1.171x` over EAGLE3, and
confidence-sizing is roughly EAGLE3 parity to `1.032x`. As with Qwen3 math K=12
bs=32, pure vLLM is faster than speculative paths in this short-output
saturated row (`1.207x` versus EAGLE3), so report this as a speculative-overhead
diagnostic rather than a final CV win.

Qwen3 MTBench K=12 bs=8 is now complete for all 10 TODO methods:
`pure_vllm=716.93`, `eagle3_oneshot=918.15`,
`cv_half_sync_simple=878.26`, `cv_half_sync_roofline=870.24`,
`cv_half_async_simple=874.54`, `cv_half_async_roofline=868.01`,
`cv_conf_sync_simple=737.93`, `cv_conf_sync_roofline=743.23`,
`cv_conf_async_simple=743.03`, and `cv_conf_async_roofline=740.83`
saturated output tok/s. Fixed-half CV is below EAGLE3 (`0.945x-0.957x`), and
confidence-sizing is lower still (`0.804x-0.809x`). This matches the small-batch
pattern: K=12 gives more removable suffix work than K=8, but at bs=8 the extra
prefix/suffix scheduling still dominates.

Qwen3 MTBench K=12 bs=16 is now complete for all 10 TODO methods:
`pure_vllm=1395.60`, `eagle3_oneshot=1235.95`,
`cv_half_sync_simple=1571.93`, `cv_half_sync_roofline=1545.76`,
`cv_half_async_simple=1560.97`, `cv_half_async_roofline=1557.99`,
`cv_conf_sync_simple=1303.16`, `cv_conf_sync_roofline=1287.87`,
`cv_conf_async_simple=1299.83`, and `cv_conf_async_roofline=1291.71`
saturated output tok/s. Fixed-half CV is `1.251x-1.272x` over EAGLE3, while
confidence-sizing is only `1.042x-1.054x`. Pure vLLM is also faster than
EAGLE3 in this short-output saturated row (`1.129x`), so this row should be
interpreted as a speculative-overhead diagnostic unless paired with a proper
MTBench judge-quality run.

Qwen3 MTBench K=12 bs=32 is now complete for all 10 TODO methods:
`pure_vllm=2638.29`, `eagle3_oneshot=1642.73`,
`cv_half_sync_simple=2121.51`, `cv_half_sync_roofline=2116.98`,
`cv_half_async_simple=2082.98`, `cv_half_async_roofline=2112.77`,
`cv_conf_sync_simple=2021.02`, `cv_conf_sync_roofline=2012.96`,
`cv_conf_async_simple=1973.79`, and `cv_conf_async_roofline=1964.75`
saturated output tok/s. Fixed-half CV is `1.268x-1.291x` over EAGLE3, and
confidence sizing is `1.196x-1.230x`. Pure vLLM is much faster than both
speculative paths in this short-output saturated row (`1.606x` versus EAGLE3),
so use it as a high-concurrency speculative-overhead diagnostic. Qwen3 planned
full-matrix rows are now complete.

Llama3.1 math K=8 bs=16 is now complete for all 10 TODO methods:
`pure_vllm=1391.87`, `eagle3_oneshot=2456.20`,
`cv_half_sync_simple=2072.55`, `cv_half_sync_roofline=2071.94`,
`cv_half_async_simple=2072.25`, `cv_half_async_roofline=2054.53`,
`cv_conf_sync_simple=1853.66`, `cv_conf_sync_roofline=1852.12`,
`cv_conf_async_simple=1850.06`, and `cv_conf_async_roofline=1761.88`
saturated output tok/s. Fixed-half CV is `0.836x-0.844x` of EAGLE3, and
confidence sizing is `0.717x-0.755x`. The common-prompt math quality follow-up
marks `cv_half_async_roofline` as a quality drop, so this K=8 Llama row is not
a valid quality-preserving CV speedup claim.

Llama3.1 math K=12 bs=16 is now complete for all 10 TODO methods:
`pure_vllm=1393.87`, `eagle3_oneshot=1773.40`,
`cv_half_sync_simple=2039.32`, `cv_half_sync_roofline=2029.18`,
`cv_half_async_simple=2028.87`, `cv_half_async_roofline=2087.13`,
`cv_conf_sync_simple=1680.19`, `cv_conf_sync_roofline=1683.39`,
`cv_conf_async_simple=1683.35`, and `cv_conf_async_roofline=1682.86`
saturated output tok/s. Fixed-half CV is `1.144x-1.177x` over EAGLE3, while
confidence sizing is slightly below EAGLE3 at `0.947x-0.949x`.
`cv_half_async_roofline` is marked `valid_quality_preserving_chunked` by the
common-prompt math quality gate.

Llama3.1 math K=8 bs=8 is now complete for all 10 TODO methods:
`pure_vllm=712.43`, `eagle3_oneshot=1460.27`,
`cv_half_sync_simple=1162.15`, `cv_half_sync_roofline=1162.72`,
`cv_half_async_simple=1167.96`, `cv_half_async_roofline=1122.73`,
`cv_conf_sync_simple=1039.85`, `cv_conf_sync_roofline=1040.33`,
`cv_conf_async_simple=1042.24`, and `cv_conf_async_roofline=980.65`
saturated output tok/s. Fixed-half CV is `0.769x-0.800x` of EAGLE3 and
confidence sizing is `0.672x-0.714x`, showing the same small-batch overhead
pattern as Qwen3 K=8 bs=8.

Llama3.1 math K=12 bs=8 is now complete for all 10 TODO methods:
`pure_vllm=713.07`, `eagle3_oneshot=1263.06`,
`cv_half_sync_simple=1112.59`, `cv_half_sync_roofline=1114.45`,
`cv_half_async_simple=1116.10`, `cv_half_async_roofline=1109.37`,
`cv_conf_sync_simple=938.52`, `cv_conf_sync_roofline=937.59`,
`cv_conf_async_simple=935.73`, and `cv_conf_async_roofline=918.07`
saturated output tok/s. Fixed-half CV is `0.878x-0.884x` of EAGLE3, and
confidence sizing is `0.727x-0.743x`. Math planned full-matrix rows are now
complete for both Qwen3 and Llama3.1; remaining planned rows are MTBench.

Llama3.1 math K=8 bs=32 is now complete for all 10 TODO methods:
`pure_vllm=2741.25`, `eagle3_oneshot=3022.88`,
`cv_half_sync_simple=3238.82`, `cv_half_sync_roofline=3221.77`,
`cv_half_async_simple=3232.27`, `cv_half_async_roofline=3223.75`,
`cv_conf_sync_simple=2936.76`, `cv_conf_sync_roofline=2934.46`,
`cv_conf_async_simple=2944.48`, and `cv_conf_async_roofline=2882.13`
saturated output tok/s. Fixed-half CV is `1.066x-1.071x` over EAGLE3, while
confidence sizing is `0.953x-0.974x`. These rows are still
`quality_unreliable_short_outputs`; the strict token-id gate previously showed
Llama3 bs=32 chunked drift, so this is performance evidence, not a final
quality-preserving claim.

Llama3.1 math K=12 bs=32 is now complete for all 10 TODO methods:
`pure_vllm=2744.54`, `eagle3_oneshot=2248.08`,
`cv_half_sync_simple=2752.24`, `cv_half_sync_roofline=2757.99`,
`cv_half_async_simple=2748.51`, `cv_half_async_roofline=2749.52`,
`cv_conf_sync_simple=2512.02`, `cv_conf_sync_roofline=2509.79`,
`cv_conf_async_simple=2499.68`, and `cv_conf_async_roofline=2495.83`
saturated output tok/s. Fixed-half CV is `1.223x-1.227x` over EAGLE3 and
confidence sizing is `1.110x-1.117x`, but pure vLLM is also `1.221x` over
EAGLE3 in this short-output saturated row. Treat this as a high-concurrency
performance diagnostic until paired with a stronger math quality gate.

Llama3.1 MTBench K=8 bs=8 is now complete for all 10 TODO methods:
`pure_vllm=711.73`, `eagle3_oneshot=1302.04`,
`cv_half_sync_simple=1069.33`, `cv_half_sync_roofline=1067.52`,
`cv_half_async_simple=1061.35`, `cv_half_async_roofline=1043.80`,
`cv_conf_sync_simple=907.59`, `cv_conf_sync_roofline=917.09`,
`cv_conf_async_simple=911.73`, and `cv_conf_async_roofline=881.14`
saturated output tok/s. Fixed-half CV is `0.802x-0.821x` of EAGLE3, and
confidence sizing is `0.677x-0.704x`. This matches the small-batch pattern:
the skipped suffix work does not cover the extra chunking/scheduling overhead.

Llama3.1 MTBench K=12 bs=8 is now complete for all 10 TODO methods:
`pure_vllm=711.67`, `eagle3_oneshot=1126.59`,
`cv_half_sync_simple=975.38`, `cv_half_sync_roofline=973.90`,
`cv_half_async_simple=981.49`, `cv_half_async_roofline=973.48`,
`cv_conf_sync_simple=812.38`, `cv_conf_sync_roofline=813.67`,
`cv_conf_async_simple=814.43`, and `cv_conf_async_roofline=799.75`
saturated output tok/s. Fixed-half CV is `0.864x-0.871x` of EAGLE3, and
confidence sizing is `0.710x-0.723x`. Remaining planned rows are Llama3.1
MTBench bs=16/32.

Llama3.1 MTBench K=8 bs=16 is now complete for all 10 TODO methods:
`pure_vllm=1386.23`, `eagle3_oneshot=2088.76`,
`cv_half_sync_simple=1878.06`, `cv_half_sync_roofline=1879.94`,
`cv_half_async_simple=1868.50`, `cv_half_async_roofline=1857.83`,
`cv_conf_sync_simple=1623.77`, `cv_conf_sync_roofline=1625.83`,
`cv_conf_async_simple=1626.74`, and `cv_conf_async_roofline=1574.68`
saturated output tok/s. Fixed-half CV is `0.889x-0.900x` of EAGLE3, and
confidence sizing is `0.754x-0.779x`.

Llama3.1 MTBench K=12 bs=16 is now complete for all 10 TODO methods:
`pure_vllm=1384.93`, `eagle3_oneshot=1528.20`,
`cv_half_sync_simple=1777.47`, `cv_half_sync_roofline=1776.24`,
`cv_half_async_simple=1773.47`, `cv_half_async_roofline=1776.03`,
`cv_conf_sync_simple=1435.90`, `cv_conf_sync_roofline=1441.64`,
`cv_conf_async_simple=1432.38`, and `cv_conf_async_roofline=1433.52`
saturated output tok/s. Fixed-half CV is `1.160x-1.163x` over EAGLE3, while
confidence sizing is `0.937x-0.943x`. Remaining planned rows are only
Llama3.1 MTBench bs=32.

Llama3.1 MTBench K=8 bs=32 is now complete for all 10 TODO methods:
`pure_vllm=2710.49`, `eagle3_oneshot=2691.43`,
`cv_half_sync_simple=2930.47`, `cv_half_sync_roofline=2922.55`,
`cv_half_async_simple=2916.68`, `cv_half_async_roofline=2930.53`,
`cv_conf_sync_simple=2594.17`, `cv_conf_sync_roofline=2592.86`,
`cv_conf_async_simple=2589.11`, and `cv_conf_async_roofline=2573.87`
saturated output tok/s. Fixed-half CV is `1.084x-1.089x` over EAGLE3, while
confidence sizing is `0.956x-0.964x`. Pure vLLM is essentially at EAGLE3 parity
in this short-output high-concurrency row (`1.007x`).

Llama3.1 MTBench K=12 bs=32 is now complete for all 10 TODO methods:
`pure_vllm=2715.90`, `eagle3_oneshot=1990.49`,
`cv_half_sync_simple=2393.39`, `cv_half_sync_roofline=2388.66`,
`cv_half_async_simple=2389.88`, `cv_half_async_roofline=2389.88`,
`cv_conf_sync_simple=2230.18`, `cv_conf_sync_roofline=2234.79`,
`cv_conf_async_simple=2229.17`, and `cv_conf_async_roofline=2223.61`
saturated output tok/s. Fixed-half CV is `1.200x-1.202x` over EAGLE3, and
confidence sizing is `1.117x-1.123x`, but pure vLLM is `1.364x` over EAGLE3.
The planned 240-row steady-state full matrix is now complete.

Latest strict math token-id slices:

```text
temp/speclink_cv_full_live_math_chunked_strict_noreplay_20260529/
temp/speclink_cv_full_live_math_exactsafe_strict_20260529/
```

The strict h<K chunked math slice has 12 rows and 8 failures. Qwen3 passes at
bs=8/16 for K=8 and K=12, but fails at bs=32; Llama3 fails strict matching for
every math chunked row in this slice. The exactsafe math slice has 12 rows and
0 failures across Qwen3/Llama3, K=8/12, and bs=8/16/32. This confirms the
fallback/one-shot guard path is exact for the math slice, while strict h<K
chunking still has batch/model-dependent token drift.

After completing the missing MTBench full-live slices, the current TODO
aggregate has all `48/48` planned full-live token-id correctness rows:
`exactsafe` passes `24/24`, while live h<K `chunked` passes `5/24` under
strict greedy settings and has 19 strict failures. Rows imported with
`greedy_eps=0.125` are retained as diagnostics but are no longer counted as
strict greedy passes. The newly completed local MTBench exactsafe rows all pass.
The newly completed local MTBench bs16/bs32 chunked rows fail for Qwen3 and Llama3.1,
consistent with the earlier math strict-gate drift: h<K chunked verification is
implemented and measurable, but it is not universally token-id exact against
EAGLE3 one-shot under the original strict greedy gate.

The current serving smoke is under:

```text
temp/todo_runner_current_imports_suffix_replay_default_v3_20260528/04_baselines/guidellm_smoke/
```

It uses steady-state saturated throughput with Qwen3/math/K=8/bs=1 and two
rows. EAGLE3 measured about `178.5` output tok/s; CV exact-safe smoke measured
about `159.1` output tok/s. This is only a serving-chain smoke and should not
be used as a final speedup claim.

The first TODO-shaped steady-state full-matrix slice is:

```text
temp/todo_runner_current_imports_suffix_replay_default_v3_20260528/runs/qwen3_8b_math_k12_bs16_eagle3_oneshot/
temp/todo_runner_current_imports_suffix_replay_default_v3_20260528/runs/qwen3_8b_math_k12_bs16_cv_half_async_roofline/
```

The first slow non-staged slice used the default conservative CV eager path:
EAGLE3 measured about `1530.6` output tok/s with `gpu_active_util=99.0`;
`cv_half_async_roofline` measured about `421.4` output tok/s with
`gpu_active_util=90.5`. That is useful as a negative control, but not the
current best performance setting.

The current best-candidate steady-state slices are:

```text
temp/speclink_cv_staged_best_candidate_qwen_llama_math_k8_bs16_quality_20260529/
temp/speclink_cv_staged_best_candidate_qwen_math_k12_bs16_quality_20260529/
temp/speclink_cv_staged_best_candidate_llama_math_k12_bs16_quality_20260529/
```

They use math/bs=16, `max_tokens=256`, a 10s warmup, 30s measurement window,
5s cooldown, `--allow-cv-cudagraph`, batched prefix/suffix, and
`SPECLINK_CV_DENSE_REALIGN_STEPS=0`. The steady-state client now writes
per-request output text and request bodies to `steady_state_requests.jsonl`, so
the matrix analyzer can run the math EM quality gate on steady-state rows.
Current measured rows:

```text
Qwen3 K=8  EAGLE3: 2021.1 tok/s; non-staged CV: 1913.8 tok/s, 0.95x, math_quality_preserved;
           staged CV: 2310.9 tok/s, 1.14x, math_quality_preserved.
Qwen3 K=12 EAGLE3: 1515.1 tok/s; non-staged CV: 1887.7 tok/s, 1.25x, math_quality_preserved;
           staged CV: 2286.1 tok/s, 1.51x, math_quality_preserved.
Llama3 K=8 EAGLE3: 2456.2 tok/s; non-staged CV: 2054.5 tok/s, 0.84x, math_quality_drop;
           staged CV: 2572.7 tok/s, 1.05x, math_quality_drop.
Llama3 K=12 EAGLE3: 1773.4 tok/s; non-staged CV: 2087.1 tok/s, 1.18x, math_quality_preserved;
            staged CV: 2621.8 tok/s, 1.48x, math_quality_drop.
```

Five imported CV rows have `speedup_claim_status=valid_quality_preserving_chunked`.
The strongest valid row is Qwen3/K=12 staged CV at `1.51x`. Llama3 staged rows
are fast but currently fail the math-quality gate, so they should not be used
as valid speedup claims. This reinforces that performance work should
prioritize staged/delayed DLM suffix drafting and batched verifier scheduling,
but quality gating must stay in the reporting loop.

Standalone full-matrix roots can be included in the TODO-level report with:

```bash
conda run -n spec python tools/speclink_cv/run_todo_experiment.py \
  --finalize-only \
  --output-root examples/evaluate/eval-guidellm/temp/todo_runner_TIMESTAMP \
  --full-matrix-import-root examples/evaluate/eval-guidellm/temp/speclink_cv_staged_best_candidate_qwen_llama_math_k8_bs16_quality_20260529 \
  --full-matrix-import-root examples/evaluate/eval-guidellm/temp/speclink_cv_staged_best_candidate_qwen_math_k12_bs16_quality_20260529 \
  --full-matrix-import-root examples/evaluate/eval-guidellm/temp/speclink_cv_staged_best_candidate_llama_math_k12_bs16_quality_20260529
```

The TODO runner's full-live correctness default now leaves
`--full-live-force-prefix-len=0`, so h is selected by the normal half policy
(`K=8 -> h=4`, `K=12 -> h=6`). Use a positive force prefix only for a targeted
diagnostic. The generated `run_full_live_correctness_gate_sliced.sh` should not
contain `--force-prefix-len` unless the force option was explicitly set.

Existing full-live correctness gates can be imported the same way:

```bash
conda run -n spec python tools/speclink_cv/run_todo_experiment.py \
  --finalize-only \
  --output-root examples/evaluate/eval-guidellm/temp/todo_runner_TIMESTAMP \
  --full-live-import-root examples/evaluate/eval-guidellm/temp/speclink_cv_current_qwen_math_k8_bs8_t16_gate_20260528
```

The current imported smoke
`temp/speclink_cv_current_qwen_math_k8_bs8_t16_gate_20260528/` covers
Qwen3/math/K=8/bs=8/max_tokens=16 with `VLLM_BATCH_INVARIANT=1`: both
`chunked` and `exactsafe` match EAGLE3 one-shot `8/8`. In the TODO report this
appears as `partial_2_of_48` full-live rows, not a completed exactness matrix.

The broader current bs=8/max_tokens=32 cross slice is:

```text
temp/speclink_cv_current_bs8_k8_k12_cross_model_data_t32_gate_20260528/
```

Command shape:

```bash
conda run -n spec python -u tools/speclink_cv/run_live_correctness_gate.py \
  --models qwen3_8b,llama3_1_8b \
  --datasets math,mtbench \
  --ks 8,12 \
  --batch-sizes 8 \
  --modes chunked,exactsafe \
  --num-prompts-per-batch \
  --max-tokens 32 \
  --env VLLM_BATCH_INVARIANT=1 \
  --output-root examples/evaluate/eval-guidellm/temp/speclink_cv_current_bs8_k8_k12_cross_model_data_t32_gate_20260528
```

It records 16 rows, 12 matched. All exactsafe rows match 8/8, and plain
chunked matches Qwen/math and Llama/MTBench, but plain chunked still fails:

```text
Qwen3/MTBench/K=8:   7/8, first mismatch token 31
Qwen3/MTBench/K=12:  7/8, first mismatch token 31
Llama3/math/K=8:     7/8, first mismatch token 18
Llama3/math/K=12:    7/8, first mismatch token 18
```

Importing this root plus the contribution root into `run_todo_experiment.py`
produces `full_live_correctness=partial_16_of_48_has_failures`. That is the
current accurate status of the exact token-id TODO gate.

The current near-tie diagnostic rerun is:

```text
temp/speclink_cv_bs8_cross_t32_greedyeps0125_gate_20260528/
```

It used the same Qwen3/Llama3.1 x math/MTBench x K=8/K=12 x bs=8 x max32
slice, but only `mode=chunked`, with `VLLM_BATCH_INVARIANT=1` and
`--greedy-eps 0.125`. All 8 chunked rows matched EAGLE3 one-shot `8/8`,
including the four rows that failed with exact argmax. This is evidence that
the short bs=8 mismatches are near-tie numerical drift, not suffix-discard
bookkeeping failure. It is still diagnostic: `greedy_eps>0` changes greedy
semantics and must not be used as a final exactness proof unless the report
explicitly scopes the claim to this near-tie tolerant mode.

When importing the old exactsafe/chunked root and the new greedy-eps root into
`run_todo_experiment.py`, pass the older root first and the newer greedy-eps
root second. The TODO runner de-duplicates full-live correctness rows by
`model,dataset,K,batch_size,mode`, so later import roots replace earlier rows
for the same case. The resulting TODO dry-run:

```text
temp/todo_runner_import_bs8_cross_t32_greedyeps0125_dryrun_20260528/
```

records `full_live_correctness=partial_16_of_48` with 16/16 matched rows:
8 exactsafe rows from the original root plus 8 near-tie-tolerant chunked rows
from the greedy-eps root. The remaining TODO correctness matrix is still open
for batch sizes 16 and 32, and for strict `greedy_eps=0` chunked rows.

Two targeted bs=16 near-tie follow-ups show the open problem:

```text
temp/speclink_cv_bs16_qwen_mtbench_t32_greedyeps0125_gate_20260528/
temp/speclink_cv_bs16_llama_math_t32_greedyeps0125_gate_20260528/
```

Both used `VLLM_BATCH_INVARIANT=1`, `--greedy-eps 0.125`, K=8/K=12,
`mode=chunked`, `--num-prompts-per-batch`, and max32 outputs. They still fail:

```text
Qwen3/MTBench/K=8/bs=16:   14/16, first mismatch token 9
Qwen3/MTBench/K=12/bs=16:  14/16, first mismatch token 9
Llama3/math/K=8/bs=16:     12/16, first mismatch token 17
Llama3/math/K=12/bs=16:    12/16, first mismatch token 17
```

Importing the bs=8 pass roots plus these bs=16 failure roots into the TODO
runner gives:

```text
temp/todo_runner_import_bs8_pass_bs16_fail_greedyeps0125_dryrun_20260528/
```

with `full_live_correctness=partial_20_of_48_has_failures`: 20/48 rows
recorded, 16 matched. This was the earlier near-tie diagnostic status:
bs=8 can be made near-tie tolerant, but bs=16 still has state/scheduler/KV
divergence that is not explained by simple argmax ties. The newer current
aggregate is the `todo_runner_current_imports_suffix_replay_default_v3_20260528`
bundle described above.

A bounded Qwen3/MTBench/K=8/bs=16/max16 debug rerun with suffix replay disabled
is archived at:

```text
temp/speclink_cv_bs16_qwen_mtbench_k8_t16_no_suffix_replay_debug_greedyeps0125_20260528/
```

It still matches only `14/16`, so suffix replay is not the whole bs=16
correctness problem. However, the trace is useful for implementation hygiene:
with `SPECLINK_CV_SUFFIX_REPLAY_ONE_SHOT_SHAPE=0`, suffix verification for
request 5 uses the expected target positions `76..79` and slots `508..511`;
the earlier replay-enabled trace reused prefix positions `72..75`. The runtime
default is therefore no suffix replay; enable replay only as an explicit
diagnostic.

Summary from the combined K=12/K=16 report:

```text
Qwen3/math/K=12/bs=16:  EAGLE3 1329.7 tok/s,
  non-staged CV 1643.1 tok/s, staged CV 1958.9 tok/s, speedup 1.47x.
Qwen3/math/K=16/bs=16:  EAGLE3 1134.0 tok/s,
  non-staged CV 1417.0 tok/s, staged CV 1691.3 tok/s, speedup 1.49x.
Llama3/math/K=12/bs=16: EAGLE3 2067.2 tok/s,
  non-staged CV 2018.0 tok/s, staged CV 2706.0 tok/s, speedup 1.31x.
Llama3/math/K=16/bs=16: EAGLE3 1861.5 tok/s,
  non-staged CV 1970.9 tok/s, staged CV 2507.0 tok/s, speedup 1.35x.
```

Interpretation: skip-suffix alone is already positive in steady-state
(`1.25x` for Qwen in this run). Staged DLM suffix saving adds another
`1.19x` for Qwen and `1.27x` for Llama over non-staged CV. The singleton-live
control is intentionally slow (`150 tok/s` for Qwen staged in this run); it
shows that the verifier chunks must be batched through vLLM's scheduler for the
theoretical saving to become serving throughput.

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

- `prediction_error_probability`: `P(pred_tokens > actual_accept_tokens)`.
- `compute_efficiency`: `E(pred_tokens / K)`.

Pareto-optimal thresholds are those not dominated by another threshold with both
lower-or-equal `prediction_error_probability` and higher-or-equal
`compute_efficiency`.

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
cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators
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
  suffix for exact TLM verification. The default suffix path verifies only the
  suffix chunk. Prefix replay is an explicit diagnostic mode
  (`SPECLINK_CV_SUFFIX_REPLAY_ONE_SHOT_SHAPE=1`), not the default.
- when `SPECLINK_CV_ROOFLINE_PACKING=1`, estimates whether the current prefix
  chunk launch is underfilled using token/sequence budget utilization; if it is
  below `SPECLINK_CV_UTIL_THRESHOLD`, it falls back to exact one-shot
  verification for that step instead of running a small prefix chunk
- when `SPECLINK_CV_ASYNC_QUEUE=1`, prefix chunks enter a live scheduler queue
  before dispatch. The queue uses selected benefit, age, token budget, sequence
  budget, and roofline utilization to decide which queued prefix chunks run in
  the current scheduler step. Age timeout prevents starvation. This is a
  conservative first live queue, not the final full cross-request packing design.
- exact-safe mode is the default. Because h<K chunked verifier shapes have
  shown greedy argmax drift versus full-K EAGLE3 one-shot, live h<K chunking is
  disabled unless `SPECLINK_CV_ALLOW_SHAPE_DRIFT_CHUNKING=1` is set. For the
  token-id exact h<K debug path, also set `VLLM_BATCH_INVARIANT=1`; the archived
  batch-invariant token gate is useful for drift debugging, not for the current
  performance path.
  Default CV rows therefore fall back to EAGLE3 one-shot and are classified as
  `invalid_no_live_chunking`, not as speedup wins.
- `SPECLINK_CV_ALLOW_BATCHED_PREFIX=1` is a separate experimental multi-prefix
  batch mode. `SPECLINK_CV_ALLOW_BATCHED_SUFFIX=1` lets accepted-prefix suffix
  verification batch as well; this is required for throughput-oriented CV runs.
- `SPECLINK_CV_GLOBAL_BATCH_BARRIER=1` is a diagnostic async-queue mode. It
  dispatches h<K prefix chunks only when every running request can enter the
  same prefix batch. It helps test whether coarse batch-step alignment fixes
  drift; it is not a performance mode.
- `SPECLINK_CV_RECOMPUTE_COMMITTED_PREFIX=1` is a diagnostic KV rollback
  mode. After an h<K prefix chunk commits tokens, the scheduler rolls
  `num_computed_tokens` back so the next TLM step recomputes those committed
  tokens and overwrites KV written by the h<K verifier shape. It is not a
  speedup mode by itself.

Current environment variables:

```text
SPECLINK_CV_ENABLE=1
SPECLINK_CV_CONFIDENCE_SIZING=0
SPECLINK_CV_ASYNC_QUEUE=0
SPECLINK_CV_ROOFLINE_PACKING=0
SPECLINK_CV_CANDIDATE_CHUNKS=1,2,4,6,8,full
SPECLINK_CV_DEFAULT_HALF_POLICY=floor
SPECLINK_CV_MIN_BENEFIT=0.0
SPECLINK_CV_FORCE_PREFIX_LEN=0
SPECLINK_CV_MAX_VERIFY_TOKENS_PER_STEP=0
SPECLINK_CV_MAX_VERIFY_SEQS_PER_STEP=0
SPECLINK_CV_ALLOW_BATCHED_PREFIX=0
SPECLINK_CV_ALLOW_BATCHED_SUFFIX=0
SPECLINK_CV_GLOBAL_BATCH_BARRIER=0
SPECLINK_CV_ALLOW_SHAPE_DRIFT_CHUNKING=0
SPECLINK_CV_SUFFIX_REPLAY_ONE_SHOT_SHAPE=0
SPECLINK_CV_CONFIRM_PREFIX_REJECT_ONE_SHOT=0
SPECLINK_CV_CONFIRM_PREFIX_ACCEPT_ONE_SHOT=0
SPECLINK_CV_CONFIRMATION_FULL_ACTIVE_SET=0
SPECLINK_CV_PREFIX_PROBE_BLOCK_ROLLBACK=0
SPECLINK_CV_PREFIX_LOW_MARGIN_FALLBACK_THRESHOLD=0.0
SPECLINK_CV_BATCH_WIDE_LOW_MARGIN_FALLBACK=0
SPECLINK_CV_BATCH_WIDE_PREFIX_REJECT_FALLBACK=0
SPECLINK_CV_RECOMPUTE_COMMITTED_PREFIX=0
SPECLINK_CV_PREFIX_NO_KV_WRITE=0
SPECLINK_CV_DENSE_REALIGN_STEPS=-1
SPECLINK_CV_PREFIX_REJECT_DENSE_REALIGN_STEPS=0
SPECLINK_CV_MAX_QUEUE_WAIT_MS=2
SPECLINK_CV_UTIL_THRESHOLD=0.6
SPECLINK_CV_CALIBRATION_PATH=
SPECLINK_CV_LOG_JSONL=/path/to/events.jsonl
SPECLINK_CV_PROFILE_JSONL=/path/to/profile.jsonl
SPECLINK_CV_DEBUG_DUMP=0
SPECLINK_CV_KV_DEBUG_TAIL_TOKENS=0
SPECLINK_CV_KV_DEBUG_MAX_LAYERS=0
SPECLINK_CV_KV_DEBUG_ROW_INDEX=-1
SPECLINK_CV_KV_DEBUG_MIN_OUTPUT_TOKENS=-1
SPECLINK_CV_KV_DEBUG_MAX_OUTPUT_TOKENS=-1
SPECLINK_CV_LOG_MAX_EVENTS=1000
SPECLINK_CV_PROFILE_MAX_EVENTS=500
SPECLINK_CV_GREEDY_EPS=0
SPECLINK_CV_DRAFT_ACCEPT_EPS=0
VLLM_BATCH_INVARIANT=0
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
By default `SPECLINK_CV_ALLOW_SHAPE_DRIFT_CHUNKING=0`, so h<K live chunking is
blocked by an exactness guard. The scheduler emits
`shape_drift_guard_one_shot` and schedules the full speculative token set. This
path is token-id exact in the current Qwen3 math bs=8 smoke, but it is not a
chunked verification speedup and the GuideLLM matrix marks it
`invalid_no_live_chunking`.
For token-id exact h<K experiments, use `VLLM_BATCH_INVARIANT=1` together with
`SPECLINK_CV_ALLOW_SHAPE_DRIFT_CHUNKING=1`. Without the batch-invariant vLLM
mode, archived h<K probes show shape-sensitive target-logit drift. With it,
`54_batch_invariant_chunked_matrix/` passes the plain chunked token-id gate
across Qwen3/Llama3.1, math/MTBench, K=8/K=12, batch size 8. Do not generalize
that to the TODO batch-size sweep: `58_batch_invariant_bs16_bs32_matrix/`
passes only 8/16 rows at bs=16/32. For performance work, leave
`VLLM_BATCH_INVARIANT=0` and judge runs by the math exact-match quality gate.
`SPECLINK_CV_GREEDY_EPS=<float>` is a diagnostic extension to
`VLLM_BATCH_INVARIANT=1`: greedy sampling selects the smallest token id whose
logit is within `eps` of the row max. Keep it at `0` unless the experiment is
explicitly studying near-tie numerical drift. On 2026-05-28, the representative
Qwen3/MTBench/K=8/bs=8/max64 `chunked_confirm_all_barrier` case failed at a
near tie (`305` vs `58003`) with `eps=0`, and passed with `eps=0.125` and
`eps=0.25`. This is not a proof of exact SpecLink-CV semantics; it is a
diagnostic way to separate near-tie numeric drift from bookkeeping bugs.
The bounded `eps=0.125` follow-up confirms that distinction: the bs=8
Qwen3/Llama3.1 x math/MTBench x K=8/K=12 x max32 gates all passed, but adding
bs=16 produced 6 failing rows out of 16 total. The combined summary is in
`examples/evaluate/eval-guidellm/temp/speclink_cv_greedy_eps0125_bs8_bs16_t32_summary_20260528/combined_summary.md`.
The failures are Qwen3/MTBench at K=8/K=12, Llama3.1/math at K=8/K=12, and
Llama3.1/MTBench at K=8/K=12. Treat this as evidence that near-tie handling
helps a representative bs=8 failure, while bs=16 still has unresolved
bookkeeping or KV-state correctness issues. A single Qwen3/MTBench/K=8/bs=16
probe with `eps=0.25` still failed (`15/16`, first mismatch at output token 9)
under
`examples/evaluate/eval-guidellm/temp/speclink_cv_greedy_eps025_qwen_mtbench_k8_bs16_t32_20260528_probe/`,
so increasing the near-tie threshold is not a complete explanation.
Later on 2026-05-28, `greedy_sample_with_preferred_tokens` was tightened for
`VLLM_BATCH_INVARIANT=1`: if `SPECLINK_CV_DRAFT_ACCEPT_EPS<=0`, it now returns
the stable greedy token and does not accept a draft token just because it is
exactly tied with the row max. Draft-favoring tie/near-tie behavior is now
opt-in via `SPECLINK_CV_DRAFT_ACCEPT_EPS>0`. The unit check is
`tools/speclink_cv/test_sampler_draft_accept_eps.py`.
A bounded `greedy_eps=0.25` matrix was run at:

```text
examples/evaluate/eval-guidellm/temp/speclink_cv_sampler_tie_eps025_bs8_bs16_t32_matrix_20260528/
```

That matrix passed all 8 Qwen3 rows for math/MTBench, K=8/K=12, bs=8/16, but
failed 6/8 Llama3.1 rows: all four math rows and MTBench bs=16 for K=8/K=12.
Treat this as a diagnostic result only. `greedy_eps` changes greedy semantics,
and for Llama math it can introduce a mismatch that exact argmax would not
necessarily introduce. Do not use `greedy_eps>0` for final correctness claims
against unmodified vLLM+EAGLE3.
The representative Llama3.1/math/K=8/bs=8/max16 debug shows the important
failure mode: token ids, positions, query_start_loc, and logical context tail
matched, but the full-K confirmation after an h<K prefix used different
physical slots/blocks from the one-shot baseline. Baseline used contiguous
slots `[111..119]` and block tail `[4,5,6,7]`; SpecLink-CV used
`[111,816..823]` and block tail `[4,5,6,51]`. The sampled token split at a
low-margin verifier row. `--prefix-no-kv-write` reproduced the same mismatch,
so target-side prefix KV writes alone are not the root cause. The current
working hypothesis is that h<K chunking changes scheduler admission/block
allocation shape, and low-margin rows can diverge even when logical metadata is
the same.
After the tie-breaking fix, the short Qwen3/MTBench/K=8/bs=16 argmax failure
at output token 9 passed:

```text
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs16_t12_tiefix_20260528/
```

The original max32 configuration still fails, so the fix is necessary but not
sufficient:

```text
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs16_t32_tiefix_20260528/
```

It reports `matched_count=13/16`, with the first mismatch at prompt 0 output
token 31. The bounded row-0 debug is:

```text
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs16_t32_row0_debug_20260528/
```

There, the baseline emits token `2213` while SpecLink-CV emits `2033`. The
failure is another low-margin verifier flip: baseline top logits are
`2213=31.75` and `2033=31.625`, while SpecLink-CV sees
`2033=31.625` and `2213=31.625`. The logical context tail matches, but physical
block allocation diverges very early: after the first decode step the one-shot
baseline row 0 uses block tail `[2,3,4,5]`, while SpecLink-CV uses
`[2,3,4,98]`; by the failing step those tails are `[4,5,117,127]` versus
`[4,98,119,129]`. KV checksums for equivalent committed historical tokens can
match across different physical blocks, but low-margin logits still drift
enough to change argmax. A follow-up bounded history-KV rerun makes this more
precise:

```text
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs16_t32_historykv_20260528/
```

At the failing verifier step, it compares the 8 logical positions before target
position 86 (`78..85`) for row 0, layer 0. The baseline and SpecLink-CV physical
slots differ, but all K/V checksums match exactly. The active request set still
differs (`15` active rows in baseline versus `13` in SpecLink-CV), and the
argmax flip is a low-margin/tie case. Treat this as evidence against a simple
"history KV content is dirty" explanation for this bounded window; active batch
shape, current slot layout, or accumulated numerical drift remain in scope.
`SPECLINK_CV_CONFIRMATION_FULL_ACTIVE_SET=1` is a diagnostic follow-up for this
case. When any request is requeued for full-K confirmation, the scheduler also
upgrades every currently running non-suffix request that still holds draft
tokens to full-K confirmation and clears its pending prefix plan. This did not
recover the Qwen3/MTBench/K=8/bs=16/max32 row-0 failure:

```text
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs16_t32_fullactive_20260528/
```

The failing step still lacks request ordinals 2 and 10 in the SpecLink-CV
active set because those requests had already left the active verifier batch;
they were not merely pending normal draft requests omitted from the
confirmation group. `SPECLINK_CV_DRAFT_ACCEPT_EPS=0.01` fixes that row-0 tie
but is not a strict correctness solution:

```text
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs16_t32_eps001_20260528/
```

With that epsilon, request 0 matches, but request 2 diverges at output token 9.
The active request set, logical context, scheduled draft tokens, and target
argmax tokens match; the CV confirmation sees draft token `1896` tied with
target token `312` at logit `31.0`, while the one-shot baseline has `312=31.0`
and `1896=30.875`. This shows fixed draft-preference epsilon can move the
acceptance boundary differently in the baseline and chunked traces.
`SPECLINK_CV_PREFIX_PROBE_BLOCK_ROLLBACK=1` is a bounded diagnostic for
discarded h<K probe state. Before a request is requeued for full-K
confirmation, and after a prefix reject skips its suffix, it truncates whole
tail KV blocks beyond the request's current `num_computed_tokens` so a prefix
probe cannot leave extra reserved block-table entries. It does not clear
partial committed blocks and should be treated as a correctness probe until a
live token gate passes.
Bounded Qwen3/MTBench/K=8/bs=16/max32 reruns with block rollback, and with
block rollback plus full-active-set confirmation, still fail 13/16. The first
mismatch remains request 0 token 31 (`2213` baseline vs `2033` CV); requests 2
and 10 are already one confirmation step ahead in the CV trace, so they are not
present in the later verifier batch for request 0. The bounded pre-target
history KV checksum window still matches exactly for request 0.
Running the same max32 case without the global batch
barrier also fails at token 31, so the issue is broader than the barrier's
batch-fill policy. The exact-safe guard still passes:

```text
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs16_t32_exactsafe_tiefix_20260528/
```

The 2026-05-28 scheduler/worker diagnostics narrowed a few secondary issues
but still did not make live h<K exact. The scheduler now debits full one-shot
shadow token budget for h<K prefix probes so waiting requests are not admitted
only because the probe is short; grouped confirmation can still batch no-spec
dense rows instead of isolating them; and the worker can selectively suppress
drafter output for only the rows affected by a SpecLink-CV realignment instead
of globally zeroing drafts for the whole batch. Bounded Qwen3/MTBench/K=8/bs=16
max32 reruns with those fixes still match only 13/16 prompts:

```text
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs16_t32_shadowbudget_20260528/
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs16_t32_shadowbudget_densemix_20260528/
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs16_t32_selective_drafter_20260528/
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs16_t32_selective_global_20260528/
```

A fresh exact-safe rerun after these diagnostics still passes token-id
correctness, confirming that the default one-shot shape guard remains intact:

```text
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs16_t32_exactsafe_afterfix_20260528/
```

Small bounded representative probes after the same fixes show where the current
h<K failure begins. Qwen3/MTBench/K=8/max32 with forced prefix length 4 passes
at bs=1 and bs=4, but fails at bs=8 with the same stable first mismatch as the
larger runs: request 0 token index 31, baseline `2213` versus CV `2033`.

```text
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs1_t32_representative_20260528/
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs4_t32_representative_20260528/
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs8_t32_representative_20260528/
```

The bs=8 `token_timeline.md` attributes the mismatch to a CV dense target step,
while the baseline is still in a full-K speculative verifier segment. The paired
`active_batch_drift.md` shows identical committed context length and identical
last 16 committed token IDs for request 0, but different active verifier shape
(baseline active request ordinals `[0,1,2,3,4,5,6]` versus CV `[0]`), different
slot/block tails, and a low-margin first-token logit tie. This is further
negative evidence for a simple token-tail or current-position bug; the remaining
suspect is accumulated batch-shape / physical KV-layout / numerical-state drift.
Further bounded bs=8/max32 probes isolate the failure more tightly:

```text
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs8_t32_dense0_20260528/
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs8_t32_batchedprefix_dense0_20260528/
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs8_t32_skiprollback_dense0_20260528/
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs8_t32_rejectconfirm_dense0_20260528/
```

Disabling suffix dense realignment moves the attribution from a dense step back
to `prefix_rejected_skip_suffix`, but the first mismatch remains token 31
(`2213` versus `2033`). Allowing batched prefix verification with the global
barrier still fails the same way. Enabling prefix skip-suffix block rollback
does release tail blocks, but it still fails token 31. Prefix-reject full-K
confirmation is the most diagnostic result: at the failing token, the CV and
baseline full-K confirmation have the same target positions and slot mapping
`[806..813]`, yet the CV target argmax is `[2033,13,576,752,...]` while the
baseline is `[2213,13,576,4024,...]`. This points past current slot mapping and
toward accumulated historical KV/hidden-state numerical drift caused by earlier
h<K/suffix execution.

A tighter bounded KV debug rerun keeps the same representative setup but dumps
only row 0, output-token window 30..32, 8 historical tokens, and the first 4
layers:

```text
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs8_t32_rejectconfirm_kvdebug_20260528/
```

It still fails 7/8 at request 0 token 31, but the bounded history KV checksum
comparison finds 0 mismatches across 32 compared entries before verifier
position 86. Logical context tail, target positions, and current slot mapping
also match. The remaining observed difference is active batch shape: baseline
verifies active ordinals `[0,1,2,3,4,5,6]`, while the CV full-K confirmation is
isolated as `[0]`. This makes the current best explanation active-batch-shape
numerical drift, not dirty historical KV in the bounded window.

Two follow-up bounded diagnostics make this more specific:

```text
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs8_t32_confirm_fullactive_afterprogress_20260528/
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs8_t32_confirmall_barrier_h4_20260528/
```

`SPECLINK_CV_CONFIRMATION_FULL_ACTIVE_SET=1` upgrades currently active draft
requests into the same full-K confirmation step, but the Qwen3/MTBench/K=8
bs=8/max32 gate still fails 7/8 at token 31. The CV confirmation now includes
more active rows than the isolated case, but request ordinal 2 has already
progressed out of the verifier batch, so the active set is still not the
EAGLE3 one-shot active set. Forcing h=4 and using `chunked_confirm_all_barrier`
also fails 7/8 at the same token; the scheduled draft IDs match baseline, but
the active set still misses request 2 and the target slot mapping differs. This
means a local "confirm this request with full K" repair is not enough for exact
token-id equivalence. Exact h<K equivalence appears to require preserving the
global verifier timeline/active batch shape, which conflicts with the intended
async chunking benefit.

`SPECLINK_CV_LOCKSTEP_ITERATION_BARRIER=1` is a stricter diagnostic for this
failure. It records each prefix verifier batch as a speculative iteration group,
holds requests that finish early, stores their next draft tokens, and releases
them only after every request in that group has resolved its prefix/suffix work.
It also allows suffix-phase requests in the same lockstep group to batch
together instead of isolating suffix verification one request at a time. The
bounded Qwen3/MTBench/K=8/bs=8/max32 lockstep runs are archived at:

```text
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs8_t32_lockstep_h4_20260528/
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs8_t32_lockstep_batchsuffix_20260528/
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs8_t32_lockstep_prefixdense8_20260528/
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs8_t32_lockstep_rejectconfirm_20260528/
```

The plain lockstep run still fails 7/8 at request 0 token 31, but its active
batch diagnostic now has equal active ordinals (`[0,1,2,3,4,5,6]` on both
sides). The remaining difference is that the CV draft for the failing prefix is
already different from one-shot: baseline verifies
`[2213,13,576,4024,...]`, while CV verifies `[2213,13,6771,752]`. Batched
suffix verification under lockstep still fails the same way. Adding 8 dense
realignment steps after prefix rejection also fails and moves the first mismatch
to a dense target step. Lockstep plus prefix-reject full-K confirmation also
fails 7/8; its first mismatch is attributed to a dense gap with verifier
positions starting at 84 rather than the baseline one-shot positions starting at
86. Treat this as evidence that the remaining problem is not just active-set
skew; h<K verifier steps are also changing the later DLM/EAGLE hidden-state or
physical slot-layout trajectory before confirmation can recover it.

The 2026-05-28 bounded proposer trace narrows this further:

```text
examples/evaluate/eval-guidellm/temp/speclink_cv_qwen_mtbench_k8_bs8_t32_proposer_debug_20260528/
```

`tools/speclink_cv/analyze_proposer_drift.py` compares only
`proposer_step_debug` rows. For request 0 at output token count 28, baseline and
CV have the same active ordinals, same sample position, and same `next_token_id`
(`19091`), but the target-hidden checksum already differs and the EAGLE draft
diverges after the first four tokens. At output token count 29, both paths still
sample `4024`, but baseline drafts `[2213,13,576,...]` while CV drafts
`[2213,13,6771,...]`; the user-visible token mismatch follows at token 31. This
means live h<K does not merely need suffix bookkeeping cleanup: EAGLE's next
draft is seeded by target hidden states that are numerically/shape dependent on
the shorter verifier forward.

Current correctness conclusion: live h<K chunking remains experimental and is
not a valid speedup-claim path for long MTBench bs=16. The correctness-safe
default remains `SPECLINK_CV_ALLOW_SHAPE_DRIFT_CHUNKING=0`, which falls back to
one-shot verification.
The best current grouped batch-wide prefix-reject fallback diagnostic is
`59_grouped_batchwide_fallback_bs16_bs32_matrix/`. With
`VLLM_BATCH_INVARIANT=1`, it improves the bs=16/32 matrix to 13/16 passing
rows, but still fails Llama/math/K=8 at bs=16 and bs=32 and
Llama/math/K=12 at bs=16. Treat it as partial correctness recovery, not a
complete or performance-valid SpecLink-CV solution.
By default `SPECLINK_CV_ALLOW_BATCHED_PREFIX=0`, so the effective live prefix
sequence budget is one even if `SPECLINK_CV_MAX_VERIFY_SEQS_PER_STEP=0` would
otherwise inherit vLLM's `max_num_seqs`. Set
`SPECLINK_CV_ALLOW_BATCHED_PREFIX=1` for throughput-oriented live h<K runs that
are judged by math quality rather than token-id equality. In sync mode with the
default `SPECLINK_CV_ALLOW_BATCHED_PREFIX=0`, the scheduler emits
`sync_conservative_fallback_one_shot` and schedules the full speculative token
set in the same batch shape as EAGLE3 one-shot.
`SPECLINK_CV_ALLOW_BATCHED_SUFFIX=1` defaults to following the prefix batching
flag. It prevents accepted-prefix suffix chunks from being isolated as singleton
scheduler steps; that isolation was a major bs=16 overhead source.
`SPECLINK_CV_GLOBAL_BATCH_BARRIER=1` only applies with
`SPECLINK_CV_ASYNC_QUEUE=1`. If any running request cannot dispatch a prefix
chunk in the current step, the scheduler emits
`global_batch_barrier_wait`; when all running requests can dispatch together it
emits `global_batch_barrier_dispatch`. Use this only through the diagnostic
`--global-batch-barrier` smoke flag or the `chunked_confirm_all_barrier` gate
mode.
Do not combine `SPECLINK_CV_GLOBAL_BATCH_BARRIER=1` with
`SPECLINK_CV_FORCE_DECODE_ISOLATION=1` for performance experiments. A bounded
probe showed that the two goals conflict: the barrier wants to fill/align the
batch while decode isolation wants to schedule a single request. The scheduler
now lets decode isolation override barrier dispatch to avoid an empty scheduling
loop, but this path is a diagnostic correctness fallback only.
`SPECLINK_CV_FORCE_PREFIX_LEN` is a diagnostic override for smoke tests only.
Leave it at `0` for the TODO ablation matrix; setting it to `K` is useful to
confirm that the one-shot fallback path remains exact, while setting it below
`K` can reproduce chunk-shape exactness failures.
`SPECLINK_CV_SUFFIX_REPLAY_ONE_SHOT_SHAPE=1` is a diagnostic mode for h<K
chunking, and is off by default. When a prefix is fully accepted, the suffix
verifier forward replays the accepted prefix tokens before the suffix. A
bounded Qwen3/MTBench/K=8/bs=16/max16 debug run showed that replay mode writes
the replayed prefix into prefix target positions again, while the no-replay
suffix verifier uses the expected suffix positions. Because replay also adds
extra verifier work, use the no-replay default for performance runs unless the
specific diagnostic needs replay.
`SPECLINK_CV_CONFIRM_PREFIX_REJECT_ONE_SHOT=1` is another diagnostic guard.
When a prefix chunk rejects, the worker masks that prefix output before local
bookkeeping, the scheduler requeues the original full-K draft, and the next
step confirms it with one-shot verification before committing tokens. This is
strictly a correctness probe; it adds an extra verifier pass and is not a valid
speedup claim.
`SPECLINK_CV_CONFIRM_PREFIX_ACCEPT_ONE_SHOT=1` applies the same discard-and-
confirm logic to fully accepted prefixes. It is useful for token-id diagnosis
because it checks whether the original full-K draft can be recovered after the
prefix forward, but it adds verifier work and is not a valid speedup path.
`SPECLINK_CV_PREFIX_LOW_MARGIN_FALLBACK_THRESHOLD` is a diagnostic verifier
margin guard. Values above `0` requeue the original full-K draft when any
prefix verifier top-1/top-2 margin is at or below the threshold; threshold
`0.5` did not recover exactness in the archived probes.
`SPECLINK_CV_BATCH_WIDE_LOW_MARGIN_FALLBACK=1` broadens that diagnostic: if
any request in the current prefix verifier batch is below the threshold, every
request in that prefix batch is routed to full-K confirmation. Threshold `1.5`
with this batch-wide mode still failed the Qwen3/math/K=8/bs=8 token-id gate,
so verifier margin is useful as a risk signal but not as a correctness fix.
`SPECLINK_CV_BATCH_WIDE_PREFIX_REJECT_FALLBACK=1` is a separate diagnostic
fallback. If any request in a prefix verifier batch rejects, every request from
that prefix batch is masked and requeued for full-K confirmation. The scheduler
then groups those requeued full-K confirmations together when possible. This
fixes the archived Qwen3/math/K=8/bs=8 smoke but is still not a general
correctness fix.
`SPECLINK_CV_DENSE_REALIGN_STEPS=-1` preserves the default conservative suffix
rejection policy: use `NUM_SPEC_TOKENS` dense TLM steps after a rejected suffix
chunk and drop the drafter output produced immediately after the suffix
verifier. Setting it to `0` disables that dense-realignment guard for diagnosis
only; setting it to `1` tests a shorter guard. Both `0` and `1` still failed
the Qwen3/math/K=8/bs=8 h<K token-id gate, so this knob is not a correctness
fix or a valid speedup mode.
`SPECLINK_CV_PREFIX_REJECT_DENSE_REALIGN_STEPS=0` is another diagnostic knob.
Positive values suppress the immediate EAGLE drafter after a prefix reject and
then run that many dense TLM realignment steps before drafting resumes. Values
`1` and `8` both failed the Qwen3/math/K=8/bs=8 h<K token-id gate, so prefix-
reject drafter suppression is not a standalone correctness fix.
`SPECLINK_CV_PREFIX_NO_KV_WRITE=1` is a diagnostic-only probe that masks KV
cache slot writes for h<K prefix verifier forwards. The 2026-05-27
Qwen3/MTBench/K=8/bs=8/64-token run
`speclink_cv_prefix_nokv_recompute_barrier_qwen_mtbench_k8_bs8_t64_20260527/`
fails `0/8` with the first mismatch at token 1. Treat this as negative
evidence: suppressing prefix-probe KV writes without a full equivalent replay
does not recover EAGLE3 one-shot semantics.

`SPECLINK_CV_PROFILE_JSONL` is active in the live scheduler path. It records
newline-delimited JSON events for:

- `schedule_step`: scheduled seq/token counts, spec-token counts,
  prefix-chunk counts, remaining token budget, and config toggles.
- `verify_chunk_queued`, `async_queue_step`, `verify_chunk_dequeued`,
  `verify_chunk_waiting`: live async queue state, selected chunks, wait time,
  dispatch reason, predicted utilization, and budget use.

Keep debug logs bounded. `SPECLINK_CV_LOG_MAX_EVENTS` and
`SPECLINK_CV_PROFILE_MAX_EVENTS` cap regular JSONL rows and write one
`jsonl_limit_reached` marker before suppressing further writes. The vLLM
runtime defaults are `log=1000` and `profile=500`; the TODO runner uses tighter
defaults (`--log-max-events=100`, `--profile-max-events=50`) and the GuideLLM
matrix runner defaults to `200/200`. Summary/figure analysis also caps profile
reading with `--analysis-profile-max-rows` so old large JSONL files are not
scanned end to end. Use `0` only for a short trusted run.
`tools/speclink_cv/live_correctness_smoke.py` prints parent-process
start/end markers and elapsed seconds for the baseline EAGLE3 child and the
SpecLink-CV child. If GPU utilization is idle between those markers, the wait
is startup/teardown around the child vLLM process rather than active decoding.
KV checksum debug is also bounded by `SPECLINK_CV_KV_DEBUG_TAIL_TOKENS`,
`SPECLINK_CV_KV_DEBUG_MAX_LAYERS`, `SPECLINK_CV_KV_DEBUG_ROW_INDEX`, and the
min/max output-token filters. Use those filters for representative failing
events instead of full-run all-layer dumps.
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
conda run -n spec python -m tools.speclink_cv.test_sampler_draft_accept_eps
```

`test_sampler_draft_accept_eps` explicitly inserts the vendored
`speculators/vllm` source root before importing vLLM so it can be run with
`python -m` from the repo root without the top-level `vllm/` directory shadowing
`vllm/vllm/__init__.py`.

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

Later batch-shape probes are archived in
`examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/`.
`29_bs1_shape_probe/` shows that plain batch-size 1 h<K chunking still fails
Hallie at token index 13. `30_bs1_confirm_all_probe/` passes 8/8 and has 46/46
comparable full-draft confirmations matching the baseline. `31_barrier_confirm_all_probe/`
uses batch size 8, batched prefix verification, the global batch barrier, and
confirm-all; it failed because the barrier did not wait for pending waiting
requests before dispatching the first prefix. After the batch-fill barrier fix,
`32_barrier_batchfill_confirm_all_probe/` passes 8/8. The true h<K shortcut is
still not exact: `33_barrier_batchfill_plain_probe/` fails 6/8, the low-margin
and accept/reject-only confirmation follow-ups fail, and forced h=1 still fails
5/8. The dense realign diagnostics also fail: `38_barrier_batchfill_dense0_probe/`
disables dense realignment and still fails 5/8; `39_barrier_batchfill_dense1_probe/`
uses one dense realign step and still fails 5/8. Prefix-reject dense realign
also fails: `40_barrier_batchfill_prefixreject_dense1_probe/` and
`41_barrier_batchfill_prefixreject_dense8_probe/` both match only 4/8. A later
offline threshold sweep on the plain trace shows threshold `1.5` would catch
both failed request ordinals, but live threshold `1.5` still fails 7/8 even
with batch-wide low-margin fallback and with low-margin candidates routed
through the prefix accept/reject confirmation branches.
Forced larger-prefix probes also do not recover exactness:
`47_barrier_batchfill_h5_probe/` matches only 6/8, while
`48_barrier_batchfill_h6_probe/` and `49_barrier_batchfill_h7_probe/` match
only 5/8. The h=6/h=7 prefix-step reports show prefix target argmax
mismatches inside matched full drafts, so the shape drift is not limited to
the h=4 prefix bonus boundary. A K=8 Qwen3/math rerun with
`VLLM_ATTENTION_BACKEND=TRITON_ATTN` is archived as
`50_barrier_batchfill_triton_k8_probe/` and still matches only 5/8. The smoke
path already uses `enforce_eager=True`, so do not treat the remaining drift as
a CUDA graph-only or default-attention-backend-only issue.

The latest Qwen3/math K=8 batch-size 8 smoke is archived as
`51_grouped_batchwide_prefixreject_probe/` and matches 8/8. The fix is narrow:
when any row in a prefix batch rejects, the scheduler requeues every row in that
prefix batch for full-K confirmation and schedules those confirmations together.
The prior implementation requeued them but then isolated each confirmation as
row 0, which reproduced the apple-orchard mismatch. Treat this as a validated
smoke for the grouped batch-wide prefix-reject fallback, not as a full
cross-model/K/batch proof yet.

Reusable GPU smoke command:

```bash
conda run -n spec python tools/speclink_cv/live_correctness_smoke.py \
  --speculator-model /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/qwen3-8b-eagle3-speculator \
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
live prefix queue. Add `--allow-batched-prefix-verification` only to reproduce
the experimental multi-prefix batch path; the default smoke uses conservative
exact mode.

For batched correctness, run the same smoke over repo-local prompts:

```bash
conda run -n spec python tools/speclink_cv/live_correctness_smoke.py \
  --speculator-model /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/qwen3-8b-eagle3-speculator \
  --prompts-jsonl examples/evaluate/eval-guidellm/data/math_reasoning.jsonl \
  --num-prompts 8 \
  --max-num-seqs 8 \
  --max-tokens 64 \
  --async-queue \
  --roofline-packing \
  --output-json examples/evaluate/eval-guidellm/temp/speclink_cv_live_batched_smoke.json
```

Add `--chat-completions` to this smoke when validating GuideLLM/OpenAI-chat
serving behavior. The chat smoke renders each prompt as a string user message
through `LLM.chat`; this is the relevant preflight for the GuideLLM
`REQUEST_TYPE=chat_completions` matrix.

Current verification note from 2026-05-26:

- Single-request Qwen3 EAGLE3 K=8 live correctness smoke passed token-id match:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_live_smoke_current_qwen_k8.json`.
- After the conservative exact cap, a 4-prompt Qwen3 EAGLE3 K=8 raw
  `LLM.generate` smoke with fixed-half sync mode passed token-id match:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_live_smoke_qwen_k8_batched_sync_conservative.json`.
- After the conservative exact cap, the same raw 4-prompt smoke with
  `--async-queue --roofline-packing` passed token-id match:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_live_smoke_qwen_k8_batched_async_roofline_conservative.json`.
- For chat-rendered prompts, `max_num_seqs=1` passed token-id match:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_live_smoke_qwen_k8_chat_string_seq_async_roofline_conservative.json`.
- For chat-rendered prompts with `max_num_seqs=4`, sync conservative mode now
  passes token-id match by falling back to full EAGLE3 one-shot verification:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_live_smoke_qwen_k8_chat_string_sync_fallback4.json`.
  Its event log contains `sync_conservative_fallback_one_shot` and no
  `prefix_scheduled` events.
- For chat-rendered prompts with `max_num_seqs=4`, async live chunking is still
  experimental. One post-suffix realignment smoke passed token-id match:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_live_smoke_qwen_k8_chat_string_postsuffix_realign_async_roofline.json`.
  A later repeat mismatched 1/4 prompts:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_live_smoke_qwen_k8_chat_string_async_roofline_final.json`.
  The repeated failure is the Hallie prompt, where the chunked path outputs
  `how many hours a week Hall` instead of one-shot EAGLE3's
  `how many hours Hallie has`.
- A later suffix-draft-drop smoke in the default math file order passed 4/4
  token-id match:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_debug_math4_chat_async_dropdraft.json`.
  Reordering the same four prompts to GuideLLM's sampled order reproduced the
  Hallie mismatch, showing that the remaining issue is batch-shape dependent
  target logits after isolation, not only stale suffix draft reuse.
- The current conservative async path therefore applies dense realignment after
  a suffix reject: it drops the same-step suffix drafter output and suppresses
  speculation for `NUM_SPEC_TOKENS` dense TLM steps before resuming chunked
  verification. This is exact in the current Qwen3 math K=8 bs=4 GuideLLM
  smoke, but it is expensive and must be counted as overhead.
- A pre-fix GuideLLM server smoke at Qwen3 math K=8 bs=4 recorded text-level
  `exact_match_vs_eagle3=0.75`:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_guidellm_conservative_smoke`.
  A later forced-isolation GuideLLM smoke recorded text-level
  `exact_match_vs_eagle3=1.0` for 4/4 requests:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_guidellm_forced_isolation_smoke`.
  The current runner smoke again records text-level
  `exact_match_vs_eagle3=0.75` for Qwen3 math K=8 bs=4
  `cv_half_async_roofline`, and the report correctly writes it to
  `correctness_warnings.csv`:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_guidellm_current_smoke`.
  Because the direct token-id async smoke is not stable, treat batched chat
  async `cv_*` serving rows as experimental unless their own exact-match gate is
  clean.
- With dense realignment enabled, the current Qwen3 math K=8 bs=4 GuideLLM
  smoke records `exact_match_vs_eagle3=1.0` for `cv_half_async_roofline` and
  `speedup_claim_status=valid_exact_chunked`, but throughput is lower than
  EAGLE3 one-shot: `speedup_vs_eagle3=0.415`, `dense_realign_steps=16`, and
  `extra_tlm_forwards_per_request=1.267`.
  Output root:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_guidellm_dense_realign_smoke`.
- Before the conservative exact cap, a 4-prompt Qwen3 EAGLE3 K=8 batched smoke
  with fixed-half sync mode produced a token mismatch on 1/4 prompts:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_live_smoke_current_qwen_k8_batched_sync_simple.json`.
- Before the conservative exact cap, the same 4-prompt smoke with
  `--async-queue --roofline-packing` also
  produced a token mismatch on 1/4 prompts:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_live_smoke_current_qwen_k8_batched_async_roofline.json`.
- Treat any batched `cv_*` GuideLLM row with `exact_match_vs_eagle3 < 1.0` as
  a correctness failure for speedup claims. The result can remain in the table,
  but it is not a valid SpecLink-CV performance win under the TODO correctness
  gate.
- The current sync conservative GuideLLM smoke is exact but intentionally
  rejected by the speedup-claim gate because it falls back to one-shot
  verification instead of doing live chunked verification:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_guidellm_sync_fallback_smoke`.
  Its `cv_half_sync_simple` row has `exact_match_vs_eagle3=1.0`,
  `fallback_ratio=1.0`, and `speedup_claim_status=invalid_no_live_chunking`.
- A later Qwen3 math K=8 batch-size 8 GuideLLM ablation pilot with the same
  regular scheduler mode for baseline and CV completed 10/10 cases:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_guidellm_qwen_math_k8_bs8_ablation_pilot_regularsched`.
  The EAGLE3 one-shot baseline reached about 898 generated tok/s. No CV row
  passed the speedup-claim gate: half-sync rows were text-mismatch warnings,
  confidence-sync rows were exact but `invalid_no_live_chunking`, and all
  async live chunking rows were `invalid_correctness_mismatch`. The async rows
  also slowed down sharply, around 138-159 generated tok/s, because dense
  realignment and queue waits dominated.
- The corresponding direct 8-prompt token-id live smoke also failed for async
  fixed-half chunking:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_live_qwen_math8_k8_async_half_tokens32.json`.
  It matched 6/8 prompts and mismatched prompts 4 and 7. This means the current
  async live CV implementation is not correctness-stable at batch size 8, even
  with conservative one-prefix scheduling and dense realignment. Do not run or
  present the full TODO matrix as a valid speedup experiment until this token-id
  correctness issue is fixed.
- After adding a guard that skips the EAGLE drafter forward during prefix-accepted
  pending-suffix, suffix-verification, and dense-realignment steps, the same
  8-prompt async fixed-half smoke still failed:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_live_qwen_math8_k8_async_half_tokens32_skipdrafter.json`.
  It improved to 7/8 token-id matches, but the remaining mismatch appears at
  token index 5 in the apple-orchard prompt. This shows that drafter KV/cache
  pollution was not the only issue.
- A forced h=1 diagnostic smoke also failed:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_live_qwen_math8_k8_async_h1_tokens32.json`.
  It matched 7/8 prompts and mismatched the Hallie prompt at token index 13.
  Because even h=1 can diverge, the remaining issue is best treated as
  verifier-shape-sensitive greedy output drift between chunked TLM verification
  and full-K EAGLE3 one-shot verification.
- A forced h=K diagnostic smoke passed token-id exactness:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_live_qwen_math8_k8_async_h8_oneshot_tokens32.json`.
  This confirms that the CV guard code does not break the one-shot fallback path;
  the correctness failure is specific to actual h<K chunked verification.
- A current-code 3-case GuideLLM rerun after the skip-drafter guard still failed
  the speedup gate for live h<K chunking:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_guidellm_qwen_math_k8_bs8_async_skipdrafter_current`.
  For Qwen3 math K=8 bs=8, `cv_half_async_simple` had
  `exact_match_vs_eagle3=0.875`, `speedup_vs_eagle3=0.221`, and
  `speedup_claim_status=invalid_correctness_mismatch`. This is the current
  evidence to use when deciding whether the TODO full matrix is ready: it is
  not ready for a valid speedup claim until h<K token-id exactness is fixed.
- A follow-up debug smoke showed that the first skip-drafter guard was too
  broad: when one request needed dense realignment, the whole batch's drafter
  was disabled, so unrelated requests could receive zero draft tokens. Dense
  realignment is now also isolated in the scheduler, preventing zero-draft
  contamination of other requests.
- After that dense-realignment isolation fix, the Qwen3 math K=8 bs=8
  32-token async fixed-half token-id smoke still failed:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_live_qwen_math8_k8_async_half_tokens32_denseiso.json`.
  It matched 6/8 prompts. The mismatches were Hallie at token index 13 and
  apple-orchard at token index 5. The event log no longer contains zero draft
  prefix chunks, so the remaining failure is not the previous batch-wide
  zero-draft bug.
- The matching current-code 3-case GuideLLM rerun after dense-realignment
  isolation is:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_guidellm_qwen_math_k8_bs8_async_denseiso_current`.
  For Qwen3 math K=8 bs=8, `cv_half_async_simple` had
  `exact_match_vs_eagle3=0.625`, `speedup_vs_eagle3=0.140`, and
  `speedup_claim_status=invalid_correctness_mismatch`.
- After adding the default exact-safe shape-drift guard, the Qwen3 math K=8
  bs=8 token-id smoke passed 8/8:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_live_qwen_math8_k8_async_default_exactsafe_tokens32.json`.
  Its event log contains `shape_drift_guard_one_shot` and no live prefix chunk.
  The corresponding GuideLLM run is:
  `examples/evaluate/eval-guidellm/temp/speclink_cv_guidellm_qwen_math_k8_bs8_default_exactsafe_current`.
  The CV row is classified as `invalid_no_live_chunking`, as intended.
  Cross-run text `exact_match_vs_eagle3` is still only 0.625; an EAGLE3-vs-EAGLE3
  repeat across separate GuideLLM runs also matched only 5/8, so this text
  metric should not be treated as the authoritative correctness gate under
  concurrent serving.
- `tools/speclink_cv/analyze_verifier_shape_drift.py` compares debug-dump
  one-shot and chunked verifier events. On the 16-token Qwen3 math bs=8 debug
  smoke, it found 26 prefix-accepted boundary events with matching full-draft
  tokens between one-shot and chunked runs; 25 matched boundary argmax, while
  one apple-orchard event differed: chunked prefix bonus argmax `594` vs
  one-shot full-K boundary target argmax `752` for full draft prefix
  `[198, 32313, 11, 1077]`.
  Artifacts:
  `examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/11_verifier_shape_drift/`.
  The newer top-k debug artifact is:
  `examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/22_verifier_shape_drift_topk/`.
  It adds target/bonus top-5 logits at the boundary. The enhanced analyzer
  compared 104 prefix target positions for matched full drafts and found zero
  prefix argmax mismatches; the remaining mismatch is at the first boundary
  position after the accepted prefix. For that mismatch, the chunked top-k
  begins `[594, 752, ...]` and one-shot top-k begins `[752, 594, ...]`; both
  top-k sets contain the other path's argmax. Treat this as evidence of a
  low-margin verifier-shape argmax flip, not as an async queue bookkeeping-only
  issue.
- The batched-prefix diagnostic exposed and fixed another worker-side hygiene
  issue: when SpecLink-CV intentionally skips the drafter for a realignment
  step, `take_draft_token_ids()` must return empty draft lists, not `[0] * K`
  token drafts. The archived rerun is:
  `examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/23_batched_prefix_emptydraft_probe/`.
  It removes zero-draft pollution but still matches only `6/8`, so the fix is
  necessary but not sufficient for live h<K correctness.
- `SPECLINK_CV_CONFIRM_PREFIX_ACCEPT_ONE_SHOT=1` is a diagnostic companion to
  `SPECLINK_CV_CONFIRM_PREFIX_REJECT_ONE_SHOT=1`. It masks fully accepted
  prefix outputs in the worker and requeues the original full-K draft before
  committing tokens. It is exposed in `live_correctness_smoke.py` as
  `--confirm-prefix-accept-one-shot`; `run_live_correctness_gate.py` mode
  `chunked_confirm_all` enables both accept and reject confirmation.
- `SPECLINK_CV_PREFIX_LOW_MARGIN_FALLBACK_THRESHOLD=<float>` is a diagnostic
  verifier-side guard. For h<K prefix chunks, the worker computes the minimum
  target/bonus top-1 minus top-2 logit margin. If the margin is at or below the
  threshold, it discards the prefix output and requeues the original full-K
  draft for one-shot confirmation. It is exposed in
  `live_correctness_smoke.py` as `--prefix-low-margin-fallback-threshold`.
- `SPECLINK_CV_GLOBAL_BATCH_BARRIER=1` is a diagnostic async-queue guard. It
  dispatches h<K prefix chunks only when every currently running request can
  enter the same prefix batch. It is exposed in `live_correctness_smoke.py` as
  `--global-batch-barrier` and in `run_live_correctness_gate.py` as mode
  `chunked_confirm_all_barrier`.
- The confirm-all probes are archived at
  `examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/24_confirm_all_isolated_probe/`
  and
  `examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/25_confirm_all_batched_probe/`.
  Isolated confirm-all matched `7/8` and still failed apple-orchard because the
  full-K confirmation ran as row 0 instead of the baseline row 7, flipping the
  same low-margin `752` vs `594` boundary. Batched confirm-all also matched
  `7/8`; it fixed apple-orchard but failed Hallie at token index 13 because
  extra prefix-probe/confirmation steps changed later step grouping and the
  subsequent target/drafter state. Treat this as evidence that next-step
  full-K confirmation is still not an exact substitute for the original
  one-shot verifier step.
- `tools/speclink_cv/analyze_prefix_step_equivalence.py` compares each h<K
  prefix probe with the matching full-K one-shot verifier event, when such a
  matching full draft still exists for the same request ordinal. It also
  compares requeued full-K confirmation target argmax vectors against the
  baseline. On `24_confirm_all_isolated_probe`, 39/49 prefix steps had matching
  baseline full drafts and 3 comparable full-K confirmations still differed in
  target argmax. On `25_confirm_all_batched_probe`, only 6/60 prefix steps had
  matching baseline full drafts; 54/60 draft sequences were already absent from
  the one-shot baseline. This separates immediate verifier-shape numerical
  drift from later state trajectory drift.
- The low-margin fallback probes are archived at
  `examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/26_low_margin_probe/`,
  `examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/27_low_margin_reject_confirm_probe/`,
  and
  `examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/28_low_margin_reject_confirm_isolated_probe/`.
  Threshold `0.5` did not restore exactness: guard-only failed apple-orchard
  at token index 5, guard plus reject confirmation failed Hallie under batched
  prefix at token index 13, and isolated guard plus reject confirmation failed
  both Hallie and apple-orchard. Treat this as evidence that a simple
  target-side margin guard is not enough to make live h<K chunking exact in
  the current vLLM/EAGLE path.
- The later margin-threshold diagnostics are archived at
  `examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/42_prefix_margin_guard_33_probe/`
  through
  `examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/46_barrier_batchwide_margin15_reuses_confirm_probe/`.
  Offline, threshold `1.5` catches both failed ordinals from the plain
  batch-fill trace but would fallback 60% of prefix chunks. Live threshold
  `1.5` improves to `7/8` but still fails apple-orchard at token index 5;
  batch-wide low-margin fallback and the refined accept/reject confirmation
  reuse path also fail `7/8`.
- Batch-shape diagnostics are archived at
  `examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/29_bs1_shape_probe/`,
  `examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/30_bs1_confirm_all_probe/`,
  and
  `examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/31_barrier_confirm_all_probe/`.
  Plain bs=1 h<K still fails Hallie at token index 13. Bs=1 confirm-all passes
  `8/8` with zero prefix-step-equivalence mismatches. Bs=8 global-batch-barrier
  confirm-all originally failed because the barrier allowed request 0 to
  dispatch while the other seven prompts were still waiting/prefilling.
- `SPECLINK_CV_GLOBAL_BATCH_BARRIER=1` now also waits for the waiting/skipped
  queues to drain before dispatching prefix chunks. The follow-up archives are
  `32_barrier_batchfill_confirm_all_probe/` through
  `37_barrier_batchfill_h1_plain_probe/`. With this batch-fill barrier,
  confirm-all passes `8/8`, but it is not a speedup path because every prefix
  output is discarded and the original full-K draft is confirmed. Plain h<K
  still fails `6/8`; low-margin threshold `0.5` fails `5/8`; accept-only and
  reject-only confirmation both fail; forced h=1 still fails `5/8`. Treat this
  as the current strongest evidence: the scheduling bug is fixed, but the real
  h<K shortcut is still not exact.
- The same audit root now includes trace-derived confidence and roofline proxy
  artifacts. `03_confidence_calibration/` contains a trace manifest, binning
  calibration model, odd-split ECE/Brier metrics, and reliability diagram.
  `07_roofline_packing/verify_cost_proxy/` contains a trace-derived verify-cost
  lookup. Treat the roofline lookup as a code-path/proxy artifact only, not as
  hardware timing or end-to-end speedup evidence.

`tools/speclink_cv/run_live_correctness_gate.py` wraps
`live_correctness_smoke.py` for repeatable token-id gates across
model/dataset/K combinations. It records per-case commands, stdout, raw
`correctness.json`, event/profile JSONL, `summary.csv`, `summary.json`, and
`report.md`; mismatch is recorded as a failed row instead of aborting the
whole suite. Use it before any GuideLLM speedup claim:

```bash
conda run -n spec python tools/speclink_cv/run_live_correctness_gate.py \
  --models qwen3_8b,llama3_1_8b \
  --datasets math,mtbench \
  --ks 8 \
  --modes chunked,exactsafe \
  --num-prompts 4 \
  --batch-size 4 \
  --max-tokens 16 \
  --output-root examples/evaluate/eval-guidellm/temp/speclink_cv_live_gate_TIMESTAMP
```

For the current exact h<K path, include the batch-invariant vLLM mode:

```bash
conda run -n spec python tools/speclink_cv/run_live_correctness_gate.py \
  --models qwen3_8b,llama3_1_8b \
  --datasets math,mtbench \
  --ks 8,12 \
  --modes chunked \
  --batch-size 8 \
  --num-prompts-per-batch \
  --max-tokens 16 \
  --allow-batched-prefix-verification \
  --env VLLM_BATCH_INVARIANT=1 \
  --output-root examples/evaluate/eval-guidellm/temp/speclink_cv_batch_invariant_gate_TIMESTAMP
```

Use `--modes chunked_confirm` to exercise
`SPECLINK_CV_CONFIRM_PREFIX_REJECT_ONE_SHOT=1` through the gate.
Use `--batch-sizes 8,16,32 --num-prompts-per-batch` to sweep TODO batch sizes
with enough prompts to actually fill each configured `max_num_seqs`; use
`--dry-run` to write commands/configs without starting vLLM.

Current live gate status from 2026-05-26 is summarized in
`examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/13_live_correctness_gate/combined_summary.csv`:
Qwen3/math h<K chunked fails `6/8` at K=8, batch size 8, max tokens 32;
Qwen3/math h<K chunked also fails `3/4` at K=12, batch size 4, max tokens 16;
Llama3.1/math h<K chunked fails `3/4`; Llama3.1/MTBench h<K chunked fails
`3/4`; Qwen3/MTBench h<K chunked passes the 4-prompt, 16-token smoke. The
exactsafe guard passes all five gates. Treat the overall result as a negative
h<K correctness gate because four small gates fail token-id equality.
A Qwen3/math K=12 probe with `VLLM_ATTENTION_BACKEND=TRITON_ATTN` also failed
`3/4` with the same Hallie prompt divergence, so do not assume switching from
the default attention backend fixes the shape-sensitive drift. The later K=8
Triton-attention batch-fill probe also failed `5/8`; this confirms the backend
switch is not a standalone correctness fix.
A Qwen3/math K=12 probe with
`SPECLINK_CV_SUFFIX_REPLAY_ONE_SHOT_SHAPE=1` also failed `3/4`:
`examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/15_suffix_replay_probe/`.
That run had zero `prefix_accepted_requeue_suffix` events and 22
`prefix_rejected_skip_suffix` events, so suffix replay did not trigger. The
remaining Hallie drift is already present in the shorter prefix verifier shape.
A Qwen3/math K=12 `chunked_confirm` probe also failed `3/4` even after worker
side masking of prefix-reject sampled tokens before local bookkeeping:
`examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/16_prefix_reject_confirm_probe/`.
It requeued 22 prefix rejects for full-K one-shot confirmation. This shows that
prefix-reject confirmation alone does not recover one-shot token identity in
the current vLLM integration, likely because the extra prefix verifier forward
still changes KV/cache or runner state that is not fully rolled back. The
default exact-safe guard was rerun after this patch and still passed Qwen3/math
K=12 `4/4`:
`examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/17_exactsafe_after_confirm_probe/`.
The current Qwen3/math K=8 batch-size sweep with true per-batch prompt pressure
is archived at
`examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/18_qwen_math_k8_batch_sweep/`.
It used batch sizes 8, 16, and 32 with matching prompt counts. h<K chunked
failed at all three sizes (`6/8`, `13/16`, `28/32`), while exactsafe passed all
three (`8/8`, `16/16`, `32/32`). A stdout parser fix in
`live_correctness_smoke.py` was needed for bs=32 because vLLM can print
shutdown logs after the child JSON payload.
The matching Qwen3/math K=12 batch-size sweep is archived at
`examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/20_qwen_math_k12_batch_sweep/`.
It also used batch sizes 8, 16, and 32 with matching prompt counts. h<K chunked
failed at all three sizes (`5/8`, `12/16`, `28/32`), while exactsafe passed all
three (`8/8`, `16/16`, `32/32`).
The current cross-model/data K=8 batch-size 8 gate is archived at
`examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/19_cross_model_dataset_k8_bs8/`.
It covers Qwen3-8B and Llama3.1-8B on math and MTBench with 8 prompts. h<K
chunked failed on Qwen3/math (`6/8`), Llama/math (`7/8`), and Llama/MTBench
(`6/8`), while Qwen3/MTBench passed (`8/8`). Exact-safe passed all four rows.
The current cross-model/data K=12 batch-size 8 gate is archived at
`examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/21_k12_cross_model_data_bs8/`.
It covers Qwen3-8B and Llama3.1-8B on math and MTBench with 8 prompts. h<K
chunked failed on Qwen3/math (`5/8`) and Llama/MTBench (`7/8`), while
Qwen3/MTBench and Llama/math passed (`8/8`). Exact-safe passed all four rows.
The newer grouped batch-wide prefix-reject fallback has been rerun as the
Qwen3/math K=8 bs=8 smoke in `51_grouped_batchwide_prefixreject_probe/` and as
the cross-model/data K=8/K=12 batch-size 8 matrix in
`52_grouped_k8_k12_cross_model_data_bs8/`. The smoke passes `8/8`; the matrix
passes `7/8` rows but fails Llama3.1/math/K=8 with `7/8` prompt matches and the
first mismatch at token index 9 (`we first need` vs `we need to first`). The
`prefix_step_equivalence.*` trace shows one prefix decision mismatch and one
prefix target-argmax mismatch: for draft prefix `[5944, 11, 584, 1205]`, the
CV prefix probe accepts all four prefix tokens while the matching EAGLE3
one-shot verifier accepts only three. Do not claim broad h<K correctness or
speedup for this path.

The follow-up worker row-order diagnostic is archived at
`53_worker_ordered_lowmargin_llama_math_k8/`. The worker now reorders the
persistent input batch by request ordinal under the global barrier, moving
request 0 from row 7 to row 0 before verification. That removes the row-binding
symptom but the low-margin full-replay probe still fails `7/8`; the paired
confidence-sizing run with `SPECLINK_CV_MIN_BENEFIT=999` forces one-shot
fallback and passes `8/8`. Treat row alignment as necessary but not sufficient:
the remaining blocker is the h<K verifier shape itself.

The batch-invariant plain-chunked matrix is archived at
`54_batch_invariant_chunked_matrix/`. It was run with
`VLLM_BATCH_INVARIANT=1`, `mode=chunked`, batched prefix verification, and
batch size 8. It passes all 8 rows across Qwen3-8B and Llama3.1-8B, math and
MTBench, and K=8 and K=12, with `8/8` prompts matched in every row. Treat
`VLLM_BATCH_INVARIANT=1` as required for the current bs=8 live h<K path until
a more targeted numerical fix exists. It is not sufficient for the full TODO
batch-size sweep.

The bs=16/32 extension is archived at
`58_batch_invariant_bs16_bs32_matrix/`. It was run with the same
`VLLM_BATCH_INVARIANT=1`, `mode=chunked`, and batched prefix verification. It
passes only 8/16 rows. All Qwen bs=32 rows fail; Qwen/MTBench/K=8/bs=16 fails;
Llama/math/K=8 fails at both bs=16 and bs=32; and Llama/math/K=12/bs=32 fails.
This keeps the full TODO correctness gate open and means no bs=16/32 h<K
GuideLLM speedup claim is valid without a scenario-specific passing token-id
gate.

The grouped batch-wide prefix-reject fallback bs=16/32 extension is archived at
`59_grouped_batchwide_fallback_bs16_bs32_matrix/`. It requeues every row in a
rejecting prefix batch for grouped full-K confirmation and passes 13/16 rows:
all Qwen rows and all Llama/MTBench rows pass, while Llama/math/K=8 fails at
bs=16 and bs=32 and Llama/math/K=12 fails at bs=16. This narrows the failure
surface but does not close the correctness gate or justify a speedup claim.

Longer generation can still break the batch-invariant bs=8 path. The
Qwen3/MTBench/K=8/bs=8 64-token gate is archived at
`60_qwen_mtbench_k8_bs8_t64_batchinv_gate/`: with
`VLLM_BATCH_INVARIANT=1`, plain h<K chunking matches only `21/24` prompts, with
the first mismatch at token index 31. The paired exact-safe fallback gate in
`61_qwen_mtbench_k8_bs8_t64_exactsafe_gate/` passes `24/24`. Do not generalize
the shorter `54_batch_invariant_chunked_matrix/` bs=8 pass to longer
generation lengths without a matching token-id gate.
The grouped batch-wide prefix-reject fallback was rerun for the same long
setting in `63_qwen_mtbench_k8_bs8_t64_grouped_fallback_gate/` after fixing a
global-batch-barrier wait bug. It completes and matches only `22/24`, so
grouped fallback also does not close the long-output MTBench correctness gap.

The matching debug run is archived at
`64_qwen_mtbench_k8_bs8_t64_debug_plain/`. Use
`tools/speclink_cv/analyze_token_timeline.py` to align final token IDs with
debug events and attribute mismatches. For request 0 in that run, the first
mismatch is token index 31: EAGLE3 one-shot emits `2213`, while SpecLink-CV
emits `2033`. The attributing CV segment is `prefix_rejected_skip_suffix`
event 286, with prefix verifier argmax `[2033, 13, 1084, 752]`; the baseline
segment covering the same index is a full-K `spec_step_output` with argmax
`[2213, 13, 576, 4024, 1736, 4658, ...]`. The paired
`65_qwen_mtbench_k8_bs8_t64_reject_confirm_gate/` run enables prefix-reject
full-K confirmation and still matches only `7/8` at first mismatch token 31.
Reject-only confirmation is not enough. The stronger
`66_qwen_mtbench_k8_bs8_t64_confirm_all_gate/` run discards both accepted and
rejected prefix outputs and requeues full-K confirmation, but still matches
only `7/8`. The `67_qwen_mtbench_k8_bs8_t64_confirm_all_barrier_gate/` run adds
the global batch barrier and also matches only `7/8`. Treat the long-output
failure as accumulated verifier/KV shape state drift rather than a missing
confirmation branch or a coarse batch-step alignment issue. The
`68_qwen_mtbench_k8_bs8_t64_prefixreject_dense8_gate/` run forces 8 dense TLM
realignment steps after every prefix reject and still matches only `7/8`, so
immediate drafter reuse after prefix rejection is not the full explanation
either. The follow-up
`69_qwen_mtbench_k8_bs8_t64_confirm_all_barrier_grouped_gate/` run groups all
forced full-K confirmations under the global batch barrier, including full-K
confirmations that are not batch-wide prefix-reject fallbacks. It also matches
only `7/8`. Its `token_timeline.*` shows the attributing CV segment is now a
full-K `spec_step_output`: CV and baseline verify the same scheduled draft IDs,
but the first target argmax differs (`2033` in CV versus `2213` in baseline).
Use `tools/speclink_cv/analyze_active_batch_drift.py` on the same trace for
the verifier batch comparison. The archived `active_batch_drift.*` output shows
both runs have active request ordinals `[0,1,2,3,4,5,6,7]`; baseline scores
`2213=31.75` and `2033=31.625`, while CV ties both at `31.625` and picks
`2033`. So isolated full-K confirmation scheduling and coarse active-set
mismatch are not the remaining explanation; treat this as accumulated
verifier/KV or numerical state drift.

New debug traces written after this point include extra
`verifier_step_debug` fields for `num_computed_tokens_cpu`,
`num_tokens_no_spec`, `worker_output_token_count`, a 16-token
`context_tail_token_ids` slice, and `block_ids_tail`. Re-run
`analyze_active_batch_drift.py` on those traces to check whether the committed
context length and token tail are identical before attributing a mismatch only
to KV numerical drift.
The first rerun with these fields is archived as
`70_qwen_mtbench_k8_bs8_t64_context_debug_gate/`; it reproduces the same
`7/8` failure at token 31 and confirms the mismatching row has identical
`num_computed_tokens_cpu=86`, `num_tokens_no_spec=87`,
`worker_output_token_count=31`, active request ordinals, scheduled draft IDs,
and final 16 committed context token IDs. This rules out committed context
length or token-tail mismatch as the remaining explanation. Its physical
`block_ids_tail` values differ between baseline and CV, so KV layout or
accumulated numerical state remains the narrowed suspect.

The 2026-05-27 follow-up diagnostics are under
`examples/evaluate/eval-guidellm/temp/` because they are non-final probes. A
full-slot reservation patch plus a batch-invariant draft-aware greedy tie
policy moved the Qwen3/MTBench/K=8/bs=8/64-token confirm-all-barrier mismatch
from token 31 to token 48, but did not make h<K exact:
`speclink_cv_tie_prefer_draft_confirm_all_barrier_20260527/` still matches
only `7/8`. The isolated confirm-all variant
`speclink_cv_tie_prefer_draft_confirm_all_isolated_20260527/` also matches only
`7/8` at the same token index. `analyze_active_batch_drift.py` shows the
failing confirmation has identical active request ordinals, context tail, and
scheduled drafts, but different physical `block_ids_tail` and different target
logits (`58003` outranks `305` in CV while the EAGLE3 one-shot baseline ties
or prefers `305`). The paired exact-safe fallback
`speclink_cv_exactsafe_after_tie_20260527/` passes `8/8`. Therefore the
current correctness conclusion remains: true live h<K chunking is not valid for
long Qwen3/MTBench bs=8 generation, and the only correctness-preserving default
is the shape-drift guard that falls back to one-shot verification.
The later `SPECLINK_CV_RECOMPUTE_COMMITTED_PREFIX=1` probe tested whether
rolling back committed-token KV after prefix chunks would repair that failure.
It did not: `speclink_cv_recompute_qwen_mtbench_k8_bs8_t64_20260527/` matches
only `3/8`, and the global-barrier variant
`speclink_cv_recompute_barrier_qwen_mtbench_k8_bs8_t64_20260527/` matches only
`6/8`, still with request 0 diverging at token 31. Its token timeline
attributes the mismatch to a dense target step, so committed-token KV
recompute alone is not enough to emulate the EAGLE3 full-K verifier shape.
The worker-ordered batched dense-realign recompute probe
`speclink_cv_batched_dense_worker_ordered_recompute_barrier_qwen_mtbench_k8_bs8_t64_20260527/`
also fails (`5/8`, first mismatch token 31). In that run the worker input batch
is explicitly reordered back to request ordinal order before dense realign, and
`active_batch_drift.md` confirms equal active request ordinals, equal
`num_computed_tokens_cpu`, equal `num_tokens_no_spec`, and the same final
16-token context tail. The remaining difference is still the physical
`block_ids_tail` and low-margin target logits, so the latest narrowed diagnosis
is KV layout or accumulated numerical state drift rather than scheduler row
order.
The prefix no-KV-write diagnostic
`speclink_cv_prefix_nokv_recompute_barrier_qwen_mtbench_k8_bs8_t64_20260527/`
is a negative probe: it masks KV writes for h<K prefix verifier forwards but
matches only `0/8` and diverges at token 1. Its active-batch drift report shows
the baseline first verifier row is isolated while CV still verifies a full
active batch, so this does not emulate the full-K one-shot path.
`tools/speclink_cv/finalize_correctness_audit.py` includes these supplemental
temp summaries in `09_reports/token_id_correctness_summary.csv` when they are
present locally, while keeping the raw probe directories under `temp/`.

Batch-invariant serving follow-ups are archived at
`55_batch_invariant_guidellm_smoke/`,
`56_batch_invariant_guidellm_qwen_math_k8_bs8_pilot/`, and
`57_batch_invariant_guidellm_prompt_gate_t32/`, plus the newer
`62_qwen_mtbench_k8_bs8_guidellm_batchinv_ablation/`. The `55` smoke validates
that vLLM serve and GuideLLM run with true h<K at Qwen3/math/K=8/bs=1, but the
CV row is slower than EAGLE3. The `56` pilot runs pure vLLM, EAGLE3, and all
8 CV ablations for Qwen3/math/K=8/bs=8 with 8 requests and 32 output tokens.
All rows complete, but every CV row is slower than EAGLE3; the best text-exact
CV row is `cv_conf_async_simple` at about `0.253x` EAGLE3 throughput. Because
several GuideLLM CV rows have text exact-match below 1.0, `57` reruns the
strict token-id gate on the exact GuideLLM prompt subset and passes `8/8` at
32 output tokens. The `62` Qwen3/MTBench/K=8/bs=8 ablation runs 24 requests
and 64 output tokens; all 10 rows complete, but every CV row is slower than
EAGLE3 and fails the GuideLLM text exact-match speedup gate. Treat GuideLLM
text exact-match as conservative serving evidence; strict correctness still
comes from token-id gates.

Use `tools/speclink_cv/finalize_correctness_audit.py` after adding new live
gate results to refresh the top-level audit report, strict token-id summary,
combined `summary_metrics`, and current patch snapshot:

```bash
conda run -n spec python tools/speclink_cv/finalize_correctness_audit.py \
  --audit-root examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036
```

The refreshed report is
`examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/09_reports/SPECLINK_CV_REPORT.md`.
It records strict token-id gate rows, including historical h<K failures,
exact-safe full-K guards marked `invalid_no_live_chunking`, and the passing
Qwen3/math K=8 bs=8 grouped batch-wide prefix-reject smoke in
`51_grouped_batchwide_prefixreject_probe/`, plus the failing broader grouped
matrix in `52_grouped_k8_k12_cross_model_data_bs8/`. The combined table is
`09_reports/summary_metrics.csv`; `09_reports/token_id_correctness_summary.csv`
contains only strict token-id gates. The diff snapshot is written to both
`patches/vllm_speclink_cv.diff` and the older compatibility name
`patches/speclink_cv.diff`. It also writes
`09_reports/TODO_REQUIREMENT_AUDIT.md`, `.csv`, and `.json`, which classify
each major `TODO.md` requirement as complete, partial, failed, or blocked. The
current audit distinguishes historical non-batch-invariant h<K failures from
the passing bs=8 `VLLM_BATCH_INVARIANT=1` smoke in
`54_batch_invariant_chunked_matrix/` and the failing bs=16/32 extension in
`58_batch_invariant_bs16_bs32_matrix/`; the full batch-size correctness gate is
not satisfied. A later grouped batch-wide prefix-reject fallback matrix in
`59_grouped_batchwide_fallback_bs16_bs32_matrix/` passes 13/16 rows at
bs=16/32 and narrows the remaining failures to Llama/math, but it is still not
a complete correctness fix. The later `64`/`65` diagnostics add token-timeline
evidence for the 64-token Qwen3/MTBench failure and show that prefix-reject
confirmation alone still fails. The later `66`/`67` diagnostics show that even
confirm-all and barrier confirm-all still fail the same long-output case.
`68` shows that prefix-reject dense8 realignment also fails. `69` shows that
grouping every forced full-K confirmation under the barrier still fails and
keeps the diagnosis on accumulated verifier/KV shape state drift. The latest
worker-ordered dense-realign recompute temp probe keeps row order aligned and
still fails, so do not treat row-order alignment alone as a correctness fix.
When the local follow-up roots are present, the same finalizer also folds
`temp/speclink_cv_math_k8_k12_bs8_bs32_staged_quality_20260528/` and
`temp/speclink_cv_contribution_ablation_k12_bs16_20260528/` into the combined
`summary_metrics.csv` and adds a `Math-Quality Performance Follow-up` section
to `SPECLINK_CV_REPORT.md`. These rows are explicitly math-quality/performance
evidence under the relaxed EM-preservation metric; they do not satisfy the
original exact greedy token-id TODO gate.

## SpecLink-CV TODO Runner

`tools/speclink_cv/run_todo_experiment.py` is the current top-level entry for
the `TODO.md` experiment bundle. It creates the required
`results/speclink_cv_TIMESTAMP/` directory tree, writes env reports, runs the
focused unit tests, optionally runs a bounded live token-id correctness gate,
optionally runs a small GuideLLM smoke, snapshots the current diff including
untracked SpecLink-CV files, and writes:

```text
09_reports/SPECLINK_CV_REPORT.md
09_reports/summary_metrics.csv
09_reports/summary_metrics.json
09_reports/TODO_REQUIREMENT_AUDIT.md
patches/vllm_speclink_cv.diff
scripts/run_full_steady_state_matrix.sh
scripts/run_full_steady_state_matrix_sliced.sh
scripts/run_full_guidellm_matrix.sh
scripts/run_full_guidellm_matrix_sliced.sh
scripts/run_full_live_correctness_gate.sh
scripts/run_full_live_correctness_gate_sliced.sh
scripts/run_math_quality_followup.sh
scripts/run_contribution_ablation.sh
```

Dry-run the bundle without starting vLLM:

```bash
conda run -n spec python -u tools/speclink_cv/run_todo_experiment.py \
  --dry-run \
  --skip-live \
  --skip-guidellm-smoke \
  --output-root examples/evaluate/eval-guidellm/temp/speclink_cv_todo_runner_dryrun_TIMESTAMP
```

Run the quick correctness bundle with bounded logs:

```bash
conda run -n spec python -u tools/speclink_cv/run_todo_experiment.py \
  --skip-guidellm-smoke \
  --output-root examples/evaluate/eval-guidellm/temp/speclink_cv_todo_runner_quick_TIMESTAMP
```

The quick default live gate is intentionally small:
Qwen3/math, K=8, `batch_size=1`, 2 prompts, 8 output tokens,
`modes=exactsafe,chunked`, forced prefix length 4, and
`VLLM_BATCH_INVARIANT=1`. It is useful evidence that the code path runs, not a
full TODO correctness proof.

For final serving throughput, the TODO runner now defaults to
`--full-benchmark-mode steady_state`, which writes
`scripts/run_full_steady_state_matrix_sliced.sh`. In that script
`batch_size=N` means closed-loop concurrency N, and `throughput` is saturated
output tokens/s counted only inside the fixed measurement window. The legacy
`scripts/run_full_guidellm_matrix_sliced.sh` file is kept as a compatibility
alias to the same command when steady-state mode is selected. Set
`--full-benchmark-mode guidellm` only for finite-request output/quality
inspection, not final serving tokens/s.

The full matrix defaults to one case per slice and reuses the same output root
with `--resume`; set `CASE_LIMIT=N`, `START_OFFSET=N`, and `MAX_SLICES=N` to
control long runs. You can also use `--case-offset/--case-limit` directly on
`run_speclink_cv_guidellm_matrix.py`. The generated matrix command passes
bounded log/profile caps (`--log-max-events`, `--profile-max-events`) and a
bounded analysis read cap (`--analysis-profile-max-rows`); keep these caps for
normal runs and only set a larger value for a short representative diagnostic.
The runner defaults to
`--enforce-eager` and `--disable-vllm-async-scheduling`; use
`--no-enforce-eager` or `--no-disable-vllm-async-scheduling` only when testing
the default vLLM serving mode rather than the conservative SpecLink-CV path.
Use `scripts/run_full_live_correctness_gate_sliced.sh` for the corresponding
48-case token-id correctness gate across the two models, two datasets, K=8/12,
batch sizes 8/16/32, and `chunked,exactsafe` modes.
`tools/speclink_cv/run_live_correctness_gate.py` records mismatches in
`summary.csv` and exits zero by default so sliced long runs can continue to
collect later configurations; pass `--strict-exit-code` when a one-off gate
should fail the shell immediately on any mismatch.
The TODO runner also writes two follow-up scripts for the current performance
direction. `scripts/run_math_quality_followup.sh` runs the relaxed
math_reasoning EM/throughput staged-CV matrix under
`05_cv_ablation/math_quality_followup/`, defaulting to Qwen3/Llama3.1, K=8/12,
batch sizes 8/16/32, EAGLE3 vs `cv_half_async_staged_simple`, and unbounded
math outputs. `scripts/run_contribution_ablation.sh` runs the skip-suffix
contribution ablation under `05_cv_ablation/contribution_ablation/`, separating
non-staged TLM suffix skip, staged DLM suffix saving, and batched scheduling.
These follow-ups are included in the TODO-level `summary_metrics.csv` when the
corresponding outputs exist, but they are explicitly relaxed math-quality
evidence and do not replace the strict token-id correctness gate.
The sliced script refreshes the matrix summary with `--analyze-only` and then
runs:

```bash
conda run -n spec python -u tools/speclink_cv/run_todo_experiment.py \
  --finalize-only \
  --output-root examples/evaluate/eval-guidellm/results/speclink_cv_TIMESTAMP
```

Use the same `--finalize-only` command manually after any direct matrix-runner
slice to restore the TODO-level report. The TODO runner stores its own config
in `logs/todo_run_config.json`; `run_speclink_cv_guidellm_matrix.py` may write
its separate matrix config to `logs/run_config.json`. `--finalize-only` reads
`logs/todo_run_config.json` first, so it preserves the original
`--full-max-requests`, `--full-max-tokens`, `--port`, and slice size instead of
rewriting the scripts with parser defaults.
`--full-live-import-root`, `--math-quality-import-root`, and
`--contribution-import-root` can be repeated to merge standalone runs into a
TODO report. Full-live correctness imports are de-duplicated by
`model,dataset,K,batch_size,mode`; later import roots replace earlier rows, and
diagnostic fields such as `greedy_eps`, `draft_accept_eps`, `profile_max_events`,
and `log_max_events` are preserved in `summary_metrics.csv`.

## SpecLink-CV GuideLLM Matrix Runner

`examples/evaluate/eval-guidellm/scripts/run_speclink_cv_guidellm_matrix.py`
starts vLLM as `python -m vllm.entrypoints.cli.main serve` using the same
Python interpreter that launched the runner. By default
`--benchmark-mode guidellm` runs finite-request GuideLLM diagnostics; use
`--benchmark-mode steady_state` for closed-loop saturated serving throughput.
The runner waits for `/health` with a proxy-safe raw socket check, parses vLLM
speculative metrics, and writes per-run logs plus
top-level `status.csv`, `summary_metrics.csv`, `summary_metrics.json`,
`report.md`, `scripts/run_commands.sh`, the TODO result tree
`00_env/` through `09_reports/`, raw run directories under `runs/`, and figure
source tables under `08_figures/`. It writes the current implementation diff
to `patches/vllm_speclink_cv.diff` and `patches/speclink_cv.diff`. For `cv_*`
methods it adds
`--no-async-scheduling` because SpecLink-CV's live prefix/suffix scheduler
logic is implemented in the regular V1 scheduler; `SPECLINK_CV_ASYNC_QUEUE` is
the experiment's own verification queue and is independent from vLLM's
scheduler async mode. By default it also sets
`SPECLINK_CV_ALLOW_SHAPE_DRIFT_CHUNKING=0`, so `cv_*` rows fall back to
one-shot verification and are marked `invalid_no_live_chunking`. Pass
`--allow-shape-drift-chunking` for live h<K. Add
`--env VLLM_BATCH_INVARIANT=1` only for token-id exact debugging; performance
rows should normally leave it off and rely on the math quality gate.
The summary includes cross-run text-level `exact_match_vs_eagle3` by aligning
GuideLLM successful requests by `request_args` and comparing each `cv_*` output
to the matching `eagle3_oneshot` output for the same model/dataset/K/batch-size
case. This text metric is useful evidence but not the authoritative
correctness gate because independent EAGLE3 GuideLLM repeats can differ under
concurrent serving. Token-id exact-match evidence still comes from
`tools/speclink_cv/live_correctness_smoke.py`. For h<K debug runs, add
`--debug-dump` and analyze the baseline/CV event JSONL with
`tools/speclink_cv/analyze_verifier_shape_drift.py` to see whether a mismatch is
coming from one-shot vs chunked verifier boundary argmax drift. Treat any
live-chunked `cv_*` row with `exact_match_vs_eagle3 < 1.0` as a correctness
warning, not as a valid speedup claim. Exact-safe fallback rows are classified
as `invalid_no_live_chunking`. The report also writes `speedup_claim_valid` and
`speedup_claim_status`; a row is eligible for best-CV selection only when
`speedup_claim_status=valid_quality_preserving_chunked`. This blocks both correctness
mismatches and conservative fallback rows that are exact only because they
reverted to one-shot verification. Rows with
`speedup_claim_status=valid_quality_preserving_chunked`
are valid performance comparisons, not guaranteed speedups. The current
conservative async path may be exact only because it pays dense realignment
overhead after suffix rejects; inspect `dense_realign_steps` and
`extra_tlm_forwards_per_request` before interpreting throughput. Also treat
async batched-chat chunking as experimental even when a small text-level
GuideLLM smoke passes; the direct token-id smoke has shown it can be
non-stable at bs=8 even after dense realignment and the skip-drafter guard.

The runner forces `NO_PROXY/no_proxy` to include local addresses and kills the
vLLM process group during cleanup. This matters in the current environment
because local HTTP proxy variables can otherwise trap `127.0.0.1` health checks,
and a failed API server can leave an EngineCore process holding GPU memory.

For final LLM serving throughput, do not use a finite request batch plus drain
makespan. Use `--benchmark-mode steady_state`, where `batch_size=N` means
closed-loop concurrency N: the client keeps N active streaming requests during
warmup and the fixed measurement window, immediately refills completed
requests, and excludes cooldown/drain from throughput. Report these rows as
`steady-state throughput at concurrency N` or `saturated output tokens/s at
concurrency N`. The formula is:

```text
output_tokens_per_second =
  output tokens emitted during the fixed measurement window /
  measurement window wall-clock seconds
```

All methods in a comparison must use the same workload, concurrency, warmup,
measurement window, random seed, and output length settings. Keep
`--steady-state-ignore-eos` enabled when fixed output length is needed so early
EOS does not make one method look artificially faster. If online serving
latency is needed, run a separate open-loop experiment and report TTFT, TPOT,
P90/P99 latency, SLO attainment, and goodput separately from saturated
tokens/s.

Example steady-state saturated run:

```bash
cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators
conda run -n spec python -u examples/evaluate/eval-guidellm/scripts/run_speclink_cv_guidellm_matrix.py \
  --benchmark-mode steady_state \
  --models qwen3_8b \
  --datasets math \
  --ks 12 \
  --batch-sizes 16 \
  --methods eagle3_oneshot,cv_half_async_staged_simple \
  --max-requests 80 \
  --max-tokens 1024 \
  --steady-state-warmup-s 30 \
  --steady-state-measurement-s 120 \
  --steady-state-cooldown-s 30 \
  --steady-state-ignore-eos \
  --disable-vllm-async-scheduling \
  --allow-shape-drift-chunking \
  --allow-batched-prefix-verification \
  --output-root examples/evaluate/eval-guidellm/temp/speclink_cv_steady_state_TIMESTAMP
```

Smoke command:

```bash
cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators
conda run -n spec python -u examples/evaluate/eval-guidellm/scripts/run_speclink_cv_guidellm_matrix.py \
  --smoke \
  --max-requests 1 \
  --enforce-eager \
  --gpu-memory-utilization 0.75 \
  --port 8051 \
  --output-root examples/evaluate/eval-guidellm/temp/speclink_cv_guidellm_smoke_TIMESTAMP
```

The direct matrix runner now defaults to `results/speclink_cv_TIMESTAMP/` when
`--output-root` is omitted.

Full TODO-shaped matrix command:

```bash
conda run -n spec python -u examples/evaluate/eval-guidellm/scripts/run_speclink_cv_guidellm_matrix.py \
  --models qwen3_8b,llama3_1_8b \
  --datasets math,mtbench \
  --ks 8,12 \
  --batch-sizes 8,16,32 \
  --methods pure_vllm,eagle3_oneshot,cv_half_sync_simple,cv_half_sync_roofline,cv_half_async_simple,cv_half_async_roofline,cv_conf_sync_simple,cv_conf_sync_roofline,cv_conf_async_simple,cv_conf_async_roofline \
  --max-requests 80 \
  --disable-vllm-async-scheduling \
  --allow-shape-drift-chunking \
  --allow-batched-prefix-verification \
  --env VLLM_BATCH_INVARIANT=1 \
  --output-root examples/evaluate/eval-guidellm/results/speclink_cv_TIMESTAMP
```

Equivalent one-command wrapper:

```bash
OUTPUT_ROOT=examples/evaluate/eval-guidellm/results/speclink_cv_TIMESTAMP \
  examples/evaluate/eval-guidellm/scripts/run_speclink_cv_guidellm_full.sh
```

The wrapper defaults to steady-state saturated throughput
(`BENCHMARK_MODE=steady_state`), `--resume`,
`DISABLE_VLLM_ASYNC_SCHEDULING=1`, `ENFORCE_EAGER=0`,
`STEADY_STATE_IGNORE_EOS=1`, `BATCH_INVARIANT=0`,
`ALLOW_SHAPE_DRIFT_CHUNKING=0`, and
`ALLOW_BATCHED_PREFIX_VERIFICATION=0`, and writes to
`results/speclink_cv_TIMESTAMP/` unless `OUTPUT_ROOT` is set. CV rows use the
conservative exact-safe one-shot fallback unless you explicitly opt into h<K.
For focused token-id h<K diagnostics, set
`BATCH_INVARIANT=1 ALLOW_SHAPE_DRIFT_CHUNKING=1
ALLOW_BATCHED_PREFIX_VERIFICATION=1`; do not treat full-matrix h<K throughput
as valid unless the matching quality/token-id gate passes. Use `ENFORCE_EAGER=0` /
`DISABLE_VLLM_ASYNC_SCHEDULING=0` only when you explicitly want the default
vLLM serving mode for the baselines. Use `CASE_OFFSET` and `CASE_LIMIT` to run
chunks of the full matrix. Use `DRY_RUN=1` to materialize commands only,
`ANALYZE_ONLY=1` to rebuild reports from existing run directories, and
`SKIP_UNIT_TESTS=1` when rerunning a narrow slice after the unit tests have
already passed.

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

GuideLLM batch-invariant status on 2026-05-27: Qwen3/math/K=8/bs=8 was rerun
with `VLLM_BATCH_INVARIANT=1`, `--allow-shape-drift-chunking`, and
`--allow-batched-prefix-verification` for pure vLLM, EAGLE3, and all 8 CV
ablations. All 10 rows completed. EAGLE3 one-shot was `402.26` output tok/s;
pure vLLM was `210.00` output tok/s. The best valid CV row in the local
GuideLLM speedup gate was `cv_conf_async_simple` at `101.67` output tok/s,
or `0.253x` EAGLE3, so this pilot shows overhead-dominated slowdown rather
than a speedup. The result is archived at:

```text
examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/56_batch_invariant_guidellm_qwen_math_k8_bs8_pilot/
```

The matching token-id gate over the exact 8 GuideLLM prompts is archived at:

```text
examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/57_batch_invariant_guidellm_prompt_gate_t32/
```

Qwen3/MTBench/K=8/bs=8 was then rerun with 24 requests and 64 output tokens in:

```text
examples/evaluate/eval-guidellm/results/speclink_cv_correctness_audit_20260526_230036/62_qwen_mtbench_k8_bs8_guidellm_batchinv_ablation/
```

EAGLE3 one-shot was `359.06` output tok/s and pure vLLM was `215.45` output
tok/s. The fastest CV row was `cv_half_async_simple` at `102.96` output tok/s
(`0.287x` EAGLE3), but all CV rows had text exact-match below `1.0`; the
paired 64-token strict gate in `60_qwen_mtbench_k8_bs8_t64_batchinv_gate/`
confirmed live h<K token-id mismatch (`21/24`). The exact-safe paired gate in
`61_qwen_mtbench_k8_bs8_t64_exactsafe_gate/` passed `24/24`.
The grouped fallback paired gate in
`63_qwen_mtbench_k8_bs8_t64_grouped_fallback_gate/` passed only `22/24`.

Implementation note: the global batch barrier should wait for waiting requests
only when the running batch has spare sequence capacity. A previous diagnostic
waited whenever `waiting_reqs > 0`; at `max_num_seqs=8` with 24 prompts this
deadlocked prefix verification because the running batch was already full and
waiting requests could not be admitted until running requests progressed.
`should_wait_for_global_batch_fill()` captures the corrected rule and is covered
by `tools/speclink_cv/test_async_queue.py`.

SpecLink-CV correctness status on 2026-05-28: keep debug/probe outputs under
`examples/evaluate/eval-guidellm/temp/`, cap `--log-max-events` and
`--profile-max-events` for representative probes, cap analysis with
`--analysis-profile-max-rows`, and avoid unbounded JSONL dumps in `results/`.
The bounded Qwen3/MTBench/K=8/bs=8/max32 proposer trace
shows plain h<K chunking still fails after the first visible divergence at token
31. The first earlier proposer drift has equal active request ordinals and equal
next token, but different target-hidden checksums and divergent EAGLE drafts,
which points to target-hidden/slot-layout/numerical trajectory drift rather than
just dirty suffix KV. A small `SPECLINK_CV_DRAFT_ACCEPT_EPS=0.01` tolerance
does not fix plain h<K (`6/8` on the representative probe). Grouped batch-wide
full-K fallback plus the same epsilon passes the short K=8/bs=8 representative
matrix across Qwen3/Llama3.1 and math/MTBench (`8/8` for each of the four
cases), but profile events show the prefix rows are discarded and requeued for
full-K confirmation, so it is a correctness fallback rather than a
suffix-pruning speedup.

Later on 2026-05-28, the attempted DLM rollback tightening was corrected. Do
not add the unverified suffix length to EAGLE's worker-side
`num_rejected_tokens_gpu`: that tensor is used by `EagleProposer` to shorten the
current drafter forward's `seq_lens`, and the current prefix target forward did
not include the suffix tokens. The scheduler/request side still discards the
suffix and rolls target KV blocks back to `request.num_computed_tokens`; stale
drafter KV past the resulting seq_len is not visible to subsequent attention.
The debug event for this corrected behavior is
`prefix_skipped_suffix_not_added_to_drafter_rollback`. The older
`prefix_reject_drafter_rollback_adjusted` interpretation should be treated as
superseded.

Slow/idle diagnostic note on 2026-05-28: a historical
`prefix_nokv_decodeiso_confirmall` probe wrote a 4.4GB profile after entering an
empty scheduler loop where `global_batch_barrier_wait_for_batch_fill` kept
waiting for queued requests while decode isolation prevented admitting them.
Current code routes force-decode-isolation prefix work through
`force_decode_isolation_dispatch`; a bounded Qwen3/MTBench/K=8/bs=2/max8 smoke
finished in 68.5s total, produced an 88K temp directory, and did not reproduce
that spin. Remaining low utilization in these gates is mostly repeated vLLM
child cold starts plus conservative correctness barriers, not active decoding.

This smoke proves the runner, vLLM server startup, GuideLLM request path, and
live `verify_chunk_*` CV profile logging work. It is not a throughput claim
because it uses one request.

SpecLink-CV performance focus after the rollback fix: do not treat the
remaining gap as a tuning-only problem. The K=16/h=8 Qwen3 math runs show that
CV can reduce target verification tokens to roughly `0.53x` of EAGLE3 one-shot
while still losing throughput at bs=16/32. Source-level suspects to check first:
the wrapper accidentally forcing `--enforce-eager`, h<K prefixes missing the
native K+1 uniform decode/CUDA-graph path, prefix dispatch underfill, full-K
DLM drafting still running every step, full one-shot slot/token-budget
reservation preventing continuous batching from using saved verifier tokens,
and suffix-phase isolation/replay when prefixes fully accept. Prefer profiling
and scheduler/model-runner changes over more threshold sweeps.

The quick non-eager smoke at:

```text
temp/speclink_cv_forward_plan_smoke_20260528/
```

confirmed `ALLOW_CV_CUDAGRAPH=1` now launches vLLM with `enforce_eager=False`
and captures CUDA graphs. The new `model_forward_plan` profile event shows
K=16/h=8 prefix verification runs as `CUDAGraphMode.PIECEWISE` with
`max_num_scheduled_tokens=9`, not as the native K+1 FULL uniform decode graph.
That makes the next source-level optimization concrete: either add a uniform
decode path/capture key for h+1 prefix verifier shapes, or restructure the
chunked verifier so h<K does not fall back to the less optimized mixed path.

The h+1 CUDA graph path was then added by making `BatchDescriptor` include
`uniform_decode_query_len`, teaching the dispatcher to capture extra
SpecLink-CV prefix query lengths, and detecting uniform h+1 prefix verifier
batches in `gpu_model_runner.py`. A smoke at:

```text
temp/speclink_cv_prefix_fullgraph_smoke_20260528/
```

uses `ALLOW_CV_CUDAGRAPH=1`, `SPECLINK_CV_PREFIX_FULL_CUDAGRAPH=1`, K=16,
and forced h=8. Its profile confirms prefix verifier forwards now dispatch as
`CUDAGraphMode.FULL` with
`BatchDescriptor(num_tokens=18, num_reqs=2, uniform=True,
uniform_decode_query_len=9, ...)` instead of being padded to the K+1 PIECEWISE
shape. This fixes one source-level overhead source; the next performance work
should measure bs=8/16/32 again and then attack remaining overheads:
underfilled prefix dispatches, full-K DLM drafting before every prefix probe,
and lookahead slot/token-budget reservation that may prevent continuous
batching from using skipped verifier tokens.

A small bs=16 A/B smoke at:

```text
temp/speclink_cv_prefix_fullgraph_perf_smoke_20260528/
```

compares EAGLE3 K=16 against CV h=8 with `SPECLINK_CV_PREFIX_FULL_CUDAGRAPH=1`
for 16 math requests and `max_tokens=128`. It is not a final quality run, but it
is useful for performance attribution:

- EAGLE3 throughput: `392.52` generated tok/s.
- CV h=8 throughput: `329.86` generated tok/s (`0.840x` EAGLE3).
- CV skipped about `0.496` of suffix verifier tokens and estimated TLM token
  ratio is `0.537x` of one-shot.
- Prefix forwards mostly use `FULL` (`prefix:FULL:11`), so the earlier
  PIECEWISE/CUDA-graph issue is no longer the dominant blocker.
- Prefix dispatch utilization is still only `0.697`, prefix full-accept is only
  `0.008`, and the DLM still drafts full K before each prefix probe.

This means the next optimization should not be more h tuning. The remaining
high-value source work is to reduce full-K DLM drafting work for likely early
rejects, make skipped verifier tokens visible to the scheduler's token/slot
budget earlier, and avoid treating suffix tokens as already-funded lookahead
work before the prefix has survived.

Staged drafting now addresses the first of those bottlenecks. The method name
is `cv_half_async_staged_simple`: the drafter first proposes only the prefix
width h and generates the suffix only after the prefix fully accepts. A small
bs=16 math A/B at:

```text
temp/speclink_cv_staged_vs_fullk_bs16_20260528/
```

used Qwen3 math, K=16, h=8, `ALLOW_CV_CUDAGRAPH=1`, and
`SPECLINK_CV_PREFIX_FULL_CUDAGRAPH=1` for 16 requests and `max_tokens=128`.
It is a performance attribution run, not a final quality run:

- EAGLE3 one-shot: `391.7` generated tok/s, active GPU util `90.4%`.
- Non-staged CV h=8: `334.1` generated tok/s (`0.853x`), active GPU util
  `88.5%`, estimated draft suffix tokens saved `0`.
- Staged CV h=8: `424.8` generated tok/s (`1.085x`), active GPU util
  `85.2%`, estimated draft suffix tokens saved `512`.

The active-window GPU utilization is already high in these short runs, so the
older "GPU idle means no speedup" diagnosis is too coarse for the active
generation window. The clearer source-level result is that full-K DLM drafting
hid much of the verifier saving; once suffix drafting is staged, the small
bs=16 smoke finally shows an end-to-end speedup.

Current bs=16 quality-preserving result after batched suffix scheduling:

```text
temp/speclink_cv_bs16_k16_staged_quality_20260528/
```

This run used Qwen3 math, K=16, h=8, batch size 16, `MAX_REQUESTS=32`,
`MAX_TOKENS=0`, `cv_half_async_staged_simple`, `ALLOW_CV_CUDAGRAPH=1`,
`BATCH_INVARIANT=0`, `SPECLINK_CV_ALLOW_BATCHED_PREFIX=1`,
`SPECLINK_CV_ALLOW_BATCHED_SUFFIX=1`, and
`SPECLINK_CV_DENSE_REALIGN_STEPS=0`.

- EAGLE3 one-shot: `785.49` aggregate generated tok/s, math EM `28/32`.
- Staged CV h=8: `975.64` aggregate generated tok/s, `1.242x` EAGLE3, math EM
  `29/32`.
- CV skipped suffix verification for about `49.0%` of speculative suffix tokens;
  estimated TLM verification token ratio was `0.539x` of one-shot.
- Staged drafting avoided about `2936` draft suffix tokens versus full-K
  drafting; estimated discarded draft ratio was still high at `0.764`.
- Active-window GPU utilization was already high for both rows (`94.1%` EAGLE3
  vs `93.0%` CV), so the speedup here is not from filling an idle GPU. It comes
  from removing full-K DLM work before prefix rejection and avoiding singleton
  suffix scheduling.

The paired Llama3.1 math run is archived as:

```text
temp/speclink_cv_llama_bs16_k16_staged_quality_20260528/
```

It uses the same K=16, h=8, bs=16, staged-CV performance settings. The result
is a useful negative quality case:

- EAGLE3 one-shot: `848.29` aggregate generated tok/s, math EM `25/32`.
- Staged CV h=8: `1313.65` aggregate generated tok/s, `1.549x` EAGLE3, math EM
  `24/32`.
- The quality gate marks this row `math_quality_drop`, so it is not a valid
  SpecLink-CV speedup claim despite higher throughput.
- qid-level output comparisons are in
  `09_reports/math_cv_wrong_outputs.md` and
  `09_reports/math_cv_drop_outputs.md`.

The math EM parser was updated on 2026-05-28 after a false negative: conclusion
sentences like `So, total is 24 + 10 = 34` must use the last number in the
sentence, not the first operand. The regression check is
`tools/speclink_cv/test_math_answer_extraction.py`.

TODO-aligned K=8/K=12 bs=16 staged-CV focused matrix:

```text
temp/speclink_cv_math_k8_k12_bs16_staged_quality_20260528/
```

This run covers Qwen3 and Llama3.1 on math, `K={8,12}`, batch size 16,
`MAX_REQUESTS=32`, `MAX_TOKENS=0`, EAGLE3 one-shot, and
`cv_half_async_staged_simple`. `FORCE_PREFIX_LEN=0` means default half prefix
selection (`K=8 -> h=4`, `K=12 -> h=6`). All four CV rows passed the math
quality gate:

| model | K | EAGLE3 tok/s | CV tok/s | speedup | EAGLE3 EM | CV EM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B | 8 | `986.15` | `1057.08` | `1.072x` | `29/32` | `29/32` |
| Qwen3-8B | 12 | `782.62` | `980.43` | `1.253x` | `28/32` | `29/32` |
| Llama3.1-8B | 8 | `1231.66` | `1323.68` | `1.075x` | `24/32` | `24/32` |
| Llama3.1-8B | 12 | `817.07` | `1415.12` | `1.732x` | `24/32` | `24/32` |

The focused matrix writes the regular report and figures under `09_reports/`
and `08_figures/`. It is still only math/bs=16, not the full TODO matrix across
MTBench and batch sizes 8/32.

The math-quality runner also writes:

```text
09_reports/steady_state_throughput.csv
09_reports/steady_state_throughput.md
```

These files parse vLLM's periodic `Avg generation throughput` logger rows and
separate steady-state samples from the fixed-request drain tail using
`Running >= ceil(batch_size * STEADY_STATE_RUNNING_FRACTION)`, default `0.8`.
This is a diagnostic only for older finite-request runs; it is not the official
saturated-throughput metric and must not be used for final serving-throughput
claims. Use the matrix runner's `--benchmark-mode steady_state` for fixed-window
closed-loop throughput. Regenerate the diagnostic without serving:

```bash
python tools/speclink_cv/analyze_steady_state_throughput.py \
  <matrix-result-root> \
  --output-dir <matrix-result-root>/09_reports
```

For the current bs=8/32 math matrix:

```text
temp/speclink_cv_math_k8_k12_bs8_bs32_staged_quality_20260528/
```

Qwen K=12 bs=32 shows a clear drain effect in the diagnostic: finite-request CV
speedup is `1.158x`, while the rolling-log active-batch speedup proxy is
`1.443x`; the CV active-batch/E2E ratio is `1.510`, so the fixed-request tail
materially lowers the GuideLLM number. Qwen K=8 bs=32 also has a large tail
ratio (`1.484`), while bs=8 rows are closer to end-to-end. Some short Llama
runs finish before vLLM emits enough periodic logger samples, so their
diagnostic fields are blank.

The all-batch staged math-quality summary combines the bs=8/32 64-request
slice with the bs=16 64-request rerun when present; otherwise it falls back to
the earlier bs=16 32-request slice:

```bash
python tools/speclink_cv/combine_math_quality_followups.py
```

Current combined output:

```text
temp/speclink_cv_math_k8_k12_all_bs_staged_quality_combined_20260528/
```

It writes `09_reports/math_quality_all_batch_summary.{csv,json,md}`,
`09_reports/math_quality_performance_diagnosis.{csv,md}`, and
`08_figures/math_quality_speedup_curve.png`. The combined table has 12 rows:
Qwen3/Llama3.1, K=8/12, batch sizes 8/16/32. With the 64-request bs=16 rerun,
11/12 rows preserve math EM under the relaxed gate and 9/12 have end-to-end
speedup over EAGLE3; mean speedup over quality-preserving rows is `1.164x`.
The quality-drop row is Llama3.1 K=8 bs=16 (`45/64` CV EM vs `47/64` EAGLE3).
The two throughput regressions are Qwen3 K=8 bs=16 (`0.948x`) and Llama3.1 K=8
bs=32 (`0.815x`). The diagnosis report explains these as K=8's smaller
removable suffix, underfilled prefix dispatch, and effective-batch or active
GPU-utilization losses. This is still a relaxed math EM performance summary,
not proof of the original strict token-id TODO gate.

Current TODO aggregate after the common-prompt math quality gate:

```text
temp/todo_runner_current_imports_suffix_replay_default_v3_20260528/
```

This report imports the steady-state best-candidate roots:

```text
temp/speclink_cv_staged_best_candidate_qwen_llama_math_k8_bs16_quality_20260529/
temp/speclink_cv_staged_best_candidate_qwen_math_k12_bs16_quality_20260529/
temp/speclink_cv_staged_best_candidate_llama_math_k12_bs16_quality_20260529/
```

`tools/speclink_cv/steady_state_openai_benchmark.py` now stores streaming text
deltas in `steady_state_requests.jsonl`, and
`run_speclink_cv_guidellm_matrix.py` scores math quality only on prompts common
to the matching EAGLE3 and CV rows. This avoids calling a row a quality drop
just because a faster steady-state method completed a different set of
requests. The refreshed TODO report has `48` full live token-id rows with `19`
strict failures, `2/2` quick serving smoke rows, `240/240` planned steady-state
full-matrix rows, `4` extra staged/best-candidate steady-state rows, `12/12`
relaxed math-quality rows, and `4` contribution-ablation rows.
The strict token-id gate is now complete as evidence but failing for h<K
chunked rows; use the math quality gate only as a relaxed performance gate for
math reasoning. `run_todo_experiment.py`
now restores import roots from `logs/*_plan.json` during `--finalize-only`, so
rerunning sliced matrix scripts will not drop standalone imported evidence from
the aggregate report. It also separates rows outside the original 240-case TODO
matrix, such as `cv_half_async_staged_simple` and historical smoke rows, into
an extra section instead of counting them toward `x/240`.

A focused strict diagnostic was added for the most conservative h<K
confirmation path:

```text
temp/speclink_cv_strict_confirmall_rollback_mode_20260529/
```

It runs Qwen3/MTBench/K=8/bs=8/max32 with `VLLM_BATCH_INVARIANT=1`,
confirm-all, global barrier, prefix-probe block rollback, full-active-set
confirmation, and lockstep iteration barriers. It still matches only `7/8`,
with request 0 diverging at token 31 (`2213` baseline versus `2033` CV). The
bounded `token_timeline.md` attributes the mismatch to a CV dense gap after the
prefix-confirm/rollback sequence, while the bounded event log reaches request 0
only through output token 12; use a larger but still bounded `--log-max-events`
only if later analysis needs the exact dense event at token 31. This is new
negative evidence that the remaining strict h<K drift is not fixed by
discarding prefix outputs, rolling back prefix-probe blocks, or regrouping
full-K confirmations. `run_live_correctness_gate.py` now has
`mode=chunked_confirm_all_rollback_barrier` to reproduce this diagnostic and
writes `strict_greedy`, `strict_matched`, and `strict_failure_reason` columns
in its summaries/reports.

The current steady-state h<K rows with
`speedup_claim_status=valid_quality_preserving_chunked` are:

| model | K | batch | method | saturated tok/s | speedup vs EAGLE3 | common-prompt math delta |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Qwen3-8B | 8 | 16 | `cv_half_async_roofline` | `1913.83` | `0.947x` | `+0.0033` |
| Qwen3-8B | 8 | 16 | `cv_half_async_staged_simple` | `2310.87` | `1.143x` | `+0.0094` |
| Qwen3-8B | 12 | 16 | `cv_half_async_roofline` | `1887.70` | `1.246x` | `-0.0042` |
| Qwen3-8B | 12 | 16 | `cv_half_async_staged_simple` | `2286.10` | `1.509x` | `-0.0084` |
| Llama3.1-8B | 12 | 16 | `cv_half_async_roofline` | `2087.13` | `1.177x` | `-0.0072` |

Llama3.1 K=8 CV rows and Llama3.1 K=12 `cv_half_async_staged_simple` are not
valid claims under the common-prompt math quality gate, even when throughput
is higher. A `valid_quality_preserving_chunked` row is a valid h<K comparison;
it is not necessarily a speedup, as shown by Qwen3 K=8 roofline at `0.947x`.

Contribution ablation entry point:

```bash
cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators
GPU_MEMORY_UTILIZATION=0.75 bash examples/evaluate/eval-guidellm/run_speclink_cv_contribution_ablation.sh
```

This script keeps the core h<K `skip_suffix` mechanism enabled in every CV
variant. It runs a batched-live group with
`eagle3_oneshot,cv_half_async_simple,cv_half_async_staged_simple` and a
singleton-live group with `cv_half_async_staged_simple`. The singleton-live
group uses `ALLOW_BATCHED_PREFIX_VERIFICATION=1`, `MAX_VERIFY_SEQS_PER_STEP=1`,
and `ALLOW_BATCHED_SUFFIX=0`; do not use
`ALLOW_BATCHED_PREFIX_VERIFICATION=0` for this ablation because that triggers
the conservative one-shot fallback and disables the h<K skip-suffix path. The
summary script is:

```bash
python tools/speclink_cv/analyze_contribution_ablation.py \
  --batched-root <output>/batched \
  --singleton-root <output>/singleton_live \
  --output-dir <output>/09_reports
```

It writes `contribution_ablation.csv` and `contribution_ablation.md`, reporting
DLM suffix-saving speedup (`staged / non-staged`) and batch-scheduling speedup
(`batched / singleton-live`) while keeping skip-suffix fixed.

Current contribution ablation result:

```text
temp/speclink_cv_contribution_ablation_k12_bs16_20260528/
```

This run used Qwen3/Llama3.1, math, K=12, bs=16, `MAX_REQUESTS=64`, and
unbounded outputs. Key rows from `09_reports/contribution_ablation.md`:

| model | EAGLE3 tok/s | non-staged batched | staged batched | staged singleton | staged speedup | DLM suffix speedup | batch sched speedup | quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3-8B | `1116.85` | `1184.87` | `1487.32` | `177.84` | `1.332x` | `1.255x` | `8.363x` | valid |
| Llama3.1-8B | `1517.19` | `1592.61` | `1839.68` | `217.60` | `1.213x` | `1.155x` | `8.454x` | math quality drop |

Interpretation:

- `skip_suffix` is not ablated; it stays enabled for all CV rows.
- Non-staged CV already benefits a little from TLM suffix skipping, but still
  drafts full K before the prefix result. Staged drafting adds another
  `1.16x-1.26x` by avoiding many DLM suffix tokens.
- Singleton-live CV keeps skip-suffix but serializes verifier scheduling. It is
  about `8.4x` slower than batched-live staged CV, showing that connecting
  prefix/suffix verification back to vLLM's batch scheduler is mandatory for
  the theoretical savings to appear end to end.
- The steady-state diagnostic for this ablation writes to
  `09_reports/steady_state_throughput.md`. For Qwen K=12 bs=16, batched staged
  CV has end-to-end speedup `1.332x` and steady-state speedup `1.421x`, while
  singleton-live staged CV is only `0.159x` end-to-end and `0.136x`
  steady-state. This means singleton-live is not just hurt by final drain; it
  is fundamentally inefficient while the batch is populated.
- Llama does not always have higher valid speedup than Qwen. In the bs8/32
  matrix, Llama K=12 bs=32 speedup is higher mostly because EAGLE3 has more
  scheduling headroom: active GPU util is `80.5%` vs Qwen's `96.3%`, and staged
  CV increases scheduled seqs/step from `20.61` to `30.25` for Llama versus
  `22.35` to `30.19` for Qwen. The skip-suffix ratios are similar
  (`0.453` Llama vs `0.458` Qwen), so the difference is not that Llama skips
  substantially more verifier suffix work.

Latest correctness/performance diagnostic, 2026-05-29:

- Final lightweight report bundle:
  `examples/evaluate/eval-guidellm/results/speclink_cv_20260529_final/`.
  It contains reports, figures, scripts, patch snapshots, unit/env evidence, and
  imported summary rows. Large raw run directories remain under `temp/` and are
  referenced by `source`/`output_root` fields in the summary tables.
- In that final bundle, use `09_reports/primary_summary_table.csv` for the
  TODO-required total table and `09_reports/best_speclink_cv_candidates.csv`
  for relaxed math-quality speedup candidates. `SPECLINK_CV_REPORT.md` has a
  `TODO Questions` section that summarizes the current answers and remaining
  correctness limitation.
- The full steady-state serving matrix has already completed (`240/240` rows).
- The strict full-live token-id audit is not fully green. A current-config
  partial refresh with `ALLOW_BATCHED_PREFIX_VERIFICATION=1` and
  `SPECLINK_CV_DENSE_REALIGN_STEPS=0` was intentionally stopped after `17`
  h<K chunked rows once remaining failures were clear: `8/17` passed and
  `9/17` failed. In the merged final report this lowers the broad strict
  failure count to `15/48`, but it still blocks exact-correct SpecLink-CV
  claims. The partial refresh lives at
  `temp/speclink_cv_full_live_chunked_currentcfg_20260529/`.
- Relaxed math-quality follow-up rows are complete (`30/30` ok). Use the
  report's math-quality table for current Qwen3/Llama math K=8/12/16 rows.
- A focused Qwen3/MTBench/K=8/bs=8 plain chunked run with the old conservative
  suffix dense-realign behavior still failed (`7/8`) because suffix rejection
  removed later draft-aware acceptance opportunities.
- The same focused run with the performance configuration
  `SPECLINK_CV_DENSE_REALIGN_STEPS=0` passed (`8/8`):
  `temp/speclink_cv_plain_chunked_dense0_qwen_mtbench_k8_bs8_20260529/`.
- For performance/quality experiments, prefer the relaxed-performance setting
  `SPECLINK_CV_DENSE_REALIGN_STEPS=0` and validate math EM, rather than forcing
  strict token-id equality with EAGLE3 one-shot.

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
