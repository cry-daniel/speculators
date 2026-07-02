#!/usr/bin/env python3
"""Run a small SR24 lossy speed/quality gate.

This script is intentionally a thin orchestrator over the existing lm-eval and
GuideLLM runners.  It implements the current SR24 optimization rule:

1. Quality is allowed to drop, but only within a configurable pp budget
   (default 8 pp on GSM8K limit=50).
2. Throughput is measured only for candidates that pass that quality gate.
3. Intermediate outputs go under results.bak/temp instead of results/.

Example:

  cd examples/evaluate/eval-guidellm
  conda run -n spec python scripts/run_sr24_lossy_speed_quality_sweep.py \
    --candidates lowresidual_gateup_riskcap2,mlpall_lowconf_prefix5_tritonoverride \
    --batch-sizes 8,16,32,64

Use --dry-run to print the exact commands without launching vLLM.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EVAL_ROOT = Path(__file__).resolve().parents[1]
SPECULATORS_ROOT = EVAL_ROOT.parents[2]
DEFAULT_OUTPUT_ROOT = EVAL_ROOT / "results.bak"
DEFAULT_TEMP_ROOT = EVAL_ROOT / "temp"
FRONT24_SERVING_POLICY = (
    EVAL_ROOT / "configs/sr24_scheduler_policy_front24_serving_bs8_64.json"
)
PREFIX2_FRONT28_PACKED_POLICY = (
    EVAL_ROOT
    / "configs/sr24_scheduler_policy_prefix2_front28_packed_k8_effective64.json"
)


@dataclass(frozen=True)
class Candidate:
    name: str
    preset: str
    note: str
    extra_quality_args: tuple[str, ...] = ()
    extra_throughput_args: tuple[str, ...] = ()


CANDIDATES: dict[str, Candidate] = {
    "criticalprefix4_bucket16_directcslt": Candidate(
        name="criticalprefix4_bucket16_directcslt",
        preset="criticalprefix4_bucket16_directcslt",
        note=(
            "quality-first row: gate_up 16-31 + down 8-15, "
            "critical-prefix residual, bucket16/direct cuSPARSELt"
        ),
    ),
    "criticalprefix4_bucket12_directcslt_aligned": Candidate(
        name="criticalprefix4_bucket12_directcslt_aligned",
        preset="criticalprefix4_bucket16_directcslt",
        note=(
            "aligned quality/throughput candidate from the historical best "
            "bucketed SR24 row: gate_up 16-31 + down 8-15, "
            "critical-prefix residual with extra_after_low=1, bucket12 dense "
            "copy, direct cuSPARSELt, and the same routing in lm-eval and "
            "GuideLLM. This rechecks whether the older near-1.2x row still "
            "holds in the current tree."
        ),
        extra_quality_args=(
            "--sr24-residual-bucket-size",
            "12",
        ),
        extra_throughput_args=(
            "--sr24-residual-bucket-size",
            "12",
        ),
    ),
    "criticalprefix4_bucket12_activeonly_directcslt": Candidate(
        name="criticalprefix4_bucket12_activeonly_directcslt",
        preset="criticalprefix4_bucket16_directcslt",
        note=(
            "same quality-safe critical-prefix layer split as bucket12, but "
            "dense correction is computed/scattered only for runtime-active "
            "bucket rows. Padding or low-importance rows keep the 2:4 sparse "
            "base result instead of being overwritten by dense."
        ),
        extra_quality_args=(
            "--sr24-residual-bucket-size",
            "12",
            "--sr24-bucket-dense-copy-active-only",
            "--sr24-bucket-dense-compute-active-only",
            "--sr24-triton-bucket-scatter",
        ),
        extra_throughput_args=(
            "--sr24-residual-bucket-size",
            "12",
            "--sr24-bucket-dense-copy-active-only",
            "--sr24-bucket-dense-compute-active-only",
            "--sr24-triton-bucket-scatter",
        ),
    ),
    "lowresidual_gateup_riskcap2": Candidate(
        name="lowresidual_gateup_riskcap2",
        preset="lowresidual_gateup_riskcap2",
        note=(
            "speed-first row: gate_up-only, prefix2 + two low-confidence "
            "draft rows, bucket8/direct cuSPARSELt"
        ),
    ),
    "lowresidual_gateup_riskcap2_rowrouted": Candidate(
        name="lowresidual_gateup_riskcap2_rowrouted",
        preset="lowresidual_gateup_riskcap2",
        note=(
            "same low-residual gate_up controller, but row-route corrected "
            "gate_up rows so important tokens skip the sparse-base branch"
        ),
        extra_quality_args=(
            "--sr24-row-routed-mlp",
            "--sr24-row-routed-mlp-min-dense-rows",
            "1",
        ),
        extra_throughput_args=(
            "--sr24-row-routed-mlp",
            "--sr24-row-routed-mlp-min-dense-rows",
            "1",
        ),
    ),
    "lowresidual_gateup_riskcap2_rowrouted_min128": Candidate(
        name="lowresidual_gateup_riskcap2_rowrouted_min128",
        preset="lowresidual_gateup_riskcap2",
        note=(
            "planner row-route gate_up only when the dense-important group has "
            "at least 128 rows; smaller groups fall back to base-first overwrite"
        ),
        extra_quality_args=(
            "--sr24-row-routed-mlp",
            "--sr24-row-routed-mlp-min-dense-rows-by-leaf",
            "gate_up_proj=128",
        ),
        extra_throughput_args=(
            "--sr24-row-routed-mlp",
            "--sr24-row-routed-mlp-min-dense-rows-by-leaf",
            "gate_up_proj=128",
        ),
    ),
    "lowresidual_gateup_riskcap2_rowrouted_min256": Candidate(
        name="lowresidual_gateup_riskcap2_rowrouted_min256",
        preset="lowresidual_gateup_riskcap2",
        note=(
            "more conservative planner row-route gate_up only for dense groups "
            "of at least 256 rows; intended to test the measured crossover"
        ),
        extra_quality_args=(
            "--sr24-row-routed-mlp",
            "--sr24-row-routed-mlp-min-dense-rows-by-leaf",
            "gate_up_proj=256",
        ),
        extra_throughput_args=(
            "--sr24-row-routed-mlp",
            "--sr24-row-routed-mlp-min-dense-rows-by-leaf",
            "gate_up_proj=256",
        ),
    ),
    "mlpall_lowconf_prefix5_tritonoverride": Candidate(
        name="mlpall_lowconf_prefix5_tritonoverride",
        preset="mlpall_lowconf_prefix5_tritonoverride",
        note=(
            "aggressive all-MLP row: all gate_up/down layers, low-confidence "
            "prefix5, bucket32, Triton bucket override"
        ),
    ),
    "mlpall_tilefill_prefix2_bucket32_cublas": Candidate(
        name="mlpall_tilefill_prefix2_bucket32_cublas",
        preset="mlpall_tilefill_prefix2_bucket32_cublas",
        note=(
            "all-MLP low-confidence prefix2 tile-fill: fixed bucket32 dense "
            "overwrite with cuBLAS, no active-only compaction and no Triton "
            "dense GEMM"
        ),
    ),
    "mlpall_lowconf_prefix5_directcslt": Candidate(
        name="mlpall_lowconf_prefix5_directcslt",
        preset="mlpall_lowconf_prefix5_tritonoverride",
        note=(
            "same all-MLP low-confidence controller, but force direct "
            "cuSPARSELt base Linear to test whether base sparse dispatch is "
            "the current bottleneck"
        ),
        extra_quality_args=("--sr24-direct-cslt-linear", ),
        extra_throughput_args=("--sr24-direct-cslt-linear", ),
    ),
    "mlpall_direct_prefix3": Candidate(
        name="mlpall_direct_prefix3",
        preset="mlpall_lowconf_prefix5_tritonoverride",
        note=(
            "all-MLP direct cuSPARSELt with mandatory dense prefix reduced "
            "from 5 to 3 draft rows"
        ),
        extra_quality_args=(
            "--sr24-direct-cslt-linear",
            "--sr24-selective-min-prefix-residual",
            "3",
        ),
        extra_throughput_args=(
            "--sr24-direct-cslt-linear",
            "--sr24-selective-min-prefix-residual",
            "3",
        ),
    ),
    "mlpall_direct_prefix2": Candidate(
        name="mlpall_direct_prefix2",
        preset="mlpall_lowconf_prefix5_tritonoverride",
        note=(
            "all-MLP direct cuSPARSELt with mandatory dense prefix reduced "
            "from 5 to 2 draft rows"
        ),
        extra_quality_args=(
            "--sr24-direct-cslt-linear",
            "--sr24-selective-min-prefix-residual",
            "2",
        ),
        extra_throughput_args=(
            "--sr24-direct-cslt-linear",
            "--sr24-selective-min-prefix-residual",
            "2",
        ),
    ),
    "mlpall_fixedprefix2_directcslt": Candidate(
        name="mlpall_fixedprefix2_directcslt",
        preset="mlpall_fixedprefix2_directcslt",
        note=(
            "score-free all-MLP direct cuSPARSELt route-table candidate: "
            "fixed prefix2 plus bonus rows, no DLM selected-probability "
            "routing"
        ),
    ),
    "lossy_prefix2_rowrouted_mlp": Candidate(
        name="lossy_prefix2_rowrouted_mlp",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "score-free all-MLP systems candidate: fixed prefix2 plus bonus "
            "rows are dense, all other rows are sparse-only, and the planner "
            "pads tiny dense groups to keep the dense branch better occupied"
        ),
    ),
    "lossy_prefix2_rowrouted_mlp_overlap": Candidate(
        name="lossy_prefix2_rowrouted_mlp_overlap",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "same fixed-prefix row-routed MLP candidate, but uses auxiliary "
            "CUDA streams to overlap dense and sparse branches; this disables "
            "the current graph-safe path and is an explicit systems ablation"
        ),
        extra_quality_args=("--sr24-route-overlap-streams", ),
        extra_throughput_args=("--sr24-route-overlap-streams", ),
    ),
    "lossy_prefix2_rowrouted_mlp_minbase256": Candidate(
        name="lossy_prefix2_rowrouted_mlp_minbase256",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "row-count-aware variant: keep the fixed-prefix dense/sparse MLP "
            "route only when the sparse-base side has at least 256 rows; "
            "smaller base groups fall back to dense to avoid underfilled "
            "semi-structured kernels"
        ),
        extra_quality_args=("--sr24-route-min-base-rows", "256"),
        extra_throughput_args=("--sr24-route-min-base-rows", "256"),
    ),
    "lossy_prefix2_rowrouted_mlp_minbase128": Candidate(
        name="lossy_prefix2_rowrouted_mlp_minbase128",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "lower-fill variant: run the sparse-base branch once the base side "
            "has at least 128 rows, so bs32 can exercise the disjoint "
            "dense/sparse MLP route instead of falling back to full dense"
        ),
        extra_quality_args=("--sr24-route-min-base-rows", "128"),
        extra_throughput_args=("--sr24-route-min-base-rows", "128"),
    ),
    "lossy_prefix2_rowrouted_mlp_minbase64": Candidate(
        name="lossy_prefix2_rowrouted_mlp_minbase64",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "aggressive-fill variant: run the sparse-base branch with at "
            "least 64 base rows, allowing bs16 and larger batches to use the "
            "sparse-only path for unimportant verifier rows"
        ),
        extra_quality_args=("--sr24-route-min-base-rows", "64"),
        extra_throughput_args=("--sr24-route-min-base-rows", "64"),
    ),
    "lossy_prefix2_rowrouted_mlp_minbase32": Candidate(
        name="lossy_prefix2_rowrouted_mlp_minbase32",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "very aggressive small-batch variant: allow the sparse-base branch "
            "with 32 base rows so bs8 can exercise sparse-only unimportant "
            "draft rows; mainly a small-kernel pressure test"
        ),
        extra_quality_args=("--sr24-route-min-base-rows", "32"),
        extra_throughput_args=("--sr24-route-min-base-rows", "32"),
    ),
    "lossy_prefix1_rowrouted_mlp_minbase128": Candidate(
        name="lossy_prefix1_rowrouted_mlp_minbase128",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "8pp-budget controller variant: protect only the first draft row "
            "plus the verifier bonus row with dense MLP, leaving more draft "
            "rows sparse-only so bs32 has enough base rows for the sparse "
            "branch at min_base_rows=128"
        ),
        extra_quality_args=(
            "--sr24-selective-min-prefix-residual",
            "1",
            "--sr24-selective-max-residual-draft-rows",
            "1",
            "--sr24-route-min-base-rows",
            "128",
        ),
        extra_throughput_args=(
            "--sr24-selective-min-prefix-residual",
            "1",
            "--sr24-selective-max-residual-draft-rows",
            "1",
            "--sr24-route-min-base-rows",
            "128",
        ),
    ),
    "lossy_prefix1_rowrouted_mlp_minbase128_overlap": Candidate(
        name="lossy_prefix1_rowrouted_mlp_minbase128_overlap",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "systems operating-point probe from the packed verifier-block "
            "microbench: protect only draft row 0 plus the verifier bonus with "
            "dense MLP, keep the remaining draft rows sparse-only, require at "
            "least 128 sparse-base rows, and overlap the dense and sparse MLP "
            "branches on CUDA streams"
        ),
        extra_quality_args=(
            "--sr24-selective-min-prefix-residual",
            "1",
            "--sr24-selective-max-residual-draft-rows",
            "1",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-route-overlap-streams",
        ),
        extra_throughput_args=(
            "--sr24-selective-min-prefix-residual",
            "1",
            "--sr24-selective-max-residual-draft-rows",
            "1",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-route-overlap-streams",
        ),
    ),
    "lossy_prefix0_rowrouted_mlp_minbase128_overlap": Candidate(
        name="lossy_prefix0_rowrouted_mlp_minbase128_overlap",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "aggressive quality-budget probe: protect only the verifier bonus "
            "row with dense MLP and route all draft rows through the 2:4 sparse "
            "MLP when the sparse-base side has at least 128 rows; this is a "
            "quality-risk ablation for the 8pp budget"
        ),
        extra_quality_args=(
            "--sr24-selective-min-prefix-residual",
            "0",
            "--sr24-selective-max-residual-draft-rows",
            "0",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-route-overlap-streams",
        ),
        extra_throughput_args=(
            "--sr24-selective-min-prefix-residual",
            "0",
            "--sr24-selective-max-residual-draft-rows",
            "0",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-route-overlap-streams",
        ),
    ),
    "lossy_prefix2_rowrouted_mlp_minbase256_noverify_sparse": Candidate(
        name="lossy_prefix2_rowrouted_mlp_minbase256_noverify_sparse",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "same min-base route-table candidate, but disables the conservative "
            "no-verify dense MLP fastpath so non-verifier rows use the 2:4 "
            "sparse base instead of being recomputed through dense weights"
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "256",
            "--no-sr24-noverify-dense-mlp-fastpath",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "256",
            "--no-sr24-noverify-dense-mlp-fastpath",
        ),
    ),
    "lossy_prefix2_rowrouted_mlp_front24_dense_noverify": Candidate(
        name="lossy_prefix2_rowrouted_mlp_front24_dense_noverify",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "layer-scoped no-verify guard: verifier rows still use disjoint "
            "fixed-prefix dense/sparse MLP routing, but no-mask/non-draft MLPs "
            "stay dense only in layers 0-23. Layers 24-31 use the 2:4 sparse "
            "base, testing whether tail no-verify work can be removed within "
            "the 8pp GSM8K budget"
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-23;down_proj=0-23",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-23;down_proj=0-23",
        ),
    ),
    "lossy_prefix2_rowrouted_mlp_front24_dense_noverify_compile": Candidate(
        name="lossy_prefix2_rowrouted_mlp_front24_dense_noverify_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "front24/prefix2 8pp-boundary candidate with vLLM default compile "
            "enabled. This is the current systems target: tail-layer "
            "noverify sparse-only creates enough sparse work to approach "
            "1.2x, while compile/graph capture should reduce Python hook and "
            "mixed-branch launch overhead."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-23;down_proj=0-23",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-23;down_proj=0-23",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_rowrouted_mlp_front20_dense_noverify_compile": Candidate(
        name="lossy_prefix2_rowrouted_mlp_front20_dense_noverify_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "front20/prefix2 quality-boundary probe: keep ordinary/no-verify "
            "MLP rows dense only in layers 0-19 and let layers 20-31 use the "
            "2:4 sparse base. This spends more of the allowed 8pp quality "
            "budget than front24 to test whether low-batch speed can approach "
            "1.2x without changing the operator."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-19;down_proj=0-19",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-19;down_proj=0-19",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_rowrouted_mlp_front22_dense_noverify_compile": Candidate(
        name="lossy_prefix2_rowrouted_mlp_front22_dense_noverify_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "front22/prefix2 boundary probe between the passing front24 and "
            "failing front20 policies. Ordinary/no-verify MLP rows stay dense "
            "in layers 0-21 and use 2:4 sparse base in layers 22-31."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-21;down_proj=0-21",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-21;down_proj=0-21",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_front24_serving_policy_compile": Candidate(
        name="lossy_prefix2_front24_serving_policy_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "front24/prefix2 8pp-boundary candidate with a serving-measured "
            "fixed-block policy: bs8/16 bypass the slow live mixed verifier "
            "MLP back to the original dense vLLM MLP, while bs32/64 keep the "
            "legacy fixed-block dense-important / 2:4-sparse mixed path. "
            "This encodes the current end-to-end evidence that important-token "
            "groups are too small at low batch until a real grouped operator "
            "exists."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-23;down_proj=0-23",
            "--sr24-scheduler-policy-path",
            str(FRONT24_SERVING_POLICY),
            "--sr24-scheduler-policy-dense-bypass",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-23;down_proj=0-23",
            "--sr24-scheduler-policy-path",
            str(FRONT24_SERVING_POLICY),
            "--sr24-scheduler-policy-dense-bypass",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_front24_outputbuf_compile": Candidate(
        name="lossy_prefix2_front24_outputbuf_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "front24/prefix2 systems ablation: keep the 8pp-boundary "
            "controller and fixed-prefix route descriptor, and add a reusable "
            "fixed-block output workspace so the MLP route assembly avoids a "
            "fresh output allocation on every layer/step."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-23;down_proj=0-23",
            "--sr24-fixed-block-output-buffer",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-23;down_proj=0-23",
            "--sr24-fixed-block-output-buffer",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_front24_tritonassemble_compile": Candidate(
        name="lossy_prefix2_front24_tritonassemble_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "front24/prefix2 systems ablation: keep the 8pp-boundary "
            "controller and use the Triton fixed-block assembly kernel instead "
            "of PyTorch slice copies for the dense-important and sparse-base "
            "MLP outputs."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-23;down_proj=0-23",
            "--sr24-triton-route-assembly",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-23;down_proj=0-23",
            "--sr24-triton-route-assembly",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_front24_reusebase_compile": Candidate(
        name="lossy_prefix2_front24_reusebase_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "front24/prefix2 fill ablation: run the 2:4 sparse MLP on the "
            "whole verifier block and overwrite the dense-important prefix2 "
            "plus bonus rows. This increases sparse branch row fill at bs64 "
            "and removes the base-row gather, but repeats sparse work for "
            "important rows."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-23;down_proj=0-23",
            "--sr24-row-routed-mlp-reuse-base-output",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-23;down_proj=0-23",
            "--sr24-row-routed-mlp-reuse-base-output",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_front24_densefill128_compile": Candidate(
        name="lossy_prefix2_front24_densefill128_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "front24/prefix2 fill ablation: promote low-priority verifier "
            "rows into the dense-important branch until each MLP call has at "
            "least 128 dense rows. This may improve dense GEMM occupancy and "
            "cannot hurt accuracy because promoted rows become exact dense."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-23;down_proj=0-23",
            "--sr24-row-routed-mlp-fixed-block-dense-fill",
            "--sr24-row-routed-mlp-min-dense-rows",
            "128",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-23;down_proj=0-23",
            "--sr24-row-routed-mlp-fixed-block-dense-fill",
            "--sr24-row-routed-mlp-min-dense-rows",
            "128",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_front24_densefill256_compile": Candidate(
        name="lossy_prefix2_front24_densefill256_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "front24/prefix2 fill ablation: promote low-priority verifier "
            "rows into the dense-important branch until each MLP call has at "
            "least 256 dense rows. This tests whether bs64 is limited by the "
            "small dense branch rather than sparse branch fill."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-23;down_proj=0-23",
            "--sr24-row-routed-mlp-fixed-block-dense-fill",
            "--sr24-row-routed-mlp-min-dense-rows",
            "256",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-23;down_proj=0-23",
            "--sr24-row-routed-mlp-fixed-block-dense-fill",
            "--sr24-row-routed-mlp-min-dense-rows",
            "256",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_front24_overlap_compile": Candidate(
        name="lossy_prefix2_front24_overlap_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "front24/prefix2 branch-concurrency ablation: launch the "
            "dense-important fixed-block MLP branch on a side CUDA stream "
            "while the 2:4 sparse-unimportant branch runs on the current "
            "stream. This tests whether bs64 is blocked by serial branch "
            "execution rather than row selection."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-23;down_proj=0-23",
            "--sr24-route-overlap-streams",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-23;down_proj=0-23",
            "--sr24-route-overlap-streams",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_rowrouted_mlp_front28_dense_noverify_compile": Candidate(
        name="lossy_prefix2_rowrouted_mlp_front28_dense_noverify_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "8pp-budget tail-sparse guard: keep ordinary/no-verify MLP rows "
            "dense in layers 0-27 and let only layers 28-31 use the 2:4 sparse "
            "base. Verifier rows still use the fixed-prefix2 disjoint route, "
            "so low-priority verifier rows are not recomputed densely after "
            "their sparse MLP branch."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-27;down_proj=0-27",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-27;down_proj=0-27",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_rowrouted_mlp_front30_dense_noverify_compile": Candidate(
        name="lossy_prefix2_rowrouted_mlp_front30_dense_noverify_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "conservative tail-sparse guard: only the final two MLP layers "
            "run ordinary/no-verify rows through 2:4 sparse, while verifier "
            "draft rows still use the fixed-prefix2 dense/sparse split. This "
            "tests whether a small quality budget can buy no-verify sparse "
            "work without the front24 quality drop."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-29;down_proj=0-29",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-29;down_proj=0-29",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix1_rowrouted_mlp_front28_dense_noverify_compile": Candidate(
        name="lossy_prefix1_rowrouted_mlp_front28_dense_noverify_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "prefix1 plus tail-sparse guard: protect only draft position 0 "
            "and the verifier bonus row with dense MLP, keep ordinary/no-verify "
            "rows dense in layers 0-27, and make layers 28-31 sparse-only for "
            "ordinary/no-verify rows. The lower dense verifier prefix increases "
            "sparse branch fill under the 8pp quality budget."
        ),
        extra_quality_args=(
            "--sr24-selective-min-prefix-residual",
            "1",
            "--sr24-selective-max-residual-draft-rows",
            "1",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-27;down_proj=0-27",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-selective-min-prefix-residual",
            "1",
            "--sr24-selective-max-residual-draft-rows",
            "1",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-27;down_proj=0-27",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix1_front28_overlap_compile": Candidate(
        name="lossy_prefix1_front28_overlap_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "front28/prefix1 branch-overlap probe: use the same 8pp-budget "
            "tail-sparse and dense-important policy as "
            "lossy_prefix1_rowrouted_mlp_front28_dense_noverify_compile, but "
            "launch the dense-important MLP branch and the 2:4 sparse "
            "unimportant MLP branch on separate CUDA streams. This directly "
            "tests whether the disjoint token data format can recover speed "
            "through branch parallelism instead of serial PyTorch launches."
        ),
        extra_quality_args=(
            "--sr24-selective-min-prefix-residual",
            "1",
            "--sr24-selective-max-residual-draft-rows",
            "1",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-27;down_proj=0-27",
            "--sr24-route-overlap-streams",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-selective-min-prefix-residual",
            "1",
            "--sr24-selective-max-residual-draft-rows",
            "1",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-27;down_proj=0-27",
            "--sr24-route-overlap-streams",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_front28_reusebase_compile": Candidate(
        name="lossy_prefix2_front28_reusebase_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "front28/prefix2 packed-operator proxy: keep the same 8pp-budget "
            "tail noverify-sparse policy, but run the 2:4 sparse MLP on the "
            "whole verifier block and overwrite the dense-important prefix2 "
            "plus bonus rows. This increases sparse branch row fill and removes "
            "the separate base-row gather, at the cost of redundant sparse "
            "work for important rows."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-27;down_proj=0-27",
            "--sr24-fixed-prefix-route-descriptor-only",
            "--sr24-row-routed-mlp-reuse-base-output",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-27;down_proj=0-27",
            "--sr24-fixed-prefix-route-descriptor-only",
            "--sr24-row-routed-mlp-reuse-base-output",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix1_front28_reusebase_compile": Candidate(
        name="lossy_prefix1_front28_reusebase_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "front28/prefix1 packed-operator proxy: protect only draft row 0 "
            "and the verifier bonus with dense MLP, run sparse MLP on the full "
            "verifier block, and overwrite the dense-important rows. This is "
            "the higher-fill variant of the quality-passing front28/prefix1 "
            "split candidate."
        ),
        extra_quality_args=(
            "--sr24-selective-min-prefix-residual",
            "1",
            "--sr24-selective-max-residual-draft-rows",
            "1",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-27;down_proj=0-27",
            "--sr24-fixed-prefix-route-descriptor-only",
            "--sr24-row-routed-mlp-reuse-base-output",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-selective-min-prefix-residual",
            "1",
            "--sr24-selective-max-residual-draft-rows",
            "1",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-27;down_proj=0-27",
            "--sr24-fixed-prefix-route-descriptor-only",
            "--sr24-row-routed-mlp-reuse-base-output",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix1_rowrouted_mlp_front30_dense_noverify_compile": Candidate(
        name="lossy_prefix1_rowrouted_mlp_front30_dense_noverify_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "prefix1 plus conservative tail-sparse guard: only layers 30-31 "
            "use sparse-only ordinary/no-verify MLP rows, and verifier draft "
            "positions after row 0 are sparse-only. This is the least "
            "aggressive candidate that still increases verifier sparse fill."
        ),
        extra_quality_args=(
            "--sr24-selective-min-prefix-residual",
            "1",
            "--sr24-selective-max-residual-draft-rows",
            "1",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-29;down_proj=0-29",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-selective-min-prefix-residual",
            "1",
            "--sr24-selective-max-residual-draft-rows",
            "1",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-29;down_proj=0-29",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix0_rowrouted_mlp_front30_dense_noverify_compile": Candidate(
        name="lossy_prefix0_rowrouted_mlp_front30_dense_noverify_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "quality-boundary tail-sparse guard: only the verifier bonus row "
            "is dense, every verifier draft row is sparse-only, and ordinary "
            "no-verify rows become sparse-only only in layers 30-31. It tests "
            "whether the 8pp budget can be spent on verifier sparse fill "
            "rather than broad no-verify sparsity."
        ),
        extra_quality_args=(
            "--sr24-selective-min-prefix-residual",
            "0",
            "--sr24-selective-max-residual-draft-rows",
            "0",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-29;down_proj=0-29",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-selective-min-prefix-residual",
            "0",
            "--sr24-selective-max-residual-draft-rows",
            "0",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-29;down_proj=0-29",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_rowrouted_mlp_front16_dense_noverify": Candidate(
        name="lossy_prefix2_rowrouted_mlp_front16_dense_noverify",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "more aggressive layer-scoped no-verify guard: keep dense no-mask "
            "MLPs only in layers 0-15 and let layers 16-31 use the 2:4 sparse "
            "base. This tests whether the earlier-layer sensitivity explains "
            "the full noverify-sparse accuracy drop"
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-15;down_proj=0-15",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-15;down_proj=0-15",
        ),
    ),
    "lossy_prefix2_rowrouted_mlp_front8_dense_noverify_compile": Candidate(
        name="lossy_prefix2_rowrouted_mlp_front8_dense_noverify_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "aggressive 8pp-budget candidate: keep dense no-mask MLPs only in "
            "layers 0-7 and let layers 8-31 use the 2:4 sparse base. The "
            "throughput path enables vLLM's default compile path because the "
            "front16 ablation showed that compile/graph integration removes "
            "most of the SR24 hook overhead."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-7;down_proj=0-7",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-7;down_proj=0-7",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_rowrouted_mlp_front8_dense_noverify_compile_minbase0": Candidate(
        name="lossy_prefix2_rowrouted_mlp_front8_dense_noverify_compile_minbase0",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "front8 compile candidate with route_min_base_rows=0. This allows "
            "small verifier sparse-base branches to run instead of falling "
            "back to full dense when the number of unimportant verifier rows "
            "is low, testing the tile-fill/underfilled-important-token concern."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "0",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-7;down_proj=0-7",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "0",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-7;down_proj=0-7",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_rowrouted_mlp_noverify_sparse_compile": Candidate(
        name="lossy_prefix2_rowrouted_mlp_noverify_sparse_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "boundary test for the 8pp budget: disable dense noverify scope "
            "entirely so no-mask/noverify MLP rows use only the 2:4 sparse "
            "base, while verifier prefix/bonus rows keep the normal selective "
            "dense protection. This replaces older noverify-sparse probes that "
            "could still fall back to full dense through the Linear path."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_rowrouted_mlp_operator_guard_compile": Candidate(
        name="lossy_prefix2_rowrouted_mlp_operator_guard_compile",
        preset="lossy_prefix2_rowrouted_mlp_operator_guard",
        note=(
            "planner-aligned guard for the fixed-prefix2 row-routed MLP path: "
            "disable dense-fill promotion and fall back to dense until the "
            "sparse-base branch has roughly the K8/bs64 row fill required by "
            "the packed-MLP microbench. It also enables the fixed-prefix route "
            "descriptor-only plan so descriptor-safe steps do not construct "
            "residual/base row-index tensors. This tests whether avoiding "
            "known underfilled mixed branches and CPU-side route tensors "
            "improves low-batch serving stability."
        ),
        extra_quality_args=(
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-fixed-prefix-route-descriptor-only",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-fixed-prefix-route-descriptor-only",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_rowrouted_mlp_operator_guard_outputbuf_compile": Candidate(
        name="lossy_prefix2_rowrouted_mlp_operator_guard_outputbuf_compile",
        preset="lossy_prefix2_rowrouted_mlp_operator_guard",
        note=(
            "same planner-aligned fixed-prefix2 row-routed MLP guard as "
            "lossy_prefix2_rowrouted_mlp_operator_guard_compile, plus a "
            "per-module fixed-block output workspace so the final dense/base "
            "assembly does not allocate a fresh output tensor each MLP call. "
            "This is a data-format/allocator-overhead ablation; it should be "
            "kept only if quality stays identical and serving throughput is "
            "measurably better."
        ),
        extra_quality_args=(
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-fixed-prefix-route-descriptor-only",
            "--sr24-fixed-block-output-buffer",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-fixed-prefix-route-descriptor-only",
            "--sr24-fixed-block-output-buffer",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_front28_policy_outputbuf_compile": Candidate(
        name="lossy_prefix2_front28_policy_outputbuf_compile",
        preset="lossy_prefix2_rowrouted_mlp_operator_guard",
        note=(
            "paired baseline for the current front28 packed-policy route: "
            "ordinary/no-verify MLP rows stay dense through layers 0-27 and "
            "use 2:4 sparse in layers 28-31, verifier rows use disjoint "
            "dense-important / sparse-unimportant fixed blocks, near-full "
            "bs64 steps are capacity-padded, and the fixed-block output "
            "workspace avoids per-call output allocation. Unlike the Triton "
            "assembly variant, final assembly remains the existing PyTorch "
            "slice-copy sequence."
        ),
        extra_quality_args=(
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-27;down_proj=0-27",
            "--sr24-scheduler-policy-path",
            str(PREFIX2_FRONT28_PACKED_POLICY),
            "--sr24-scheduler-policy-near-full-tolerance",
            "2",
            "--sr24-fixed-block-capacity-padding",
            "--sr24-fixed-block-output-buffer",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-27;down_proj=0-27",
            "--sr24-scheduler-policy-path",
            str(PREFIX2_FRONT28_PACKED_POLICY),
            "--sr24-scheduler-policy-near-full-tolerance",
            "2",
            "--sr24-fixed-block-capacity-padding",
            "--sr24-fixed-block-output-buffer",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_front28_policy_outputbuf_tritonassemble_compile": Candidate(
        name="lossy_prefix2_front28_policy_outputbuf_tritonassemble_compile",
        preset="lossy_prefix2_rowrouted_mlp_operator_guard",
        note=(
            "current 8pp-budget front28 systems target with the refreshed "
            "prefix2/K8 packed policy: ordinary/no-verify MLP rows stay dense "
            "through layers 0-27 and use 2:4 sparse in layers 28-31, verifier "
            "rows use disjoint dense-important / sparse-unimportant fixed "
            "blocks, near-full bs64 steps are capacity-padded, and Triton "
            "fixed-block assembly writes into the reusable output workspace. "
            "This measures whether allocator-free fused assembly helps the "
            "best current live route before implementing a grouped queue."
        ),
        extra_quality_args=(
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-27;down_proj=0-27",
            "--sr24-scheduler-policy-path",
            str(PREFIX2_FRONT28_PACKED_POLICY),
            "--sr24-scheduler-policy-near-full-tolerance",
            "2",
            "--sr24-fixed-block-capacity-padding",
            "--sr24-fixed-block-output-buffer",
            "--sr24-triton-route-assembly",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "gate_up_proj=0-27;down_proj=0-27",
            "--sr24-scheduler-policy-path",
            str(PREFIX2_FRONT28_PACKED_POLICY),
            "--sr24-scheduler-policy-near-full-tolerance",
            "2",
            "--sr24-fixed-block-capacity-padding",
            "--sr24-fixed-block-output-buffer",
            "--sr24-triton-route-assembly",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_rowrouted_mlp_verifier_only_outputbuf_compile": Candidate(
        name="lossy_prefix2_rowrouted_mlp_verifier_only_outputbuf_compile",
        preset="lossy_prefix2_rowrouted_mlp_operator_guard",
        note=(
            "quality-gated systems candidate: keep ordinary/non-verify rows "
            "on the dense TLM path, route only verifier draft rows through the "
            "fixed-prefix2 dense/sparse MLP split, and use the per-module "
            "fixed-block output workspace. This avoids the measured quality "
            "collapse from making all no-mask rows sparse while still testing "
            "the PPoPP-style data-format change where important verifier "
            "tokens do dense work and low-priority verifier tokens do 2:4 "
            "sparse work without a later dense correction."
        ),
        extra_quality_args=(
            "--sr24-fixed-prefix-route-descriptor-only",
            "--sr24-fixed-block-output-buffer",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-fixed-prefix-route-descriptor-only",
            "--sr24-fixed-block-output-buffer",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_verifier_only_outputbuf_smallm_alg1_t256_compile": Candidate(
        name="lossy_prefix2_verifier_only_outputbuf_smallm_alg1_t256_compile",
        preset="lossy_prefix2_rowrouted_mlp_operator_guard",
        note=(
            "same verifier-only fixed-prefix2/output-buffer policy, but route "
            "direct cuSPARSELt sparse Linear calls with rows <=256 to alg_id=1. "
            "The fixed-prefix2 base-row sweep showed alg1 improves rows "
            "48/96/192 while rows 384 should stay on the default alg0, so this "
            "is the shape-aware single-block small-M operator candidate."
        ),
        extra_quality_args=(
            "--sr24-fixed-prefix-route-descriptor-only",
            "--sr24-fixed-block-output-buffer",
            "--sr24-cslt-small-m-alg-id-enable",
            "--sr24-cslt-small-m-threshold",
            "256",
            "--sr24-cslt-small-m-alg-id",
            "1",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-fixed-prefix-route-descriptor-only",
            "--sr24-fixed-block-output-buffer",
            "--sr24-cslt-small-m-alg-id-enable",
            "--sr24-cslt-small-m-threshold",
            "256",
            "--sr24-cslt-small-m-alg-id",
            "1",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_verifier_only_outputbuf_leafalg_pair_compile": Candidate(
        name="lossy_prefix2_verifier_only_outputbuf_leafalg_pair_compile",
        preset="lossy_prefix2_rowrouted_mlp_operator_guard",
        note=(
            "verifier-only fixed-prefix2/output-buffer policy with projection-aware "
            "cuSPARSELt alg selection. The alg-pair microbench shows rows48 wants "
            "gate_up/down alg1, rows96/192 want gate_up alg0 plus down alg1, and "
            "rows384 should stay alg0. This candidate encodes that without doing "
            "sparse work again for dense-important or non-verifier tokens."
        ),
        extra_quality_args=(
            "--sr24-fixed-prefix-route-descriptor-only",
            "--sr24-fixed-block-output-buffer",
            "--sr24-cslt-small-m-alg-id-enable",
            "--sr24-cslt-small-m-threshold",
            "64",
            "--sr24-cslt-small-m-alg-id",
            "1",
            "--sr24-cslt-small-m-threshold-by-leaf",
            "down_proj=256",
            "--sr24-cslt-small-m-alg-id-by-leaf",
            "down_proj=1",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-fixed-prefix-route-descriptor-only",
            "--sr24-fixed-block-output-buffer",
            "--sr24-cslt-small-m-alg-id-enable",
            "--sr24-cslt-small-m-threshold",
            "64",
            "--sr24-cslt-small-m-alg-id",
            "1",
            "--sr24-cslt-small-m-threshold-by-leaf",
            "down_proj=256",
            "--sr24-cslt-small-m-alg-id-by-leaf",
            "down_proj=1",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix1_verifier_only_outputbuf_leafalg_pair_compile": Candidate(
        name="lossy_prefix1_verifier_only_outputbuf_leafalg_pair_compile",
        preset="lossy_prefix2_rowrouted_mlp_operator_guard",
        note=(
            "verifier-only fixed-prefix1/output-buffer policy with the same "
            "projection-aware cuSPARSELt alg selection as the prefix2 row. "
            "Only draft position 0 plus the verifier bonus row are dense; "
            "later verifier draft rows are 2:4 sparse-only, while ordinary "
            "no-mask decode rows remain dense. This isolates the quality/speed "
            "effect of reducing important verifier tokens without making "
            "non-verifier decode sparse."
        ),
        extra_quality_args=(
            "--sr24-selective-min-prefix-residual",
            "1",
            "--sr24-selective-max-residual-draft-rows",
            "1",
            "--sr24-fixed-prefix-route-descriptor-only",
            "--sr24-fixed-block-output-buffer",
            "--sr24-cslt-small-m-alg-id-enable",
            "--sr24-cslt-small-m-threshold",
            "64",
            "--sr24-cslt-small-m-alg-id",
            "1",
            "--sr24-cslt-small-m-threshold-by-leaf",
            "down_proj=256",
            "--sr24-cslt-small-m-alg-id-by-leaf",
            "down_proj=1",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-selective-min-prefix-residual",
            "1",
            "--sr24-selective-max-residual-draft-rows",
            "1",
            "--sr24-fixed-prefix-route-descriptor-only",
            "--sr24-fixed-block-output-buffer",
            "--sr24-cslt-small-m-alg-id-enable",
            "--sr24-cslt-small-m-threshold",
            "64",
            "--sr24-cslt-small-m-alg-id",
            "1",
            "--sr24-cslt-small-m-threshold-by-leaf",
            "down_proj=256",
            "--sr24-cslt-small-m-alg-id-by-leaf",
            "down_proj=1",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix0_verifier_only_outputbuf_leafalg_pair_compile": Candidate(
        name="lossy_prefix0_verifier_only_outputbuf_leafalg_pair_compile",
        preset="lossy_prefix2_rowrouted_mlp_operator_guard",
        note=(
            "verifier-only fixed-prefix0/output-buffer policy with projection-aware "
            "cuSPARSELt alg selection. Only the verifier bonus row is dense; "
            "all verifier draft rows are 2:4 sparse-only, while ordinary "
            "no-mask decode rows remain dense. This is the aggressive lower "
            "quality-boundary check under the 8pp accuracy budget."
        ),
        extra_quality_args=(
            "--sr24-selective-min-prefix-residual",
            "0",
            "--sr24-selective-max-residual-draft-rows",
            "0",
            "--sr24-fixed-prefix-route-descriptor-only",
            "--sr24-fixed-block-output-buffer",
            "--sr24-cslt-small-m-alg-id-enable",
            "--sr24-cslt-small-m-threshold",
            "64",
            "--sr24-cslt-small-m-alg-id",
            "1",
            "--sr24-cslt-small-m-threshold-by-leaf",
            "down_proj=256",
            "--sr24-cslt-small-m-alg-id-by-leaf",
            "down_proj=1",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-selective-min-prefix-residual",
            "0",
            "--sr24-selective-max-residual-draft-rows",
            "0",
            "--sr24-fixed-prefix-route-descriptor-only",
            "--sr24-fixed-block-output-buffer",
            "--sr24-cslt-small-m-alg-id-enable",
            "--sr24-cslt-small-m-threshold",
            "64",
            "--sr24-cslt-small-m-alg-id",
            "1",
            "--sr24-cslt-small-m-threshold-by-leaf",
            "down_proj=256",
            "--sr24-cslt-small-m-alg-id-by-leaf",
            "down_proj=1",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_rowrouted_mlp_descriptor_reusebase_compile": Candidate(
        name="lossy_prefix2_rowrouted_mlp_descriptor_reusebase_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "descriptor-only fixed-prefix reuse-base ablation: run the 2:4 "
            "sparse base MLP on the full verifier block to increase sparse "
            "branch row fill, then overwrite only the dense-important "
            "prefix/bonus rows. This spends extra sparse work on important "
            "rows and is meant to test whether larger sparse-M occupancy can "
            "beat the disjoint split branch."
        ),
        extra_quality_args=(
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-fixed-prefix-route-descriptor-only",
            "--sr24-row-routed-mlp-reuse-base-output",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-fixed-prefix-route-descriptor-only",
            "--sr24-row-routed-mlp-reuse-base-output",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_rowrouted_mlp_descriptor_inputbuf_compile": Candidate(
        name="lossy_prefix2_rowrouted_mlp_descriptor_inputbuf_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "descriptor-only fixed-prefix input-buffer ablation: keep the "
            "disjoint dense-important and sparse-unimportant MLP branches, but "
            "assemble branch inputs into reusable graph-stable temporary "
            "buffers instead of allocating torch.cat/reshape temporaries."
        ),
        extra_quality_args=(
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-fixed-prefix-route-descriptor-only",
            "--sr24-fixed-block-input-buffer",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-fixed-prefix-route-descriptor-only",
            "--sr24-fixed-block-input-buffer",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_rowrouted_mlp_noverify_sparse_compile_minbase256": Candidate(
        name="lossy_prefix2_rowrouted_mlp_noverify_sparse_compile_minbase256",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "same noverify-sparse boundary as "
            "lossy_prefix2_rowrouted_mlp_noverify_sparse_compile, but requires "
            "at least 256 sparse-base rows before splitting the gate_up MLP. "
            "Below that fill level it falls back to the full sparse-base plus "
            "dense-overwrite path, which should preserve CUDA Graph/full-row "
            "cuSPARSELt efficiency instead of launching underfilled sparse "
            "GEMMs for the unimportant rows."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "256",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "256",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_noverify_sparse_fillaware_bs64_compile": Candidate(
        name="lossy_prefix2_noverify_sparse_fillaware_bs64_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "fill-aware live operating point from the useful-row coalescing "
            "microbench: keep the quality-safe noverify-sparse fixed-prefix2 "
            "semantics, but only enable SR24 serving when client concurrency "
            "is at least 64. Smaller batch sizes use the dense EAGLE3 server "
            "because the current split sparse/dense MLP branch is underfilled "
            "there. This is a throughput guard, not a claimed low-batch 2:4 "
            "speedup."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "384",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "384",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-min-enabled-batch-size",
            "64",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_noverify_sparse_gateup384_compile": Candidate(
        name="lossy_prefix2_noverify_sparse_gateup384_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "planner-fallback probe: keep the noverify-sparse fixed-prefix2 "
            "quality boundary, but require at least 384 sparse/base rows for "
            "the gate_up row-routed MLP branch. Smaller gate/up sparse "
            "branches fall back to full dense, while other leaves can keep the "
            "global route_min_base_rows setting. This is the first live use of "
            "SPECLINK_SR24_ROUTE_MIN_BASE_ROWS_BY_LEAF."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "0",
            "--sr24-route-min-base-rows-by-leaf",
            "gate_up_proj=384",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "0",
            "--sr24-route-min-base-rows-by-leaf",
            "gate_up_proj=384",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_gateup384_tritonassemble_compile": Candidate(
        name="lossy_prefix2_gateup384_tritonassemble_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "operator-overhead probe combining the gate_up per-leaf min-base "
            "planner fallback with Triton fixed-block final assembly. The "
            "routing/quality policy is unchanged from gateup384: first two "
            "draft rows plus the verifier bonus row are dense, later draft "
            "rows are 2:4 sparse-only, and underfilled gate/up sparse branches "
            "fall back to dense."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "0",
            "--sr24-route-min-base-rows-by-leaf",
            "gate_up_proj=384",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-triton-route-assembly",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "0",
            "--sr24-route-min-base-rows-by-leaf",
            "gate_up_proj=384",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-triton-route-assembly",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_rowrouted_mlp_noverify_sparse_compile_smallm_alg1": Candidate(
        name="lossy_prefix2_rowrouted_mlp_noverify_sparse_compile_smallm_alg1",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "same noverify-sparse compile boundary, but use cuSPARSELt alg_id=1 "
            "for direct sparse Linear calls with rows <=96. The operator "
            "microbench showed this helps bs8-shaped full sparse MLPs while "
            "keeping larger rows on the default alg_id=0."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-cslt-small-m-alg-id-enable",
            "--sr24-cslt-small-m-threshold",
            "96",
            "--sr24-cslt-small-m-alg-id",
            "1",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-cslt-small-m-alg-id-enable",
            "--sr24-cslt-small-m-threshold",
            "96",
            "--sr24-cslt-small-m-alg-id",
            "1",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_noverify_sparse_smallm_alg1_t160_compile": Candidate(
        name="lossy_prefix2_noverify_sparse_smallm_alg1_t160_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "same noverify-sparse compile boundary, but extend the direct "
            "cuSPARSELt alg_id=1 small-M override to rows <=160. The K8 "
            "operator sweep showed alg1 is also better around rows=144, so "
            "this tests whether bs16-shaped sparse branches can improve "
            "without changing the quality policy."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-cslt-small-m-alg-id-enable",
            "--sr24-cslt-small-m-threshold",
            "160",
            "--sr24-cslt-small-m-alg-id",
            "1",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-cslt-small-m-alg-id-enable",
            "--sr24-cslt-small-m-threshold",
            "160",
            "--sr24-cslt-small-m-alg-id",
            "1",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_rowrouted_mlp_noverify_sparse_compile_minbase0": Candidate(
        name="lossy_prefix2_rowrouted_mlp_noverify_sparse_compile_minbase0",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "same noverify-sparse boundary as "
            "lossy_prefix2_rowrouted_mlp_noverify_sparse_compile, but sets "
            "route_min_base_rows=0. This tests whether low-batch runs should "
            "still execute the sparse-unimportant branch instead of falling "
            "back when the 2:4 row count is small."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "0",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "0",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_noverify_sparse_densefill64_compile": Candidate(
        name="lossy_prefix2_noverify_sparse_densefill64_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "PPoPP-style fixed-block route-table probe: important prefix2 plus "
            "bonus rows go dense, unimportant rows go 2:4 sparse-only, but the "
            "fixed-block operator may promote adjacent base rows to dense until "
            "the dense branch has at least 64 rows. Promoted rows are a "
            "tile-fill optimization, not required for correctness."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "0",
            "--sr24-row-routed-mlp-min-dense-rows",
            "64",
            "--sr24-row-routed-mlp-fixed-block-dense-fill",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "0",
            "--sr24-row-routed-mlp-min-dense-rows",
            "64",
            "--sr24-row-routed-mlp-fixed-block-dense-fill",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_noverify_sparse_densefill128_compile": Candidate(
        name="lossy_prefix2_noverify_sparse_densefill128_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "same fixed-block noverify-sparse data layout, but promote enough "
            "adjacent base rows to give the dense-important branch at least "
            "128 rows. This tests whether filling the dense Tensor Core branch "
            "helps more than preserving every unimportant verifier row as "
            "sparse-only."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "0",
            "--sr24-row-routed-mlp-min-dense-rows",
            "128",
            "--sr24-row-routed-mlp-fixed-block-dense-fill",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "0",
            "--sr24-row-routed-mlp-min-dense-rows",
            "128",
            "--sr24-row-routed-mlp-fixed-block-dense-fill",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_rowrouted_mlp_noverify_sparse_overlap_compile": Candidate(
        name="lossy_prefix2_rowrouted_mlp_noverify_sparse_overlap_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "same noverify-sparse boundary as "
            "lossy_prefix2_rowrouted_mlp_noverify_sparse_compile, but enables "
            "the fixed-block dense/sparse overlap stream path. This tests "
            "whether important dense rows and unimportant 2:4 rows can run "
            "concurrently once rows are disjoint and noverify rows are not "
            "recomputed dense."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-route-overlap-streams",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-route-overlap-streams",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_rowrouted_mlp_noverify_sparse_tritonassemble_compile": Candidate(
        name="lossy_prefix2_rowrouted_mlp_noverify_sparse_tritonassemble_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "noverify-sparse compile boundary with a fused Triton fixed-block "
            "final assembly. This replaces the three PyTorch copy_ kernels "
            "used to restore [request, draft-position] row order after the "
            "dense-important and sparse-unimportant MLP branches."
        ),
        extra_quality_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-triton-route-assembly",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-triton-route-assembly",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_gateup_only_noverify_sparse_compile": Candidate(
        name="lossy_prefix2_gateup_only_noverify_sparse_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "gate_up-only variant of the noverify-sparse boundary: route "
            "important gate_up rows through dense, leave unimportant/noverify "
            "gate_up rows sparse-only, and keep down_proj on the original "
            "dense vLLM path. This tests whether removing the small-M sparse "
            "down branch is a better 8pp-budget operating point."
        ),
        extra_quality_args=(
            "--sr24-target-leafs",
            "gate_up_proj",
            "--sr24-residual-target-leafs",
            "gate_up_proj",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-target-leafs",
            "gate_up_proj",
            "--sr24-residual-target-leafs",
            "gate_up_proj",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_gateup_only_noverify_sparse_minbase256_compile": Candidate(
        name="lossy_prefix2_gateup_only_noverify_sparse_minbase256_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "gate_up-only noverify-sparse candidate with a 256-row sparse-base "
            "fill threshold. When the unimportant/base side is smaller than "
            "256 rows, it avoids row-splitting and falls back to full sparse "
            "base plus dense overwrite, targeting the underfilled gate_up "
            "sparse GEMM shown by the bs64 breakdown."
        ),
        extra_quality_args=(
            "--sr24-target-leafs",
            "gate_up_proj",
            "--sr24-residual-target-leafs",
            "gate_up_proj",
            "--sr24-route-min-base-rows",
            "256",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-target-leafs",
            "gate_up_proj",
            "--sr24-residual-target-leafs",
            "gate_up_proj",
            "--sr24-route-min-base-rows",
            "256",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix1_gateup_only_noverify_sparse_minbase128_compile": Candidate(
        name="lossy_prefix1_gateup_only_noverify_sparse_minbase128_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "8pp-budget gate_up-only candidate: protect only draft position 0 "
            "plus the verifier bonus row with dense gate_up, leave the other "
            "draft/noverify rows sparse-only, and require at least 128 base "
            "rows before row-splitting. This tests whether fewer important "
            "tokens gives enough sparse fill without violating GSM8K quality."
        ),
        extra_quality_args=(
            "--sr24-target-leafs",
            "gate_up_proj",
            "--sr24-residual-target-leafs",
            "gate_up_proj",
            "--sr24-selective-min-prefix-residual",
            "1",
            "--sr24-selective-max-residual-draft-rows",
            "1",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-target-leafs",
            "gate_up_proj",
            "--sr24-residual-target-leafs",
            "gate_up_proj",
            "--sr24-selective-min-prefix-residual",
            "1",
            "--sr24-selective-max-residual-draft-rows",
            "1",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix1_mlp_noverify_sparse_minbase128_compile": Candidate(
        name="lossy_prefix1_mlp_noverify_sparse_minbase128_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "8pp-budget all-MLP candidate: protect only draft position 0 plus "
            "the verifier bonus row with dense MLP, route all other draft and "
            "noverify rows through 2:4, and require at least 128 base rows "
            "before row-splitting. This is the broader speed candidate if the "
            "quality gate tolerates prefix1 protection."
        ),
        extra_quality_args=(
            "--sr24-selective-min-prefix-residual",
            "1",
            "--sr24-selective-max-residual-draft-rows",
            "1",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-selective-min-prefix-residual",
            "1",
            "--sr24-selective-max-residual-draft-rows",
            "1",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix0_mlp_noverify_sparse_minbase128_compile": Candidate(
        name="lossy_prefix0_mlp_noverify_sparse_minbase128_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "quality-budget boundary for the disjoint MLP data format: protect "
            "only the verifier bonus row with dense MLP, route every draft row "
            "through 2:4 sparse-only, and require at least 128 base rows before "
            "using the split sparse/dense branch. This intentionally allows "
            "accuracy loss up to the configured 8pp gate."
        ),
        extra_quality_args=(
            "--sr24-selective-min-prefix-residual",
            "0",
            "--sr24-selective-max-residual-draft-rows",
            "0",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-selective-min-prefix-residual",
            "0",
            "--sr24-selective-max-residual-draft-rows",
            "0",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_down_only_noverify_sparse_compile": Candidate(
        name="lossy_prefix2_down_only_noverify_sparse_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "down-proj-only systems probe: keep gate/up on the normal dense "
            "vLLM path, route important down_proj rows through dense, and route "
            "unimportant/noverify down_proj rows through 2:4 sparse-only. This "
            "tests the breakdown hypothesis that down_proj mixed routing is a "
            "better operator shape than gate/up row splitting."
        ),
        extra_quality_args=(
            "--sr24-target-leafs",
            "down_proj",
            "--sr24-residual-target-leafs",
            "down_proj",
            "--sr24-row-routed-down-linear",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-target-leafs",
            "down_proj",
            "--sr24-residual-target-leafs",
            "down_proj",
            "--sr24-row-routed-down-linear",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_down_front24_compile": Candidate(
        name="lossy_prefix2_down_front24_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "down-proj-only tail-layer systems candidate: keep gate/up dense, "
            "keep ordinary/no-verify down_proj rows dense in layers 0-23, and "
            "use 2:4 sparse-only for no-verify down_proj rows in layers 24-31. "
            "Verifier down_proj rows still use fixed-prefix2 dense-important "
            "routing. This targets the component result where down_proj has "
            "the best 2:4 base speedup without splitting gate_up."
        ),
        extra_quality_args=(
            "--sr24-target-leafs",
            "down_proj",
            "--sr24-residual-target-leafs",
            "down_proj",
            "--sr24-row-routed-down-linear",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "down_proj=0-23",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-target-leafs",
            "down_proj",
            "--sr24-residual-target-leafs",
            "down_proj",
            "--sr24-row-routed-down-linear",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "down_proj=0-23",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_down_front16_compile": Candidate(
        name="lossy_prefix2_down_front16_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "more aggressive down-proj-only tail-layer candidate: keep "
            "ordinary/no-verify down_proj rows dense only in layers 0-15 and "
            "use sparse-only for layers 16-31. This tests whether down_proj "
            "can spend more of the 8pp accuracy budget than all-MLP front24."
        ),
        extra_quality_args=(
            "--sr24-target-leafs",
            "down_proj",
            "--sr24-residual-target-leafs",
            "down_proj",
            "--sr24-row-routed-down-linear",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "down_proj=0-15",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-target-leafs",
            "down_proj",
            "--sr24-residual-target-leafs",
            "down_proj",
            "--sr24-row-routed-down-linear",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "down_proj=0-15",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_down_front12_compile": Candidate(
        name="lossy_prefix2_down_front12_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "down-proj-only quality-boundary probe: keep gate/up dense, keep "
            "ordinary/no-verify down_proj rows dense only in layers 0-11, and "
            "use 2:4 sparse-only for layers 12-31. This spends more quality "
            "budget than down-front16 while avoiding the expensive gate_up "
            "split."
        ),
        extra_quality_args=(
            "--sr24-target-leafs",
            "down_proj",
            "--sr24-residual-target-leafs",
            "down_proj",
            "--sr24-row-routed-down-linear",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "down_proj=0-11",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-target-leafs",
            "down_proj",
            "--sr24-residual-target-leafs",
            "down_proj",
            "--sr24-row-routed-down-linear",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "down_proj=0-11",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_down_front14_compile": Candidate(
        name="lossy_prefix2_down_front14_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "down-proj-only boundary probe between the passing down-front16 "
            "and failing down-front12 policies. Gate/up stays dense; "
            "ordinary/no-verify down_proj rows stay dense in layers 0-13 and "
            "use 2:4 sparse base in layers 14-31."
        ),
        extra_quality_args=(
            "--sr24-target-leafs",
            "down_proj",
            "--sr24-residual-target-leafs",
            "down_proj",
            "--sr24-row-routed-down-linear",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "down_proj=0-13",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-target-leafs",
            "down_proj",
            "--sr24-residual-target-leafs",
            "down_proj",
            "--sr24-row-routed-down-linear",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "down_proj=0-13",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_down_front14_outputbuf_compile": Candidate(
        name="lossy_prefix2_down_front14_outputbuf_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "same 8pp-boundary down-proj-only policy as "
            "lossy_prefix2_down_front14_compile, plus the fixed-block output "
            "workspace. This keeps dense-important and 2:4 sparse-base rows "
            "disjoint while avoiding a fresh MLP output allocation in the "
            "fixed-block row-routed path."
        ),
        extra_quality_args=(
            "--sr24-target-leafs",
            "down_proj",
            "--sr24-residual-target-leafs",
            "down_proj",
            "--sr24-row-routed-down-linear",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "down_proj=0-13",
            "--sr24-fixed-block-output-buffer",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-target-leafs",
            "down_proj",
            "--sr24-residual-target-leafs",
            "down_proj",
            "--sr24-row-routed-down-linear",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "down_proj=0-13",
            "--sr24-fixed-block-output-buffer",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix2_down_front14_outputbuf_overlap_compile": Candidate(
        name="lossy_prefix2_down_front14_outputbuf_overlap_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "concurrency ablation for the 8pp-boundary down-proj-only policy: "
            "use the fixed-block output workspace and launch dense-important "
            "and 2:4 sparse-base rows on separate CUDA streams. This is a "
            "systems probe for overlap potential, not the default path unless "
            "the end-to-end result beats the non-overlap output-buffer row."
        ),
        extra_quality_args=(
            "--sr24-target-leafs",
            "down_proj",
            "--sr24-residual-target-leafs",
            "down_proj",
            "--sr24-row-routed-down-linear",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "down_proj=0-13",
            "--sr24-fixed-block-output-buffer",
            "--sr24-route-overlap-streams",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-target-leafs",
            "down_proj",
            "--sr24-residual-target-leafs",
            "down_proj",
            "--sr24-row-routed-down-linear",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "down_proj=0-13",
            "--sr24-fixed-block-output-buffer",
            "--sr24-route-overlap-streams",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix1_down_only_noverify_sparse_compile": Candidate(
        name="lossy_prefix1_down_only_noverify_sparse_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "down-proj-only variant with fewer important rows: protect draft "
            "position 0 plus the verifier bonus row, leaving more draft rows "
            "on the 2:4 sparse-only down_proj path to improve sparse branch "
            "fill under the 8pp quality budget."
        ),
        extra_quality_args=(
            "--sr24-target-leafs",
            "down_proj",
            "--sr24-residual-target-leafs",
            "down_proj",
            "--sr24-row-routed-down-linear",
            "--sr24-selective-min-prefix-residual",
            "1",
            "--sr24-selective-max-residual-draft-rows",
            "1",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-target-leafs",
            "down_proj",
            "--sr24-residual-target-leafs",
            "down_proj",
            "--sr24-row-routed-down-linear",
            "--sr24-selective-min-prefix-residual",
            "1",
            "--sr24-selective-max-residual-draft-rows",
            "1",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
    ),
    "lossy_prefix0_down_only_noverify_sparse_compile": Candidate(
        name="lossy_prefix0_down_only_noverify_sparse_compile",
        preset="lossy_prefix2_rowrouted_mlp",
        note=(
            "aggressive down-proj-only quality-boundary probe: only the verifier "
            "bonus row is dense, all draft/noverify down_proj rows are 2:4 "
            "sparse-only, and the split branch is used only when it has at "
            "least 128 base rows."
        ),
        extra_quality_args=(
            "--sr24-target-leafs",
            "down_proj",
            "--sr24-residual-target-leafs",
            "down_proj",
            "--sr24-row-routed-down-linear",
            "--sr24-selective-min-prefix-residual",
            "0",
            "--sr24-selective-max-residual-draft-rows",
            "0",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
        extra_throughput_args=(
            "--sr24-target-leafs",
            "down_proj",
            "--sr24-residual-target-leafs",
            "down_proj",
            "--sr24-row-routed-down-linear",
            "--sr24-selective-min-prefix-residual",
            "0",
            "--sr24-selective-max-residual-draft-rows",
            "0",
            "--sr24-route-min-base-rows",
            "128",
            "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
            "none",
            "--sr24-default-vllm-compile",
        ),
    ),
    "gateup_channel_pair_dense25_eager": Candidate(
        name="gateup_channel_pair_dense25_eager",
        preset="manual",
        note=(
            "channel-split gate_up MLP candidate: keep the highest-norm 25% "
            "intermediate channels dense, run the remaining gate/up channels "
            "through a full-row 2:4 sparse branch, and keep down_proj dense. "
            "This avoids row-split tiny-M sparse GEMMs and removes dense "
            "recompute for the sparse channels."
        ),
        extra_quality_args=(
            "--sr24-target-leafs",
            "gate_up_proj",
            "--sr24-residual-target-leafs",
            "none",
            "--sr24-residual-device",
            "cuda",
            "--sr24-gate-up-split",
            "channel_pair",
            "--sr24-gate-up-channel-dense-fraction",
            "0.25",
            "--sr24-gate-up-channel-strategy",
            "norm",
            "--sr24-reduce-cpu-sync",
            "--no-sr24-sync-mask-state",
            "--sr24-static-mask-buffer",
            "--sr24-batched-mask-builder",
            "--sr24-disable-runtime-stats",
        ),
        extra_throughput_args=(
            "--sr24-target-leafs",
            "gate_up_proj",
            "--sr24-residual-target-leafs",
            "none",
            "--sr24-residual-device",
            "cuda",
            "--sr24-gate-up-split",
            "channel_pair",
            "--sr24-gate-up-channel-dense-fraction",
            "0.25",
            "--sr24-gate-up-channel-strategy",
            "norm",
            "--sr24-reduce-cpu-sync",
            "--no-sr24-sync-mask-state",
            "--sr24-static-mask-buffer",
            "--sr24-batched-mask-builder",
            "--sr24-disable-runtime-stats",
        ),
    ),
    "gateup_channel_pair_dense50_eager": Candidate(
        name="gateup_channel_pair_dense50_eager",
        preset="manual",
        note=(
            "quality-biased channel-split gate_up MLP candidate: keep the "
            "highest-norm 50% intermediate channels dense and run the rest "
            "through full-row 2:4 sparse. It tests whether a larger dense "
            "channel budget stays within the 8pp quality gate while still "
            "leaving enough sparse work to matter."
        ),
        extra_quality_args=(
            "--sr24-target-leafs",
            "gate_up_proj",
            "--sr24-residual-target-leafs",
            "none",
            "--sr24-residual-device",
            "cuda",
            "--sr24-gate-up-split",
            "channel_pair",
            "--sr24-gate-up-channel-dense-fraction",
            "0.50",
            "--sr24-gate-up-channel-strategy",
            "norm",
            "--sr24-reduce-cpu-sync",
            "--no-sr24-sync-mask-state",
            "--sr24-static-mask-buffer",
            "--sr24-batched-mask-builder",
            "--sr24-disable-runtime-stats",
        ),
        extra_throughput_args=(
            "--sr24-target-leafs",
            "gate_up_proj",
            "--sr24-residual-target-leafs",
            "none",
            "--sr24-residual-device",
            "cuda",
            "--sr24-gate-up-split",
            "channel_pair",
            "--sr24-gate-up-channel-dense-fraction",
            "0.50",
            "--sr24-gate-up-channel-strategy",
            "norm",
            "--sr24-reduce-cpu-sync",
            "--no-sr24-sync-mask-state",
            "--sr24-static-mask-buffer",
            "--sr24-batched-mask-builder",
            "--sr24-disable-runtime-stats",
        ),
    ),
    "fixedprefix2_directcslt_mlp16_31": Candidate(
        name="fixedprefix2_directcslt_mlp16_31",
        preset="mlpall_fixedprefix2_directcslt",
        note=(
            "score-free fixed-prefix2 route table with direct cuSPARSELt, but "
            "only MLP layers 16-31 are attached to SR24 dense residual weights; "
            "earlier MLP layers stay on the original dense vLLM path. This is a "
            "quality/scope reference, not a sparse-only storage-reduction path"
        ),
        extra_quality_args=(
            "--sr24-residual-layer-ids-by-leaf",
            "gate_up_proj=16-31;down_proj=16-31",
        ),
        extra_throughput_args=(
            "--sr24-residual-layer-ids-by-leaf",
            "gate_up_proj=16-31;down_proj=16-31",
        ),
    ),
    "fixedprefix2_directcslt_mlp24_31": Candidate(
        name="fixedprefix2_directcslt_mlp24_31",
        preset="mlpall_fixedprefix2_directcslt",
        note=(
            "narrower fixed-prefix2 route-table reference: only MLP layers "
            "24-31 are attached to SR24 dense residual weights, while earlier "
            "MLP layers stay on the original dense vLLM path. This tests "
            "residual scope, not sparse-only tail execution"
        ),
        extra_quality_args=(
            "--sr24-residual-layer-ids-by-leaf",
            "gate_up_proj=24-31;down_proj=24-31",
        ),
        extra_throughput_args=(
            "--sr24-residual-layer-ids-by-leaf",
            "gate_up_proj=24-31;down_proj=24-31",
        ),
    ),
    "fixedprefix2_directcslt_mlp0_15_base16_31": Candidate(
        name="fixedprefix2_directcslt_mlp0_15_base16_31",
        preset="mlpall_fixedprefix2_directcslt",
        note=(
            "front-sensitive layer split: MLP layers 0-15 keep fixed-prefix2 "
            "dense residual correction, while layers 16-31 are explicit "
            "sparse-only base layers to reduce residual storage and work"
        ),
        extra_quality_args=(
            "--sr24-residual-layer-ids-by-leaf",
            "gate_up_proj=0-15;down_proj=0-15",
            "--sr24-base-only-layer-ids-by-leaf",
            "gate_up_proj=16-31;down_proj=16-31",
        ),
        extra_throughput_args=(
            "--sr24-residual-layer-ids-by-leaf",
            "gate_up_proj=0-15;down_proj=0-15",
            "--sr24-base-only-layer-ids-by-leaf",
            "gate_up_proj=16-31;down_proj=16-31",
        ),
    ),
    "fixedprefix2_directcslt_mlp0_23_base24_31": Candidate(
        name="fixedprefix2_directcslt_mlp0_23_base24_31",
        preset="mlpall_fixedprefix2_directcslt",
        note=(
            "quality-biased layer split: MLP layers 0-23 keep fixed-prefix2 "
            "dense residual correction, while only layers 24-31 become "
            "sparse-only base layers"
        ),
        extra_quality_args=(
            "--sr24-residual-layer-ids-by-leaf",
            "gate_up_proj=0-23;down_proj=0-23",
            "--sr24-base-only-layer-ids-by-leaf",
            "gate_up_proj=24-31;down_proj=24-31",
        ),
        extra_throughput_args=(
            "--sr24-residual-layer-ids-by-leaf",
            "gate_up_proj=0-23;down_proj=0-23",
            "--sr24-base-only-layer-ids-by-leaf",
            "gate_up_proj=24-31;down_proj=24-31",
        ),
    ),
    "mlpall_direct_prefix2_activeonly": Candidate(
        name="mlpall_direct_prefix2_activeonly",
        preset="mlpall_lowconf_prefix5_tritonoverride",
        note=(
            "all-MLP direct cuSPARSELt prefix2, with dense correction computed "
            "only for runtime-active bucket rows so unimportant rows remain "
            "sparse-only; this is the variable-shape operator ablation"
        ),
        extra_quality_args=(
            "--sr24-direct-cslt-linear",
            "--sr24-selective-min-prefix-residual",
            "2",
            "--sr24-bucket-dense-copy",
            "--sr24-bucket-dense-copy-active-only",
            "--sr24-bucket-dense-compute-active-only",
            "--sr24-triton-bucket-scatter",
        ),
        extra_throughput_args=(
            "--sr24-direct-cslt-linear",
            "--sr24-selective-min-prefix-residual",
            "2",
            "--sr24-bucket-dense-copy",
            "--sr24-bucket-dense-copy-active-only",
            "--sr24-bucket-dense-compute-active-only",
            "--sr24-triton-bucket-scatter",
        ),
    ),
    "mlpall_direct_prefix2_activefused": Candidate(
        name="mlpall_direct_prefix2_activefused",
        preset="mlpall_lowconf_prefix5_tritonoverride",
        note=(
            "all-MLP direct cuSPARSELt prefix2, but keep bucket correction "
            "shape-stable via fixed bucket rows plus a GPU active mask in the "
            "Triton fused GEMM+scatter path"
        ),
        extra_quality_args=(
            "--sr24-direct-cslt-linear",
            "--sr24-selective-min-prefix-residual",
            "2",
            "--sr24-bucket-dense-copy",
            "--sr24-bucket-dense-copy-active-only",
            "--sr24-bucket-dense-compute-active-only",
            "--sr24-triton-bucket-dense-gemm",
            "--sr24-bucket-dense-active-mask-fused",
        ),
        extra_throughput_args=(
            "--sr24-direct-cslt-linear",
            "--sr24-selective-min-prefix-residual",
            "2",
            "--sr24-bucket-dense-copy",
            "--sr24-bucket-dense-copy-active-only",
            "--sr24-bucket-dense-compute-active-only",
            "--sr24-triton-bucket-dense-gemm",
            "--sr24-bucket-dense-active-mask-fused",
        ),
    ),
    "gateup_res16_25_base26_31_critical4": Candidate(
        name="gateup_res16_25_base26_31_critical4",
        preset="gateup_res16_25_base26_31_critical4",
        note=(
            "low-storage gate_up split: residual-correct layers 16-25 and "
            "leave layers 26-31 sparse-only, using critical_prefix+extra4. "
            "Historical GSM8K/Minerva-50 quality was within 8pp with about "
            "1.25x dense storage instead of 1.625x."
        ),
    ),
    "gateup_res16_25_base26_31_critical4_smallrow160": Candidate(
        name="gateup_res16_25_base26_31_critical4_smallrow160",
        preset="gateup_res16_25_base26_31_critical4_smallrow160",
        note=(
            "same low-storage split, but isolate the low-batch sparse-base "
            "bottleneck by falling back to dense only for no-residual "
            "gate_up_proj steps with <=160 rows"
        ),
    ),
    "mlpall_direct_prefix0": Candidate(
        name="mlpall_direct_prefix0",
        preset="mlpall_lowconf_prefix5_tritonoverride",
        note=(
            "all-MLP direct cuSPARSELt with no mandatory dense prefix; only "
            "low-confidence/bonus rows are corrected"
        ),
        extra_quality_args=(
            "--sr24-direct-cslt-linear",
            "--sr24-selective-min-prefix-residual",
            "0",
        ),
        extra_throughput_args=(
            "--sr24-direct-cslt-linear",
            "--sr24-selective-min-prefix-residual",
            "0",
        ),
    ),
    "fixedprefix4_bucket16_directcslt": Candidate(
        name="fixedprefix4_bucket16_directcslt",
        preset="fixedprefix4_bucket16_directcslt",
        note=(
            "low-score-overhead row: fixed first-four draft rows plus bonus, "
            "same gate_up/down scope as bucket16"
        ),
    ),
    "criticalprefix4_bucket16_directcslt_rowrouted_min128": Candidate(
        name="criticalprefix4_bucket16_directcslt_rowrouted_min128",
        preset="criticalprefix4_bucket16_directcslt",
        note=(
            "critical-prefix quality candidate with shape-aware row routing for "
            "gate_up/down only when dense groups have at least 128 rows"
        ),
        extra_quality_args=(
            "--sr24-row-routed-mlp",
            "--sr24-row-routed-down-linear",
            "--sr24-row-routed-mlp-min-dense-rows-by-leaf",
            "gate_up_proj=128;down_proj=128",
        ),
        extra_throughput_args=(
            "--sr24-row-routed-mlp",
            "--sr24-row-routed-down-linear",
            "--sr24-row-routed-mlp-min-dense-rows-by-leaf",
            "gate_up_proj=128;down_proj=128",
        ),
    ),
    "fixedprefix4_bucket16_directcslt_rowrouted_min128": Candidate(
        name="fixedprefix4_bucket16_directcslt_rowrouted_min128",
        preset="fixedprefix4_bucket16_directcslt",
        note=(
            "fixed-prefix low-score-overhead candidate with the same min128 "
            "shape-aware row-routing planner"
        ),
        extra_quality_args=(
            "--sr24-row-routed-mlp",
            "--sr24-row-routed-down-linear",
            "--sr24-row-routed-mlp-min-dense-rows-by-leaf",
            "gate_up_proj=128;down_proj=128",
        ),
        extra_throughput_args=(
            "--sr24-row-routed-mlp",
            "--sr24-row-routed-down-linear",
            "--sr24-row-routed-mlp-min-dense-rows-by-leaf",
            "gate_up_proj=128;down_proj=128",
        ),
    ),
    "down0_15_fixedprefix4_directcslt": Candidate(
        name="down0_15_fixedprefix4_directcslt",
        preset="down0_15_fixedprefix4_directcslt",
        note=(
            "focused down-proj row: only early down_proj layers, fixed "
            "first-four residual rows"
        ),
    ),
}


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def quote_cmd(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(
    command: list[str],
    *,
    cwd: Path,
    commands: list[str],
    dry_run: bool,
) -> None:
    rendered = f"(cd {shlex.quote(str(cwd))} && {quote_cmd(command)})"
    commands.append(rendered)
    print(rendered, flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd), check=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def quality_command(args: argparse.Namespace, candidate: Candidate,
                    output_dir: Path, port_base: int) -> list[str]:
    return [
        sys.executable,
        str(EVAL_ROOT / "scripts/run_lm_eval_accuracy.py"),
        "--mode",
        "dense_baseline,speclink_t08",
        "--task",
        args.quality_task,
        "--models",
        args.model,
        "--limit",
        str(args.quality_limit),
        "--max-new-tokens",
        str(args.quality_max_new_tokens),
        "--num-spec-tokens",
        str(args.num_spec_tokens),
        "--max-context-length",
        str(args.max_context_length),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--port-base",
        str(port_base),
        "--sr24-preset",
        candidate.preset,
        "--output-dir",
        str(output_dir),
        "--aggregate",
        *candidate.extra_quality_args,
    ]


def throughput_command(args: argparse.Namespace, candidate: Candidate,
                       final_root: Path, work_root: Path,
                       port_base: int) -> list[str]:
    return [
        sys.executable,
        str(EVAL_ROOT / "scripts/run_llama31_vllm_fastdraft_smurfs_eagle3_matrix.py"),
        "--methods",
        "dense_baseline,speclink_t08",
        "--datasets",
        args.throughput_datasets,
        "--batch-sizes",
        args.batch_sizes,
        "--repeats",
        str(args.repeats),
        "--fixed-total-requests",
        str(args.fixed_total_requests),
        "--max-tokens",
        str(args.throughput_max_tokens),
        "--max-model-len",
        str(args.max_context_length),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--warmup-s",
        str(args.warmup_s),
        "--measurement-s",
        str(args.measurement_s),
        "--cooldown-s",
        str(args.cooldown_s),
        "--port-base",
        str(port_base),
        "--eagle3-k",
        str(args.num_spec_tokens),
        "--sr24-preset",
        candidate.preset,
        "--final-root",
        str(final_root),
        "--work-root",
        str(work_root),
        *candidate.extra_throughput_args,
    ]


def summarize_quality(path: Path) -> dict[str, Any]:
    rows = read_csv(path / "summary.csv")
    dense = next((row for row in rows if row.get("mode") == "dense_baseline"), {})
    speclink = next((row for row in rows if row.get("mode") == "speclink_t08"), {})
    return {
        "quality_summary_csv": str((path / "summary.csv").resolve()),
        "dense_score": to_float(dense.get("score")),
        "speclink_score": to_float(speclink.get("score")),
        "delta_pp_vs_dense": to_float(speclink.get("delta_pp_vs_dense")),
        "samples": to_float(speclink.get("samples")),
        "pair_reg": to_float(speclink.get("dense_correct_experimental_wrong")),
        "pair_imp": to_float(speclink.get("dense_wrong_experimental_correct")),
        "status": speclink.get("status", ""),
    }


def summarize_throughput(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path / "summary.csv")
    dense_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("method") != "dense_baseline" or row.get("status") != "ok":
            continue
        dense_by_key[(row.get("dataset", ""), row.get("batch_size", ""))] = row
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("method") != "speclink_t08" or row.get("status") != "ok":
            continue
        key = (row.get("dataset", ""), row.get("batch_size", ""))
        dense = dense_by_key.get(key, {})
        dense_total = to_float(dense.get("total_output_tokens_per_second"))
        dense_full = to_float(dense.get("full_batch_output_tokens_per_second"))
        sparse_total = to_float(row.get("total_output_tokens_per_second"))
        sparse_full = to_float(row.get("full_batch_output_tokens_per_second"))
        out.append({
            "dataset": key[0],
            "batch_size": int(float(key[1])) if key[1] else "",
            "dense_total_tps": dense_total,
            "speclink_total_tps": sparse_total,
            "total_speedup": (
                sparse_total / dense_total
                if sparse_total is not None and dense_total else None
            ),
            "dense_full_batch_tps": dense_full,
            "speclink_full_batch_tps": sparse_full,
            "full_batch_speedup": (
                sparse_full / dense_full
                if sparse_full is not None and dense_full else None
            ),
            "sr24_target_leafs": row.get("sr24_target_leafs", ""),
            "sr24_residual_leafs": row.get("sr24_residual_target_leafs", ""),
            "sr24_residual_bucket_size": row.get("sr24_residual_bucket_size", ""),
            "sr24_cudagraph_modes": (
                row.get("server_cudagraph_profile_counts")
                or row.get("sr24_cudagraph_mode_counts")
                or ""
            ),
        })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    number = to_float(value)
    return "" if number is None else f"{number:.{digits}f}"


def write_report(path: Path, rows: list[dict[str, Any]],
                 args: argparse.Namespace) -> None:
    lines = [
        "# SR24 Lossy Speed/Quality Sweep",
        "",
        "This run gates SR24 candidates with a bounded accuracy loss before "
        "spending GPU time on throughput.",
        "",
        "## Configuration",
        "",
        f"- model: `{args.model}`",
        f"- quality task: `{args.quality_task}`",
        f"- quality limit: `{args.quality_limit}`",
        f"- quality budget: `{args.max_accuracy_drop_pp:.2f} pp`",
        f"- throughput datasets: `{args.throughput_datasets}`",
        f"- batch sizes: `{args.batch_sizes}`",
        f"- K: `{args.num_spec_tokens}`",
        "",
        "## Candidate Summary",
        "",
        "| candidate | quality pass | dense acc | SR24 acc | delta pp | pair reg/imp | best full speedup | best total speedup | note |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {candidate} | {quality_pass} | {dense} | {sr24} | {delta} | "
            "{pair} | {full} | {total} | {note} |".format(
                candidate=row["candidate"],
                quality_pass="yes" if row.get("quality_pass") else "no",
                dense=fmt(row.get("dense_score")),
                sr24=fmt(row.get("speclink_score")),
                delta=fmt(row.get("delta_pp_vs_dense")),
                pair=(
                    f"{fmt(row.get('pair_reg'), 0)}/"
                    f"{fmt(row.get('pair_imp'), 0)}"
                ),
                full=fmt(row.get("best_full_batch_speedup")),
                total=fmt(row.get("best_total_speedup")),
                note=row.get("note", ""),
            )
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Passing this script is not a final matrix claim; it is the first "
        "small-scale gate for the 8pp-loss optimization path.",
        "- `best full speedup` removes request-drain tail effects; `best total "
        "speedup` is stricter end-to-end fixed-request throughput.",
        "- Candidates that fail quality are skipped by default, because lower "
        "dense correction would only be useful if a later controller recovers "
        "accuracy elsewhere.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SR24 candidate accuracy gate and optional throughput.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--candidates",
        default="criticalprefix4_bucket16_directcslt,lowresidual_gateup_riskcap2,mlpall_lowconf_prefix5_tritonoverride",
        help=f"Comma-separated candidates. Known: {','.join(sorted(CANDIDATES))}",
    )
    parser.add_argument("--model", default="llama3_1_8b")
    parser.add_argument("--quality-task", default="gsm8k_cot")
    parser.add_argument("--quality-limit", type=int, default=50)
    parser.add_argument("--quality-max-new-tokens", type=int, default=512)
    parser.add_argument("--max-accuracy-drop-pp", type=float, default=8.0)
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--throughput-datasets", default="math_reasoning")
    parser.add_argument("--batch-sizes", default="8,16,32,64")
    parser.add_argument("--fixed-total-requests", type=int, default=128)
    parser.add_argument("--throughput-max-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--num-spec-tokens", type=int, default=8)
    parser.add_argument("--max-context-length", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--measurement-s", type=float, default=8.0)
    parser.add_argument("--cooldown-s", type=float, default=1.0)
    parser.add_argument("--port-base", type=int, default=8720)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    selected: list[Candidate] = []
    for name in csv_list(args.candidates):
        candidate = CANDIDATES.get(name)
        if candidate is None:
            raise SystemExit(
                f"unknown candidate {name!r}; known={','.join(sorted(CANDIDATES))}"
            )
        selected.append(candidate)
    run_id = f"sr24_lossy_speed_quality_sweep_{timestamp()}"
    output_root = (args.output_root or (DEFAULT_OUTPUT_ROOT / run_id)).resolve()
    work_root = (args.work_root or (DEFAULT_TEMP_ROOT / f"{run_id}_work")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    commands: list[str] = []
    summary_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected):
        quality_dir = output_root / "quality" / candidate.name
        throughput_dir = output_root / "throughput" / candidate.name
        candidate_work = work_root / candidate.name
        quality_pass = True
        quality_summary: dict[str, Any] = {}
        if not args.skip_quality:
            run_command(
                quality_command(
                    args,
                    candidate,
                    quality_dir,
                    args.port_base + index * 20,
                ),
                cwd=EVAL_ROOT,
                commands=commands,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                quality_summary = summarize_quality(quality_dir)
                delta = quality_summary.get("delta_pp_vs_dense")
                status_ok = quality_summary.get("status") == "ok"
                quality_pass = (
                    status_ok
                    and delta is not None
                    and float(delta) >= -float(args.max_accuracy_drop_pp) - 1e-6
                )
        throughput_rows: list[dict[str, Any]] = []
        if not args.skip_throughput and quality_pass:
            run_command(
                throughput_command(
                    args,
                    candidate,
                    throughput_dir,
                    candidate_work,
                    args.port_base + index * 20 + 5,
                ),
                cwd=EVAL_ROOT,
                commands=commands,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                throughput_rows = summarize_throughput(throughput_dir)
                write_csv(output_root / f"{candidate.name}_throughput_summary.csv",
                          throughput_rows)

        best_full = max(
            (
                value
                for value in (
                    to_float(row.get("full_batch_speedup"))
                    for row in throughput_rows
                )
                if value is not None
            ),
            default=None,
        )
        best_total = max(
            (
                value
                for value in (
                    to_float(row.get("total_speedup"))
                    for row in throughput_rows
                )
                if value is not None
            ),
            default=None,
        )
        row = {
            "candidate": candidate.name,
            "preset": candidate.preset,
            "quality_pass": quality_pass,
            "note": candidate.note,
            "quality_dir": str(quality_dir.resolve()),
            "throughput_dir": str(throughput_dir.resolve()),
            "best_full_batch_speedup": best_full,
            "best_total_speedup": best_total,
            **quality_summary,
        }
        summary_rows.append(row)
        write_csv(output_root / "summary.csv", summary_rows)
        write_report(output_root / "report.md", summary_rows, args)

    (output_root / "commands.sh").write_text(
        "# Commands executed by run_sr24_lossy_speed_quality_sweep.py\n"
        + "\n".join(commands)
        + "\n",
        encoding="utf-8",
    )
    config = vars(args).copy()
    config["output_root"] = str(output_root.resolve())
    config["work_root"] = str(work_root.resolve())
    config["candidates_resolved"] = [candidate.name for candidate in selected]
    (output_root / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(output_root.resolve())


if __name__ == "__main__":
    main()
