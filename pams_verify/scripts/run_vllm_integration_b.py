#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pams.configs import EXPERIMENTS, register_experiment, write_json
from pams.vllm_hooks import inspect_vllm_for_pams_hooks


PROPOSED_DIFF = """diff --git a/vllm/entrypoints/openai/cli_args.py b/vllm/entrypoints/openai/cli_args.py
--- a/vllm/entrypoints/openai/cli_args.py
+++ b/vllm/entrypoints/openai/cli_args.py
@@
+    parser.add_argument("--pams-verify-config", default=None)
+
diff --git a/vllm/attention/backends/abstract.py b/vllm/attention/backends/abstract.py
--- a/vllm/attention/backends/abstract.py
+++ b/vllm/attention/backends/abstract.py
@@
+    # Prototype only: carry an optional verifier block mask for decode verification.
+    pams_block_mask: Optional[Any] = None
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--apply", action="store_true", help="Do not use in shared env unless you intend to patch vLLM source.")
    args = parser.parse_args()
    inspected = inspect_vllm_for_pams_hooks()
    exp = EXPERIMENTS / "08_vllm_integration_b_attention_patch"
    (exp / "raw").mkdir(parents=True, exist_ok=True)
    (exp / "parsed").mkdir(parents=True, exist_ok=True)
    (exp / "figures").mkdir(parents=True, exist_ok=True)
    (exp / "raw" / "proposed_pams_vllm_patch.diff").write_text(PROPOSED_DIFF, encoding="utf-8")
    result = {
        "integration": "B_attention_patch_sparse_verifier",
        "apply_requested": bool(args.apply),
        "patched_vllm": False,
        "compiled": False,
        "ran_live_vllm": False,
        "arbitrary_block_mask_supported": inspected["arbitrary_block_mask_supported"],
        "pams_feature_flag_present": inspected["has_pams_verify_config_arg"],
        "vllm_inspection": inspected,
        "outcome": "unsupported_installed_backend_no_patch_applied",
        "limitation": "Installed vLLM does not expose an arbitrary verifier block-mask path; patching site-packages was not performed from the shared sandbox.",
    }
    write_json(exp / "parsed" / "integration_b_result.json", result)
    summary = [
        "# Integration B: Attention Patch / Sparse Verifier Path",
        "",
        "Attempted by inspecting the installed vLLM package and writing a minimal proposed diff showing the required feature flag and mask carrier.",
        "",
        "Result: unsupported in the current installed vLLM path. No live sparse verifier patch compiled or ran.",
        "",
        f"- vLLM package path: `{inspected['package_path']}`",
        f"- PAMS feature flag already present: `{inspected['has_pams_verify_config_arg']}`",
        f"- Arbitrary block mask support detected: `{inspected['arbitrary_block_mask_supported']}`",
        f"- Proposed diff: `raw/proposed_pams_vllm_patch.diff`",
    ]
    register_experiment(
        "08_vllm_integration_b_attention_patch",
        config=vars(args),
        command="python scripts/run_vllm_integration_b.py --target-model Qwen/Qwen3-8B",
        status="attempted_unsupported_backend",
        summary="\n".join(summary),
        metadata_extra=result,
        model_name=args.target_model,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
