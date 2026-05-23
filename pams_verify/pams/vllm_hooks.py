from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any


def inspect_vllm_for_pams_hooks() -> dict[str, Any]:
    result: dict[str, Any] = {
        "vllm_imported": False,
        "version": "unavailable",
        "package_path": None,
        "attention_modules_checked": [],
        "has_pams_verify_config_arg": False,
        "arbitrary_block_mask_supported": False,
        "notes": [],
    }
    try:
        import vllm

        result["vllm_imported"] = True
        result["version"] = str(getattr(vllm, "__version__", "unknown"))
        result["package_path"] = str(Path(inspect.getfile(vllm)).resolve().parent)
    except Exception as exc:
        result["notes"].append(f"vllm import failed: {type(exc).__name__}: {exc}")
        return result

    module_names = [
        "vllm.v1.attention",
        "vllm.v1.attention.backends.flash_attn",
        "vllm.v1.attention.backends.flex_attention",
        "vllm.v1.attention.ops.triton_unified_attention",
        "vllm.model_executor.layers.attention",
        "vllm.transformers_utils.configs.speculators",
        "vllm.v1.spec_decode",
    ]
    for name in module_names:
        try:
            module = __import__(name, fromlist=["dummy"])
            source_path = Path(inspect.getfile(module)).resolve()
            text = source_path.read_text(encoding="utf-8", errors="ignore")
            result["attention_modules_checked"].append(str(source_path))
            if "pams-verify-config" in text or "PAMS_VERIFY_ENABLE" in text:
                result["has_pams_verify_config_arg"] = True
            if "block_mask" in text or "sparse_mask" in text:
                result["notes"].append(f"{name} contains mask-like identifiers; these are backend/local masks, not a PAMS verifier-mask API.")
            if name.endswith("flash_attn") and "block_table" in text and "flash_attn_varlen_func" in text:
                result["notes"].append(
                    "vllm.v1.attention.backends.flash_attn is the live Qwen3-8B path; it forwards paged KV block_table/seqused_k into flash_attn_varlen_func, but has no request-time arbitrary verifier block-mask field."
                )
        except Exception as exc:
            result["notes"].append(f"{name} unavailable: {type(exc).__name__}: {exc}")
    result["notes"].append(
        "No installed vLLM feature flag for PAMS was found. Arbitrary verifier block masks are treated as unsupported until a local vLLM source patch is applied."
    )
    return result
