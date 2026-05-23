from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .configs import write_json


QWEN3_8B_FALLBACK = {
    "num_hidden_layers": 36,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "parameter_count": 8_200_000_000,
    "max_position_embeddings": 40960,
}


@dataclass
class HardwareInfo:
    cuda_available: bool
    gpu_count: int
    gpu_name: str
    total_vram_gb: float | None
    driver_version: str
    cuda_version: str
    torch_version: str
    vllm_version: str
    recommended_dtype: str
    safe_max_model_len_candidates: list[int]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cuda_available": self.cuda_available,
            "gpu_count": self.gpu_count,
            "gpu_name": self.gpu_name,
            "total_vram_gb": self.total_vram_gb,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "torch_version": self.torch_version,
            "vllm_version": self.vllm_version,
            "recommended_dtype": self.recommended_dtype,
            "safe_max_model_len_candidates": self.safe_max_model_len_candidates,
            "notes": self.notes,
        }


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=20).stdout
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def detect_hardware(save_path: Path | None = None) -> HardwareInfo:
    notes: list[str] = []
    cuda_available = False
    gpu_count = 0
    torch_version = "unavailable"
    torch_cuda = "unavailable"
    try:
        import torch

        torch_version = torch.__version__
        torch_cuda = str(torch.version.cuda)
        cuda_available = bool(torch.cuda.is_available())
        gpu_count = int(torch.cuda.device_count())
    except Exception as exc:
        notes.append(f"torch import or CUDA query failed: {type(exc).__name__}: {exc}")

    vllm_version = "unavailable"
    try:
        import vllm

        vllm_version = str(getattr(vllm, "__version__", "unknown"))
    except Exception as exc:
        notes.append(f"vllm import failed: {type(exc).__name__}: {exc}")

    smi = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu_name = "unknown"
    total_vram_gb: float | None = None
    driver_version = "unknown"
    if smi and not smi.startswith("ERROR") and "NVIDIA-SMI has failed" not in smi:
        first = smi.strip().splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) >= 3:
            gpu_name = parts[0]
            try:
                total_vram_gb = round(float(parts[1]) / 1024.0, 2)
            except ValueError:
                total_vram_gb = None
            driver_version = parts[2]
            if gpu_count == 0:
                gpu_count = len(smi.strip().splitlines())
                notes.append("nvidia-smi sees GPU(s) but torch CUDA is unavailable in this process.")
    else:
        notes.append(f"nvidia-smi query failed: {smi.strip()[:200]}")

    smi_header = _run(["nvidia-smi"])
    cuda_version = torch_cuda
    match = re.search(r"CUDA Version:\s*([0-9.]+)", smi_header)
    if match:
        cuda_version = match.group(1)

    recommended_dtype = "bfloat16" if "RTX 5090" in gpu_name or "H100" in gpu_name or "A100" in gpu_name else "float16"
    candidates = [2048, 4096]
    if total_vram_gb and total_vram_gb >= 24:
        candidates.append(8192)
    if total_vram_gb and total_vram_gb >= 40:
        candidates.append(16384)
    elif gpu_name == "NVIDIA GeForce RTX 5090":
        candidates.append(16384)
        notes.append("16384 should still be gated by memory estimator headroom on RTX 5090.")

    info = HardwareInfo(
        cuda_available=cuda_available,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        total_vram_gb=total_vram_gb,
        driver_version=driver_version,
        cuda_version=cuda_version,
        torch_version=torch_version,
        vllm_version=vllm_version,
        recommended_dtype=recommended_dtype,
        safe_max_model_len_candidates=candidates,
        notes=notes,
    )
    if save_path is not None:
        write_json(save_path, info.to_dict())
    return info


def load_model_config(model: str) -> dict[str, Any]:
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True, local_files_only=True)
        payload = cfg.to_dict()
    except Exception:
        payload = dict(QWEN3_8B_FALLBACK)
    for key, value in QWEN3_8B_FALLBACK.items():
        payload.setdefault(key, value)
    payload.setdefault("parameter_count", QWEN3_8B_FALLBACK["parameter_count"])
    return payload


def dtype_bytes(dtype: str) -> int:
    lowered = dtype.lower()
    if lowered in {"float32", "fp32"}:
        return 4
    if lowered in {"bfloat16", "bf16", "float16", "fp16", "half"}:
        return 2
    if lowered in {"fp8", "float8", "int8"}:
        return 1
    return 2


def estimate_memory(
    model: str,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
    max_num_seqs: int = 16,
) -> dict[str, Any]:
    hw = detect_hardware()
    cfg = load_model_config(model)
    bytes_per_value = dtype_bytes(dtype)
    layers = int(cfg.get("num_hidden_layers", QWEN3_8B_FALLBACK["num_hidden_layers"]))
    kv_heads = int(cfg.get("num_key_value_heads", cfg.get("num_attention_heads", 8)))
    head_dim = int(cfg.get("head_dim", int(cfg.get("hidden_size", 4096)) // int(cfg.get("num_attention_heads", 32))))
    params = int(cfg.get("parameter_count", QWEN3_8B_FALLBACK["parameter_count"]))
    weight_gb = params * bytes_per_value / (1024**3)
    kv_bytes_per_token = layers * 2 * kv_heads * head_dim * bytes_per_value
    total_vram_gb = hw.total_vram_gb or 32.0
    usable_gb = total_vram_gb * gpu_memory_utilization
    runtime_overhead_gb = max(3.0, total_vram_gb * 0.12)
    available_kv_gb = max(0.0, usable_gb - weight_gb - runtime_overhead_gb)
    max_kv_tokens = int(available_kv_gb * (1024**3) / kv_bytes_per_token) if kv_bytes_per_token else 0
    requested_kv_tokens = max_model_len * max_num_seqs
    headroom = (max_kv_tokens - requested_kv_tokens) / max(requested_kv_tokens, 1)

    recommended_len = 4096
    if max_kv_tokens >= 8192 * max(4, min(max_num_seqs, 8)):
        recommended_len = 8192
    if max_kv_tokens >= int(16384 * max(1, min(max_num_seqs, 4)) * 1.2):
        recommended_len = 16384
    if "RTX 5090" in hw.gpu_name and recommended_len > 16384:
        recommended_len = 16384

    recommendation_notes = [
        "Start smoke tests at max_model_len=4096 and gpu_memory_utilization=0.85.",
        "Try 8192 only after smoke passes.",
    ]
    if recommended_len < 16384:
        recommendation_notes.append("Do not run 16384 by default because estimated headroom is below 20%.")
    else:
        recommendation_notes.append("16384 is allowed only for low max_num_seqs and after smoke passes.")
    recommendation_notes.append("Never run 32768 context by default on RTX 5090.")

    return {
        "model": model,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "gpu_name": hw.gpu_name,
        "total_vram_gb": total_vram_gb,
        "model_weight_memory_gb": round(weight_gb, 3),
        "kv_bytes_per_token": kv_bytes_per_token,
        "kv_mib_per_1024_tokens": round(kv_bytes_per_token * 1024 / (1024**2), 3),
        "runtime_overhead_gb": round(runtime_overhead_gb, 3),
        "available_kv_gb": round(available_kv_gb, 3),
        "max_kv_tokens_under_budget": max_kv_tokens,
        "requested_kv_tokens": requested_kv_tokens,
        "requested_headroom_ratio": round(headroom, 3),
        "recommended_max_model_len": recommended_len,
        "recommended_max_num_seqs": max(1, min(max_num_seqs, math.floor(max_kv_tokens / max(recommended_len, 1)))),
        "safe_8192": max_kv_tokens >= 8192 * max(1, min(max_num_seqs, 8)),
        "safe_16384_with_20pct_headroom": max_kv_tokens >= int(16384 * max(1, min(max_num_seqs, 4)) * 1.2),
        "oom_degrade_order": ["max_num_seqs", "max_model_len", "num_prompts", "dtype_or_kv_cache_dtype"],
        "notes": hw.notes + recommendation_notes,
        "model_config_used": {
            "num_hidden_layers": layers,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "parameter_count": params,
        },
    }

