#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pams.configs import EXPERIMENTS, register_experiment
from pams.memory import detect_hardware


def main() -> None:
    out = EXPERIMENTS / "00_env" / "hardware.json"
    info = detect_hardware(out)
    status = "completed" if info.gpu_count > 0 else "blocked_no_gpu_visible"
    summary = [
        "# Phase 0 Hardware Detection",
        "",
        f"- CUDA available to torch: `{info.cuda_available}`",
        f"- GPU count: `{info.gpu_count}`",
        f"- GPU: `{info.gpu_name}`",
        f"- VRAM GB: `{info.total_vram_gb}`",
        f"- Driver: `{info.driver_version}`",
        f"- CUDA: `{info.cuda_version}`",
        f"- Torch: `{info.torch_version}`",
        f"- vLLM: `{info.vllm_version}`",
        f"- Recommended dtype: `{info.recommended_dtype}`",
        f"- Safe candidates: `{info.safe_max_model_len_candidates}`",
        "",
        "Notes:",
    ]
    summary.extend(f"- {note}" for note in info.notes)
    register_experiment(
        "00_env",
        config={"script": "detect_hardware.py"},
        command="python scripts/detect_hardware.py",
        status=status,
        summary="\n".join(summary),
        metadata_extra={"hardware_json": str(out), **info.to_dict()},
    )
    print(info.to_dict())


if __name__ == "__main__":
    main()

