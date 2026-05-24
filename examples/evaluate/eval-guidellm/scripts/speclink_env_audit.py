#!/usr/bin/env python3
"""Write the SpecLink environment audit required by speclink.md."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]


def run_cmd(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc), "stdout": "", "stderr": ""}
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def count_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return sum(1 for _line in handle)


def model_status(path_or_id: str) -> dict[str, Any]:
    path = Path(path_or_id)
    return {
        "value": path_or_id,
        "exists_as_path": path.exists(),
        "is_dir": path.is_dir(),
        "is_hf_id": "/" in path_or_id and not path.exists(),
    }


def peagle_patch_report() -> str:
    lines: list[str] = []
    spec_dir = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    spec_dir = spec_dir / "site-packages" / "vllm" / "transformers_utils" / "configs" / "speculators"
    lines.append(f"speculators_config_dir={spec_dir}")
    lines.append(f"exists={spec_dir.exists()}")
    for filename in ("algos.py", "base.py"):
        path = spec_dir / filename
        lines.append(f"\n[{filename}]")
        if not path.exists():
            lines.append("missing")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        checks = {
            "supported_peagle": "peagle" in text,
            "method_eagle3": "eagle3" in text,
            "parallel_drafting": "parallel_drafting" in text,
            "pard_token": "pard_token" in text,
            "draft_vocab_size": "draft_vocab_size" in text,
            "norm_before_residual": "norm_before_residual" in text,
            "norm_before_fc": "norm_before_fc" in text,
            "eagle_aux_hidden_state_layer_ids": "eagle_aux_hidden_state_layer_ids" in text,
        }
        for key, value in checks.items():
            lines.append(f"{key}={value}")
        for backup in sorted(path.parent.glob(f"{filename}.bak-peagle*")):
            lines.append(f"backup={backup.name}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Path to env.json")
    parser.add_argument(
        "--dataset",
        default=str(EVAL_DIR / "data" / "math_reasoning.jsonl"),
        help="Dataset path to audit",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset = Path(args.dataset)

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        torch = None
        torch_error = repr(exc)
    else:
        torch_error = None

    audit = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "git_commit": run_cmd(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT),
        "git_diff_stat": run_cmd(["git", "diff", "--stat"], cwd=REPO_ROOT),
        "python_path": sys.executable,
        "python_version": sys.version,
        "torch_version": getattr(torch, "__version__", None) if torch else None,
        "torch_import_error": torch_error,
        "vllm_version": package_version("vllm"),
        "guidellm_version": package_version("guidellm"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_available": bool(torch.cuda.is_available()) if torch else None,
        "torch_cuda_device_count": torch.cuda.device_count() if torch else None,
        "nvidia_smi_path": shutil.which("nvidia-smi"),
        "nvidia_smi": run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        )
        if shutil.which("nvidia-smi")
        else None,
        "models": {
            "BASE_MODEL": model_status(os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")),
            "EAGLE3_SPECULATOR_MODEL": model_status(
                os.environ.get(
                    "EAGLE3_SPECULATOR_MODEL",
                    "/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/qwen3-8b-eagle3-speculator",
                )
            ),
            "PEAGLE_SPECULATOR_MODEL": model_status(
                os.environ.get(
                    "PEAGLE_SPECULATOR_MODEL",
                    "/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/qwen3-8b-peagle-speculator",
                )
            ),
        },
        "dataset": {
            "path": str(dataset),
            "exists": dataset.exists(),
            "bytes": dataset.stat().st_size if dataset.exists() else None,
            "lines": count_lines(dataset),
        },
    }

    out.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    patch_out = out.parent / "peagle_patch_check.txt"
    patch_out.write_text(peagle_patch_report(), encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {patch_out}")


if __name__ == "__main__":
    main()
