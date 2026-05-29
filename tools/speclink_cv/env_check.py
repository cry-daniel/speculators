#!/usr/bin/env python3
"""Collect environment information for SpecLink-CV experiments."""

from __future__ import annotations

import argparse
import json
import os
import textwrap
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "examples" / "evaluate" / "eval-guidellm"
MODELS_ROOT = REPO_ROOT.parent / "models"
EVAL_DATA_ROOT = EVAL_ROOT / "data"


def run(cmd: list[str], timeout: int = 60, cwd: Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"cmd": cmd, "error": repr(exc)}


def import_report() -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name in ["torch", "guidellm"]:
        try:
            module = __import__(name)
            try:
                version = metadata.version(name)
            except Exception:
                version = getattr(module, "__version__", "unknown")
            report[name] = {
                "version": version,
                "file": getattr(module, "__file__", None),
            }
        except Exception as exc:  # noqa: BLE001
            report[name] = {"error": repr(exc)}
    try:
        vllm_probe = run(
            [
                sys.executable,
                "-c",
                "import pathlib, vllm; print(getattr(vllm, '__version__', 'unknown')); print(pathlib.Path(vllm.__file__).resolve() if getattr(vllm, '__file__', None) else None)",
            ],
            cwd=EVAL_ROOT,
        )
        lines = (vllm_probe.get("stdout") or "").splitlines()
        report["vllm"] = {
            "version": lines[0] if lines else "unknown",
            "file": lines[1] if len(lines) > 1 else None,
            "probe": vllm_probe,
        }
    except Exception as exc:  # noqa: BLE001
        report["vllm"] = {"error": repr(exc)}
    try:
        import torch

        report["torch_cuda"] = {
            "cuda_version": torch.version.cuda,
            "is_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "name": torch.cuda.get_device_name(idx),
                    "total_memory": torch.cuda.get_device_properties(idx).total_memory,
                }
                for idx in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:  # noqa: BLE001
        report["torch_cuda"] = {"error": repr(exc)}
    return report


def vllm_help_probe() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        "--help",
    ]
    result = run(command, timeout=90, cwd=EVAL_ROOT)
    output = (result.get("stdout") or "") + "\n" + (result.get("stderr") or "")
    lower = output.lower()
    flags = {
        "command": command,
        "returncode": result.get("returncode"),
        "supports_max_num_batched_tokens": "--max-num-batched-tokens" in lower,
        "supports_max_num_seqs": "--max-num-seqs" in lower,
        "supports_chunked_prefill": "chunked-prefill" in lower,
        "supports_no_async_scheduling": "--no-async-scheduling" in lower,
        "supports_speculative_config": "--speculative-config" in lower,
    }
    if all(value for key, value in flags.items() if key.startswith("supports_")):
        flags["source"] = "vllm serve --help"
        return flags

    # In sandboxed Codex commands vLLM may fail while constructing the parser
    # because no CUDA device is visible. Fall back to source inspection so the
    # report still captures whether this local checkout exposes the needed args.
    source_text = ""
    for path in [
        REPO_ROOT / "vllm/vllm/engine/arg_utils.py",
        REPO_ROOT / "vllm/vllm/entrypoints/openai/cli_args.py",
    ]:
        if path.exists():
            source_text += path.read_text(encoding="utf-8", errors="ignore") + "\n"
    source_lower = source_text.lower()
    for flag, key in [
        ("--max-num-batched-tokens", "supports_max_num_batched_tokens"),
        ("--max-num-seqs", "supports_max_num_seqs"),
        ("chunked-prefill", "supports_chunked_prefill"),
        ("--speculative-config", "supports_speculative_config"),
    ]:
        if not flags[key]:
            flags[key] = flag in source_lower
    if not flags["supports_no_async_scheduling"]:
        flags["supports_no_async_scheduling"] = (
            "--no-async-scheduling" in source_lower
            or "--async-scheduling" in source_lower
        )
    flags["source"] = "source_fallback_after_help_probe"
    flags["help_error_tail"] = output[-2000:]
    return flags


def speculative_support_report() -> dict[str, Any]:
    code = textwrap.dedent(
        """
        import json
        from vllm.transformers_utils.configs.speculators import algos
        from vllm.config.speculative import SpeculativeConfig

        fields = getattr(SpeculativeConfig, "model_fields", None)
        if fields is None:
            fields = getattr(SpeculativeConfig, "__fields__", {})
        print(json.dumps({
            "supported_speculator_types": sorted(algos.SUPPORTED_SPECULATORS_TYPES.keys()),
            "supports_eagle3": "eagle3" in algos.SUPPORTED_SPECULATORS_TYPES,
            "supports_peagle": "peagle" in algos.SUPPORTED_SPECULATORS_TYPES,
            "speculative_config_fields": sorted(fields.keys()),
        }))
        """
    )
    result = run([sys.executable, "-c", code], timeout=60, cwd=EVAL_ROOT)
    report: dict[str, Any] = {"command": result.get("cmd", []), "returncode": result.get("returncode")}
    if result.get("returncode") == 0 and result.get("stdout"):
        try:
            payload = json.loads(result["stdout"].splitlines()[-1])
            report.update(payload)
        except Exception as exc:  # noqa: BLE001
            report["error"] = repr(exc)
    else:
        report["error"] = result.get("stderr")
    return report


def path_report() -> dict[str, Any]:
    model_paths = {
        "qwen3_eagle3_speculator": os.environ.get(
            "QWEN3_8B_EAGLE3_SPECULATOR_MODEL",
            os.environ.get(
                "EAGLE3_SPECULATOR_MODEL",
                str(MODELS_ROOT / "qwen3-8b-eagle3-speculator"),
            ),
        ),
        "qwen3_peagle_speculator": os.environ.get(
            "QWEN3_8B_PEAGLE_SPECULATOR_MODEL",
            os.environ.get(
                "PEAGLE_SPECULATOR_MODEL",
                str(MODELS_ROOT / "qwen3-8b-peagle-speculator"),
            ),
        ),
        "llama3_1_8b_eagle3_speculator": os.environ.get(
            "LLAMA3_1_8B_EAGLE3_SPECULATOR_MODEL",
            os.environ.get(
                "LLAMA_EAGLE3_SPECULATOR_MODEL",
                str(MODELS_ROOT / "llama-3.1-8b-eagle3-speculator"),
            ),
        ),
    }
    dataset_paths = {
        "math_reasoning": os.environ.get(
            "MATH_REASONING_DATASET",
            os.environ.get(
                "MATH_DATASET",
                str(EVAL_DATA_ROOT / "math_reasoning.jsonl"),
            ),
        ),
        "mt_bench": os.environ.get(
            "MTBENCH_DATASET",
            str(EVAL_DATA_ROOT / "mt_bench.jsonl"),
        ),
    }
    paths = {
        **{name: Path(value) for name, value in model_paths.items()},
        **{name: Path(value) for name, value in dataset_paths.items()},
        "vendored_vllm": REPO_ROOT / "vllm",
    }
    return {
        name: {"path": str(path), "exists": path.exists()}
        for name, path in paths.items()
    }


def missing_paths(report: dict[str, Any]) -> list[str]:
    miss: list[str] = []
    for name, info in report.items():
        if not info.get("exists"):
            miss.append(name)
    return miss


def collect(strict: bool = False) -> dict[str, Any]:
    nvidia = ["nvidia-smi"] if shutil.which("nvidia-smi") else ["bash", "-lc", "command -v nvidia-smi"]
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "date": datetime.now().isoformat(),
        "repo_root": str(REPO_ROOT),
        "python": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "git": {
            "branch": run(["git", "branch", "--show-current"]),
            "commit": run(["git", "rev-parse", "HEAD"]),
            "status": run(["git", "status", "--short", "--branch"]),
            "diff_stat": run(["git", "diff", "--stat"]),
        },
        "imports": import_report(),
        "paths": path_report(),
        "nvidia_smi": run(nvidia, timeout=30),
        "vllm_cli_support": vllm_help_probe(),
        "speculative_config": speculative_support_report(),
        "analysis": {
            "missing_paths": missing_paths(path_report()),
        },
    }


def validate_or_exit(report: dict[str, Any], strict: bool = False) -> None:
    if not strict:
        return
    missing = report.get("analysis", {}).get("missing_paths", [])
    errors = []
    if missing:
        errors.append("missing required paths: " + ", ".join(missing))
    speculative = report.get("speculative_config", {})
    if not speculative.get("supports_eagle3"):
        errors.append("vLLM EAGLE3 speculative config is unavailable")
    cli = report.get("vllm_cli_support", {})
    for key in [
        "supports_max_num_batched_tokens",
        "supports_max_num_seqs",
        "supports_no_async_scheduling",
        "supports_speculative_config",
    ]:
        if not cli.get(key):
            errors.append(f"vLLM CLI missing {key}")
    if errors:
        raise SystemExit("Environment validation failed: " + "; ".join(errors))


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = ["# SpecLink-CV Environment Report", ""]
    lines.append(f"- hostname: {report['hostname']}")
    lines.append(f"- date: {report['date']}")
    lines.append(f"- repo root: `{report['repo_root']}`")
    lines.append(f"- python: `{report['python']}`")
    lines.append(f"- conda env: `{report.get('conda_env')}`")
    lines.append(f"- git branch: `{report['git']['branch'].get('stdout', '')}`")
    lines.append(f"- git commit: `{report['git']['commit'].get('stdout', '')}`")
    lines.append("")
    lines.append("## Imports")
    for name, info in report["imports"].items():
        lines.append(f"- {name}: `{info}`")
    lines.append("")
    lines.append("## Paths")
    for name, info in report["paths"].items():
        lines.append(f"- {name}: `{info['path']}` exists={info['exists']}")
    analysis = report.get("analysis", {})
    missing = analysis.get("missing_paths", [])
    lines.extend(["", "## Required Asset Check", f"- missing: {', '.join(missing) if missing else 'none'}"])
    lines.append("")
    lines.append("## vLLM CLI Flags")
    vllm_help = report.get("vllm_cli_support", {})
    for key, value in vllm_help.items():
        if key == "command":
            continue
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Speculative Config")
    for key, value in report.get("speculative_config", {}).items():
        if key == "command":
            continue
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## nvidia-smi")
    lines.append("```text")
    lines.append(report["nvidia_smi"].get("stdout") or report["nvidia_smi"].get("stderr") or str(report["nvidia_smi"]))
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = collect(strict=True)
    validate_or_exit(report, strict=True)
    (args.output_dir / "env_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(report, args.output_dir / "env_report.md")
    print(f"[INFO] Wrote {args.output_dir / 'env_report.md'}")


if __name__ == "__main__":
    main()
