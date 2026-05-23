from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency fallback
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
EXPERIMENTS = ROOT / "experiments"
REPORTS = ROOT / "reports"

EXPERIMENT_NAMES = [
    "00_env",
    "01_dense_baselines",
    "02_trace_collection",
    "03_union_problem",
    "04_acceptance_prior",
    "05_mask_planner_offline",
    "06_sparse_kernel_microbench",
    "07_vllm_integration_a_scheduler_hook",
    "08_vllm_integration_b_attention_patch",
    "09_vllm_integration_c_fallback_prefilter",
    "10_end2end",
    "11_correctness_quality",
    "12_ablations",
    "13_failures_oom",
]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_text(cmd: list[str], cwd: Path | None = None, timeout: int = 30) -> str:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return proc.stdout.strip()
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def git_commit_hash() -> str:
    out = run_text(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    if out.startswith("ERROR") or "fatal:" in out.lower():
        return "unknown"
    return out.splitlines()[-1].strip()


def import_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_jsonable(payload)
    if yaml is not None:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def to_jsonable(payload: Any) -> Any:
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {str(k): to_jsonable(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [to_jsonable(v) for v in payload]
    if isinstance(payload, set):
        return sorted(to_jsonable(v) for v in payload)
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def base_metadata(
    *,
    status: str,
    model_name: str = "Qwen/Qwen3-8B",
    model_dtype: str = "bfloat16",
    max_model_len: int | None = None,
    vllm_launch_args: list[str] | None = None,
    seed: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "git_commit_hash": git_commit_hash(),
        "vllm_version": import_version("vllm"),
        "pytorch_version": import_version("torch"),
        "cuda_version": _torch_cuda_version(),
        "gpu_name": os.environ.get("PAMS_GPU_NAME", "unknown"),
        "gpu_vram": os.environ.get("PAMS_GPU_VRAM", "unknown"),
        "model_name": model_name,
        "model_dtype": model_dtype,
        "max_model_len": max_model_len,
        "vllm_launch_args": vllm_launch_args or [],
        "seed": seed,
        "timestamp": utc_timestamp(),
        "run_status": status,
        "hostname": platform.node(),
    }
    if extra:
        meta.update(extra)
    return meta


def _torch_cuda_version() -> str:
    try:
        import torch

        return str(torch.version.cuda)
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def ensure_experiment_dir(name: str) -> Path:
    path = EXPERIMENTS / name
    for child in ["raw", "parsed", "figures"]:
        (path / child).mkdir(parents=True, exist_ok=True)
    return path


def register_experiment(
    name: str,
    *,
    config: dict[str, Any],
    command: str,
    status: str,
    summary: str,
    stdout: str = "",
    stderr: str = "",
    metadata_extra: dict[str, Any] | None = None,
    model_name: str = "Qwen/Qwen3-8B",
    model_dtype: str = "bfloat16",
    max_model_len: int | None = None,
    seed: int = 0,
    vllm_launch_args: list[str] | None = None,
) -> Path:
    path = ensure_experiment_dir(name)
    write_yaml(path / "config.yaml", config)
    (path / "command.sh").write_text(command.rstrip() + "\n", encoding="utf-8")
    (path / "stdout.log").write_text(stdout, encoding="utf-8")
    (path / "stderr.log").write_text(stderr, encoding="utf-8")
    (path / "summary.md").write_text(summary.rstrip() + "\n", encoding="utf-8")
    write_json(
        path / "metadata.json",
        base_metadata(
            status=status,
            model_name=model_name,
            model_dtype=model_dtype,
            max_model_len=max_model_len,
            vllm_launch_args=vllm_launch_args,
            seed=seed,
            extra=metadata_extra,
        ),
    )
    return path


def ensure_all_experiments_scaffolded() -> None:
    for name in EXPERIMENT_NAMES:
        path = ensure_experiment_dir(name)
        if not (path / "config.yaml").exists():
            register_experiment(
                name,
                config={"experiment": name, "registered": False},
                command="# Not run yet",
                status="not_run",
                summary=f"# {name}\n\nThis experiment has not been run yet.",
            )


def failure_record(kind: str, message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "kind": kind,
        "message": message,
        "context": context or {},
        "timestamp": utc_timestamp(),
    }
