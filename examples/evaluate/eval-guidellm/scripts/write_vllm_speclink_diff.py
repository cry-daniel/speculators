#!/usr/bin/env python3
"""Write patches/vllm-speclink.diff from installed vLLM backups."""

from __future__ import annotations

import difflib
from pathlib import Path

import vllm


def baseline_backup(path: Path) -> Path:
    backups = sorted(path.parent.glob(f"{path.name}.bak-speclink-*"))
    if not backups:
        raise FileNotFoundError(f"no speclink backup for {path}")
    return backups[0]


def diff_pair(original: Path, patched: Path) -> list[str]:
    old_lines = original.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines = patched.read_text(encoding="utf-8").splitlines(keepends=True)
    rel = patched.relative_to(Path(vllm.__file__).resolve().parent)
    return list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/vllm/{rel}",
            tofile=f"b/vllm/{rel}",
        )
    )


def main() -> None:
    vllm_root = Path(vllm.__file__).resolve().parent
    paths = [
        vllm_root / "v1" / "core" / "sched" / "scheduler.py",
        vllm_root / "v1" / "engine" / "core.py",
        vllm_root / "v1" / "worker" / "gpu_model_runner.py",
    ]
    output = Path("patches/vllm-speclink.diff")
    output.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    for path in paths:
        chunks.extend(diff_pair(baseline_backup(path), path))
        if chunks and not chunks[-1].endswith("\n"):
            chunks[-1] += "\n"
        chunks.append("\n")
    output.write_text("".join(chunks), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
