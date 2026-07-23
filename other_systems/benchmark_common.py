"""Shared adapters and formal timing helpers for external N:M systems."""

from __future__ import annotations

import csv
import gc
import math
import os
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.nn.functional as F

from speculators.speclink import TP1_FUSED_WEIGHT_SHAPES

from .nm import NMFormat, apply_nm_mask, parse_nm
from .runtime import (
    FlashLLMWeight,
    SparTAWeight,
    SpInferWeight,
    flash_llm_linear,
    prepare_flash_llm,
    prepare_sparta,
    prepare_spinfer,
    sparta_linear,
    spinfer_linear,
)


SYSTEMS = ("flash_llm", "spinfer", "sparta")
DEFAULT_MODELS = ("qwen3_8b", "llama3_1_8b")
PROJECTIONS = ("qkv", "o", "gate_up", "down")
M_VALUES = (512, 1024, 2048)
BATCH_SIZES = (64, 128, 256)
FORMATS = ("5:8", "3:4")
SPLIT_CANDIDATES = (1, 2, 4, 8)


@dataclass(frozen=True)
class PreparedSystem:
    system: str
    weight: FlashLLMWeight | SpInferWeight | SparTAWeight
    split_k: int = 1

    def linear(self, x: torch.Tensor) -> torch.Tensor:
        if self.system == "flash_llm":
            assert isinstance(self.weight, FlashLLMWeight)
            return flash_llm_linear(x, self.weight, split_k=self.split_k)
        if self.system == "spinfer":
            assert isinstance(self.weight, SpInferWeight)
            return spinfer_linear(x, self.weight, split_k=self.split_k)
        if self.system == "sparta":
            assert isinstance(self.weight, SparTAWeight)
            return sparta_linear(
                x, self.weight, residual_split_k=self.split_k
            )
        raise ValueError(self.system)

    def with_split(self, split_k: int) -> "PreparedSystem":
        return PreparedSystem(self.system, self.weight, split_k)


@dataclass(frozen=True)
class TimingSummary:
    median_us: float
    p10_us: float
    p90_us: float
    min_us: float
    max_us: float
    mean_us: float

    def as_dict(self) -> dict[str, float]:
        return {
            "median_us": self.median_us,
            "p10_us": self.p10_us,
            "p90_us": self.p90_us,
            "min_us": self.min_us,
            "max_us": self.max_us,
            "mean_us": self.mean_us,
        }


@dataclass
class Captured:
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor


def parse_csv(value: str, allowed: Sequence[str], label: str) -> tuple[str, ...]:
    selected = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(selected) - set(allowed))
    if not selected or unknown:
        raise ValueError(f"invalid {label}: {unknown or value!r}")
    return selected


def assert_gpu_idle(device_index: int) -> None:
    """Fail before CUDA initialization if another compute process is present."""

    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(device_index),
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if processes:
        raise RuntimeError(
            "formal benchmark requires an idle GPU; found: "
            + "; ".join(processes)
        )


def environment_report(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "pid": os.getpid(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": properties.total_memory,
    }


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(map(float, values))
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return (
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def summarize(values: Sequence[float]) -> TimingSummary:
    return TimingSummary(
        median_us=statistics.median(values),
        p10_us=percentile(values, 0.1),
        p90_us=percentile(values, 0.9),
        min_us=min(values),
        max_us=max(values),
        mean_us=statistics.mean(values),
    )


def make_nm_weight(
    model: str,
    projection: str,
    fmt: str | NMFormat,
    *,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    n, k = TP1_FUSED_WEIGHT_SHAPES[model][projection]
    generator = torch.Generator(device=device)
    generator.manual_seed(
        _stable_seed(seed, model, projection, parse_nm(fmt).label)
    )
    scale = 1.0 / math.sqrt(k)
    dense = torch.randn(
        (n, k), dtype=torch.bfloat16, device=device, generator=generator
    ).mul_(scale)
    return apply_nm_mask(dense, fmt)


def make_input(
    model: str,
    projection: str,
    m: int,
    *,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    _, k = TP1_FUSED_WEIGHT_SHAPES[model][projection]
    generator = torch.Generator(device=device)
    generator.manual_seed(_stable_seed(seed, model, projection, m, "input"))
    return torch.randn(
        (m, k), dtype=torch.bfloat16, device=device, generator=generator
    ).contiguous()


def _stable_seed(base: int, *parts: object) -> int:
    import hashlib

    digest = hashlib.sha256(
        "|".join((str(base), *(str(part) for part in parts))).encode()
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def prepare_system(
    system: str,
    weight_nm: torch.Tensor,
    fmt: str | NMFormat,
) -> PreparedSystem:
    if system == "flash_llm":
        weight = prepare_flash_llm(weight_nm)
    elif system == "spinfer":
        weight = prepare_spinfer(weight_nm)
    elif system == "sparta":
        weight = prepare_sparta(weight_nm, fmt)
    else:
        raise ValueError(system)
    return PreparedSystem(system, weight)


def capture(fn: Callable[[], torch.Tensor], warmup: int = 5) -> Captured:
    output: torch.Tensor | None = None
    for _ in range(warmup):
        output = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    assert output is not None
    graph.replay()
    torch.cuda.synchronize()
    return Captured(graph, output)


def select_split(
    prepared: PreparedSystem,
    x: torch.Tensor,
    candidates: Iterable[int] = SPLIT_CANDIDATES,
    *,
    warmup: int = 5,
    repeats: int = 20,
) -> tuple[PreparedSystem, list[dict[str, float | int]]]:
    valid = [
        split
        for split in candidates
        if split >= 1 and split <= x.shape[1] // 64
    ]
    samples: list[dict[str, float | int]] = []
    best_split = valid[0]
    best_latency = float("inf")
    for split in valid:
        candidate = prepared.with_split(split)
        captured = capture(lambda candidate=candidate: candidate.linear(x), warmup)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            captured.graph.replay()
        end.record()
        end.synchronize()
        latency = 1000.0 * float(start.elapsed_time(end)) / repeats
        samples.append({"split_k": split, "latency_us": latency})
        if latency < best_latency:
            best_latency = latency
            best_split = split
    return prepared.with_split(best_split), samples


def formal_measure(
    captured: Captured,
    eviction: torch.Tensor,
    *,
    warmup: int,
    trials: int,
    replays: int,
) -> tuple[TimingSummary, list[float]]:
    for _ in range(warmup):
        captured.graph.replay()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(trials):
        # Each independent remeasurement starts after a 256 MiB cache eviction.
        # The following large replay interval amortizes event and launch noise.
        eviction.add_(1)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            captured.graph.replay()
        end.record()
        end.synchronize()
        values.append(1000.0 * float(start.elapsed_time(end)) / replays)
    return summarize(values), values


def correctness(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 0.2,
    rtol: float = 0.1,
) -> dict[str, float | bool]:
    difference = (actual.float() - expected.float()).abs()
    return {
        "correct": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def release(*objects: object) -> None:
    del objects
    gc.collect()
    torch.cuda.empty_cache()


def dense_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(x, weight)
