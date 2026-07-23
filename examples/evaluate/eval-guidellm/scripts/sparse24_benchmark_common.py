"""Shared deterministic data, CUDA Graph, timing, and GPU-state helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch

from speculators.speclink import (
    TokenRoute,
    assert_24_weight,
    route_from_indices,
)


@dataclass(frozen=True, slots=True)
class ShapeCase:
    model: str
    projection: str
    m: int
    k: int
    n: int

    @property
    def key(self) -> str:
        return f"{self.model}__{self.projection}__m{self.m}"


@dataclass(slots=True)
class CapturedGraph:
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    unroll: int = 1


@dataclass(slots=True)
class MultiStreamResources:
    capture_stream: Any
    dense_stream: Any
    sparse_stream: Any
    fork_event: Any
    dense_done_event: Any
    sparse_done_event: Any


def stable_seed(base_seed: int, *parts: object) -> int:
    digest = hashlib.sha256(
        (str(base_seed) + "|" + "|".join(map(str, parts))).encode()
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def fraction_text(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def route_key(rows: int, fraction: Fraction) -> str:
    return f"m{rows}__f{fraction.numerator}_{fraction.denominator}"


def generate_routes(
    m_values: Iterable[int], fractions: Iterable[Fraction], seed: int
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for rows in sorted(set(m_values)):
        for fraction in fractions:
            count = fraction * rows
            if count.denominator != 1:
                raise ValueError(f"{fraction} does not produce an integer for M={rows}")
            generator = torch.Generator(device="cpu")
            route_seed = stable_seed(seed, "route", rows, fraction_text(fraction))
            generator.manual_seed(route_seed)
            dense = torch.randperm(rows, generator=generator)[: int(count)].sort().values
            mask = torch.zeros(rows, dtype=torch.bool)
            mask[dense] = True
            sparse = (~mask).nonzero(as_tuple=False).flatten()
            dense_list = list(map(int, dense.tolist()))
            records[route_key(rows, fraction)] = {
                "rows": rows,
                "fraction": fraction_text(fraction),
                "dense_count": len(dense_list),
                "sparse_count": int(sparse.numel()),
                "seed": route_seed,
                "dense_indices": dense_list,
                "sparse_indices": list(map(int, sparse.tolist())),
                "dense_indices_sha256": hashlib.sha256(
                    json.dumps(dense_list, separators=(",", ":")).encode()
                ).hexdigest(),
            }
    return {"base_seed": seed, "routes": records}


def route_from_record(record: dict[str, Any], device: torch.device) -> TokenRoute:
    rows = int(record["rows"])
    dense = torch.tensor(record["dense_indices"], device=device, dtype=torch.int64)
    sparse = torch.tensor(record["sparse_indices"], device=device, dtype=torch.int64)
    mask = torch.zeros(rows, device=device, dtype=torch.bool)
    mask[dense] = True
    return route_from_indices(rows, dense, sparse, dense_mask=mask, validate=True)


def make_synthetic_weight(
    case: ShapeCase, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(stable_seed(seed, "weight", case.model, case.projection))
    scale = 1.0 / math.sqrt(case.k)
    weight = torch.randn(
        (case.n, case.k), device=device, dtype=torch.bfloat16, generator=generator
    ).mul_(scale)
    weight.masked_fill_(weight.eq(0), scale)
    weight24 = torch.zeros_like(weight)
    for start in range(0, case.n, 512):
        stop = min(case.n, start + 512)
        source = weight[start:stop].view(stop - start, case.k // 4, 4)
        target = weight24[start:stop].view(stop - start, case.k // 4, 4)
        keep = source.abs().topk(2, dim=-1).indices
        target.scatter_(-1, keep, source.gather(-1, keep))
    assert_24_weight(weight24)
    return weight.contiguous(), weight24.contiguous()


def make_input(
    case: ShapeCase, seed: int, device: torch.device, *, purpose: str
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(
        stable_seed(seed, purpose, case.model, case.projection, case.m)
    )
    return torch.randn(
        (case.m, case.k), device=device, dtype=torch.bfloat16, generator=generator
    ).contiguous()


def capture_graph(
    fn: Callable[[], torch.Tensor], *, warmup: int, unroll: int = 1
) -> CapturedGraph:
    if warmup <= 0 or unroll <= 0:
        raise ValueError("warmup and unroll must be positive")
    output: torch.Tensor | None = None
    for _ in range(warmup):
        output = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(unroll):
            output = fn()
    assert output is not None
    graph.replay()
    torch.cuda.synchronize()
    return CapturedGraph(graph, output, unroll)


def create_multistream_resources(
    device: torch.device,
    *,
    dense_priority: int = 0,
    sparse_priority: int = 0,
) -> MultiStreamResources:
    """Create graph-safe streams; lower numeric priorities run first."""

    capture = torch.cuda.Stream(device=device)
    dense = torch.cuda.Stream(device=device, priority=dense_priority)
    sparse = torch.cuda.Stream(device=device, priority=sparse_priority)
    handles = {
        int(stream.cuda_stream)
        for stream in (capture, dense, sparse, torch.cuda.default_stream(device))
    }
    if len(handles) != 4:
        raise RuntimeError("capture/dense/sparse streams must be distinct and non-default")
    event = lambda: torch.cuda.Event(enable_timing=False, external=False)
    return MultiStreamResources(capture, dense, sparse, event(), event(), event())


def launch_two_branch_concurrent(
    dense_call: Callable[[], torch.Tensor],
    sparse_call: Callable[[], torch.Tensor],
    output: torch.Tensor,
    resources: MultiStreamResources,
    *,
    device: torch.device,
) -> torch.Tensor:
    origin = torch.cuda.current_stream(device)
    resources.fork_event.record(origin)
    resources.dense_stream.wait_event(resources.fork_event)
    resources.sparse_stream.wait_event(resources.fork_event)
    with torch.cuda.stream(resources.dense_stream):
        dense_result = dense_call()
    resources.dense_done_event.record(resources.dense_stream)
    with torch.cuda.stream(resources.sparse_stream):
        sparse_result = sparse_call()
    resources.sparse_done_event.record(resources.sparse_stream)
    origin.wait_event(resources.dense_done_event)
    origin.wait_event(resources.sparse_done_event)
    if dense_result.data_ptr() != output.data_ptr() or sparse_result.data_ptr() != output.data_ptr():
        raise RuntimeError("branch did not preserve the shared output")
    return output


def capture_multistream_graph(
    fn: Callable[[], torch.Tensor],
    resources: MultiStreamResources,
    *,
    warmup: int,
    device: torch.device,
) -> CapturedGraph:
    if warmup <= 0:
        raise ValueError("warmup must be positive")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    output: torch.Tensor | None = None
    with torch.cuda.graph(graph, stream=resources.capture_stream):
        output = fn()
    assert output is not None
    graph.replay()
    torch.cuda.synchronize(device)
    return CapturedGraph(graph, output)


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(map(float, values))
    if not ordered:
        raise ValueError("percentile requires samples")
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] if lower == upper else (
        ordered[lower] * (upper - position) + ordered[upper] * (position - lower)
    )


def summarize(values: Sequence[float]) -> dict[str, float]:
    return {
        "median_us": statistics.median(values),
        "p10_us": percentile(values, 0.1),
        "p90_us": percentile(values, 0.9),
        "min_us": min(values),
        "max_us": max(values),
        "mean_us": statistics.mean(values),
    }


def steady_graph_sample(captured: CapturedGraph, replays: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        captured.graph.replay()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) * 1000.0 / replays


def gpu_processes(device_index: int) -> list[str]:
    raw = subprocess.check_output(
        ["nvidia-smi", "-i", str(device_index),
         "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], text=True
    ).strip()
    return [line.strip() for line in raw.splitlines() if line.strip()]


def idle_state(device_index: int) -> dict[str, Any]:
    return {"device_index": device_index, "compute_processes": gpu_processes(device_index)}


def require_idle_gpu(device_index: int) -> dict[str, Any]:
    state = idle_state(device_index)
    if state["compute_processes"]:
        raise RuntimeError(f"GPU {device_index} is busy: {state['compute_processes']}")
    samples: list[int] = []
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        raw = subprocess.check_output(
            ["nvidia-smi", "-i", str(device_index),
             "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        samples.append(int(raw.splitlines()[0]))
        if len(samples) >= 3 and all(value < 5 for value in samples[-3:]):
            state["utilization_samples"] = samples
            return state
        time.sleep(0.1)
    raise RuntimeError(f"GPU {device_index} did not settle: {samples}")


def gpu_identity(device_index: int) -> dict[str, str]:
    raw = subprocess.check_output(
        ["nvidia-smi", "-i", str(device_index),
         "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    name, uuid, driver = (part.strip() for part in raw.split(",", maxsplit=2))
    return {"name": name, "uuid": uuid, "driver_version": driver}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty CSV")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


__all__ = [name for name in globals() if not name.startswith("_")]
