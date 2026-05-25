# SPDX-License-Identifier: Apache-2.0
"""SpecLink-only timing helpers for local motivation experiments."""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch


_verify_collector: ContextVar[dict[str, list[Any]] | None] = ContextVar(
    "speclink_verify_collector", default=None
)


@dataclass
class _CpuInterval:
    start: float
    end: float


@dataclass
class _CudaInterval:
    start: torch.cuda.Event
    end: torch.cuda.Event


class _VerifyTimer:
    def __init__(self, collector: dict[str, list[Any]], name: str):
        self.collector = collector
        self.name = name
        self.interval: _CpuInterval | _CudaInterval | None = None

    def __enter__(self):
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self.interval = _CudaInterval(start, end)
        else:
            self.interval = _CpuInterval(time.perf_counter(), 0.0)
        return self

    def __exit__(self, exc_type, exc, tb):
        interval = self.interval
        if interval is None:
            return False
        if isinstance(interval, _CudaInterval):
            interval.end.record()
        else:
            interval.end = time.perf_counter()
        self.collector.setdefault(self.name, []).append(interval)
        return False


def verify_detail_enabled() -> bool:
    return (
        os.getenv("SPECLINK_BREAKDOWN", "0") == "1"
        and os.getenv("SPECLINK_BREAKDOWN_VERIFY_DETAIL", "0") == "1"
    )


def begin_verify_detail() -> tuple[dict[str, list[Any]] | None, Any]:
    if not verify_detail_enabled():
        return None, None
    collector: dict[str, list[Any]] = {}
    token = _verify_collector.set(collector)
    return collector, token


def end_verify_detail(token: Any) -> None:
    if token is not None:
        _verify_collector.reset(token)


def verify_timer(name: str):
    collector = _verify_collector.get()
    if collector is None:
        return nullcontext()
    return _VerifyTimer(collector, name)


def snapshot_verify_detail(collector: dict[str, list[Any]] | None) -> dict[str, float]:
    if not collector:
        return {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    out: dict[str, float] = {}
    for name, intervals in collector.items():
        total = 0.0
        for interval in intervals:
            if isinstance(interval, _CudaInterval):
                total += float(interval.start.elapsed_time(interval.end))
            else:
                total += (interval.end - interval.start) * 1000.0
        out[name] = total
    return out
