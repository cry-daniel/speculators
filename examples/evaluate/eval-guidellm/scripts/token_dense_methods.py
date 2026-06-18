#!/usr/bin/env python3
"""Shared method parsing and vLLM env setup for token-dense accuracy runs."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from run_structured_24_spec_quality import add_local_no_proxy


TOKEN_DENSE_METHODS = [f"token_dense_t{i:02d}" for i in range(11)]
DEFAULT_TOKEN_DENSE_METHODS = ",".join(TOKEN_DENSE_METHODS)


@dataclass(frozen=True)
class MethodConfig:
    label: str
    base_method: str
    policy: str = "all_sparse"
    keep_n: int = 0
    token_dense_threshold: float | None = None


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_method_config(label: str) -> MethodConfig:
    if label == "activation_aware":
        return MethodConfig(label=label, base_method="activation_aware")
    match = re.fullmatch(r"activation_aware_(keep_first_last|keep_first|keep_last)_(\d+)", label)
    if match:
        return MethodConfig(
            label=label,
            base_method="activation_aware",
            policy=match.group(1),
            keep_n=int(match.group(2)),
        )
    token_dense_match = re.fullmatch(r"token_dense_t(\d{1,3})", label)
    if token_dense_match:
        raw = int(token_dense_match.group(1))
        threshold = raw / 10.0 if raw <= 10 else raw / 100.0
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"unsupported token-dense threshold in {label!r}")
        return MethodConfig(
            label=label,
            base_method="token_dense",
            token_dense_threshold=threshold,
        )
    raise ValueError(
        f"unsupported method label {label!r}; use activation_aware or token_dense_t00-t10"
    )


def method_env(
    args: Any,
    *,
    model_label: str,
    method: MethodConfig,
    stats_path: Path,
) -> dict[str, str]:
    env = add_local_no_proxy(os.environ.copy())
    env["SPECLINK_TOKEN_DENSE_ENABLE"] = "0"
    if method.label == "dense":
        env["SPECLINK_STRUCTURED_24_ENABLE"] = "0"
        return env

    env.update(
        {
            "SPECLINK_STRUCTURED_24_ENABLE": "1",
            "SPECLINK_STRUCTURED_24_MODEL_LABEL": model_label,
            "SPECLINK_STRUCTURED_24_CALIBRATION_CACHE_ROOT": str(
                args.calibration_cache_root.resolve()
            ),
            "SPECLINK_STRUCTURED_24_POLICY": method.policy,
            "SPECLINK_STRUCTURED_24_KEEP_N": str(method.keep_n),
            "SPECLINK_STRUCTURED_24_STATS_PATH": str(stats_path.resolve()),
        }
    )
    if method.base_method == "token_dense":
        env.update(
            {
                "SPECLINK_TOKEN_DENSE_ENABLE": "1",
                "SPECLINK_TOKEN_DENSE_MODE": "high_confidence_dense",
                "SPECLINK_TOKEN_DENSE_THRESHOLD": str(
                    method.token_dense_threshold
                    if method.token_dense_threshold is not None
                    else 0.7
                ),
                "SPECLINK_TOKEN_DENSE_STATS_PATH": str(
                    (stats_path.parent / "token_dense_stats.jsonl").resolve()
                ),
                "SPECLINK_TOKEN_DENSE_STATS_DETAIL": "0",
            }
        )
    return env

