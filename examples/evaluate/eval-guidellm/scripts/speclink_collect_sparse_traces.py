#!/usr/bin/env python3
"""Normalize live SpecLink traces into sparse challenge trace JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_KEYS = {"candidates", "accept_probs"}


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def normalize(row: dict[str, Any], source: str) -> dict[str, Any]:
    missing = sorted(key for key in REQUIRED_KEYS if key not in row)
    if missing:
        raise ValueError(f"trace row missing required keys: {missing}")
    candidate_source = row.get("candidate_source", "unknown")
    if candidate_source == "random":
        raise ValueError("random candidate traces are not valid experiment evidence")
    accept_probs = [float(x) for x in row["accept_probs"]]
    rho = row.get("rho") or compute_rho(accept_probs)
    risk = row.get("risk") or [
        rho_i * 4.0 * prob * (1.0 - prob)
        for rho_i, prob in zip(rho, accept_probs, strict=True)
    ]
    return {
        "request_id": row.get("request_id"),
        "step": row.get("step"),
        "prompt_len": row.get("prompt_len"),
        "generated_len": row.get("decode_len"),
        "num_spec_tokens": len(row["candidates"]),
        "block_size": row.get("block_size"),
        "draft_tokens": row.get("draft_tokens", []),
        "accepted_prefix_len": row.get("accepted_prefix_len"),
        "a_i": accept_probs,
        "rho_i": rho,
        "risk_i": risk,
        "candidates": row["candidates"],
        "candidate_source": candidate_source,
        "trace_source": source,
    }


def compute_rho(accept_probs: list[float]) -> list[float]:
    out: list[float] = []
    prefix = 1.0
    for prob in accept_probs:
        out.append(prefix)
        prefix *= max(0.0, min(1.0, prob))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-trace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    source = str(args.live_trace)
    rows = [normalize(row, source) for row in read_jsonl(Path(args.live_trace), args.limit)]
    write_jsonl(Path(args.out), rows)
    print(f"wrote {len(rows)} normalized sparse traces to {args.out}")


if __name__ == "__main__":
    main()
