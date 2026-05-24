#!/usr/bin/env python3
"""Copy the first N JSONL records for bounded smoke benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    copied = 0
    with Path(args.input).open("r", encoding="utf-8") as src:
        with output.open("w", encoding="utf-8") as dst:
            for line in src:
                if not line.strip():
                    continue
                dst.write(line)
                copied += 1
                if copied >= args.limit:
                    break
    print(f"wrote {copied} rows to {output}")


if __name__ == "__main__":
    main()
