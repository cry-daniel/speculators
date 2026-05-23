#!/usr/bin/env python3
"""Create a P-EAGLE checkpoint that lets vLLM share target weights."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import safe_open, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove selected duplicate P-EAGLE weights so vLLM shares them "
            "from the target model at load time."
        )
    )
    parser.add_argument("--peagle-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--drop-embed-tokens", action="store_true")
    parser.add_argument("--drop-lm-head", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    peagle_dir = args.peagle_dir
    output_dir = args.output_dir
    model_path = peagle_dir / "model.safetensors"
    config_path = peagle_dir / "config.json"

    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not args.drop_embed_tokens and not args.drop_lm_head:
        raise ValueError("Select at least one weight group to drop.")

    output_dir.mkdir(parents=True, exist_ok=True)

    dropped: list[str] = []
    tensors = {}
    with safe_open(model_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if args.drop_embed_tokens and key == "embed_tokens.weight":
                dropped.append(key)
                continue
            if args.drop_lm_head and key == "lm_head.weight":
                dropped.append(key)
                continue
            tensors[key] = handle.get_tensor(key)

    save_file(
        tensors,
        output_dir / "model.safetensors",
        metadata={"dropped_for_target_sharing": ",".join(dropped)},
    )

    config = json.loads(config_path.read_text())
    config["shared_target_weight_note"] = (
        "Selected P-EAGLE weights were removed so vLLM shares target weights."
    )
    config["shared_target_weight_dropped"] = dropped
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    for filename in ("config.py", ".gitattributes", "val_metrics.json"):
        src = peagle_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)

    print(f"Wrote {output_dir}")
    print("dropped=" + ",".join(dropped))


if __name__ == "__main__":
    main()
