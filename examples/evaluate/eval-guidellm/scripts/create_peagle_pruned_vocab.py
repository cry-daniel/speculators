#!/usr/bin/env python3
"""Create a P-EAGLE checkpoint with a smaller draft vocabulary."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file


def load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    return tensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prune a full-vocabulary P-EAGLE lm_head to the target-token subset "
            "defined by another checkpoint's d2t offset mapping."
        )
    )
    parser.add_argument("--peagle-dir", required=True, type=Path)
    parser.add_argument("--mapping-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--metadata-note",
        default="P-EAGLE lm_head pruned with EAGLE3 d2t mapping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    peagle_dir = args.peagle_dir
    mapping_dir = args.mapping_dir
    output_dir = args.output_dir

    peagle_model = peagle_dir / "model.safetensors"
    mapping_model = mapping_dir / "model.safetensors"
    peagle_config = peagle_dir / "config.json"

    if not peagle_model.is_file():
        raise FileNotFoundError(peagle_model)
    if not mapping_model.is_file():
        raise FileNotFoundError(mapping_model)
    if not peagle_config.is_file():
        raise FileNotFoundError(peagle_config)

    output_dir.mkdir(parents=True, exist_ok=True)

    peagle_tensors = load_safetensors(peagle_model)
    mapping_tensors = load_safetensors(mapping_model)
    if "lm_head.weight" not in peagle_tensors:
        raise KeyError("P-EAGLE checkpoint does not contain lm_head.weight")
    if "d2t" not in mapping_tensors:
        raise KeyError("mapping checkpoint does not contain d2t")

    d2t = mapping_tensors["d2t"].to(dtype=torch.long)
    draft_vocab_size = int(d2t.numel())
    selected_target_ids = torch.arange(draft_vocab_size, dtype=torch.long) + d2t

    lm_head = peagle_tensors["lm_head.weight"]
    if int(selected_target_ids.max()) >= lm_head.shape[0]:
        raise ValueError("d2t mapping selects token IDs outside P-EAGLE lm_head")

    pruned_tensors = dict(peagle_tensors)
    pruned_tensors["lm_head.weight"] = lm_head.index_select(0, selected_target_ids)
    pruned_tensors["d2t"] = d2t
    if "t2d" in mapping_tensors:
        pruned_tensors["t2d"] = mapping_tensors["t2d"].to(dtype=torch.bool)

    save_file(
        pruned_tensors,
        output_dir / "model.safetensors",
        metadata={"note": args.metadata_note},
    )

    config = json.loads(peagle_config.read_text())
    config["draft_vocab_size"] = draft_vocab_size
    config["pruned_draft_vocab_source"] = str(mapping_dir)
    config["pruned_draft_vocab_size"] = draft_vocab_size
    config["pruned_draft_vocab_note"] = args.metadata_note
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    for filename in ("config.py", ".gitattributes", "val_metrics.json"):
        src = peagle_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)

    print(f"Wrote {output_dir}")
    print(f"draft_vocab_size={draft_vocab_size}")
    print(f"lm_head.weight={tuple(pruned_tensors['lm_head.weight'].shape)}")


if __name__ == "__main__":
    main()
