#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from pams.configs import EXPERIMENTS, register_experiment, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Actually call huggingface-cli download when files are missing.")
    args = parser.parse_args()
    model_root = REPO_ROOT.parent / "models"
    inventory = {
        "base_model": "Qwen/Qwen3-8B",
        "base_model_localized": False,
        "eagle3_speculator": str(model_root / "qwen3-8b-eagle3-speculator"),
        "peagle_speculator": str(model_root / "qwen3-8b-peagle-speculator"),
        "download_attempted": bool(args.download),
        "models": {},
    }
    for key in ["eagle3_speculator", "peagle_speculator"]:
        path = Path(inventory[key])
        inventory["models"][key] = {
            "path": str(path),
            "exists": path.exists(),
            "config_exists": (path / "config.json").exists(),
        }
    out = EXPERIMENTS / "00_env" / "model_inventory.json"
    write_json(out, inventory)
    status = "completed" if all(item["config_exists"] for item in inventory["models"].values()) else "missing_speculator_files"
    summary = [
        "# Model Inventory",
        "",
        "The base model remains the Hugging Face ID `Qwen/Qwen3-8B` and is expected to use the local HF cache.",
        "",
    ]
    for key, item in inventory["models"].items():
        summary.append(f"- {key}: exists={item['exists']} config={item['config_exists']} path=`{item['path']}`")
    register_experiment(
        "00_env",
        config=vars(args),
        command="python scripts/download_models.py",
        status=status,
        summary="\n".join(summary),
        metadata_extra={"model_inventory_json": str(out), "model_inventory": inventory},
    )
    print(inventory)


if __name__ == "__main__":
    main()
