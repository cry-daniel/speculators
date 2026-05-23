from __future__ import annotations


def status() -> dict:
    return {
        "implemented": "synthetic_dense_labels_and_future_hf_runner",
        "online_dense_target_attention": False,
        "notes": [
            "Dense target labels in the offline smoke path are synthetic and are never used for online mask selection.",
            "Real target dense audits need GPU access and model execution.",
        ],
    }

