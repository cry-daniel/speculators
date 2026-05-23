from __future__ import annotations


def triton_available() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except Exception:
        return False


def implementation_status() -> dict:
    if not triton_available():
        return {
            "available": False,
            "label": "reference_only",
            "reason": "triton import failed; using torch gather+dense reference path",
        }
    return {
        "available": False,
        "label": "reference_only",
        "reason": "custom Triton kernel was not implemented; sparse_attention_ref is used for measured reference overhead",
    }

