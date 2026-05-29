#!/usr/bin/env python3
"""Unit checks for GuideLLM math answer extraction."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_SCRIPT = (
    REPO_ROOT
    / "examples/evaluate/eval-guidellm/scripts/run_speclink_cv_guidellm_matrix.py"
)


def load_matrix_module():
    spec = importlib.util.spec_from_file_location("speclink_cv_matrix", MATRIX_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load matrix module: {MATRIX_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = load_matrix_module()


def test_final_equation_after_so_uses_result_not_first_operand() -> None:
    text = (
        "So, the total number of utensils Jonathan has remaining is "
        "24 (measuring cups) + 10 (measuring spoons) = 34."
    )
    assert M.extract_predicted_answer(text) == "34"


def test_therefore_sentence_uses_last_number() -> None:
    text = (
        "Therefore, the cost is 12 dollars for the first box and "
        "8 dollars for the second, so the answer is 20."
    )
    assert M.extract_predicted_answer(text) == "20"


def test_boxed_answer_preferred() -> None:
    assert M.extract_predicted_answer("Work... \\boxed{1,234}") == "1234"


def test_hash_answer_preferred() -> None:
    assert M.extract_predicted_answer("#### 56") == "56"


def main() -> int:
    test_final_equation_after_so_uses_result_not_first_operand()
    test_therefore_sentence_uses_last_number()
    test_boxed_answer_preferred()
    test_hash_answer_preferred()
    print("[PASS] test_math_answer_extraction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
