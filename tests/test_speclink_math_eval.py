from speculators.speclink.math_eval import (
    extract_final_answer,
    flexible_answer_equal,
)


def test_extract_hash_answer():
    assert extract_final_answer("work\n#### 42") == "42"


def test_extract_answer_is():
    assert extract_final_answer("The answer is 42.") == "42"


def test_normalizes_dollar_and_commas():
    assert flexible_answer_equal(extract_final_answer("#### $1,234"), "1234")


def test_extract_negative_decimal():
    assert extract_final_answer("Final answer: -3.5.") == "-3.5"


def test_extract_fraction():
    assert extract_final_answer("The answer is 1/2.") == "1/2"


def test_fraction_numeric_equivalence():
    assert flexible_answer_equal("1/2", "0.5")


def test_decimal_fraction_numeric_equivalence():
    assert flexible_answer_equal("1.5/3", "0.5")


def test_invalid_answer():
    assert extract_final_answer("No numeric final answer here.") is None
