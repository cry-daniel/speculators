from __future__ import annotations

import re
from typing import Any


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "").replace("$", "")
    text = text.rstrip(".")
    if re.fullmatch(r"[-+]?\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def extract_final_answer(text: str | None) -> str | None:
    if not text:
        return None
    if "####" in text:
        lines = text.rsplit("####", 1)[-1].strip().splitlines()
        if lines:
            normalized = normalize_number(lines[0])
            if normalized:
                return normalized
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        normalized = normalize_number(boxed[-1])
        if normalized:
            return normalized
    numbers = re.findall(r"[-+]?\$?\d[\d,]*(?:\.\d+)?", text)
    if numbers:
        return normalize_number(numbers[-1])
    return None


def _first_text_field(doc: dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = doc.get(name)
        if isinstance(value, list):
            value = value[0] if value else ""
        if value is not None:
            text = str(value)
            if text:
                return text
    return ""


def doc_to_text(doc: dict[str, Any]) -> str:
    prompt = _first_text_field(doc, ("prompt", "question", "input"))
    return f"Question:\n{prompt}\n\nAnswer:"


def doc_to_target(doc: dict[str, Any]) -> str:
    target = _first_text_field(doc, ("answer", "output", "response", "target", "reference"))
    return extract_final_answer(target) or target


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    prediction = extract_final_answer(results[0] if results else "")
    gold = doc_to_target(doc)
    return {"exact_match": 1.0 if prediction is not None and prediction == gold else 0.0}
