"""Math-answer extraction and dataset helpers for SpecLink experiments."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

PROMPT_FIELDS = ("prompt", "question", "instruction", "input", "messages")
REFERENCE_FIELDS = (
    "answer",
    "response",
    "output",
    "target",
    "reference",
    "ground_truth",
)

NUMBER_RE = re.compile(
    r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?"
)
HASH_ANSWER_RE = re.compile(r"####\s*([^\n\r]+)")
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
ANSWER_IS_RE = re.compile(r"(?i)(?:final\s+answer|answer)\s*(?:is|:)?\s*([^\n\r]+)")


@dataclass(frozen=True)
class DatasetRecord:
    """A normalized math benchmark record."""

    id: str
    prompt: str
    reference_raw: str
    raw: dict[str, Any]


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_present(record: dict[str, Any], fields: Sequence[str]) -> tuple[str, Any] | None:
    for field in fields:
        if field in record and record[field] not in (None, ""):
            return field, record[field]
    return None


def messages_to_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return str(messages)
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            parts.append(str(message))
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    text_parts.append(str(item.get("text", item)))
                else:
                    text_parts.append(str(item))
            content = " ".join(text_parts)
        parts.append(str(content))
    return "\n".join(part for part in parts if part)


def coerce_reference(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return ""
        return str(value[0])
    if isinstance(value, dict):
        for key in REFERENCE_FIELDS:
            if key in value:
                return coerce_reference(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def normalize_record(record: dict[str, Any], index: int) -> DatasetRecord:
    prompt_item = first_present(record, PROMPT_FIELDS)
    reference_item = first_present(record, REFERENCE_FIELDS)
    if prompt_item is None:
        raise ValueError(f"record {index} has no prompt-like field")
    if reference_item is None:
        raise ValueError(f"record {index} has no reference-like field")

    prompt_field, prompt_value = prompt_item
    if prompt_field == "messages":
        prompt = messages_to_text(prompt_value)
    else:
        prompt = str(prompt_value)

    record_id = record.get("id", record.get("question_id", index))
    return DatasetRecord(
        id=str(record_id),
        prompt=prompt,
        reference_raw=coerce_reference(reference_item[1]),
        raw=record,
    )


def load_dataset(path: str | Path, limit: int | None = None) -> list[DatasetRecord]:
    return [
        normalize_record(record, index)
        for index, record in enumerate(read_jsonl(path, limit=limit))
    ]


def build_math_prompt(problem: str) -> str:
    return (
        "Solve the following math problem. Show reasoning briefly, and put the "
        "final answer in the format #### <answer>.\n\n"
        f"{problem}"
    )


def normalize_answer(answer: str | None) -> str:
    if answer is None:
        return ""
    text = str(answer).strip()
    text = text.replace("\\,", "")
    text = text.replace("$", "")
    text = text.replace(",", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\(", "").replace("\\)", "")
    boxed = BOXED_RE.search(text)
    if boxed:
        text = boxed.group(1)
    text = text.strip().strip("`*_ ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .;:,\n\t")
    if text.endswith("%"):
        text = text[:-1].strip()
    text = re.sub(r"\s*/\s*", "/", text)
    return text


def _first_number(text: str) -> str | None:
    match = NUMBER_RE.search(text)
    return match.group(0) if match else None


def _last_number(text: str) -> str | None:
    matches = NUMBER_RE.findall(text)
    return matches[-1] if matches else None


def extract_final_answer(text: str | None) -> str | None:
    if not text:
        return None
    source = str(text)

    match = HASH_ANSWER_RE.search(source)
    if match:
        answer = _first_number(match.group(1)) or match.group(1)
        return normalize_answer(answer)

    boxed = BOXED_RE.findall(source)
    if boxed:
        answer = _first_number(boxed[-1]) or boxed[-1]
        return normalize_answer(answer)

    answer_matches = ANSWER_IS_RE.findall(source)
    if answer_matches:
        answer = _first_number(answer_matches[-1])
        if answer is not None:
            return normalize_answer(answer)

    answer = _last_number(source)
    if answer is None:
        return None
    return normalize_answer(answer)


def _to_number(value: str) -> Fraction | Decimal | None:
    normalized = normalize_answer(value)
    if not normalized:
        return None
    try:
        if "/" in normalized:
            numerator, denominator = normalized.split("/", 1)
            return Fraction(numerator) / Fraction(denominator)
        return Decimal(normalized)
    except (InvalidOperation, ValueError, ZeroDivisionError, TypeError):
        return None


def flexible_answer_equal(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return False
    left_norm = normalize_answer(left)
    right_norm = normalize_answer(right)
    if left_norm == right_norm:
        return True
    left_num = _to_number(left_norm)
    right_num = _to_number(right_norm)
    if left_num is None or right_num is None:
        return False
    return left_num == right_num


def strict_answer_equal(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return False
    return normalize_answer(left) == normalize_answer(right)


def output_equivalence(left: str | None, right: str | None) -> tuple[bool, bool]:
    if left is None or right is None:
        return False, False
    return left == right, normalize_answer(left) == normalize_answer(right)
