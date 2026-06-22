import collections
import re
import string

MAX_CONTEXT_CHARS = 10_000


def _truncate_context(context: str) -> str:
    if len(context) <= MAX_CONTEXT_CHARS:
        return context
    marker = "\n\n[...]\n\n"
    head_chars = 7_000
    tail_chars = MAX_CONTEXT_CHARS - head_chars - len(marker)
    return context[:head_chars].rstrip() + marker + context[-tail_chars:].lstrip()


def doc_to_text(doc) -> str:
    context = _truncate_context(str(doc.get("context", "")).strip())
    question = str(doc.get("question", "")).strip()
    answer_prefix = str(doc.get("answer_prefix") or "Answer:").strip()
    return f"{context}\n\n{question}\n{answer_prefix}"


def _normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def _f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = _normalize_answer(prediction).split()
    ground_truth_tokens = _normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    same = sum(common.values())
    if not prediction_tokens or not ground_truth_tokens:
        return float(prediction_tokens == ground_truth_tokens)
    if same == 0:
        return 0.0
    precision = same / len(prediction_tokens)
    recall = same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def get_qa_f1_with_score(doc, results) -> dict[str, float]:
    prediction = str(results[0]).strip() if results else ""
    answers = doc.get("answers") or []
    if isinstance(answers, str):
        answers = [answers]
    score = max((_f1_score(prediction, str(answer)) for answer in answers), default=0.0)
    return {"score": score, "qa_f1_score": score}
