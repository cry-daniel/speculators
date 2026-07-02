#!/usr/bin/env python3
"""Compare token/logprob traces from SR24 replay JSON files.

The replay runner writes one JSON file per mode. This script is intentionally
offline: it does not launch vLLM. It aligns generated token ids by position,
finds the first divergence against a reference replay, and writes the local
logprob/top-token evidence needed to diagnose whether selective SR24 diverges
while dense/all-corrected remain aligned.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("input must be label:/path/to/replay.json")
    label, raw_path = value.split(":", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("input label cannot be empty")
    path = Path(raw_path).expanduser()
    return label, path


def load_replay(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("token_ids"), list):
        raise ValueError(f"{path} does not look like a replay JSON: missing token_ids")
    return data


def token_ids(data: dict[str, Any]) -> list[int]:
    return [int(item) for item in data.get("token_ids", [])]


def logprob_row(data: dict[str, Any], position: int) -> dict[str, Any]:
    rows = data.get("logprobs") or []
    if position < 0 or position >= len(rows):
        return {}
    row = rows[position]
    return row if isinstance(row, dict) else {}


def token_entries(data: dict[str, Any], position: int) -> dict[str, dict[str, Any]]:
    row = logprob_row(data, position)
    tokens = row.get("tokens")
    return tokens if isinstance(tokens, dict) else {}


def selected_logprob_info(
    data: dict[str, Any],
    position: int,
    token_id: int | None,
) -> dict[str, Any]:
    if token_id is None:
        return {}
    entries = token_entries(data, position)
    value = entries.get(str(int(token_id)))
    return value if isinstance(value, dict) else {}


def top_token_info(data: dict[str, Any], position: int) -> tuple[int | None, dict[str, Any]]:
    entries = token_entries(data, position)
    if not entries:
        return None, {}

    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[float, float]:
        _, info = item
        rank = info.get("rank")
        if rank is None:
            rank_key = float("inf")
        else:
            try:
                rank_key = float(rank)
            except (TypeError, ValueError):
                rank_key = float("inf")
        logprob = info.get("logprob")
        try:
            logprob_key = -float(logprob)
        except (TypeError, ValueError):
            logprob_key = float("inf")
        return rank_key, logprob_key

    token_id, info = min(entries.items(), key=sort_key)
    try:
        return int(token_id), info
    except ValueError:
        return None, info


def decoded(info: dict[str, Any]) -> str:
    value = info.get("decoded_token")
    if value is None:
        return ""
    return str(value).replace("\n", "\\n")


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def find_first_divergence(
    reference: list[int],
    candidates: dict[str, list[int]],
) -> int | None:
    max_len = max([len(reference), *(len(ids) for ids in candidates.values())])
    for pos in range(max_len):
        ref_tok = reference[pos] if pos < len(reference) else None
        for ids in candidates.values():
            cand_tok = ids[pos] if pos < len(ids) else None
            if cand_tok != ref_tok:
                return pos
    return None


def build_position_rows(
    labeled: list[tuple[str, dict[str, Any]]],
    *,
    reference_label: str,
    max_positions: int | None,
) -> list[dict[str, Any]]:
    ids_by_label = {label: token_ids(data) for label, data in labeled}
    reference_ids = ids_by_label[reference_label]
    max_len = max(len(ids) for ids in ids_by_label.values())
    if max_positions is not None:
        max_len = min(max_len, int(max_positions))

    rows: list[dict[str, Any]] = []
    for pos in range(max_len):
        ref_token = reference_ids[pos] if pos < len(reference_ids) else None
        row: dict[str, Any] = {
            "position": pos,
            "reference_token_id": ref_token,
        }
        any_diverged = False
        for label, data in labeled:
            ids = ids_by_label[label]
            selected = ids[pos] if pos < len(ids) else None
            selected_info = selected_logprob_info(data, pos, selected)
            top_id, top_info = top_token_info(data, pos)
            ref_info = selected_logprob_info(data, pos, ref_token)
            diverged = selected != ref_token
            any_diverged = any_diverged or diverged
            prefix = f"{label}_"
            row[prefix + "token_id"] = selected
            row[prefix + "decoded"] = decoded(selected_info)
            row[prefix + "logprob"] = selected_info.get("logprob")
            row[prefix + "rank"] = selected_info.get("rank")
            row[prefix + "top_token_id"] = top_id
            row[prefix + "top_decoded"] = decoded(top_info)
            row[prefix + "top_logprob"] = top_info.get("logprob")
            row[prefix + "top_rank"] = top_info.get("rank")
            row[prefix + "reference_token_logprob"] = ref_info.get("logprob")
            row[prefix + "reference_token_rank"] = ref_info.get("rank")
            row[prefix + "diverged"] = int(diverged)
        row["any_diverged"] = int(any_diverged)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    labeled: list[tuple[str, dict[str, Any]]],
    rows: list[dict[str, Any]],
    *,
    reference_label: str,
) -> None:
    ids_by_label = {label: token_ids(data) for label, data in labeled}
    reference_ids = ids_by_label[reference_label]
    first_divergence = find_first_divergence(
        reference_ids,
        {label: ids for label, ids in ids_by_label.items() if label != reference_label},
    )
    labels = [label for label, _ in labeled]
    lines = [
        "# SR24 Replay Logit Comparison",
        "",
        f"- reference: `{reference_label}`",
        f"- first divergence position: `{first_divergence if first_divergence is not None else 'none'}`",
        "",
        "## Inputs",
        "",
        "| label | mode | doc | tokens | finish reason | path |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for label, data in labeled:
        lines.append(
            "| {label} | `{mode}` | {doc} | {tokens} | {finish} | `{path}` |".format(
                label=label,
                mode=data.get("mode", ""),
                doc=data.get("doc_id", ""),
                tokens=len(token_ids(data)),
                finish=data.get("finish_reason", ""),
                path=data.get("_path", ""),
            )
        )

    if first_divergence is None:
        lines.extend(
            [
                "",
                "All provided replays selected the same generated token ids over the compared length.",
            ]
        )
    else:
        window = [
            row for row in rows
            if max(0, first_divergence - 2) <= int(row["position"]) <= first_divergence + 2
        ]
        lines.extend(
            [
                "",
                "## Divergence Window",
                "",
                "| pos | " + " | ".join(
                    f"{label} token/top/ref-logprob" for label in labels
                ) + " |",
                "| ---: | " + " | ".join("---" for _ in labels) + " |",
            ]
        )
        for row in window:
            cells = []
            for label in labels:
                prefix = f"{label}_"
                selected = row.get(prefix + "token_id")
                selected_lp = fmt(row.get(prefix + "logprob"))
                top = row.get(prefix + "top_token_id")
                top_lp = fmt(row.get(prefix + "top_logprob"))
                ref_lp = fmt(row.get(prefix + "reference_token_logprob"))
                decoded_token = str(row.get(prefix + "decoded") or "")
                top_decoded = str(row.get(prefix + "top_decoded") or "")
                cells.append(
                    "`{selected}` {decoded} lp={lp}; top `{top}` {top_decoded} lp={top_lp}; ref_lp={ref_lp}".format(
                        selected=selected,
                        decoded=decoded_token,
                        lp=selected_lp,
                        top=top,
                        top_decoded=top_decoded,
                        top_lp=top_lp,
                        ref_lp=ref_lp,
                    )
                )
            lines.append(f"| {row['position']} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Per-Mode Agreement",
            "",
            "| label | first divergence vs reference | selected tokens identical to reference |",
            "| --- | ---: | ---: |",
        ]
    )
    for label, ids in ids_by_label.items():
        if label == reference_label:
            lines.append(f"| {label} | n/a | 1 |")
            continue
        div = find_first_divergence(reference_ids, {label: ids})
        identical = int(div is None)
        lines.append(f"| {label} | {div if div is not None else 'none'} | {identical} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=parse_labeled_path,
        required=True,
        help="Replay JSON as label:/abs/or/relative/path. Pass at least two.",
    )
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-positions", type=int, default=None)
    args = parser.parse_args()

    if len(args.input) < 2:
        raise SystemExit("provide at least two --input values")
    labels = [label for label, _ in args.input]
    if len(set(labels)) != len(labels):
        raise SystemExit("input labels must be unique")
    if args.reference_label not in set(labels):
        raise SystemExit("--reference-label must match one input label")

    labeled: list[tuple[str, dict[str, Any]]] = []
    for label, path in args.input:
        data = load_replay(path)
        data["_path"] = str(path.resolve())
        labeled.append((label, data))

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = build_position_rows(
        labeled,
        reference_label=args.reference_label,
        max_positions=args.max_positions,
    )
    write_csv(args.output_root / "position_logprobs.csv", rows)
    write_report(
        args.output_root / "report.md",
        labeled,
        rows,
        reference_label=args.reference_label,
    )
    print((args.output_root / "report.md").resolve())


if __name__ == "__main__":
    main()
