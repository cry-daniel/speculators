#!/usr/bin/env python3
"""Summarize Nsight Systems sqlite exports for SpecLink-CV runs.

Typical use:

    python tools/speclink_cv/nsys_profile_summary.py \
      --run-root examples/evaluate/eval-guidellm/temp/speclink_cv_nsys_profile_run1

The input is the output root produced by
`examples/evaluate/eval-guidellm/scripts/run_speclink_cv_qwen_math.py
--nsys-profile`. The summary is intentionally compact: it extracts the
hardware counters needed to understand whether suffix skipping turns into
actual saturated throughput.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any


METRIC_IDS = {
    6: "gr_active_pct",
    7: "sms_active_pct",
    8: "sm_issue_pct",
    9: "tensor_active_pct",
    19: "dram_read_pct",
    20: "dram_write_pct",
}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * pct)
    return ordered[max(0, min(len(ordered) - 1, index))]


def union_busy_and_gaps(intervals: list[tuple[int, int]]) -> tuple[int, list[int]]:
    if not intervals:
        return 0, []
    intervals.sort()
    busy = 0
    gaps: list[int] = []
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start > current_end:
            busy += current_end - current_start
            gaps.append(start - current_end)
            current_start, current_end = start, end
        elif end > current_end:
            current_end = end
    busy += current_end - current_start
    return busy, gaps


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summarize_cv_profile(run_dir: Path) -> dict[str, Any]:
    rows = iter_jsonl(run_dir / "speclink_cv_profile.jsonl")
    prefix_results = [
        row
        for row in rows
        if row.get("event") == "verify_chunk_result"
        and row.get("phase") == "prefix"
    ]
    forward_plans = [
        row for row in rows if row.get("event") == "model_forward_plan"
    ]
    phase_modes: dict[str, int] = {}
    for row in forward_plans:
        key = f"{row.get('phase') or 'unknown'}:{row.get('cudagraph_mode') or 'unknown'}"
        phase_modes[key] = phase_modes.get(key, 0) + 1
    skipped_suffix = sum(float(row.get("skipped_suffix_tokens") or 0) for row in prefix_results)
    suffix_total = sum(float(row.get("suffix_len") or 0) for row in prefix_results)
    return {
        "profile_events": len(rows),
        "skipped_suffix_ratio": skipped_suffix / suffix_total if suffix_total else "",
        "phase_cudagraph_counts": json.dumps(phase_modes, sort_keys=True),
    }


def metric_average(conn: sqlite3.Connection, metric_id: int) -> float:
    row = conn.execute(
        "select avg(value) from GPU_METRICS where metricId = ?",
        (metric_id,),
    ).fetchone()
    return float(row[0] or 0.0)


def api_sum(
    conn: sqlite3.Connection,
    name_like: str,
) -> tuple[int, float]:
    row = conn.execute(
        """
        select count(*), coalesce(sum(r.end - r.start), 0)
        from CUPTI_ACTIVITY_KIND_RUNTIME r
        left join StringIds s on r.nameId = s.id
        where s.value like ?
        """,
        (name_like,),
    ).fetchone()
    return int(row[0] or 0), float(row[1] or 0.0) / 1e9


def memcpy_api_by_kind(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        select coalesce(e.label, 'unknown') as kind,
               count(*) as calls,
               coalesce(sum(r.end - r.start), 0) as api_ns,
               coalesce(sum(m.bytes), 0) as num_bytes
        from CUPTI_ACTIVITY_KIND_RUNTIME r
        join StringIds s on r.nameId = s.id
        join CUPTI_ACTIVITY_KIND_MEMCPY m on r.correlationId = m.correlationId
        left join ENUM_CUDA_MEMCPY_OPER e on m.copyKind = e.id
        where s.value like 'cudaMemcpyAsync%'
        group by kind
        """
    ).fetchall()
    return {
        str(kind): {
            "calls": int(calls or 0),
            "api_s": float(api_ns or 0.0) / 1e9,
            "bytes": int(num_bytes or 0),
        }
        for kind, calls, api_ns, num_bytes in rows
    }


def top_d2h_copy_sizes(conn: sqlite3.Connection, limit: int = 5) -> str:
    rows = conn.execute(
        """
        select m.bytes,
               count(*) as calls,
               coalesce(sum(r.end - r.start), 0) as api_ns
        from CUPTI_ACTIVITY_KIND_RUNTIME r
        join StringIds s on r.nameId = s.id
        join CUPTI_ACTIVITY_KIND_MEMCPY m on r.correlationId = m.correlationId
        where s.value like 'cudaMemcpyAsync%' and m.copyKind = 2
        group by m.bytes
        order by api_ns desc
        limit ?
        """,
        (limit,),
    ).fetchall()
    return json.dumps(
        [
            {
                "bytes": int(num_bytes or 0),
                "calls": int(calls or 0),
                "api_s": round(float(api_ns or 0.0) / 1e9, 6),
            }
            for num_bytes, calls, api_ns in rows
        ],
        sort_keys=True,
    )


def top_kernel_total(conn: sqlite3.Connection) -> tuple[str, int, float]:
    row = conn.execute(
        """
        select coalesce(s.value, cast(k.demangledName as text)) as name,
               count(*) as calls,
               coalesce(sum(k.end - k.start), 0) as total_ns
        from CUPTI_ACTIVITY_KIND_KERNEL k
        left join StringIds s on k.demangledName = s.id
        group by name
        order by total_ns desc
        limit 1
        """
    ).fetchone()
    if not row:
        return "", 0, 0.0
    return str(row[0]), int(row[1]), float(row[2] or 0.0) / 1e9


def summarize_sqlite(sqlite_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(sqlite_path)
    intervals = [
        (int(start), int(end))
        for start, end in conn.execute(
            "select start, end from CUPTI_ACTIVITY_KIND_KERNEL order by start"
        )
    ]
    if intervals:
        first = min(start for start, _ in intervals)
        last = max(end for _, end in intervals)
        kernel_span_s = (last - first) / 1e9
    else:
        kernel_span_s = 0.0
    kernel_busy_ns, gaps_ns = union_busy_and_gaps(intervals)
    kernel_total_ns = conn.execute(
        "select coalesce(sum(end - start), 0) from CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchone()[0]
    runtime_total_ns = conn.execute(
        "select coalesce(sum(end - start), 0) from CUPTI_ACTIVITY_KIND_RUNTIME"
    ).fetchone()[0]
    launch_calls, launch_s = api_sum(conn, "cudaLaunchKernel%")
    graph_calls, graph_s = api_sum(conn, "cudaGraphLaunch%")
    sync_calls, sync_s = api_sum(conn, "cudaStreamSynchronize%")
    memcpy_calls, memcpy_s = api_sum(conn, "cudaMemcpyAsync%")
    memcpy_kinds = memcpy_api_by_kind(conn)
    top_kernel, top_kernel_calls, top_kernel_s = top_kernel_total(conn)
    out: dict[str, Any] = {
        "sqlite": str(sqlite_path),
        "kernel_span_s": kernel_span_s,
        "kernel_total_s": float(kernel_total_ns or 0.0) / 1e9,
        "kernel_count": len(intervals),
        "kernel_busy_pct": (100.0 * kernel_busy_ns / (kernel_span_s * 1e9))
        if kernel_span_s
        else 0.0,
        "global_gap_total_s": sum(gaps_ns) / 1e9,
        "global_gap_avg_us": (sum(gaps_ns) / len(gaps_ns) / 1e3)
        if gaps_ns
        else 0.0,
        "global_gap_p90_us": percentile([gap / 1e3 for gap in gaps_ns], 0.90),
        "global_gap_p99_us": percentile([gap / 1e3 for gap in gaps_ns], 0.99),
        "runtime_api_total_s": float(runtime_total_ns or 0.0) / 1e9,
        "cuda_launch_kernel_calls": launch_calls,
        "cuda_launch_kernel_s": launch_s,
        "cuda_graph_launch_calls": graph_calls,
        "cuda_graph_launch_s": graph_s,
        "cuda_stream_sync_calls": sync_calls,
        "cuda_stream_sync_s": sync_s,
        "cuda_memcpy_async_calls": memcpy_calls,
        "cuda_memcpy_async_s": memcpy_s,
        "d2h_memcpy_calls": memcpy_kinds.get("Device-to-Host", {}).get(
            "calls", 0
        ),
        "d2h_memcpy_api_s": memcpy_kinds.get("Device-to-Host", {}).get(
            "api_s", 0.0
        ),
        "h2d_memcpy_calls": memcpy_kinds.get("Host-to-Device", {}).get(
            "calls", 0
        ),
        "h2d_memcpy_api_s": memcpy_kinds.get("Host-to-Device", {}).get(
            "api_s", 0.0
        ),
        "d2d_memcpy_calls": memcpy_kinds.get("Device-to-Device", {}).get(
            "calls", 0
        ),
        "d2d_memcpy_api_s": memcpy_kinds.get("Device-to-Device", {}).get(
            "api_s", 0.0
        ),
        "top_d2h_copy_sizes": top_d2h_copy_sizes(conn),
        "top_kernel": top_kernel[:120],
        "top_kernel_calls": top_kernel_calls,
        "top_kernel_s": top_kernel_s,
    }
    for metric_id, key in METRIC_IDS.items():
        out[key] = metric_average(conn, metric_id)
    conn.close()
    return out


def find_case_dirs(run_root: Path) -> list[Path]:
    runs_dir = run_root / "runs"
    if not runs_dir.exists():
        raise SystemExit(f"missing runs directory: {runs_dir}")
    return sorted(path for path in runs_dir.iterdir() if path.is_dir())


def summarize_case(case_dir: Path) -> dict[str, Any]:
    sqlite_paths = sorted((case_dir / "nsys").glob("*.sqlite"))
    if not sqlite_paths:
        raise SystemExit(f"missing Nsight sqlite under {case_dir / 'nsys'}")
    config = read_json(case_dir / "config.json")
    result = read_json(case_dir / "steady_state_results.json")
    row = {
        "method": config.get("method") or case_dir.name,
        "throughput_tok_s": result.get("output_tokens_per_second", ""),
        "measurement_s": result.get("measurement_s", ""),
        "measurement_output_tokens": result.get("measurement_output_tokens", ""),
    }
    row.update(summarize_cv_profile(case_dir))
    row.update(summarize_sqlite(sqlite_paths[0]))
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 3) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# SpecLink-CV Nsight Profile Summary",
        "",
        "All throughput values are saturated output tokens/s from the fixed measurement window.",
        "Nsight counters cover the profiler capture range around the benchmark, so they include warmup/measurement/cooldown capture overhead.",
        "",
        "| Method | Tok/s | Kernel busy % | SM active % | Tensor active % | DRAM read % | Kernel count | CUDA launches | Graph launches | Stream sync s | D2H API s | Gap total s | Skip suffix | CG modes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        launches = int(row.get("cuda_launch_kernel_calls") or 0)
        graph = int(row.get("cuda_graph_launch_calls") or 0)
        lines.append(
            "| {method} | {tok} | {busy} | {sm} | {tensor} | {dram} | {kernels} | {launches} | {graph} | {sync} | {d2h} | {gap} | {skip} | `{modes}` |".format(
                method=row["method"],
                tok=fmt(row.get("throughput_tok_s"), 2),
                busy=fmt(row.get("kernel_busy_pct"), 1),
                sm=fmt(row.get("sms_active_pct"), 1),
                tensor=fmt(row.get("tensor_active_pct"), 1),
                dram=fmt(row.get("dram_read_pct"), 1),
                kernels=int(row.get("kernel_count") or 0),
                launches=launches,
                graph=graph,
                sync=fmt(row.get("cuda_stream_sync_s"), 3),
                d2h=fmt(row.get("d2h_memcpy_api_s"), 3),
                gap=fmt(row.get("global_gap_total_s"), 3),
                skip=fmt(row.get("skipped_suffix_ratio"), 3),
                modes=row.get("phase_cudagraph_counts", ""),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation hints:",
            "",
            "- A high `Skip suffix` value only measures skipped suffix verifier chunks; it is not the fraction of total GPU work removed.",
            "- If kernel busy, SM active, or Tensor active drop while kernel/API counts rise, suffix skipping is being diluted by smaller verifier waves and launch/synchronization overhead.",
            "- `CG modes` should be checked to see whether prefix verifier chunks enter FULL CUDA graph or PIECEWISE/non-graph fallback.",
            "- `D2H API s` is often the clearest sign that CPU-side prefix/sampler bookkeeping is waiting on GPU results; inspect `top_d2h_copy_sizes` in the CSV for the dominant copy sizes.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv", dest="csv_path", type=Path, default=None)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    rows = [summarize_case(case_dir) for case_dir in find_case_dirs(run_root)]
    order = {
        "eagle3_oneshot": 0,
        "cv_half_async_staged_simple": 1,
        "cv_wavefront_staged": 2,
    }
    rows.sort(key=lambda row: (order.get(str(row.get("method")), 100), str(row.get("method"))))
    csv_path = args.csv_path or run_root / "hardware_profile_summary.csv"
    md_path = args.output or run_root / "hardware_profile_summary.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
