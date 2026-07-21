"""Nsight Compute CSV parsing and Blackwell metric extraction."""

from __future__ import annotations

import csv
import io
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Any

NCU = Path("/usr/local/cuda-13.2/bin/ncu")

BYTE_SCALES = {
    "byte": Decimal(1),
    "Kbyte": Decimal(1000),
    "Mbyte": Decimal(1_000_000),
    "Gbyte": Decimal(1_000_000_000),
}
TIME_TO_US = {
    "ns": Decimal("0.001"),
    "us": Decimal(1),
    "ms": Decimal(1000),
    "s": Decimal(1_000_000),
}


def ncu_records(report: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    command = [
        str(NCU),
        "--import",
        str(report),
        "--csv",
        "--page",
        "raw",
        "--print-units",
        "auto",
        "--print-kernel-base",
        "demangled",
    ]
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rows = list(csv.reader(io.StringIO(completed.stdout)))
    header_index = next(
        index
        for index, row in enumerate(rows)
        if row[:2] == ["ID", "Process ID"]
    )
    header = rows[header_index]
    unit_row = rows[header_index + 1]
    units = dict(zip(header, unit_row, strict=False))
    records: list[dict[str, str]] = []
    for row in rows[header_index + 2 :]:
        if len(row) == len(header) and row and row[0] != "":
            records.append(dict(zip(header, row, strict=True)))
    if not records:
        raise RuntimeError(f"NCU report contains no records: {report}")
    return records, units


def decimal_value(record: dict[str, str], metric: str) -> Decimal | None:
    raw = record.get(metric, "").strip()
    if raw in {"", "n/a", "N/A", "(!) n/a"}:
        return None
    return Decimal(raw)


def converted_metric(
    record: dict[str, str],
    units: dict[str, str],
    output_name: str,
    metric: str,
) -> float | int | None:
    value = decimal_value(record, metric)
    if value is None:
        return None
    unit = units.get(metric, "")
    if output_name.endswith("_bytes"):
        scale = BYTE_SCALES.get(unit)
        if scale is None:
            raise ValueError(f"unsupported byte unit {unit!r} for {metric}")
        return int(value * scale)
    if output_name == "duration_us":
        scale = TIME_TO_US.get(unit)
        if scale is None:
            raise ValueError(f"unsupported time unit {unit!r} for {metric}")
        return float(value * scale)
    if output_name.endswith("_sectors") or output_name in {
        "registers_per_thread",
        "hmma_instructions",
        "hmma_sparse_ops",
        "hmma_dense_ops",
        "shared_bank_conflicts",
        "shared_load_bank_conflicts",
        "shared_store_bank_conflicts",
        "shared_load_requests",
        "shared_load_wavefronts",
        "local_load_sectors",
        "local_store_sectors",
    }:
        return int(value)
    return float(value)


RAW_METRICS = {
    "dram_read_bytes": "dram__bytes_op_read.sum",
    "dram_write_bytes": "dram__bytes_op_write.sum",
    "l2_read_sectors": "lts__t_sectors_srcunit_tex_op_read.sum",
    "l2_read_hit_sectors": "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "l2_read_miss_sectors": "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
    "tensor_active_pct": (
        "sm__pipe_tensor_subpipe_hmma_cycles_active."
        "avg.pct_of_peak_sustained_elapsed"
    ),
    "hmma_instruction_throughput_pct": (
        "sm__inst_executed_pipe_tensor_subpipe_hmma."
        "avg.pct_of_peak_sustained_elapsed"
    ),
    "dense_bf16_hmma_ops": (
        "sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_"
        "sparsity_off.sum"
    ),
    "sparse_bf16_hmma_ops": (
        "sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_"
        "sparsity_on.sum"
    ),
    "dense_bf16_hmma_util_pct": (
        "sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_"
        "sparsity_off.sum.pct_of_peak_sustained_elapsed"
    ),
    "sparse_bf16_hmma_util_pct": (
        "sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_"
        "sparsity_on.sum.pct_of_peak_sustained_elapsed"
    ),
    "active_warps_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "active_warps_per_cycle": "sm__warps_active.avg.per_cycle_active",
    "eligible_warps_per_cycle": "smsp__warps_eligible.avg.per_cycle_active",
    "issue_active_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "shared_bank_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "shared_ld_bank_conflicts": (
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
    ),
    "shared_st_bank_conflicts": (
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"
    ),
    "shared_ldgsts_bank_conflicts": (
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts.sum"
    ),
    "shared_load_wavefronts": (
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"
    ),
    "shared_wavefronts": "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
    "shared_store_wavefronts": (
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"
    ),
    "cp_async_sass_wavefronts": (
        "smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_"
        "op_ldgsts.sum"
    ),
    "ldgsts_global_read_bytes": (
        "sm__sass_l1tex_m_xbar2l1tex_read_bytes_mem_global_"
        "op_ldgsts_cache_bypass.sum"
    ),
    "ldgsts_instructions": "smsp__inst_executed_op_ldgsts.sum",
    "stall_barrier_per_issue": (
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"
    ),
    "stall_long_scoreboard_per_issue": (
        "smsp__average_warps_issue_stalled_long_scoreboard_"
        "per_issue_active.ratio"
    ),
    "stall_short_scoreboard_per_issue": (
        "smsp__average_warps_issue_stalled_short_scoreboard_"
        "per_issue_active.ratio"
    ),
    "stall_mio_throttle_per_issue": (
        "smsp__average_warps_issue_stalled_mio_throttle_"
        "per_issue_active.ratio"
    ),
    "stall_math_pipe_per_issue": (
        "smsp__average_warps_issue_stalled_math_pipe_throttle_"
        "per_issue_active.ratio"
    ),
    "stall_not_selected_per_issue": (
        "smsp__average_warps_issue_stalled_not_selected_"
        "per_issue_active.ratio"
    ),
    "stall_wait_per_issue": (
        "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"
    ),
    "registers_per_thread": "launch__registers_per_thread",
    "dynamic_shared_mem_bytes": "launch__shared_mem_per_block_dynamic",
    "local_load_sectors": "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "local_store_sectors": "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
    "register_spill_instructions": "sass__inst_executed_register_spilling",
    "local_spill_requests": "derived__local_spilling_requests",
}

INTEGER_FIELDS = {
    "dram_read_bytes",
    "dram_write_bytes",
    "l2_read_sectors",
    "l2_read_hit_sectors",
    "l2_read_miss_sectors",
    "dense_bf16_hmma_ops",
    "sparse_bf16_hmma_ops",
    "shared_bank_conflicts",
    "shared_ld_bank_conflicts",
    "shared_st_bank_conflicts",
    "shared_ldgsts_bank_conflicts",
    "shared_wavefronts",
    "shared_load_wavefronts",
    "shared_store_wavefronts",
    "cp_async_sass_wavefronts",
    "ldgsts_global_read_bytes",
    "ldgsts_instructions",
    "registers_per_thread",
    "dynamic_shared_mem_bytes",
    "local_load_sectors",
    "local_store_sectors",
    "register_spill_instructions",
    "local_spill_requests",
}


def value(record: dict[str, str], metric: str) -> float | int | None:
    raw = decimal_value(record, metric)
    if raw is None:
        return None
    return float(raw)


def byte_value(
    record: dict[str, str], units: dict[str, str], metric: str
) -> int | None:
    raw = decimal_value(record, metric)
    if raw is None:
        return None
    unit = units.get(metric, "").removesuffix("/block")
    scale = {
        "byte": 1,
        "Kbyte": 1_000,
        "Mbyte": 1_000_000,
        "Gbyte": 1_000_000_000,
    }.get(unit)
    if scale is None:
        raise ValueError(f"unsupported byte unit {unit!r} for {metric}")
    return int(raw * scale)


def extract(report: Path, method: str) -> dict[str, Any]:
    records, units = ncu_records(report)
    if len(records) != 1:
        raise RuntimeError(f"expected one record in {report}, got {len(records)}")
    record = records[0]
    row: dict[str, Any] = {
        "method": method,
        "report": report.name,
        "kernel_name": record.get("Kernel Name", ""),
        "grid_size": record.get("Grid Size", ""),
        "block_size": record.get("Block Size", ""),
        "duration_us": converted_metric(
            record,
            units,
            "duration_us",
            "gpu__time_duration.sum",
        ),
    }
    for output_name, metric in RAW_METRICS.items():
        if output_name.endswith("_bytes"):
            metric_value = byte_value(record, units, metric)
        else:
            metric_value = value(record, metric)
        if metric_value is not None and output_name in INTEGER_FIELDS:
            metric_value = int(metric_value)
        row[output_name] = metric_value
    sectors = row["l2_read_sectors"]
    hits = row["l2_read_hit_sectors"]
    row["l2_read_bytes"] = None if sectors is None else 32 * sectors
    row["l2_read_hit_pct"] = (
        None if not sectors or hits is None else 100.0 * hits / sectors
    )
    row["ldgsts_wavefronts_per_instruction"] = divide(
        row["cp_async_sass_wavefronts"], row["ldgsts_instructions"]
    )
    return row


def divide(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    return float(numerator) / float(denominator)


__all__ = [
    "RAW_METRICS", "byte_value", "converted_metric",
    "decimal_value", "extract", "ncu_records",
]
