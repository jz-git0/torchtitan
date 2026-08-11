# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measure communication/computation overlap in an Nsight Systems report.

Nsight's timeline shows overlap visually; this script quantifies it over
measured steps and aggregates all ranks. Kernel names, rather than stream ids,
identify DeepEP dispatch/combine and NCCL collectives from TP or DP.

The metrics narrow down a cause but do not prove one. A comm_hidden value near
zero means communication did not coincide with compute. A large gpu_idle can
indicate a launch-bound workload or an explicit synchronization gap.

Usage:
    python -m torchtitan.experiments.moe_ladder.analyze_overlap report.nsys-rep

The analyzer exports only the measured NVTX window and required SQLite tables
to a temporary directory. Existing SQLite exports are also accepted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import statistics
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

# Nsight's shortName drops DeepEP's namespace and template arguments.
COMM_KERNEL_MARKERS = ("nccl", "dispatch_impl", "combine_impl")
WINDOW_TABLES = ("StringIds", "NVTX_EVENTS")
ANALYSIS_TABLES = (*WINDOW_TABLES, "CUPTI_ACTIVITY_KIND_KERNEL")
_GLOBAL_ID_PROCESS_MASK = 0x00FFFFFF


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Input: process argv.
    Output: namespace with the Nsight Systems or SQLite report paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument(
        "--step-range",
        default="measure_step",
        help="NVTX range substring bounding the measured region",
    )
    parser.add_argument(
        "--results",
        nargs="*",
        type=Path,
        default=(),
        help=(
            "Profiler JSON to join on variant name, adding latency and MFU "
            "so one table carries step cost and communication together."
        ),
    )
    return parser.parse_args()


def _load_results(paths: tuple[Path, ...]) -> dict[str, dict]:
    """Index profiler JSON results by variant.

    Input: paths to result files, each holding one result object or a list.
    Output: mapping from variant name to its result dictionary.
    """
    results: dict[str, dict] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        for item in payload if isinstance(payload, list) else [payload]:
            results[item["variant"]] = item
    return results


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping intervals.

    Input: unsorted (start, end) pairs in nanoseconds.
    Output: disjoint intervals sorted by start.
    """
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _total(intervals: list[tuple[int, int]]) -> int:
    """Return the summed length of disjoint intervals."""
    return sum(end - start for start, end in intervals)


def _finite_mean(values: list[float]) -> float:
    """Average finite values, returning NaN when none exist."""
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else float("nan")


def _intersection(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> int:
    """Return the overlapping length of two disjoint interval lists.

    Input: two lists of disjoint intervals sorted by start.
    Output: total nanoseconds covered by both.
    """
    i = j = 0
    total = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            total += end - start
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def _clip(
    intervals: list[tuple[int, int]],
    windows: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Clip intervals to the union of measured step ranges."""
    clipped = []
    for start, end in intervals:
        for window_start, window_end in windows:
            clipped_start = max(start, window_start)
            clipped_end = min(end, window_end)
            if clipped_end > clipped_start:
                clipped.append((clipped_start, clipped_end))
    return _merge(clipped)


def _process_id(global_id: int) -> int:
    """Decode the process id stored in an Nsight global id."""
    return (global_id >> 24) & _GLOBAL_ID_PROCESS_MASK


def _measure_ranges(
    cursor: sqlite3.Cursor,
    step_range: str,
    *,
    process_ids: set[int] | None = None,
) -> tuple[list[tuple[int, int]], int, str]:
    """Return measured step ranges, step count, and variant name.

    Input: cursor, NVTX range substring, and optional process ids.
    Output ranges are nanoseconds; variant comes from `prefix/variant/step`.
    """
    rows = cursor.execute(
        """
        SELECT n.start, n.end, n.globalTid, COALESCE(n.text, s.value)
        FROM NVTX_EVENTS n
        LEFT JOIN StringIds s ON s.id = n.textId
        WHERE COALESCE(n.text, s.value) LIKE ?
        """,
        (f"%{step_range}%",),
    ).fetchall()
    if process_ids is not None:
        rows = [row for row in rows if _process_id(row[2]) in process_ids]
    if not rows:
        scope = (
            f" for process ids {sorted(process_ids)}" if process_ids is not None else ""
        )
        raise ValueError(f"no NVTX range matching {step_range!r}{scope} in report")
    per_tid: dict[int, int] = {}
    for _, _, tid, _ in rows:
        per_tid[tid] = per_tid.get(tid, 0) + 1
    parts = rows[0][3].split("/")
    variant = parts[-2] if len(parts) >= 2 else "unknown"
    return (
        _merge([(row[0], row[1]) for row in rows]),
        max(per_tid.values()),
        variant,
    )


def _measure_window(
    cursor: sqlite3.Cursor,
    step_range: str,
    *,
    process_ids: set[int] | None = None,
) -> tuple[int, int, int, str]:
    """Return enclosing bounds for a filtered Nsight export."""
    ranges, steps, variant = _measure_ranges(
        cursor, step_range, process_ids=process_ids
    )
    return ranges[0][0], ranges[-1][1], steps, variant


def _device_kernels(
    cursor: sqlite3.Cursor, device: int, windows: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split one device's kernels into communication and computation.

    Input: cursor, CUDA device id, and measured step ranges.
    Output: (comm_intervals, compute_intervals), each merged and disjoint.
    """
    start, end = windows[0][0], windows[-1][1]
    predicate = " OR ".join("LOWER(s.value) LIKE ?" for _ in COMM_KERNEL_MARKERS)
    params = [f"%{marker}%" for marker in COMM_KERNEL_MARKERS]
    comm = cursor.execute(
        f"""
        SELECT MAX(k.start, ?), MIN(k.end, ?)
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.shortName
        WHERE k.deviceId = ? AND k.end > ? AND k.start < ? AND ({predicate})
        """,
        [start, end, device, start, end, *params],
    ).fetchall()
    compute = cursor.execute(
        f"""
        SELECT MAX(k.start, ?), MIN(k.end, ?)
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.shortName
        WHERE k.deviceId = ? AND k.end > ? AND k.start < ?
          AND NOT ({predicate})
        """,
        [start, end, device, start, end, *params],
    ).fetchall()
    return _clip(comm, windows), _clip(compute, windows)


def _kind_overlap(
    cursor: sqlite3.Cursor,
    device: int,
    windows: list[tuple[int, int]],
    marker: str,
    compute: list[tuple[int, int]],
) -> tuple[int, int]:
    """Return busy and hidden nanoseconds for one communication kernel kind.

    Splitting dispatch from combine identifies which collective a schedule
    hides. Although DeepEP dispatch is one indivisible API phase, the complete
    phase can overlap independent work when launched on the MoE side stream.

    Input: cursor, device id, step ranges, kernel-name substring, and merged
    computation intervals for the device.
    Output: (busy_ns, hidden_ns) for kernels matching the marker.
    """
    start, end = windows[0][0], windows[-1][1]
    intervals = _clip(
        cursor.execute(
            """
            SELECT MAX(k.start, ?), MIN(k.end, ?)
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = k.shortName
            WHERE k.deviceId = ? AND k.end > ? AND k.start < ?
              AND LOWER(s.value) LIKE ?
            """,
            (start, end, device, start, end, f"%{marker}%"),
        ).fetchall(),
        windows,
    )
    return _total(intervals), _intersection(intervals, compute)


def analyze(report: Path, step_range: str) -> dict[str, float | str]:
    """Print per-device overlap statistics for one nsys sqlite export.

    Input: path to the export and the NVTX range bounding measured steps.
    Output: a summary row averaged over devices, for the combined table.
    """
    connection = sqlite3.connect(f"file:{report}?mode=ro", uri=True)
    cursor = connection.cursor()
    try:
        _, _, variant = _measure_ranges(cursor, step_range)
        devices = [
            row[0]
            for row in cursor.execute(
                "SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY 1"
            )
        ]
        print(f"\n{os.path.basename(report)}  variant={variant}")
        header = (
            f"  {'dev':>3} {'comm_us':>9} {'compute_us':>11} {'overlap_us':>11} "
            f"{'comm_hidden':>12} {'gpu_idle':>9} {'dispatch_hid':>13} "
            f"{'combine_hid':>12}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        totals: dict[str, list[float]] = {
            k: [] for k in ("comm", "compute", "hidden", "idle")
        }
        for device in devices:
            process_ids = {
                _process_id(row[0])
                for row in cursor.execute(
                    "SELECT DISTINCT globalPid "
                    "FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = ?",
                    (device,),
                )
            }
            windows, steps, _ = _measure_ranges(
                cursor, step_range, process_ids=process_ids
            )
            comm, compute = _device_kernels(cursor, device, windows)
            comm_ns = _total(comm)
            compute_ns = _total(compute)
            overlap_ns = _intersection(comm, compute)
            busy_ns = _total(_merge(comm + compute))
            span_ns = _total(windows)
            hidden = 100.0 * overlap_ns / comm_ns if comm_ns else float("nan")
            idle = 100.0 * (span_ns - busy_ns) / span_ns if span_ns else float("nan")
            kinds = {}
            for marker in ("dispatch_impl", "combine_impl"):
                busy, hid = _kind_overlap(cursor, device, windows, marker, compute)
                kinds[marker] = 100.0 * hid / busy if busy else float("nan")
            print(
                f"  {device:>3} {comm_ns / steps / 1e3:>9.1f} "
                f"{compute_ns / steps / 1e3:>11.1f} "
                f"{overlap_ns / steps / 1e3:>11.1f} "
                f"{hidden:>11.1f}% {idle:>8.1f}% "
                f"{kinds['dispatch_impl']:>12.1f}% "
                f"{kinds['combine_impl']:>11.1f}%"
            )
            totals["comm"].append(comm_ns / steps / 1e3)
            totals["compute"].append(compute_ns / steps / 1e3)
            totals["hidden"].append(hidden)
            totals["idle"].append(idle)
        return {
            "variant": variant,
            "comm_us": _finite_mean(totals["comm"]),
            "compute_us": _finite_mean(totals["compute"]),
            "comm_hidden_pct": _finite_mean(totals["hidden"]),
            "gpu_idle_pct": _finite_mean(totals["idle"]),
        }
    finally:
        connection.close()


def _run_nsys_export(
    report: Path,
    output: Path,
    *,
    tables: tuple[str, ...],
    time_range: tuple[int, int] | None = None,
) -> None:
    """Export selected Nsight Systems tables to SQLite.

    Input: source report, output path, table names, and optional time bounds.
    Output: writes one SQLite export.
    """
    command = [
        "nsys",
        "export",
        "--type=sqlite",
        "--force-overwrite=true",
        "--quiet=true",
        f"--tables={','.join(tables)}",
    ]
    if time_range is not None:
        command.append(f"--times={time_range[0]}/{time_range[1]}")
    command.extend(("--output", str(output), str(report)))
    subprocess.run(command, check=True)


def _export_measured_window(report: Path, output_dir: Path, step_range: str) -> Path:
    """Create a minimal SQLite export for the measured NVTX window.

    Input: .nsys-rep path, temporary output directory, and NVTX substring.
    Output: path to a temporary SQLite report containing the required rows.
    """
    stem = report.name.removesuffix(".nsys-rep")
    ranges = output_dir / f"{stem}.ranges.sqlite"
    _run_nsys_export(report, ranges, tables=WINDOW_TABLES)
    connection = sqlite3.connect(f"file:{ranges}?mode=ro", uri=True)
    try:
        start, end, _, _ = _measure_window(connection.cursor(), step_range)
    finally:
        connection.close()

    output = output_dir / f"{stem}.sqlite"
    _run_nsys_export(
        report,
        output,
        tables=ANALYSIS_TABLES,
        time_range=(start, end),
    )
    return output


def analyze_report(report: Path, step_range: str) -> dict[str, float | str]:
    """Analyze an .nsys-rep or an existing SQLite export."""
    if not report.name.endswith(".nsys-rep"):
        return analyze(report, step_range)
    with TemporaryDirectory(prefix="moe_ladder_nsys_") as temp_dir:
        export = _export_measured_window(report, Path(temp_dir), step_range)
        return analyze(export, step_range)


def _print_combined(rows: list[dict], results: dict[str, dict]) -> None:
    """Print one table joining measured latency with the comm breakdown.

    Input: per-report summary rows and profiler results indexed by variant.
    Output: writes the combined table to stdout when results are provided.
    """
    if not results:
        print("\nPass --results <profile.json> to add latency and MFU.")
        return
    header = (
        f"\n{'variant':16s}{'median_ms':>11}{'mfu%':>7}{'total_s':>9}"
        f"{'comm_ms':>9}{'comm_share':>12}{'comm_hidden':>13}{'gpu_idle':>10}"
    )
    print(header)
    print("-" * (len(header) - 1))
    for row in sorted(rows, key=lambda r: r["variant"]):
        item = results.get(str(row["variant"]))
        if item is None:
            continue
        comm_ms = float(row["comm_us"]) / 1e3
        share = 100.0 * comm_ms / item["median_ms"]
        print(
            f"{row['variant']:16s}{item['median_ms']:>11.2f}"
            f"{item.get('mfu_pct', float('nan')):>7.1f}"
            f"{item.get('total_measured_s', float('nan')):>9.2f}"
            f"{comm_ms:>9.2f}{share:>11.1f}%"
            f"{float(row['comm_hidden_pct']):>12.1f}%"
            f"{float(row['gpu_idle_pct']):>9.1f}%"
        )
    print(
        "\ncomm_ms is per-step device time in DeepEP and NCCL communication "
        "kernels;\ncomm_share is that as a fraction of the measured step."
    )


def main() -> None:
    """Analyze every report given on the command line."""
    args = _parse_args()
    print(
        "Per-step device time (us). comm_hidden is the fraction of "
        "communication\nthat ran while computation was also resident; "
        "gpu_idle is the share of the\nmeasured window with no kernel at all.\n"
        "dispatch_hid/combine_hid split the hidden fraction by collective."
    )
    rows = [analyze_report(report, args.step_range) for report in args.reports]
    _print_combined(rows, _load_results(tuple(args.results)))


if __name__ == "__main__":
    main()
