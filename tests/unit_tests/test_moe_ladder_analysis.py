# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sqlite3
from pathlib import Path

import pytest
import torchtitan.experiments.moe_ladder.analyze_overlap as overlap_analysis
import torchtitan.experiments.moe_ladder.nvtx as nvtx
from torchtitan.experiments.moe_ladder.analyze_overlap import (
    _device_kernels,
    _finite_mean,
    _intersection,
    _kind_overlap,
    _merge,
    _process_id,
    analyze,
)


def test_interval_operations() -> None:
    assert _merge([(5, 10), (1, 3), (3, 6), (12, 14)]) == [(1, 10), (12, 14)]
    assert _intersection([(1, 8), (10, 12)], [(0, 2), (4, 11)]) == 6
    assert _process_id((1 << 55) | (2 << 24) | 17) == 2


def test_analyze_uses_each_rank_measurement_window(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "rank0.sqlite"
    connection = sqlite3.connect(report)
    cursor = connection.cursor()
    cursor.executescript(
        """
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS (
            start INTEGER, end INTEGER, globalTid INTEGER,
            text TEXT, textId INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER, end INTEGER, deviceId INTEGER,
            shortName INTEGER, correlationId INTEGER, globalPid INTEGER
        );
        """
    )
    cursor.executemany(
        "INSERT INTO StringIds VALUES (?, ?)",
        [
            (1, "dispatch_impl"),
            (2, "combine_impl"),
            (3, "gemm_kernel"),
        ],
    )
    cursor.executemany(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?)",
        [
            (0, 100, 0, "moe_ladder/parallel/measure_step", None),
            (200, 300, 0, "moe_ladder/parallel/measure_step", None),
            (20, 120, 2 << 24, "moe_ladder/parallel/measure_step", None),
            (220, 320, 2 << 24, "moe_ladder/parallel/measure_step", None),
        ],
    )
    cursor.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?)",
        [
            (10, 50, 0, 1, 1, 0),
            (30, 70, 0, 3, 2, 0),
            (130, 190, 0, 3, 3, 0),
            (210, 230, 0, 2, 4, 0),
            (220, 260, 0, 3, 5, 0),
            (30, 70, 1, 1, 11, 2 << 24),
            (50, 90, 1, 3, 12, 2 << 24),
            (230, 250, 1, 2, 13, 2 << 24),
            (240, 280, 1, 3, 14, 2 << 24),
        ],
    )
    connection.commit()
    windows = [(0, 100), (200, 300)]
    comm, compute = _device_kernels(cursor, device=0, windows=windows)
    assert comm == [(10, 50), (210, 230)]
    assert compute == [(30, 70), (220, 260)]
    assert _kind_overlap(cursor, 0, windows, "dispatch_impl", compute) == (40, 20)
    assert _kind_overlap(cursor, 0, windows, "combine_impl", compute) == (20, 10)
    connection.close()

    row = analyze(report, "measure_step")
    report_header = next(
        line for line in capsys.readouterr().out.splitlines() if "variant=" in line
    )

    assert report_header.endswith("variant=parallel")
    assert row["variant"] == "parallel"
    assert row["comm_us"] == pytest.approx(0.03)
    assert row["compute_us"] == pytest.approx(0.04)
    assert row["comm_hidden_pct"] == pytest.approx(50.0)
    assert row["gpu_idle_pct"] == pytest.approx(45.0)


def test_finite_mean_ignores_devices_without_communication() -> None:
    assert _finite_mean([float("nan"), 50.0]) == 50.0
    assert _finite_mean([float("nan")]) != _finite_mean([float("nan")])


def test_nvtx_ranges_require_explicit_enable(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_call(*args) -> None:
        raise AssertionError(f"unexpected NVTX call: {args}")

    monkeypatch.setattr(nvtx, "_NVTX_ENABLED", False)
    monkeypatch.setattr(nvtx.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(nvtx.torch.cuda.nvtx, "range_push", unexpected_call)
    monkeypatch.setattr(nvtx.torch.cuda.nvtx, "range_pop", unexpected_call)

    with nvtx.nvtx_range("disabled"):
        pass


def test_nvtx_ranges_emit_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(nvtx, "_NVTX_ENABLED", True)
    monkeypatch.setattr(nvtx.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        nvtx.torch.cuda.nvtx,
        "range_push",
        lambda name: calls.append(("push", name)),
    )
    monkeypatch.setattr(
        nvtx.torch.cuda.nvtx,
        "range_pop",
        lambda: calls.append(("pop", None)),
    )

    with nvtx.nvtx_range("enabled"):
        pass

    assert calls == [("push", "enabled"), ("pop", None)]


def test_export_measured_window_is_time_and_table_filtered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, Path, tuple[str, ...], tuple[int, int] | None]] = []

    def fake_run_nsys_export(
        report: Path,
        output: Path,
        *,
        tables: tuple[str, ...],
        time_range: tuple[int, int] | None = None,
    ) -> None:
        calls.append((report, output, tables, time_range))
        if tables == overlap_analysis.WINDOW_TABLES:
            connection = sqlite3.connect(output)
            connection.executescript(
                """
                CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
                CREATE TABLE NVTX_EVENTS (
                    start INTEGER, end INTEGER, globalTid INTEGER,
                    text TEXT, textId INTEGER
                );
                INSERT INTO NVTX_EVENTS VALUES (
                    10, 110, 0, 'moe_ladder/parallel/measure_step', NULL
                );
                INSERT INTO NVTX_EVENTS VALUES (
                    110, 210, 0, 'moe_ladder/parallel/measure_step', NULL
                );
                """
            )
            connection.close()
        else:
            output.touch()

    monkeypatch.setattr(overlap_analysis, "_run_nsys_export", fake_run_nsys_export)
    report = tmp_path / "parallel.nsys-rep"
    output = overlap_analysis._export_measured_window(report, tmp_path, "measure_step")

    assert calls[0][2] == overlap_analysis.WINDOW_TABLES
    assert calls[0][3] is None
    assert calls[1][2] == overlap_analysis.ANALYSIS_TABLES
    assert calls[1][3] == (10, 210)
    assert output == calls[1][1]


def test_analyze_report_removes_temporary_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exports: list[Path] = []

    def fake_export(report: Path, output_dir: Path, step_range: str) -> Path:
        output = output_dir / "minimal.sqlite"
        output.touch()
        exports.append(output)
        return output

    def fake_analyze(report: Path, step_range: str) -> dict[str, float | str]:
        assert report.exists()
        return {"variant": "parallel"}

    monkeypatch.setattr(overlap_analysis, "_export_measured_window", fake_export)
    monkeypatch.setattr(overlap_analysis, "analyze", fake_analyze)

    row = overlap_analysis.analyze_report(
        tmp_path / "parallel.nsys-rep", "measure_step"
    )

    assert row == {"variant": "parallel"}
    assert not exports[0].exists()
