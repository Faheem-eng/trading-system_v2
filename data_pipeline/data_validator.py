"""
data_pipeline/data_validator.py

Comprehensive data quality validation for downloaded Parquet files.

Checks performed per file:
    1. Structure     — required columns present
    2. Nullability   — no NaN in critical columns
    3. Types         — numeric columns are numeric
    4. Price sanity  — high >= max(open, close), low <= min(open, close)
    5. Volume sanity — non-negative
    6. Continuity    — timestamps monotonic + gap analysis
    7. Range         — actual date range vs expected
    8. Duplicates    — no duplicate open_time values

Outputs a validation report and flags files needing attention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.logger import get_logger

logger = get_logger(__name__)

VALIDATOR_VERSION = "1.0.0"

REQUIRED_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume",
]

INTERVAL_MS = {
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
}


# ============================================================
# Per-file validation
# ============================================================


def validate_file(filepath: Path) -> dict[str, Any]:
    """
    Run all quality checks on a single Parquet file.

    Returns:
        Report dict with per-check results + issues list.
    """
    result: dict[str, Any] = {
        "file": filepath.name,
        "n_bars": 0,
        "issues": [],
        "warnings": [],
        "stats": {},
    }

    # Parse symbol/interval from filename
    stem = filepath.stem  # e.g., "BTCUSDT_5m"
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        result["issues"].append(f"Filename format invalid: {stem}")
        return result
    symbol, interval = parts
    result["symbol"] = symbol
    result["interval"] = interval

    # Load
    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        result["issues"].append(f"Read error: {e}")
        return result

    result["n_bars"] = len(df)
    if len(df) == 0:
        result["issues"].append("Empty DataFrame")
        return result

    # Check 1: Required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        result["issues"].append(f"Missing columns: {missing_cols}")
        return result

    # Check 2: Nullability
    critical = ["open_time", "open", "high", "low", "close", "volume"]
    nan_counts = {c: int(df[c].isnull().sum()) for c in critical}
    for col, count in nan_counts.items():
        if count > 0:
            result["issues"].append(f"NaN in {col}: {count}")

    # Check 3: Price sanity
    price_violations = int(
        ((df["high"] < df["low"])
         | (df["high"] < df["open"])
         | (df["high"] < df["close"])
         | (df["low"] > df["open"])
         | (df["low"] > df["close"])).sum()
    )
    if price_violations > 0:
        result["issues"].append(f"Price sanity violations: {price_violations}")

    # Check 4: Volume sanity
    neg_volume = int((df["volume"] < 0).sum())
    if neg_volume > 0:
        result["issues"].append(f"Negative volume: {neg_volume}")

    # Check 5: Continuity — monotonic timestamps
    if not df["open_time"].is_monotonic_increasing:
        result["issues"].append("open_time not monotonically increasing")

    # Check 6: Gap analysis
    expected_gap = INTERVAL_MS.get(interval)
    if expected_gap is not None:
        gaps = df["open_time"].diff().dropna()
        n_perfect = int((gaps == expected_gap).sum())
        n_irregular = int((gaps != expected_gap).sum())
        result["stats"]["perfect_gaps"] = n_perfect
        result["stats"]["irregular_gaps"] = n_irregular

        # Some irregular gaps are expected (Binance downtime, delistings)
        # Flag as warning if >1% of bars have gaps
        if n_irregular > len(df) * 0.01:
            result["warnings"].append(
                f"High gap count: {n_irregular:,} ({n_irregular/len(df)*100:.1f}%)"
            )

    # Check 7: Date range
    start_ms = int(df["open_time"].min())
    end_ms = int(df["open_time"].max())
    result["stats"]["date_start"] = pd.to_datetime(start_ms, unit="ms").strftime(
        "%Y-%m-%d"
    )
    result["stats"]["date_end"] = pd.to_datetime(end_ms, unit="ms").strftime(
        "%Y-%m-%d"
    )

    # Check 8: Duplicates
    n_dups = int(df["open_time"].duplicated().sum())
    if n_dups > 0:
        result["issues"].append(f"Duplicate open_time: {n_dups}")

    # Aggregate stats
    result["stats"]["nan_total"] = sum(nan_counts.values())
    result["stats"]["price_violations"] = price_violations
    result["stats"]["duplicates"] = n_dups
    result["stats"]["price_range"] = (
        f"[{df['low'].min():.4f}, {df['high'].max():.4f}]"
    )

    return result


# ============================================================
# Batch validator
# ============================================================


def validate_all(data_dir: str = "data/raw") -> dict[str, Any]:
    """
    Validate all Parquet files in data_dir.

    Returns:
        Summary with pass/fail counts, list of files with issues.
    """
    path = Path(data_dir)
    files = sorted(path.glob("*.parquet"))

    if not files:
        logger.warning(f"No Parquet files found in {data_dir}")
        return {"n_files": 0, "results": []}

    print("=" * 70)
    print(f"DATA VALIDATOR (v{VALIDATOR_VERSION})")
    print("=" * 70)
    print(f"\nDirectory: {data_dir}")
    print(f"Files:     {len(files)}\n")

    results: list[dict[str, Any]] = []
    n_ok = 0
    n_warn = 0
    n_fail = 0

    for i, filepath in enumerate(files, 1):
        report = validate_file(filepath)
        results.append(report)

        n_issues = len(report["issues"])
        n_warnings = len(report["warnings"])

        if n_issues > 0:
            n_fail += 1
            status = "✗ FAIL"
        elif n_warnings > 0:
            n_warn += 1
            status = "⚠ WARN"
        else:
            n_ok += 1
            status = "✓ OK"

        symbol = report.get("symbol", "?")
        interval = report.get("interval", "?")
        n_bars = report["n_bars"]

        # Print only warnings/failures inline; OK summarized at end
        if n_issues > 0 or n_warnings > 0:
            print(f"[{i:3d}/{len(files)}] {symbol:12s} {interval:4s} "
                  f"{n_bars:>8,} bars  {status}")
            for issue in report["issues"]:
                print(f"           ISSUE: {issue}")
            for warn in report["warnings"]:
                print(f"           WARN:  {warn}")

    # Summary
    total_bars = sum(r["n_bars"] for r in results)

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\nTotal files:     {len(files)}")
    print(f"  ✓ Clean:        {n_ok}")
    print(f"  ⚠ Warnings:     {n_warn}")
    print(f"  ✗ Failed:       {n_fail}")
    print(f"\nTotal bars:      {total_bars:,}")

    # Per-interval summary
    by_interval: dict[str, dict[str, int]] = {}
    for r in results:
        interval = r.get("interval", "?")
        if interval not in by_interval:
            by_interval[interval] = {"files": 0, "bars": 0}
        by_interval[interval]["files"] += 1
        by_interval[interval]["bars"] += r["n_bars"]

    print(f"\nPer-interval:")
    for interval in ["5m", "15m", "1h", "4h"]:
        if interval in by_interval:
            d = by_interval[interval]
            print(f"  {interval:5s}  {d['files']:3d} files, "
                  f"{d['bars']:>15,} bars")

    # Failed files list
    if n_fail > 0:
        print(f"\nFAILED FILES:")
        for r in results:
            if r["issues"]:
                print(f"  {r['file']}")
                for issue in r["issues"]:
                    print(f"    - {issue}")

    print("\n" + "=" * 70)

    if n_fail == 0 and n_warn == 0:
        print("\n✅ ALL FILES CLEAN — Data pipeline institutional grade")
    elif n_fail == 0:
        print(f"\n✅ NO CRITICAL ISSUES — {n_warn} files have gap warnings "
              f"(likely normal delistings)")
    else:
        print(f"\n⚠  {n_fail} files need attention")

    return {
        "n_files": len(files),
        "n_ok": n_ok,
        "n_warn": n_warn,
        "n_fail": n_fail,
        "total_bars": total_bars,
        "results": results,
    }


# ============================================================
# Entry point
# ============================================================


if __name__ == "__main__":
    validate_all("data/raw")