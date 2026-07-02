"""
data_pipeline/batch_collector.py

Production-grade batch data collector.

Orchestrates downloading of all symbols × all intervals with:
    - Progress tracking (per symbol/interval)
    - Graceful error handling (one failure doesn't kill batch)
    - Resume capability (skip already-downloaded files)
    - Summary report at the end
    - ETA estimation

Reads config from configs/symbols_config.yaml.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from core.config_loader import load_config
from core.logger import get_logger
from data_pipeline.binance_downloader import collect_symbol_interval

logger = get_logger(__name__)

BATCH_COLLECTOR_VERSION = "1.0.0"


# ============================================================
# Symbol universe helpers
# ============================================================


def flatten_symbols(symbols_config: dict[str, Any]) -> list[tuple[str, str]]:
    """
    Flatten tiered symbol dict into [(tier, symbol), ...] list.

    Preserves tier info for reporting.
    """
    result: list[tuple[str, str]] = []
    for tier, symbol_list in symbols_config.items():
        if not isinstance(symbol_list, list):
            continue
        for symbol in symbol_list:
            result.append((tier, symbol))
    return result


# ============================================================
# Batch orchestrator
# ============================================================


def run_batch_collection(
    resume: bool = True,
    verify: bool = True,
) -> dict[str, Any]:
    """
    Download all symbols × all intervals per config.

    Args:
        resume: If True, skip files that already exist on disk.
        verify: If True, verify SHA256 checksums.

    Returns:
        Summary report dict with per-symbol/interval stats.
    """
    # Load config
    cfg = load_config("symbols_config")
    symbols_by_tier = cfg["symbols"]
    collection = cfg["collection"]

    intervals: list[str] = collection["intervals"]
    start_date: str = collection["start_date"]
    end_date_cfg = collection["end_date"]
    end_date: str = end_date_cfg if end_date_cfg else time.strftime("%Y-%m-%d")
    output_dir: str = collection["output_dir"]

    # Flatten symbols
    symbol_list = flatten_symbols(symbols_by_tier)
    n_symbols = len(symbol_list)
    n_intervals = len(intervals)
    total_jobs = n_symbols * n_intervals

    print("=" * 70)
    print(f"BATCH COLLECTION (v{BATCH_COLLECTOR_VERSION})")
    print("=" * 70)
    print(f"\nSymbols:   {n_symbols}")
    print(f"Intervals: {n_intervals} ({', '.join(intervals)})")
    print(f"Total jobs: {total_jobs}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Output:    {output_dir}")
    print(f"Resume:    {resume} (skip existing files)")
    print(f"Verify:    {verify} (SHA256 checksums)")
    print(f"\n{'=' * 70}\n")

    # Prepare output dir
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Tracking
    results: list[dict[str, Any]] = []
    t_start = time.time()
    job_num = 0

    for tier, symbol in symbol_list:
        for interval in intervals:
            job_num += 1
            parquet_file = out_path / f"{symbol}_{interval}.parquet"

            job_prefix = f"[{job_num:3d}/{total_jobs}] {symbol:12s} {interval:4s}"

            # Resume: skip if already downloaded
            if resume and parquet_file.exists():
                try:
                    existing = pd.read_parquet(parquet_file)
                    n_bars = len(existing)
                    print(f"{job_prefix} ⏭ SKIP (exists: {n_bars:,} bars)")
                    results.append({
                        "tier": tier,
                        "symbol": symbol,
                        "interval": interval,
                        "status": "skipped",
                        "bars": n_bars,
                        "elapsed_s": 0.0,
                    })
                    continue
                except Exception:
                    # Corrupt existing file — re-download
                    pass

            # Download
            job_t0 = time.time()
            try:
                df = collect_symbol_interval(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date,
                    output_dir=output_dir,
                    verify=verify,
                )
                elapsed = time.time() - job_t0

                if df is None or len(df) == 0:
                    print(f"{job_prefix} ✗ NO DATA ({elapsed:.1f}s)")
                    results.append({
                        "tier": tier,
                        "symbol": symbol,
                        "interval": interval,
                        "status": "no_data",
                        "bars": 0,
                        "elapsed_s": elapsed,
                    })
                else:
                    print(f"{job_prefix} ✓ {len(df):>7,} bars ({elapsed:.1f}s)")
                    results.append({
                        "tier": tier,
                        "symbol": symbol,
                        "interval": interval,
                        "status": "ok",
                        "bars": len(df),
                        "elapsed_s": elapsed,
                    })

            except Exception as e:
                elapsed = time.time() - job_t0
                print(f"{job_prefix} ✗ ERROR: {e} ({elapsed:.1f}s)")
                results.append({
                    "tier": tier,
                    "symbol": symbol,
                    "interval": interval,
                    "status": "error",
                    "bars": 0,
                    "elapsed_s": elapsed,
                    "error": str(e),
                })

            # ETA
            elapsed_total = time.time() - t_start
            avg_per_job = elapsed_total / job_num
            remaining = (total_jobs - job_num) * avg_per_job
            if job_num % 10 == 0:
                print(f"      ... elapsed {elapsed_total/60:.1f} min, "
                      f"ETA {remaining/60:.1f} min\n")

    total_time = time.time() - t_start
    return _build_summary(results, total_time)


# ============================================================
# Summary reporting
# ============================================================


def _build_summary(
    results: list[dict[str, Any]], total_time: float
) -> dict[str, Any]:
    """Build human-readable summary from job results."""
    df = pd.DataFrame(results)

    n_ok = int((df["status"] == "ok").sum())
    n_skip = int((df["status"] == "skipped").sum())
    n_nodata = int((df["status"] == "no_data").sum())
    n_error = int((df["status"] == "error").sum())
    total_bars = int(df["bars"].sum())

    print("\n" + "=" * 70)
    print("BATCH COLLECTION SUMMARY")
    print("=" * 70)
    print(f"\nTotal jobs:        {len(df)}")
    print(f"  ✓ Downloaded:    {n_ok}")
    print(f"  ⏭ Skipped:       {n_skip}")
    print(f"  ✗ No data:       {n_nodata}")
    print(f"  ✗ Errors:        {n_error}")
    print(f"\nTotal bars collected: {total_bars:,}")
    print(f"Total time:           {total_time/60:.1f} min "
          f"({total_time:.0f}s)")

    # Per-tier breakdown
    print("\nPer-tier bars:")
    tier_bars = df.groupby("tier")["bars"].sum().sort_values(ascending=False)
    for tier, bars in tier_bars.items():
        print(f"  {tier:20s} {int(bars):>12,} bars")

    # Per-interval breakdown
    print("\nPer-interval bars:")
    int_bars = df.groupby("interval")["bars"].sum()
    for interval, bars in int_bars.items():
        print(f"  {interval:5s} {int(bars):>12,} bars")

    # Errors detail
    if n_error > 0:
        print("\nErrors detail:")
        errors = df[df["status"] == "error"]
        for _, row in errors.iterrows():
            print(f"  {row['symbol']:12s} {row['interval']:4s} "
                  f"→ {row.get('error', 'unknown')}")

    # No-data detail
    if n_nodata > 0:
        print("\nNo-data (symbols too new for start date):")
        nodata = df[df["status"] == "no_data"]
        for _, row in nodata.iterrows():
            print(f"  {row['symbol']:12s} {row['interval']:4s}")

    print("\n" + "=" * 70)

    return {
        "n_ok": n_ok,
        "n_skip": n_skip,
        "n_nodata": n_nodata,
        "n_error": n_error,
        "total_bars": total_bars,
        "total_time_s": total_time,
        "results": results,
    }


# ============================================================
# Entry point
# ============================================================


if __name__ == "__main__":
    summary = run_batch_collection(resume=True, verify=True)

    if summary["n_error"] > 0:
        print(f"\n⚠  {summary['n_error']} errors occurred — check log")
    else:
        print(f"\n✅ Batch collection complete — {summary['total_bars']:,} "
              f"bars across {summary['n_ok'] + summary['n_skip']} datasets")