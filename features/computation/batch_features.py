"""
features/computation/batch_features.py

Batch feature computation — runs the feature engine on all symbols.

Reads symbol universe from configs/symbols_config.yaml,
computes all 116 features for each, saves to data/features/.

Robustness:
    - Per-symbol try/except (one failure doesn't kill batch)
    - Resume capability (skip already-processed symbols)
    - Progress + ETA reporting
    - Summary at the end
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from core.config_loader import load_config
from core.logger import get_logger
from features.computation.feature_engine import (
    process_symbol, ALL_FEATURE_NAMES,
)

logger = get_logger(__name__)

BATCH_FEATURES_VERSION = "1.0.0"


def flatten_symbols(symbols_config: dict[str, Any]) -> list[str]:
    """Flatten tiered symbol dict into a flat symbol list."""
    result: list[str] = []
    for _tier, symbol_list in symbols_config.items():
        if isinstance(symbol_list, list):
            result.extend(symbol_list)
    return result


def run_batch_features(
    resume: bool = True,
    raw_dir: str = "data/raw",
    output_dir: str = "data/features",
) -> dict[str, Any]:
    """
    Compute features for all symbols in the universe.

    Args:
        resume: Skip symbols whose feature file already exists.
        raw_dir: Raw data directory.
        output_dir: Feature output directory.

    Returns:
        Summary dict.
    """
    cfg = load_config("symbols_config")
    symbols = flatten_symbols(cfg["symbols"])
    n = len(symbols)

    print("=" * 70)
    print(f"BATCH FEATURE COMPUTATION (v{BATCH_FEATURES_VERSION})")
    print("=" * 70)
    print(f"\nSymbols:  {n}")
    print(f"Features: {len(ALL_FEATURE_NAMES)} per symbol")
    print(f"Resume:   {resume}")
    print(f"Output:   {output_dir}\n")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    t_start = time.time()

    for i, symbol in enumerate(symbols, 1):
        prefix = f"[{i:2d}/{n}] {symbol:12s}"

        # Resume: skip existing
        feature_file = out_path / f"{symbol}_5m.parquet"
        if resume and feature_file.exists():
            try:
                existing = pd.read_parquet(feature_file)
                print(f"{prefix} ⏭ SKIP ({len(existing):,} bars)")
                results.append({
                    "symbol": symbol, "status": "skipped",
                    "bars": len(existing), "features": 0, "elapsed_s": 0.0,
                })
                continue
            except Exception:
                pass  # corrupt, recompute

        # Process
        result = process_symbol(symbol, raw_dir, output_dir)
        results.append(result)

        status = result["status"]
        bars = result.get("bars", 0)
        feats = result.get("features", 0)
        elapsed_s = result.get("elapsed_s", 0.0)
        if status == "ok":
            print(f"{prefix} ✓ {bars:>7,} bars, "
                  f"{feats} feats ({elapsed_s:.0f}s)")
        elif status == "skipped":
            print(f"{prefix} ⏭ SKIP (data missing)")
        else:
            print(f"{prefix} ✗ ERROR: {result.get('error', '?')}")

        # ETA every 5 symbols
        if i % 5 == 0:
            elapsed = time.time() - t_start
            avg = elapsed / i
            eta = (n - i) * avg
            print(f"      ... {elapsed/60:.1f} min elapsed, "
                  f"ETA {eta/60:.1f} min\n")

    total_time = time.time() - t_start

    # Summary
    df = pd.DataFrame(results)
    n_ok = int((df["status"] == "ok").sum())
    n_skip = int((df["status"] == "skipped").sum())
    n_err = int((df["status"] == "error").sum())
    total_bars = int(df["bars"].sum())

    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    print(f"\nTotal symbols:  {n}")
    print(f"  ✓ Processed:  {n_ok}")
    print(f"  ⏭ Skipped:    {n_skip}")
    print(f"  ✗ Errors:     {n_err}")
    print(f"\nTotal bars:     {total_bars:,}")
    print(f"Total time:     {total_time/60:.1f} min")

    if n_err > 0:
        print(f"\nErrors:")
        for _, row in df[df["status"] == "error"].iterrows():
            print(f"  {row['symbol']}: {row.get('error', '?')}")

    print("\n" + "=" * 70)
    if n_err == 0:
        print("✅ BATCH COMPLETE — all features computed")
    else:
        print(f"⚠  {n_err} errors — check log")

    return {
        "n_ok": n_ok, "n_skip": n_skip, "n_err": n_err,
        "total_bars": total_bars, "total_time_s": total_time,
    }


if __name__ == "__main__":
    run_batch_features(resume=True)