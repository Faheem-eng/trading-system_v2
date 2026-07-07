"""
features/computation/batch_features_1h.py

Batch 1h feature computation for multi-symbol validation.

Computes 1h features for a curated set of liquid symbols to test
whether the 1h edge (found on BTC) generalizes across the market.

Uses a focused symbol set (most liquid) rather than all 56 —
enough for statistical validation without excessive compute.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from core.logger import get_logger
from features.computation.feature_engine_1h import process_symbol_1h

logger = get_logger(__name__)

BATCH_1H_VERSION = "1.0.0"

# Curated liquid symbols across sectors (for edge validation)
VALIDATION_SYMBOLS = [
    "BTCUSDT",   # already done, will skip
    "ETHUSDT",   # large cap
    "BNBUSDT",   # large cap
    "XRPUSDT",   # large cap
    "SOLUSDT",   # layer1
    "ADAUSDT",   # layer1
    "AVAXUSDT",  # layer1
    "LINKUSDT",  # infra
    "DOTUSDT",   # layer1
    "MATICUSDT", # layer2
    "LTCUSDT",   # established
    "ATOMUSDT",  # layer1
]


def run_batch_1h(
    symbols: list[str] | None = None,
    resume: bool = True,
    output_dir: str = "data/features_1h",
) -> dict[str, Any]:
    """Compute 1h features for validation symbols."""
    if symbols is None:
        symbols = VALIDATION_SYMBOLS

    n = len(symbols)
    print("=" * 70)
    print(f"BATCH 1H FEATURES — Multi-Symbol Validation (v{BATCH_1H_VERSION})")
    print("=" * 70)
    print(f"\nSymbols: {n}")
    print(f"Output:  {output_dir}\n")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = []
    t_start = time.time()

    for i, symbol in enumerate(symbols, 1):
        prefix = f"[{i:2d}/{n}] {symbol:12s}"

        feature_file = out_path / f"{symbol}_1h.parquet"
        if resume and feature_file.exists():
            existing = pd.read_parquet(feature_file)
            print(f"{prefix} ⏭ SKIP ({len(existing):,} bars)")
            results.append({"symbol": symbol, "status": "skipped"})
            continue

        result = process_symbol_1h(symbol, output_dir=output_dir)
        results.append(result)

        if result["status"] == "ok":
            print(f"{prefix} ✓ {result['bars']:,} bars ({result['elapsed_s']:.0f}s)")
        elif result["status"] == "skipped":
            print(f"{prefix} ⏭ SKIP (data missing)")
        else:
            print(f"{prefix} ✗ ERROR: {result.get('error', '?')}")

    total_time = time.time() - t_start

    df = pd.DataFrame(results)
    n_ok = int((df["status"] == "ok").sum())
    n_skip = int((df["status"] == "skipped").sum())
    n_err = int((df["status"] == "error").sum())

    print("\n" + "=" * 70)
    print(f"DONE: {n_ok} computed, {n_skip} skipped, {n_err} errors "
          f"({total_time/60:.1f} min)")
    print("=" * 70)

    return {"n_ok": n_ok, "n_skip": n_skip, "n_err": n_err}


if __name__ == "__main__":
    run_batch_1h()