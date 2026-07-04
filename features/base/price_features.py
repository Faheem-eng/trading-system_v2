"""
features/base/price_features.py

Module 1: Price-based features.

15 features capturing what price is doing at each bar.

Feature categories:
    - Returns (log returns at multiple horizons)
    - Ranges (high-low, close position within range)
    - Gaps (open vs previous close)
    - Body/wick ratios (candle anatomy)

All features are:
    - Scale-invariant (normalized by close or ATR)
    - Look-ahead safe (only uses data up to current bar)
    - Cross-symbol compatible (BTC $65K to PEPE $0.000012)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.logger import get_logger
import features.base.utils

logger = get_logger(__name__)

PRICE_FEATURES_VERSION = "1.0.0"


# ============================================================
# Main computation
# ============================================================


def compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 15 price-based features.

    Args:
        df: DataFrame with columns [open, high, low, close]

    Returns:
        df with 15 new feature columns added.

    Features (all scale-invariant):
        1.  log_return_1        — 1-bar log return
        2.  log_return_5        — 5-bar log return
        3.  log_return_15       — 15-bar log return
        4.  log_return_60       — 60-bar log return
        5.  hl_range_norm       — (high-low) / close
        6.  oc_range_norm       — (close-open) / close
        7.  close_pos_in_range  — Where close is in bar's range (0=low, 1=high)
        8.  gap_norm            — (open - prev_close) / prev_close
        9.  upper_wick_norm     — Upper wick / total range
        10. lower_wick_norm     — Lower wick / total range
        11. body_ratio          — Body size / total range
        12. price_vs_ma_20      — (close - MA20) / MA20
        13. price_vs_ma_50      — (close - MA50) / MA50
        14. dist_from_high_20   — (high_20 - close) / close
        15. dist_from_low_20    — (close - low_20) / close
    """
    if df.empty:
        raise ValueError("Cannot compute features on empty DataFrame")

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()

    # Extract as numpy arrays (type-safe operations)
    open_arr = out["open"].to_numpy(dtype="float64")
    high_arr = out["high"].to_numpy(dtype="float64")
    low_arr = out["low"].to_numpy(dtype="float64")
    close_arr = out["close"].to_numpy(dtype="float64")
    idx = out.index

    # Small epsilon to prevent division by zero
    eps = 1e-10

    # --- Returns (features 1-4) ---
    close_series = out["close"]
    out["log_return_1"] = features.base.utils.log_return(close_series, 1)
    out["log_return_5"] = features.base.utils.log_return(close_series, 5)
    out["log_return_15"] = features.base.utils.log_return(close_series, 15)
    out["log_return_60"] = features.base.utils.log_return(close_series, 60)

    # --- Ranges (features 5-7) ---
    # High-low range normalized
    hl_range = high_arr - low_arr
    out["hl_range_norm"] = features.base.utils.series_from_array(
        hl_range / (close_arr + eps), idx
    )

    # Open-close range normalized
    oc_range = close_arr - open_arr
    out["oc_range_norm"] = features.base.utils.series_from_array(
        oc_range / (close_arr + eps), idx
    )

    # Close position within bar's range (0 = at low, 1 = at high)
    close_pos = (close_arr - low_arr) / (hl_range + eps)
    out["close_pos_in_range"] = features.base.utils.series_from_array(close_pos, idx)

    # --- Gap (feature 8) ---
    prev_close_arr = np.concatenate([[np.nan], close_arr[:-1]])
    gap = (open_arr - prev_close_arr) / (prev_close_arr + eps)
    out["gap_norm"] = features.base.utils.series_from_array(gap, idx)

    # --- Candle anatomy (features 9-11) ---
    # Upper wick: high - max(open, close)
    body_high = np.maximum(open_arr, close_arr)
    upper_wick = high_arr - body_high
    out["upper_wick_norm"] = features.base.utils.series_from_array(
        upper_wick / (hl_range + eps), idx
    )

    # Lower wick: min(open, close) - low
    body_low = np.minimum(open_arr, close_arr)
    lower_wick = body_low - low_arr
    out["lower_wick_norm"] = features.base.utils.series_from_array(
        lower_wick / (hl_range + eps), idx
    )

    # Body ratio: |close - open| / (high - low)
    body_size = np.abs(close_arr - open_arr)
    out["body_ratio"] = features.base.utils.series_from_array(
        body_size / (hl_range + eps), idx
    )

    # --- Price vs moving averages (features 12-13) ---
    # Simple MA (using pandas for rolling)
    ma_20 = out["close"].rolling(window=20, min_periods=20).mean()
    ma_50 = out["close"].rolling(window=50, min_periods=50).mean()

    out["price_vs_ma_20"] = (out["close"] - ma_20) / (ma_20 + eps)
    out["price_vs_ma_50"] = (out["close"] - ma_50) / (ma_50 + eps)

    # --- Distance from recent extremes (features 14-15) ---
    high_20 = features.base.utils.safe_rolling_max(out["high"], 20)
    low_20 = features.base.utils.safe_rolling_min(out["low"], 20)

    out["dist_from_high_20"] = (high_20 - out["close"]) / (out["close"] + eps)
    out["dist_from_low_20"] = (out["close"] - low_20) / (out["close"] + eps)

    return out


# ============================================================
# Feature list (for validation/documentation)
# ============================================================


PRICE_FEATURE_NAMES = [
    "log_return_1",
    "log_return_5",
    "log_return_15",
    "log_return_60",
    "hl_range_norm",
    "oc_range_norm",
    "close_pos_in_range",
    "gap_norm",
    "upper_wick_norm",
    "lower_wick_norm",
    "body_ratio",
    "price_vs_ma_20",
    "price_vs_ma_50",
    "dist_from_high_20",
    "dist_from_low_20",
]


# ============================================================
# Self-test
# ============================================================


if __name__ == "__main__":
    from pathlib import Path

    print("=" * 70)
    print(f"Price Features Smoke Test (v{PRICE_FEATURES_VERSION})")
    print("=" * 70)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    test_file = PROJECT_ROOT / "data" / "raw" / "BTCUSDT_5m.parquet"

    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        exit(1)

    print("\nLoading BTCUSDT 5m data...")
    df = pd.read_parquet(test_file)
    print(f"  Loaded {len(df):,} bars × {len(df.columns)} columns")

    # Use last 10K bars for speed
    df_test = df.iloc[-10000:].reset_index(drop=True).copy()
    print(f"  Test subset: {len(df_test):,} bars")

    print("\nComputing 15 price features...")
    import time
    t0 = time.time()
    result = compute_price_features(df_test)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s ({len(df_test)/elapsed:,.0f} bars/sec)")

    # Verify all features present
    print("\nFeatures created:")
    missing = []
    for fname in PRICE_FEATURE_NAMES:
        if fname in result.columns:
            print(f"  ✅ {fname}")
        else:
            print(f"  ❌ {fname} MISSING")
            missing.append(fname)

    if missing:
        print(f"\n❌ Missing features: {missing}")
        exit(1)

    # Statistics
    print("\nFeature statistics (mean, std, min, max):")
    print("-" * 70)
    for fname in PRICE_FEATURE_NAMES:
        values = result[fname].dropna()
        if len(values) == 0:
            print(f"  {fname:22s} ALL NaN")
            continue
        print(f"  {fname:22s} "
              f"mean={values.mean():+.5f}  "
              f"std={values.std():.5f}  "
              f"[{values.min():+.4f}, {values.max():+.4f}]")

    # NaN check
    print("\nNaN check (expected in warmup period):")
    print("-" * 70)
    for fname in PRICE_FEATURE_NAMES:
        n_nan = int(result[fname].isnull().sum())
        first_valid = result[fname].first_valid_index()
        print(f"  {fname:22s} {n_nan:4d} NaN, "
              f"first valid at bar {first_valid}")

    print("\n✅ Price features module working — 15 features generated")