"""
features/base/volume_features.py

Module 2: Volume-based features.

10 features capturing volume dynamics and buy/sell pressure.

Feature categories:
    - Volume magnitude (normalized)
    - Volume flow (buy/sell pressure via taker data)
    - Volume-price relationships (money flow)
    - Volume regime (rising/falling activity)

Key insight from quantitative finance:
    Volume reveals CONVICTION behind price moves.
    Same 1% price change means different things at 10x volume vs 0.1x volume.

All features are:
    - Scale-invariant (BTC vs DOGE volumes comparable)
    - Look-ahead safe
    - Cumulative-drift-free (slope-based where needed)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.logger import get_logger
from features.base.utils import (
    safe_rolling_mean,
    safe_rolling_std,
    series_from_array,
    slope_of_cumulative,
    zscore,
)

logger = get_logger(__name__)

VOLUME_FEATURES_VERSION = "1.0.0"


# ============================================================
# Main computation
# ============================================================


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 10 volume-based features.

    Args:
        df: DataFrame with columns [close, volume, taker_buy_volume,
                                     quote_volume]

    Returns:
        df with 10 new feature columns.

    Features (all scale-invariant):
        1.  volume_zscore_20      — Rolling z-score of volume
        2.  volume_ratio_20       — volume / mean(volume_20)
        3.  volume_slope_20       — Rate of change of volume
        4.  taker_buy_ratio       — taker_buy_vol / total_vol
                                    (buy pressure indicator)
        5.  taker_buy_ratio_ma_20 — Smoothed buy pressure
        6.  quote_vol_ratio       — quote_vol / (close × volume)
                                    (avg trade size proxy)
        7.  volume_price_corr_20  — Rolling corr(volume, log_return)
        8.  obv_slope_20          — On-Balance Volume slope (drift-free)
        9.  volume_regime         — z-scored current vs historical
        10. large_bar_indicator   — 1 if volume > 2σ above mean
    """
    if df.empty:
        raise ValueError("Cannot compute features on empty DataFrame")

    required = ["close", "volume", "taker_buy_volume", "quote_volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    eps = 1e-10

    volume = out["volume"]
    close = out["close"]
    taker_buy = out["taker_buy_volume"]
    quote_vol = out["quote_volume"]

    # ============================================================
    # Feature 1: Volume z-score (20-bar)
    # ============================================================
    # Standardizes volume: mean=0, std=1 over rolling window.
    # Makes BTC and DOGE volumes DIRECTLY COMPARABLE.
    out["volume_zscore_20"] = zscore(volume, window=20)

    # ============================================================
    # Feature 2: Volume ratio (current / 20-bar avg)
    # ============================================================
    # Simple multiplier: 1.0 = average, 2.0 = double normal, etc.
    # Intuitive interpretation for humans + ML models.
    vol_mean_20 = safe_rolling_mean(volume, 20)
    out["volume_ratio_20"] = volume / (vol_mean_20 + eps)

    # ============================================================
    # Feature 3: Volume slope (rate of change)
    # ============================================================
    # Is volume trending up or down?
    # Positive = increasing activity, negative = drying up.
    out["volume_slope_20"] = slope_of_cumulative(volume, window=20) / (
        vol_mean_20 + eps
    )

    # ============================================================
    # Feature 4: Taker buy ratio (buy pressure)
    # ============================================================
    # Binance provides "taker buy volume" = aggressive buyers.
    # ratio > 0.5 → more aggressive buying
    # ratio < 0.5 → more aggressive selling
    # THIS IS ORDER FLOW SIGNAL — very valuable.
    out["taker_buy_ratio"] = taker_buy / (volume + eps)

    # ============================================================
    # Feature 5: Smoothed taker buy ratio (20-bar MA)
    # ============================================================
    # Removes noise from feature 4.
    # Reveals SUSTAINED buy/sell pressure.
    out["taker_buy_ratio_ma_20"] = safe_rolling_mean(
        out["taker_buy_ratio"], 20
    )

    # ============================================================
    # Feature 6: Average trade size proxy
    # ============================================================
    # quote_volume = volume × avg_price (approximately)
    # Ratio reveals avg trade size (institutional vs retail).
    # High = large institutional trades, Low = retail flurry.
    out["quote_vol_ratio"] = quote_vol / ((close * volume) + eps)

    # ============================================================
    # Feature 7: Volume-price correlation (rolling)
    # ============================================================
    # Positive corr: volume rises with price → bullish accumulation
    # Negative corr: volume rises when price falls → distribution
    # Zero corr: no clear pattern
    log_returns = pd.Series(
    np.log(close / close.shift(1)), index=close.index, dtype="float64"
)
# Rolling correlation between log returns and volume
    out["volume_price_corr_20"] = (
    log_returns.rolling(window=20, min_periods=20)
    .corr(volume)
)

    # ============================================================
    # Feature 8: OBV slope (drift-free!)
    # ============================================================
    # OBV = cumulative sum of signed volume
    # LESSON FROM V1: Raw OBV drifts, USELESS for ML.
    # Solution: Use SLOPE of OBV (stationary).
    close_arr = close.to_numpy(dtype="float64")
    volume_arr = volume.to_numpy(dtype="float64")

    # Sign of price change (+1 up, -1 down, 0 flat)
    price_change = np.diff(close_arr, prepend=close_arr[0])
    signs = np.sign(price_change)
    signed_volume = signs * volume_arr

    # Cumulative sum (this is raw OBV — DON'T use directly)
    obv_arr = np.cumsum(signed_volume)
    obv_series = series_from_array(obv_arr, close.index)

    # Slope of OBV = stationary, ML-friendly
    obv_slope = slope_of_cumulative(obv_series, window=20)
    # Normalize by rolling volume mean (scale-invariance across symbols)
    out["obv_slope_20"] = obv_slope / (vol_mean_20 + eps)

    # ============================================================
    # Feature 9: Volume regime (long-term z-score)
    # ============================================================
    # Same as feature 1 but longer window (100 bars).
    # Detects LONGER-TERM volume regimes:
    #   High z-score = high-activity regime (news, breakout)
    #   Low z-score = low-activity regime (consolidation, holidays)
    out["volume_regime"] = zscore(volume, window=100)

    # ============================================================
    # Feature 10: Large bar indicator (binary signal)
    # ============================================================
    # Binary: 1 if this bar is unusually high volume, else 0.
    # ML models love binary indicators for regime shifts.
    vol_mean = safe_rolling_mean(volume, 20)
    vol_std = safe_rolling_std(volume, 20)
    threshold = vol_mean + 2 * vol_std
    out["large_bar_indicator"] = (volume > threshold).astype("float64")

    return out


# ============================================================
# Feature list
# ============================================================


VOLUME_FEATURE_NAMES = [
    "volume_zscore_20",
    "volume_ratio_20",
    "volume_slope_20",
    "taker_buy_ratio",
    "taker_buy_ratio_ma_20",
    "quote_vol_ratio",
    "volume_price_corr_20",
    "obv_slope_20",
    "volume_regime",
    "large_bar_indicator",
]


# ============================================================
# Self-test
# ============================================================


if __name__ == "__main__":
    from pathlib import Path
    import time

    print("=" * 70)
    print(f"Volume Features Smoke Test (v{VOLUME_FEATURES_VERSION})")
    print("=" * 70)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    test_file = PROJECT_ROOT / "data" / "raw" / "BTCUSDT_5m.parquet"

    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        exit(1)

    print("\nLoading BTCUSDT 5m data...")
    df = pd.read_parquet(test_file)
    print(f"  Loaded {len(df):,} bars × {len(df.columns)} columns")

    df_test = df.iloc[-10000:].reset_index(drop=True).copy()
    print(f"  Test subset: {len(df_test):,} bars")

    print("\nComputing 10 volume features...")
    t0 = time.time()
    result = compute_volume_features(df_test)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s ({len(df_test)/elapsed:,.0f} bars/sec)")

    # Verify all features
    print("\nFeatures created:")
    for fname in VOLUME_FEATURE_NAMES:
        status = "✅" if fname in result.columns else "❌"
        print(f"  {status} {fname}")

    # Statistics
    print("\nFeature statistics:")
    print("-" * 70)
    for fname in VOLUME_FEATURE_NAMES:
        values = result[fname].dropna()
        if len(values) == 0:
            print(f"  {fname:25s} ALL NaN")
            continue
        print(f"  {fname:25s} "
              f"mean={values.mean():+.4f}  "
              f"std={values.std():.4f}  "
              f"[{values.min():+.3f}, {values.max():+.3f}]")

    # NaN check
    print("\nNaN check:")
    print("-" * 70)
    for fname in VOLUME_FEATURE_NAMES:
        n_nan = int(result[fname].isnull().sum())
        first_valid = result[fname].first_valid_index()
        print(f"  {fname:25s} {n_nan:4d} NaN, "
              f"first valid at bar {first_valid}")

    # Sanity checks specific to volume
    print("\nSanity checks:")
    print("-" * 70)

    # Taker buy ratio should be in [0, 1]
    tbr = result["taker_buy_ratio"].dropna()
    tbr_valid = bool((tbr >= 0).all() and (tbr <= 1).all())
    print(f"  taker_buy_ratio in [0,1]:      {'✅' if tbr_valid else '❌'} "
          f"(mean={tbr.mean():.3f}, expected ≈0.5)")

    # Volume z-score should have mean ≈ 0, std ≈ 1 (after warmup)
    vz = result["volume_zscore_20"].dropna()
    print(f"  volume_zscore mean:            {vz.mean():+.4f} "
          f"(expected ≈0)")
    print(f"  volume_zscore std:             {vz.std():.4f} "
          f"(expected ≈1)")

    # Large bar indicator should be binary
    lbi = result["large_bar_indicator"].dropna()
    is_binary = bool(((lbi == 0) | (lbi == 1)).all())
    print(f"  large_bar_indicator binary:    {'✅' if is_binary else '❌'} "
          f"(fires ~{lbi.mean()*100:.1f}% of bars)")

    # OBV slope should be stationary (not drifting)
    obv = result["obv_slope_20"].dropna()
    obv_mean = obv.mean()
    print(f"  obv_slope stationary:          {'✅' if abs(obv_mean) < 1 else '⚠'} "
          f"(mean={obv_mean:+.4f}, should be near 0)")

    print("\n✅ Volume features module working — 10 features generated")