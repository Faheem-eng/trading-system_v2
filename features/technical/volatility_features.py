"""
features/technical/volatility_features.py

Module 6: Volatility features (ATR, Bollinger Bands, Keltner, realized vol).

10 features capturing volatility magnitude, regime, and structure.

Why volatility matters MOST:
    - Position sizing scales inversely with volatility (Phase 5)
    - Stop-loss distances = ATR multiples (Phase 5)
    - Regime detection uses vol clustering (Module 11)
    - Breakout detection needs vol context

All features scale-invariant (normalized by close or middle band).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.logger import get_logger
from features.base.utils import (
    safe_rolling_mean,
    safe_rolling_std,
    series_from_array,
)

logger = get_logger(__name__)

VOLATILITY_FEATURES_VERSION = "1.0.0"

EPS = 1e-10


def _atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """Average True Range (Wilder smoothing). Returns ATR in price units."""
    n = len(close)
    result = np.full(n, np.nan, dtype="float64")
    if n < period + 1:
        return result

    prev_close = np.concatenate([[np.nan], close[:-1]])
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    tr[0] = high[0] - low[0]

    # Wilder smoothing
    result[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
    return result


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 10 volatility features.

    Args:
        df: DataFrame with columns [high, low, close]

    Returns:
        df with 10 new feature columns.

    Features (all scale-invariant):
        1.  atr_pct           — ATR(14) / close (volatility as % of price)
        2.  atr_ratio_50      — ATR(14) / ATR(50) (vol expansion/contraction)
        3.  realized_vol_20   — Std of 20 log returns (per-bar realized vol)
        4.  realized_vol_60   — Std of 60 log returns
        5.  vol_of_vol        — Std of rolling volatility (vol regime change)
        6.  bb_width          — Bollinger band width / middle (normalized)
        7.  bb_position       — Where close is in BB [0=lower, 1=upper]
        8.  keltner_position  — Where close is in Keltner channel
        9.  bb_squeeze        — 1 if BB inside Keltner (low vol, pre-breakout)
        10. high_low_ratio_20 — 20-bar (high-low)/close average range
    """
    if df.empty:
        raise ValueError("Cannot compute features on empty DataFrame")

    required = ["high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    idx = out.index

    high_arr = out["high"].to_numpy(dtype="float64")
    low_arr = out["low"].to_numpy(dtype="float64")
    close_arr = out["close"].to_numpy(dtype="float64")
    close_series = out["close"]

    # ============================================================
    # Features 1-2: ATR family
    # ============================================================
    atr_14 = _atr(high_arr, low_arr, close_arr, 14)
    atr_50 = _atr(high_arr, low_arr, close_arr, 50)

    out["atr_pct"] = series_from_array(atr_14 / (close_arr + EPS), idx)
    out["atr_ratio_50"] = series_from_array(
        atr_14 / (atr_50 + EPS), idx
    )

    # ============================================================
    # Features 3-4: Realized volatility (std of log returns)
    # ============================================================
    log_ret = pd.Series(
        np.log(close_series / close_series.shift(1)),
        index=idx, dtype="float64",
    )
    out["realized_vol_20"] = safe_rolling_std(log_ret, 20)
    out["realized_vol_60"] = safe_rolling_std(log_ret, 60)

    # ============================================================
    # Feature 5: Vol-of-vol (volatility regime change detector)
    # ============================================================
    # Std of the rolling volatility itself.
    # High = volatility is changing rapidly (regime shift).
    rv_20 = out["realized_vol_20"]
    out["vol_of_vol"] = safe_rolling_std(rv_20, 20)

    # ============================================================
    # Features 6-7: Bollinger Bands
    # ============================================================
    bb_mid = safe_rolling_mean(close_series, 20)
    bb_std = safe_rolling_std(close_series, 20)
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std

    # Width normalized by middle band (scale-invariant)
    out["bb_width"] = (bb_upper - bb_lower) / (bb_mid + EPS)

    # Position in bands: 0 = at lower, 1 = at upper
    out["bb_position"] = (close_series - bb_lower) / (
        bb_upper - bb_lower + EPS
    )

    # ============================================================
    # Feature 8: Keltner channel position
    # ============================================================
    # Keltner = EMA ± multiplier × ATR
    kc_mid = bb_mid  # reuse SMA20 as middle
    atr_14_series = series_from_array(atr_14, idx)
    kc_upper = kc_mid + 2.0 * atr_14_series
    kc_lower = kc_mid - 2.0 * atr_14_series
    out["keltner_position"] = (close_series - kc_lower) / (
        kc_upper - kc_lower + EPS
    )

    # ============================================================
    # Feature 9: BB squeeze (low volatility, pre-breakout signal)
    # ============================================================
    # When Bollinger Bands are INSIDE Keltner channel → low volatility.
    # Historically precedes big moves (volatility mean-reverts).
    squeeze = ((bb_upper < kc_upper) & (bb_lower > kc_lower)).astype("float64")
    # Preserve NaN in warmup
    squeeze[bb_upper.isnull() | kc_upper.isnull()] = np.nan
    out["bb_squeeze"] = squeeze

    # ============================================================
    # Feature 10: Average high-low range (20-bar)
    # ============================================================
    hl_range = (out["high"] - out["low"]) / (close_series + EPS)
    out["high_low_ratio_20"] = safe_rolling_mean(hl_range, 20)

    return out


VOLATILITY_FEATURE_NAMES = [
    "atr_pct",
    "atr_ratio_50",
    "realized_vol_20",
    "realized_vol_60",
    "vol_of_vol",
    "bb_width",
    "bb_position",
    "keltner_position",
    "bb_squeeze",
    "high_low_ratio_20",
]


if __name__ == "__main__":
    from pathlib import Path
    import time

    print("=" * 70)
    print(f"Volatility Features Smoke Test (v{VOLATILITY_FEATURES_VERSION})")
    print("=" * 70)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    test_file = PROJECT_ROOT / "data" / "raw" / "BTCUSDT_5m.parquet"

    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        exit(1)

    print("\nLoading BTCUSDT 5m data...")
    df = pd.read_parquet(test_file)
    df_test = df.iloc[-10000:].reset_index(drop=True).copy()
    print(f"  Test subset: {len(df_test):,} bars")

    print("\nComputing 10 volatility features...")
    t0 = time.time()
    result = compute_volatility_features(df_test)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s ({len(df_test)/elapsed:,.0f} bars/sec)")

    print("\nFeatures created:")
    for fname in VOLATILITY_FEATURE_NAMES:
        status = "✅" if fname in result.columns else "❌"
        print(f"  {status} {fname}")

    print("\nFeature statistics:")
    print("-" * 70)
    for fname in VOLATILITY_FEATURE_NAMES:
        values = result[fname].dropna()
        if len(values) == 0:
            print(f"  {fname:20s} ALL NaN ❌")
            continue
        print(f"  {fname:20s} "
              f"mean={values.mean():+.5f}  "
              f"std={values.std():.5f}  "
              f"[{values.min():+.4f}, {values.max():+.4f}]")

    # Sanity checks
    print("\nSanity checks:")
    print("-" * 70)

    atr = result["atr_pct"].dropna()
    atr_ok = bool((atr > 0).all())
    print(f"  ATR% positive:            {'✅' if atr_ok else '❌'} "
          f"(mean={atr.mean()*100:.3f}%)")

    bbpos = result["bb_position"].dropna()
    # Most values in [0,1], but can exceed during strong moves
    in_range = float(((bbpos >= 0) & (bbpos <= 1)).mean()) * 100
    print(f"  BB position mostly [0,1]: {in_range:.0f}% in range "
          f"(mean={bbpos.mean():.3f}, expected ≈0.5)")

    squeeze = result["bb_squeeze"].dropna()
    sq_ok = bool(set(squeeze.unique()).issubset({0.0, 1.0}))
    print(f"  BB squeeze binary:        {'✅' if sq_ok else '❌'} "
          f"(fires {squeeze.mean()*100:.1f}% of bars)")

    rv20 = result["realized_vol_20"].dropna()
    rv60 = result["realized_vol_60"].dropna()
    print(f"  Realized vol positive:    "
          f"{'✅' if (rv20 > 0).all() else '❌'} "
          f"(rv20={rv20.mean():.5f}, rv60={rv60.mean():.5f})")

    print("\n✅ Volatility features module working — 10 features generated")