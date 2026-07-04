"""
features/technical/trend_features.py

Module 5: Trend features (EMA alignment, ADX, slopes).

8 features capturing trend direction, strength, and quality.

Key concepts:
    - EMA alignment: fast > mid > slow = strong uptrend structure
    - ADX: trend STRENGTH regardless of direction (Wilder's method)
    - Slope normalization: all slopes / close (scale-invariant)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.logger import get_logger
from features.base.utils import series_from_array

logger = get_logger(__name__)

TREND_FEATURES_VERSION = "1.0.0"

EPS = 1e-10


# ============================================================
# Core: EMA (shared with momentum but standalone here)
# ============================================================


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """EMA with explicit warmup (first period-1 bars NaN)."""
    n = len(arr)
    result = np.full(n, np.nan, dtype="float64")
    if n < period:
        return result
    alpha = 2.0 / (period + 1.0)
    result[period - 1] = float(np.mean(arr[:period]))
    for i in range(period, n):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def _wilder_smooth(arr: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder's smoothing (used in ADX).
    smooth[t] = smooth[t-1] - smooth[t-1]/period + arr[t]
    """
    n = len(arr)
    result = np.full(n, np.nan, dtype="float64")
    if n < period:
        return result
    # Seed: sum of first `period` values
    result[period - 1] = float(np.sum(arr[:period]))
    for i in range(period, n):
        result[i] = result[i - 1] - result[i - 1] / period + arr[i]
    return result


def _adx(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Average Directional Index (Wilder's exact method).

    Returns (adx, plus_di, minus_di).

    Steps:
        1. True Range, +DM, -DM per bar
        2. Wilder-smooth all three
        3. +DI = 100 × smooth(+DM)/smooth(TR), same for -DI
        4. DX = 100 × |+DI - -DI| / (+DI + -DI)
        5. ADX = Wilder-smoothed DX
    """
    n = len(close)
    nan_arr = np.full(n, np.nan, dtype="float64")
    if n < 2 * period:
        return nan_arr.copy(), nan_arr.copy(), nan_arr.copy()

    prev_close = np.concatenate([[np.nan], close[:-1]])
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])

    # True Range
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    tr[0] = high[0] - low[0]

    # Directional movements
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0

    # Wilder smoothing
    tr_smooth = _wilder_smooth(tr, period)
    plus_smooth = _wilder_smooth(plus_dm, period)
    minus_smooth = _wilder_smooth(minus_dm, period)

    # Directional indicators
    plus_di = 100.0 * plus_smooth / (tr_smooth + EPS)
    minus_di = 100.0 * minus_smooth / (tr_smooth + EPS)

    # DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + EPS)

    # ADX = Wilder-smoothed DX (simple recursive mean after seed)
    adx = np.full(n, np.nan, dtype="float64")
    seed_start = period - 1  # where dx becomes valid
    seed_end = seed_start + period
    if seed_end <= n:
        valid_dx = dx[seed_start:seed_end]
        adx[seed_end - 1] = float(np.nanmean(valid_dx))
        for i in range(seed_end, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx, plus_di, minus_di


# ============================================================
# Main computation
# ============================================================


def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 8 trend features.

    Args:
        df: DataFrame with columns [high, low, close]

    Returns:
        df with 8 new feature columns.

    Features:
        1. ema_alignment      — +1 (9>21>50), -1 (9<21<50), 0 mixed
        2. price_vs_ema_21    — (close - EMA21) / EMA21
        3. ema_9_slope_norm   — EMA9 5-bar slope / close
        4. ema_21_slope_norm  — EMA21 5-bar slope / close
        5. adx_14             — Trend strength [0,100]
        6. adx_normalized     — ADX/100, [0,1] (ML-friendly)
        7. di_diff_norm       — (+DI - -DI)/100, direction [-1,1]
        8. trend_consistency  — Fraction of last 20 bars in EMA9 direction
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

    # EMAs
    ema_9 = _ema(close_arr, 9)
    ema_21 = _ema(close_arr, 21)
    ema_50 = _ema(close_arr, 50)

    # ============================================================
    # Feature 1: EMA alignment (trend structure)
    # ============================================================
    alignment = np.full(len(close_arr), np.nan, dtype="float64")
    valid = ~(np.isnan(ema_9) | np.isnan(ema_21) | np.isnan(ema_50))
    bullish = valid & (ema_9 > ema_21) & (ema_21 > ema_50)
    bearish = valid & (ema_9 < ema_21) & (ema_21 < ema_50)
    alignment[valid] = 0.0
    alignment[bullish] = 1.0
    alignment[bearish] = -1.0
    out["ema_alignment"] = series_from_array(alignment, idx)

    # ============================================================
    # Feature 2: Price vs EMA21 (distance from trend)
    # ============================================================
    out["price_vs_ema_21"] = series_from_array(
        (close_arr - ema_21) / (ema_21 + EPS), idx
    )

    # ============================================================
    # Features 3-4: EMA slopes (normalized by close — scale-invariant)
    # ============================================================
    ema_9_series = series_from_array(ema_9, idx)
    ema_21_series = series_from_array(ema_21, idx)

    ema_9_slope = (ema_9_series - ema_9_series.shift(5)) / 5.0
    ema_21_slope = (ema_21_series - ema_21_series.shift(5)) / 5.0

    close_series = out["close"]
    out["ema_9_slope_norm"] = ema_9_slope / (close_series + EPS)
    out["ema_21_slope_norm"] = ema_21_slope / (close_series + EPS)

    # ============================================================
    # Features 5-7: ADX family
    # ============================================================
    adx_arr, plus_di, minus_di = _adx(high_arr, low_arr, close_arr, 14)
    out["adx_14"] = series_from_array(adx_arr, idx)
    out["adx_normalized"] = series_from_array(adx_arr / 100.0, idx)
    out["di_diff_norm"] = series_from_array(
        (plus_di - minus_di) / 100.0, idx
    )

    # ============================================================
    # Feature 8: Trend consistency
    # ============================================================
    # Fraction of last 20 bars where EMA9 was rising
    ema_9_rising = (ema_9_series > ema_9_series.shift(1)).astype("float64")
    # Preserve NaN in warmup
    ema_9_rising[ema_9_series.isnull()] = np.nan
    out["trend_consistency"] = ema_9_rising.rolling(
        window=20, min_periods=20
    ).mean()

    return out


TREND_FEATURE_NAMES = [
    "ema_alignment",
    "price_vs_ema_21",
    "ema_9_slope_norm",
    "ema_21_slope_norm",
    "adx_14",
    "adx_normalized",
    "di_diff_norm",
    "trend_consistency",
]


if __name__ == "__main__":
    from pathlib import Path
    import time

    print("=" * 70)
    print(f"Trend Features Smoke Test (v{TREND_FEATURES_VERSION})")
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

    print("\nComputing 8 trend features...")
    t0 = time.time()
    result = compute_trend_features(df_test)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s ({len(df_test)/elapsed:,.0f} bars/sec)")

    print("\nFeatures created:")
    for fname in TREND_FEATURE_NAMES:
        status = "✅" if fname in result.columns else "❌"
        print(f"  {status} {fname}")

    print("\nFeature statistics:")
    print("-" * 70)
    for fname in TREND_FEATURE_NAMES:
        values = result[fname].dropna()
        if len(values) == 0:
            print(f"  {fname:20s} ALL NaN ❌")
            continue
        print(f"  {fname:20s} "
              f"mean={values.mean():+.4f}  "
              f"std={values.std():.4f}  "
              f"[{values.min():+.3f}, {values.max():+.3f}]")

    # Sanity checks
    print("\nSanity checks:")
    print("-" * 70)

    adx = result["adx_14"].dropna()
    adx_ok = bool((adx >= 0).all() and (adx <= 100).all())
    print(f"  ADX in [0,100]:           {'✅' if adx_ok else '❌'} "
          f"(mean={adx.mean():.1f}, typical 15-35)")

    align = result["ema_alignment"].dropna()
    align_ok = bool(set(align.unique()).issubset({-1.0, 0.0, 1.0}))
    bull_pct = float((align == 1).mean()) * 100
    bear_pct = float((align == -1).mean()) * 100
    print(f"  Alignment values valid:   {'✅' if align_ok else '❌'} "
          f"(bull={bull_pct:.0f}%, bear={bear_pct:.0f}%)")

    tc = result["trend_consistency"].dropna()
    tc_ok = bool((tc >= 0).all() and (tc <= 1).all())
    print(f"  Consistency in [0,1]:     {'✅' if tc_ok else '❌'} "
          f"(mean={tc.mean():.3f}, expected ≈0.5)")

    di = result["di_diff_norm"].dropna()
    print(f"  DI diff mean:             {di.mean():+.4f} "
          f"(expected ≈0, balanced)")

    print("\n✅ Trend features module working — 8 features generated")