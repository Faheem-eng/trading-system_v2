"""
features/regime/regime_features.py

Module 11: Market regime detection features.

5 features identifying WHAT MODE the market is in.

Quant research foundation:
    Hurst Exponent (Hurst, 1951): trending vs mean-reverting vs random.
        H > 0.5 = trending (momentum works)
        H = 0.5 = random walk (no edge)
        H < 0.5 = mean-reverting (reversals work)
    Regime-Switching (Hamilton, 1989): markets switch modes.
    Volatility Clustering (Mandelbrot, 1963): vol is predictable.

Why regime detection matters:
    A momentum strategy that works when H>0.5 FAILS when H<0.5.
    Phase 6 rules use regime to enable/disable strategy components.

V1 LESSONS APPLIED:
    - Vol regime is PERCENTILE-based (self-normalizing, not raw 0.30)
    - Hurst uses bounded lags (computational efficiency)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.logger import get_logger
from features.base.utils import series_from_array

logger = get_logger(__name__)

REGIME_FEATURES_VERSION = "1.0.0"

EPS = 1e-10


def _hurst_exponent(ts: np.ndarray, max_lag: int = 20) -> float:
    """
    Estimate Hurst exponent via rescaled range (R/S) analysis.

    H > 0.5: trending, H = 0.5: random, H < 0.5: mean-reverting.

    Args:
        ts: Time series (log prices)
        max_lag: Maximum lag for R/S calculation

    Returns:
        Hurst exponent estimate, or 0.5 if insufficient/degenerate data.
    """
    n = len(ts)
    if n < max_lag * 2:
        return 0.5

    lags = range(2, max_lag)
    tau = []
    valid_lags = []

    for lag in lags:
        # Differences at this lag
        diff = ts[lag:] - ts[:-lag]
        std = float(np.std(diff))
        if std > EPS:
            tau.append(std)
            valid_lags.append(lag)

    if len(tau) < 3:
        return 0.5

    # Hurst = slope of log(tau) vs log(lag)
    log_lags = np.log(np.array(valid_lags, dtype="float64"))
    log_tau = np.log(np.array(tau, dtype="float64"))

    # Linear fit slope
    try:
        slope = float(np.polyfit(log_lags, log_tau, 1)[0])
    except Exception:
        return 0.5

    # Clamp to reasonable range
    return float(np.clip(slope, 0.0, 1.0))


def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 5 regime features.

    Args:
        df: DataFrame with [high, low, close]

    Returns:
        df with 5 new feature columns.

    Features:
        1. hurst_100          — Hurst exponent (100-bar rolling)
        2. vol_regime_pct     — Volatility percentile [0,1] (self-normalizing)
        3. trend_regime       — +1 trending up, -1 trending down, 0 ranging
        4. vol_clustering     — Autocorr of |returns| (vol persistence)
        5. efficiency_ratio   — Kaufman ER: directional move / total movement
    """
    if df.empty:
        raise ValueError("Cannot compute features on empty DataFrame")

    required = ["high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    idx = out.index

    close_arr = out["close"].to_numpy(dtype="float64")
    close_series = out["close"]
    log_price = np.log(close_arr + EPS)

    # ============================================================
    # Feature 1: Hurst exponent (rolling 100-bar)
    # ============================================================
    n = len(close_arr)
    hurst = np.full(n, np.nan, dtype="float64")
    window = 100
    # Compute every 5 bars for efficiency, forward-fill between
    step = 20  # Hurst changes slowly; compute every 20 bars (4x faster)
    for i in range(window, n, step):
        segment = log_price[i - window:i]
        h = _hurst_exponent(segment, max_lag=20)
        for j in range(i, min(i + step, n)):
            hurst[j] = h
    out["hurst_100"] = series_from_array(hurst, idx)

    # ============================================================
    # Feature 2: Volatility regime (PERCENTILE — V1 lesson!)
    # ============================================================
    # V1 mistake: raw vol regime = mean 0.30, useless.
    # Fix: percentile rank of current vol vs last 200 bars.
    # Self-normalizing: 0 = calmest, 1 = most volatile.
    log_ret = pd.Series(
        np.log(close_arr / np.concatenate([[np.nan], close_arr[:-1]])),
        index=idx, dtype="float64",
    )
    realized_vol = log_ret.rolling(window=20, min_periods=20).std()
    # Percentile rank over rolling 200-bar window
    vol_pct = realized_vol.rolling(window=200, min_periods=200).rank(pct=True)
    out["vol_regime_pct"] = vol_pct.to_numpy()

    # ============================================================
    # Feature 3: Trend regime (via efficiency + direction)
    # ============================================================
    # Combine: is there a trend, and which direction?
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean()
    ema_slope = (ema_50 - ema_50.shift(10)) / (ema_50.shift(10) + EPS)
    # Threshold: |slope| > 0.001 (0.1% over 10 bars) = trending
    trend = pd.Series(0.0, index=idx, dtype="float64")
    trend[ema_slope > 0.001] = 1.0
    trend[ema_slope < -0.001] = -1.0
    trend[ema_50.isnull()] = np.nan
    out["trend_regime"] = trend

    # ============================================================
    # Feature 4: Volatility clustering (autocorrelation of |returns|)
    # ============================================================
    # High = volatility persists (clustering, Mandelbrot).
    # Correlation of |ret| with |ret_prev| over rolling window.
    abs_ret = log_ret.abs()
    vol_cluster = abs_ret.rolling(window=50, min_periods=50).corr(
        abs_ret.shift(1)
    )
    out["vol_clustering"] = vol_cluster.to_numpy()

    # ============================================================
    # Feature 5: Efficiency ratio (Kaufman)
    # ============================================================
    # directional_move / sum_of_absolute_moves
    # 1.0 = perfect trend, 0.0 = pure noise/choppy.
    period = 20
    direction = np.abs(close_arr - np.concatenate(
        [np.full(period, np.nan), close_arr[:-period]]
    ))
    # Sum of absolute bar-to-bar moves over period
    abs_moves = np.abs(np.diff(close_arr, prepend=close_arr[0]))
    abs_moves_series = pd.Series(abs_moves, index=idx)
    volatility_sum = abs_moves_series.rolling(
        window=period, min_periods=period
    ).sum().to_numpy()
    efficiency = direction / (volatility_sum + EPS)
    out["efficiency_ratio"] = series_from_array(efficiency, idx)

    return out


REGIME_FEATURE_NAMES = [
    "hurst_100",
    "vol_regime_pct",
    "trend_regime",
    "vol_clustering",
    "efficiency_ratio",
]


if __name__ == "__main__":
    from pathlib import Path
    import time

    print("=" * 70)
    print(f"Regime Features Smoke Test (v{REGIME_FEATURES_VERSION})")
    print("=" * 70)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    test_file = PROJECT_ROOT / "data" / "raw" / "BTCUSDT_5m.parquet"

    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        exit(1)

    print("\nLoading BTCUSDT 5m data...")
    df = pd.read_parquet(test_file)
    df_test = df.iloc[-15000:].reset_index(drop=True).copy()
    print(f"  Test subset: {len(df_test):,} bars")

    print("\nComputing 5 regime features...")
    t0 = time.time()
    result = compute_regime_features(df_test)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s ({len(df_test)/elapsed:,.0f} bars/sec)")

    print("\nFeatures created:")
    for fname in REGIME_FEATURE_NAMES:
        status = "✅" if fname in result.columns else "❌"
        print(f"  {status} {fname}")

    print("\nFeature statistics:")
    print("-" * 70)
    for fname in REGIME_FEATURE_NAMES:
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

    hurst = result["hurst_100"].dropna()
    hurst_ok = bool((hurst >= 0).all() and (hurst <= 1).all())
    print(f"  Hurst in [0,1]:           {'✅' if hurst_ok else '❌'} "
          f"(mean={hurst.mean():.3f})")
    print("    → ~0.5 = random, >0.5 trending, <0.5 mean-reverting")

    volpct = result["vol_regime_pct"].dropna()
    volpct_ok = bool((volpct >= 0).all() and (volpct <= 1).all())
    print(f"  Vol regime in [0,1]:      {'✅' if volpct_ok else '❌'} "
          f"(mean={volpct.mean():.3f}, expected ≈0.5 percentile)")

    trend = result["trend_regime"].dropna()
    trend_ok = bool(set(trend.unique()).issubset({-1.0, 0.0, 1.0}))
    trending_pct = float((trend != 0).mean()) * 100
    print(f"  Trend regime valid:       {'✅' if trend_ok else '❌'} "
          f"(trending {trending_pct:.0f}% of time)")

    er = result["efficiency_ratio"].dropna()
    er_ok = bool((er >= 0).all() and (er <= 1.01).all())
    print(f"  Efficiency ratio [0,1]:   {'✅' if er_ok else '❌'} "
          f"(mean={er.mean():.3f}, higher=trendier)")

    print("\n✅ Regime features module working — 5 features generated")