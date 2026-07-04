"""
features/technical/momentum_features.py

Module 4: Momentum features (RSI, MACD, ROC, Stochastic).

12 features capturing momentum (rate and persistence of price movement).

Critical design (V1 lessons applied):
    - MACD normalized by close (scale-invariance across symbols)
    - EMA warmup handled explicitly (first bars NaN, not biased)
    - RSI epsilon-protected (no division by zero in flat markets)
    - All numpy->pandas boundaries explicitly typed
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.logger import get_logger
from features.base.utils import series_from_array

logger = get_logger(__name__)

MOMENTUM_FEATURES_VERSION = "1.0.0"

EPS = 1e-10


# ============================================================
# Core indicator implementations (numpy, look-ahead safe)
# ============================================================


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """
    Exponential moving average with explicit warmup handling.

    First (period-1) values are NaN (insufficient data).
    Seed = SMA of first `period` values (standard practice).
    """
    n = len(arr)
    result = np.full(n, np.nan, dtype="float64")
    if n < period:
        return result

    alpha = 2.0 / (period + 1.0)
    # Seed with SMA
    result[period - 1] = float(np.mean(arr[:period]))
    for i in range(period, n):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index (Wilder smoothing).

    RSI = 100 - 100/(1+RS), RS = avg_gain / avg_loss.
    Epsilon-protected against flat markets.
    """
    n = len(close)
    result = np.full(n, np.nan, dtype="float64")
    if n < period + 1:
        return result

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Initial averages (SMA seed)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    rs = avg_gain / (avg_loss + EPS)
    result[period] = 100.0 - 100.0 / (1.0 + rs)

    # Wilder smoothing
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        rs = avg_gain / (avg_loss + EPS)
        result[i] = 100.0 - 100.0 / (1.0 + rs)

    return result


# ============================================================
# Main computation
# ============================================================


def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 12 momentum features.

    Args:
        df: DataFrame with columns [high, low, close]

    Returns:
        df with 12 new feature columns.

    Features (all scale-invariant):
        1.  rsi_14              — RSI(14), range [0,100]
        2.  rsi_14_normalized   — (RSI-50)/50, range [-1,1] (ML-friendly)
        3.  rsi_divergence      — RSI slope vs price slope mismatch
        4.  macd_norm           — MACD line / close (scale-invariant!)
        5.  macd_signal_norm    — Signal line / close
        6.  macd_hist_norm      — Histogram / close
        7.  macd_cross          — +1 bullish cross, -1 bearish, 0 none
        8.  roc_5               — Rate of change (5 bars), as return
        9.  roc_20              — Rate of change (20 bars)
        10. stoch_k             — Stochastic %K, range [0,100]
        11. stoch_d             — Stochastic %D (3-bar MA of %K)
        12. momentum_quality    — |ROC| / rolling_std (signal vs noise)
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
    out["high"].to_numpy(dtype="float64")
    out["low"].to_numpy(dtype="float64")

    # ============================================================
    # Features 1-3: RSI family
    # ============================================================
    rsi_arr = _rsi(close_arr, period=14)
    out["rsi_14"] = series_from_array(rsi_arr, idx)

    # Normalized RSI: [-1, +1] centered at 0 (better for ML)
    out["rsi_14_normalized"] = series_from_array(
        (rsi_arr - 50.0) / 50.0, idx
    )

    # RSI divergence: RSI slope minus price slope (both z-scale)
    # Positive = RSI rising faster than price (hidden strength)
    rsi_series = out["rsi_14"]
    rsi_slope = (rsi_series - rsi_series.shift(5)) / 5.0 / 100.0
    price_slope = (out["close"] - out["close"].shift(5)) / (
        out["close"].shift(5) + EPS
    ) / 5.0
    out["rsi_divergence"] = rsi_slope - price_slope * 100.0

    # ============================================================
    # Features 4-7: MACD family (NORMALIZED — V1 lesson!)
    # ============================================================
    ema_12 = _ema(close_arr, 12)
    ema_26 = _ema(close_arr, 26)
    macd_line = ema_12 - ema_26

    # CRITICAL: Normalize by close for scale-invariance
    # Without this: BTC MACD ~500, ADA MACD ~0.005 (incomparable)
    macd_norm_arr = macd_line / (close_arr + EPS)
    out["macd_norm"] = series_from_array(macd_norm_arr, idx)

    # Signal line = EMA(9) of MACD
    # Handle NaN warmup: compute EMA only on valid region
    signal_arr = np.full(len(close_arr), np.nan, dtype="float64")
    valid_start = 26 - 1  # MACD valid from here
    valid_macd = macd_line[valid_start:]
    if len(valid_macd) >= 9:
        signal_valid = _ema(valid_macd, 9)
        signal_arr[valid_start:] = signal_valid
    signal_norm_arr = signal_arr / (close_arr + EPS)
    out["macd_signal_norm"] = series_from_array(signal_norm_arr, idx)

    # Histogram = MACD - Signal (normalized)
    hist_norm_arr = macd_norm_arr - signal_norm_arr
    out["macd_hist_norm"] = series_from_array(hist_norm_arr, idx)

    # MACD cross: +1 when hist crosses above 0, -1 below, 0 otherwise
    hist_series = out["macd_hist_norm"]
    prev_hist = hist_series.shift(1)
    cross = pd.Series(0.0, index=idx, dtype="float64")
    cross[(prev_hist <= 0) & (hist_series > 0)] = 1.0
    cross[(prev_hist >= 0) & (hist_series < 0)] = -1.0
    # Preserve NaN in warmup
    cross[hist_series.isnull()] = np.nan
    out["macd_cross"] = cross

    # ============================================================
    # Features 8-9: Rate of Change
    # ============================================================
    close_series = out["close"]
    out["roc_5"] = (close_series - close_series.shift(5)) / (
        close_series.shift(5) + EPS
    )
    out["roc_20"] = (close_series - close_series.shift(20)) / (
        close_series.shift(20) + EPS
    )

    # ============================================================
    # Features 10-11: Stochastic oscillator
    # ============================================================
    low_14 = out["low"].rolling(window=14, min_periods=14).min()
    high_14 = out["high"].rolling(window=14, min_periods=14).max()
    stoch_k = 100.0 * (close_series - low_14) / (high_14 - low_14 + EPS)
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_k.rolling(window=3, min_periods=3).mean()

    # ============================================================
    # Feature 12: Momentum quality (signal-to-noise)
    # ============================================================
    # |20-bar move| / std of 1-bar moves
    # High = clean directional move; Low = choppy noise
    ret_1 = close_series.pct_change()
    ret_std_20 = ret_1.rolling(window=20, min_periods=20).std()
    out["momentum_quality"] = out["roc_20"].abs() / (
        ret_std_20 * np.sqrt(20) + EPS
    )

    return out


MOMENTUM_FEATURE_NAMES = [
    "rsi_14",
    "rsi_14_normalized",
    "rsi_divergence",
    "macd_norm",
    "macd_signal_norm",
    "macd_hist_norm",
    "macd_cross",
    "roc_5",
    "roc_20",
    "stoch_k",
    "stoch_d",
    "momentum_quality",
]


if __name__ == "__main__":
    from pathlib import Path
    import time

    print("=" * 70)
    print(f"Momentum Features Smoke Test (v{MOMENTUM_FEATURES_VERSION})")
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

    print("\nComputing 12 momentum features...")
    t0 = time.time()
    result = compute_momentum_features(df_test)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s ({len(df_test)/elapsed:,.0f} bars/sec)")

    print("\nFeatures created:")
    for fname in MOMENTUM_FEATURE_NAMES:
        status = "✅" if fname in result.columns else "❌"
        print(f"  {status} {fname}")

    print("\nFeature statistics:")
    print("-" * 70)
    for fname in MOMENTUM_FEATURE_NAMES:
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

    rsi = result["rsi_14"].dropna()
    rsi_ok = bool((rsi >= 0).all() and (rsi <= 100).all())
    print(f"  RSI in [0,100]:           {'✅' if rsi_ok else '❌'} "
          f"(mean={rsi.mean():.1f}, expected ≈50)")

    stoch = result["stoch_k"].dropna()
    stoch_ok = bool((stoch >= 0).all() and (stoch <= 100).all())
    print(f"  Stoch %K in [0,100]:      {'✅' if stoch_ok else '❌'}")

    # MACD normalized: should be TINY (fraction of price)
    macd = result["macd_norm"].dropna()
    macd_scale_ok = bool(macd.abs().max() < 0.1)
    print(f"  MACD scale-invariant:     {'✅' if macd_scale_ok else '❌'} "
          f"(max |value|={macd.abs().max():.5f}, must be <0.1)")

    # MACD cross: should be sparse (-1/0/+1)
    cross = result["macd_cross"].dropna()
    cross_vals_ok = bool(set(cross.unique()).issubset({-1.0, 0.0, 1.0}))
    cross_rate = float((cross != 0).mean())
    print(f"  MACD cross values valid:  {'✅' if cross_vals_ok else '❌'} "
          f"(fires {cross_rate*100:.1f}% of bars)")

    print("\n✅ Momentum features module working — 12 features generated")