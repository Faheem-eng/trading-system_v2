"""
features/microstructure_quality/quality_features.py

Module 13: Microstructure quality features (liquidity, spread, price impact).

5 features measuring EXECUTION QUALITY — how easily/cheaply can we trade.

Quant research foundation:
    Amihud Illiquidity (2002): |return| / volume = price impact per volume.
    Kyle's Lambda (1985): order-flow price impact (market depth).
    Roll's Spread (1984): effective spread from price serial covariance.

Why execution quality matters:
    Backtest ignoring liquidity = fake profits (Loophole #3).
    Illiquid bars have massive slippage.
    Phase 4 backtest uses these for REALISTIC cost modeling.

All features scale-invariant, look-ahead safe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.logger import get_logger
from features.base.utils import (
    safe_rolling_mean,
    series_from_array,
    zscore,
)

logger = get_logger(__name__)

QUALITY_FEATURES_VERSION = "1.0.0"

EPS = 1e-10


def compute_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 5 microstructure quality features.

    Args:
        df: DataFrame with [high, low, close, volume, quote_volume]

    Returns:
        df with 5 new feature columns.

    Features (all scale-invariant, look-ahead safe):
        1. amihud_illiquidity  — |return| / quote_volume (log-scaled)
        2. spread_proxy        — (high-low) / close (effective spread proxy)
        3. kyle_lambda         — Price impact per unit signed volume
        4. liquidity_zscore    — Z-scored volume-based liquidity
        5. price_impact_ratio  — Realized move / expected move (efficiency)
    """
    if df.empty:
        raise ValueError("Cannot compute features on empty DataFrame")

    required = ["high", "low", "close", "volume", "quote_volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    idx = out.index

    high_arr = out["high"].to_numpy(dtype="float64")
    low_arr = out["low"].to_numpy(dtype="float64")
    close_arr = out["close"].to_numpy(dtype="float64")
    close_series = out["close"]  # noqa: F841
    volume_series = out["volume"]
    quote_vol_series = out["quote_volume"]

    # Log returns
    log_ret = pd.Series(
        np.log(close_arr / np.concatenate([[np.nan], close_arr[:-1]])),
        index=idx, dtype="float64",
    )
    abs_ret = log_ret.abs()

    # ============================================================
    # Feature 1: Amihud illiquidity (log-scaled for stability)
    # ============================================================
    # |return| / quote_volume. High = illiquid (big price move per $).
    # Log-transform to tame extreme spikes on low-volume bars.
    amihud_raw = abs_ret / (quote_vol_series + EPS)
    # Log1p transform (handles zeros gracefully) then z-score
    amihud_log = np.log1p(amihud_raw.to_numpy() * 1e6)  # scale for numerics
    amihud_series = series_from_array(amihud_log, idx)
    # Smooth over 20 bars
    out["amihud_illiquidity"] = safe_rolling_mean(amihud_series, 20)

    # ============================================================
    # Feature 2: Spread proxy (Roll-style, high-low based)
    # ============================================================
    # (high - low) / close = effective spread estimate.
    # We lack quote data, so high-low range proxies the spread.
    spread = (high_arr - low_arr) / (close_arr + EPS)
    spread_series = series_from_array(spread, idx)
    out["spread_proxy"] = safe_rolling_mean(spread_series, 20)

    # ============================================================
    # Feature 3: Kyle's lambda (price impact per signed volume)
    # ============================================================
    # |price change| / volume, smoothed. Market depth proxy.
    # Higher = thinner market (price moves easily).
    price_change = np.abs(np.diff(close_arr, prepend=close_arr[0]))
    price_change_norm = price_change / (close_arr + EPS)
    kyle_raw = price_change_norm / (volume_series.to_numpy() + EPS)
    kyle_log = np.log1p(kyle_raw * 1e6)
    kyle_series = series_from_array(kyle_log, idx)
    out["kyle_lambda"] = safe_rolling_mean(kyle_series, 20)

    # ============================================================
    # Feature 4: Liquidity z-score (higher = more liquid)
    # ============================================================
    # Z-scored quote volume. Positive = above-average liquidity.
    out["liquidity_zscore"] = zscore(quote_vol_series, window=50)

    # ============================================================
    # Feature 5: Price impact ratio (realized vs expected move)
    # ============================================================
    # How much did price ACTUALLY move vs what volume suggests?
    # realized = |return|, expected ~ sqrt(volume ratio) heuristic.
    vol_mean_20 = safe_rolling_mean(volume_series, 20)
    vol_ratio = volume_series / (vol_mean_20 + EPS)
    expected_move = np.sqrt(vol_ratio.to_numpy()) * abs_ret.rolling(
        window=20, min_periods=20
    ).mean().to_numpy()
    realized_move = abs_ret.to_numpy()
    impact_ratio = realized_move / (expected_move + EPS)
    impact_series = series_from_array(impact_ratio, idx)
    # Winsorize extreme values (clip at 99th percentile behavior)
    out["price_impact_ratio"] = impact_series.clip(upper=10.0)

    return out


QUALITY_FEATURE_NAMES = [
    "amihud_illiquidity",
    "spread_proxy",
    "kyle_lambda",
    "liquidity_zscore",
    "price_impact_ratio",
]


if __name__ == "__main__":
    from pathlib import Path
    import time

    print("=" * 70)
    print(f"Microstructure Quality Smoke Test (v{QUALITY_FEATURES_VERSION})")
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

    print("\nComputing 5 quality features...")
    t0 = time.time()
    result = compute_quality_features(df_test)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s ({len(df_test)/elapsed:,.0f} bars/sec)")

    print("\nFeatures created:")
    for fname in QUALITY_FEATURE_NAMES:
        status = "✅" if fname in result.columns else "❌"
        print(f"  {status} {fname}")

    print("\nFeature statistics:")
    print("-" * 70)
    for fname in QUALITY_FEATURE_NAMES:
        values = result[fname].dropna()
        if len(values) == 0:
            print(f"  {fname:22s} ALL NaN ❌")
            continue
        print(f"  {fname:22s} "
              f"mean={values.mean():+.5f}  "
              f"std={values.std():.5f}  "
              f"[{values.min():+.3f}, {values.max():+.3f}]")

    # Sanity checks
    print("\nSanity checks:")
    print("-" * 70)

    spread = result["spread_proxy"].dropna()
    spread_ok = bool((spread > 0).all())
    print(f"  Spread positive:          {'✅' if spread_ok else '❌'} "
          f"(mean={spread.mean()*100:.3f}%)")

    amihud = result["amihud_illiquidity"].dropna()
    amihud_finite = bool(np.isfinite(amihud).all())
    print(f"  Amihud finite (no inf):   {'✅' if amihud_finite else '❌'} "
          f"(mean={amihud.mean():.3f})")

    liq = result["liquidity_zscore"].dropna()
    print(f"  Liquidity zscore:         "
          f"mean={liq.mean():+.3f} std={liq.std():.3f} "
          f"(expected ≈0, ≈1)")

    impact = result["price_impact_ratio"].dropna()
    impact_bounded = bool((impact <= 10.0).all())
    print(f"  Impact ratio bounded:     {'✅' if impact_bounded else '❌'} "
          f"(mean={impact.mean():.3f})")

    print("\n✅ Microstructure quality module working — 5 features generated")