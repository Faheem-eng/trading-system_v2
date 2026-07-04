"""
features/cross_asset/cross_asset_features.py

Module 10: Cross-asset features (BTC correlation, market beta, relative strength).

6 features measuring an asset's relationship to BTC (the market driver).

Quant research foundation:
    CAPM (Sharpe, 1964): return = alpha + beta×market + residual
    - beta = systematic risk (moves with BTC)
    - residual = idiosyncratic alpha (the true edge)

Why this matters (Loophole #7 defense):
    Fake diversification = trading 10 coins all 0.8+ correlated with BTC.
    This module quantifies BTC dependence so Phase 5 can build
    TRULY diversified, correlation-adjusted portfolios.

All features look-ahead safe (rolling windows, past data only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.logger import get_logger
from features.base.utils import series_from_array  # noqa: F401

logger = get_logger(__name__)

CROSS_ASSET_FEATURES_VERSION = "1.0.0"

EPS = 1e-10


def compute_cross_asset_features(
    df: pd.DataFrame,
    btc_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute all 6 cross-asset features relative to BTC.

    Args:
        df: Asset DataFrame with [open_time, close]
        btc_df: BTC DataFrame with [open_time, close]

    Returns:
        df with 6 new feature columns.

    Features (all look-ahead safe):
        1. btc_correlation_60  — Rolling 60-bar corr(asset_ret, btc_ret)
        2. btc_beta_60         — Rolling 60-bar beta (systematic risk)
        3. relative_strength   — asset_return - btc_return (outperformance)
        4. rs_slope_20         — Slope of relative strength (momentum vs BTC)
        5. residual_return     — asset_ret - beta×btc_ret (idiosyncratic alpha)
        6. btc_lead_lag        — Correlation of asset_ret with btc_ret.shift(1)
                                 (does asset follow BTC with lag?)
    """
    if df.empty or btc_df.empty:
        raise ValueError("Cannot compute cross-asset on empty DataFrames")

    for col in ["open_time", "close"]:
        if col not in df.columns:
            raise ValueError(f"Asset missing column: {col}")
        if col not in btc_df.columns:
            raise ValueError(f"BTC missing column: {col}")

    out = df.copy()

    # ============================================================
    # Align asset and BTC on open_time (critical for correctness)
    # ============================================================
    btc_slim = btc_df[["open_time", "close"]].copy()
    btc_slim = btc_slim.rename(columns={"close": "btc_close"})

    merged = out[["open_time", "close"]].merge(
        btc_slim, on="open_time", how="left"
    )

    # Compute log returns for both
    asset_ret = pd.Series(
        np.log(merged["close"].to_numpy(dtype="float64")
               / merged["close"].shift(1).to_numpy(dtype="float64")),
        index=merged.index, dtype="float64",
    )
    btc_ret = pd.Series(
        np.log(merged["btc_close"].to_numpy(dtype="float64")
               / merged["btc_close"].shift(1).to_numpy(dtype="float64")),
        index=merged.index, dtype="float64",
    )

    # ============================================================
    # Feature 1: BTC correlation (rolling 60-bar)
    # ============================================================
    corr_60 = asset_ret.rolling(window=60, min_periods=60).corr(btc_ret)
    out["btc_correlation_60"] = corr_60.to_numpy()

    # ============================================================
    # Feature 2: BTC beta (rolling 60-bar)
    # beta = cov(asset, btc) / var(btc)
    # ============================================================
    cov_60 = asset_ret.rolling(window=60, min_periods=60).cov(btc_ret)
    var_btc_60 = btc_ret.rolling(window=60, min_periods=60).var()
    beta_60 = cov_60 / (var_btc_60 + EPS)
    out["btc_beta_60"] = beta_60.to_numpy()

    # ============================================================
    # Feature 3: Relative strength (outperformance vs BTC)
    # ============================================================
    rel_strength = asset_ret - btc_ret
    out["relative_strength"] = rel_strength.to_numpy()

    # ============================================================
    # Feature 4: Relative strength slope (momentum vs BTC)
    # ============================================================
    # Cumulative RS, then slope (drift-free via difference)
    rs_cum = rel_strength.cumsum()
    rs_slope = (rs_cum - rs_cum.shift(20)) / 20.0
    out["rs_slope_20"] = rs_slope.to_numpy()

    # ============================================================
    # Feature 5: Residual return (idiosyncratic alpha)
    # asset_ret - beta × btc_ret = pure asset-specific move
    # THIS IS THE TRUE EDGE (market-neutral component)
    # ============================================================
    residual = asset_ret - beta_60 * btc_ret
    out["residual_return"] = residual.to_numpy()

    # ============================================================
    # Feature 6: BTC lead-lag (does asset follow BTC with delay?)
    # Correlation of asset return with PREVIOUS bar's BTC return.
    # High = asset lags BTC (predictable follow-through).
    # ============================================================
    btc_ret_lagged = btc_ret.shift(1)
    lead_lag = asset_ret.rolling(window=60, min_periods=60).corr(
        btc_ret_lagged
    )
    out["btc_lead_lag"] = lead_lag.to_numpy()

    return out


CROSS_ASSET_FEATURE_NAMES = [
    "btc_correlation_60",
    "btc_beta_60",
    "relative_strength",
    "rs_slope_20",
    "residual_return",
    "btc_lead_lag",
]


if __name__ == "__main__":
    from pathlib import Path
    import time

    print("=" * 70)
    print(f"Cross-Asset Features Smoke Test (v{CROSS_ASSET_FEATURES_VERSION})")
    print("=" * 70)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    asset_file = PROJECT_ROOT / "data" / "raw" / "ETHUSDT_5m.parquet"
    btc_file = PROJECT_ROOT / "data" / "raw" / "BTCUSDT_5m.parquet"

    if not asset_file.exists() or not btc_file.exists():
        print("❌ Need ETHUSDT and BTCUSDT 5m files")
        exit(1)

    print("\nLoading ETHUSDT (asset) + BTCUSDT (market)...")
    asset = pd.read_parquet(asset_file)
    btc = pd.read_parquet(btc_file)
    print(f"  ETH: {len(asset):,} bars, BTC: {len(btc):,} bars")

    # Use aligned recent subset
    asset_test = asset.iloc[-20000:].reset_index(drop=True).copy()
    btc_test = btc.iloc[-20000:].reset_index(drop=True).copy()
    print(f"  Test: {len(asset_test):,} bars each")

    print("\nComputing 6 cross-asset features (ETH vs BTC)...")
    t0 = time.time()
    result = compute_cross_asset_features(asset_test, btc_test)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s")

    print("\nFeatures created:")
    for fname in CROSS_ASSET_FEATURE_NAMES:
        status = "✅" if fname in result.columns else "❌"
        print(f"  {status} {fname}")

    print("\nFeature statistics:")
    print("-" * 70)
    for fname in CROSS_ASSET_FEATURE_NAMES:
        values = result[fname].dropna()
        if len(values) == 0:
            print(f"  {fname:22s} ALL NaN ❌")
            continue
        print(f"  {fname:22s} "
              f"mean={values.mean():+.4f}  "
              f"std={values.std():.4f}  "
              f"[{values.min():+.3f}, {values.max():+.3f}]")

    # Sanity checks
    print("\nSanity checks:")
    print("-" * 70)

    corr = result["btc_correlation_60"].dropna()
    corr_ok = bool((corr >= -1.01).all() and (corr <= 1.01).all())
    print(f"  Correlation in [-1,1]:    {'✅' if corr_ok else '❌'} "
          f"(mean={corr.mean():.3f})")
    print("    → ETH-BTC corr should be HIGH (0.6-0.9, they move together)")

    beta = result["btc_beta_60"].dropna()
    print(f"  Beta mean:                {beta.mean():.3f} "
          f"(ETH beta typically 0.8-1.3)")

    rs = result["relative_strength"].dropna()
    print(f"  Relative strength mean:   {rs.mean():+.5f} "
          f"(≈0 = ETH tracks BTC)")

    resid = result["residual_return"].dropna()
    print(f"  Residual return mean:     {resid.mean():+.5f} "
          f"(≈0 = idiosyncratic, market-neutral)")

    print("\n✅ Cross-asset features module working — 6 features generated")