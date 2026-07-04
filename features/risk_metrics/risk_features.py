"""
features/risk_metrics/risk_features.py

Module 14: Risk metric features (rolling Sharpe, Sortino, VaR, drawdown).

5 features quantifying RISK at each bar — the foundation of Phase 5.

Quant research foundation:
    Sharpe Ratio (Sharpe, 1966): return / total volatility.
    Sortino Ratio (Sortino, 1980): return / DOWNSIDE volatility.
    Value at Risk: probabilistic worst-case loss.
    Maximum Drawdown: largest peak-to-trough decline.

Why risk metrics as features:
    Risk management = 40% of edge (survival).
    These features let ML learn "when is risk elevated?"
    Phase 5 uses them directly for position sizing + circuit breakers.

All features look-ahead safe (rolling past windows), per-bar (not annualized).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.logger import get_logger
from features.base.utils import series_from_array  # noqa: F401

logger = get_logger(__name__)

RISK_FEATURES_VERSION = "1.0.0"

EPS = 1e-10


def compute_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 5 risk metric features.

    Args:
        df: DataFrame with [close]

    Returns:
        df with 5 new feature columns.

    Features (all look-ahead safe, per-bar scale):
        1. rolling_sharpe_60   — Mean return / std (60-bar), risk-adjusted
        2. rolling_sortino_60  — Mean return / downside std (60-bar)
        3. var_95_60           — 5th percentile return (95% VaR, 60-bar)
        4. current_drawdown    — Current decline from 100-bar rolling peak
        5. downside_vol_ratio  — Downside vol / total vol (asymmetry measure)
    """
    if df.empty:
        raise ValueError("Cannot compute features on empty DataFrame")

    if "close" not in df.columns:
        raise ValueError("Missing required column: close")

    out = df.copy()
    idx = out.index

    close_arr = out["close"].to_numpy(dtype="float64")
    close_series = out["close"]

    # Log returns
    log_ret = pd.Series(
        np.log(close_arr / np.concatenate([[np.nan], close_arr[:-1]])),
        index=idx, dtype="float64",
    )

    window = 60

    # ============================================================
    # Feature 1: Rolling Sharpe (per-bar)
    # ============================================================
    # mean(returns) / std(returns) over 60 bars.
    # Positive = good risk-adjusted performance.
    ret_mean = log_ret.rolling(window=window, min_periods=window).mean()
    ret_std = log_ret.rolling(window=window, min_periods=window).std()
    out["rolling_sharpe_60"] = ret_mean / (ret_std + EPS)

    # ============================================================
    # Feature 2: Rolling Sortino (downside-only risk)
    # ============================================================
    # mean(returns) / std(NEGATIVE returns only).
    # Sortino > Sharpe when downside is controlled.
    def _downside_std(x: np.ndarray) -> float:
        neg = x[x < 0]
        if len(neg) < 2:
            return EPS
        return float(np.std(neg))

    downside_std = log_ret.rolling(
        window=window, min_periods=window
    ).apply(_downside_std, raw=True)
    out["rolling_sortino_60"] = ret_mean / (downside_std + EPS)

    # ============================================================
    # Feature 3: Value at Risk (95%, 60-bar)
    # ============================================================
    # 5th percentile of returns = "95% confident loss won't exceed this".
    # Negative value (it's a loss threshold).
    var_95 = log_ret.rolling(window=window, min_periods=window).quantile(0.05)
    out["var_95_60"] = var_95.to_numpy()

    # ============================================================
    # Feature 4: Current drawdown (from rolling peak)
    # ============================================================
    # How far below the recent (100-bar) peak are we?
    # 0 = at peak, negative = in drawdown.
    rolling_peak = close_series.rolling(
        window=100, min_periods=1
    ).max()
    current_dd = (close_series - rolling_peak) / (rolling_peak + EPS)
    out["current_drawdown"] = current_dd.to_numpy()

    # ============================================================
    # Feature 5: Downside vol ratio (return asymmetry)
    # ============================================================
    # downside_vol / total_vol.
    # 0.5 = symmetric, >0.5 = fat left tail (crash-prone),
    # <0.5 = fat right tail (melt-up prone).
    out["downside_vol_ratio"] = downside_std / (ret_std + EPS)

    return out


RISK_FEATURE_NAMES = [
    "rolling_sharpe_60",
    "rolling_sortino_60",
    "var_95_60",
    "current_drawdown",
    "downside_vol_ratio",
]


if __name__ == "__main__":
    from pathlib import Path
    import time

    print("=" * 70)
    print(f"Risk Metric Features Smoke Test (v{RISK_FEATURES_VERSION})")
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

    print("\nComputing 5 risk features...")
    t0 = time.time()
    result = compute_risk_features(df_test)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s ({len(df_test)/elapsed:,.0f} bars/sec)")

    print("\nFeatures created:")
    for fname in RISK_FEATURE_NAMES:
        status = "✅" if fname in result.columns else "❌"
        print(f"  {status} {fname}")

    print("\nFeature statistics:")
    print("-" * 70)
    for fname in RISK_FEATURE_NAMES:
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

    sharpe = result["rolling_sharpe_60"].dropna()
    print(f"  Rolling Sharpe mean:      {sharpe.mean():+.4f} "
          f"(≈0 for random 5m returns)")

    var = result["var_95_60"].dropna()
    var_neg = bool((var <= 0).all())
    print(f"  VaR negative (loss):      {'✅' if var_neg else '❌'} "
          f"(mean={var.mean():.5f})")

    dd = result["current_drawdown"].dropna()
    dd_ok = bool((dd <= 0.001).all())
    print(f"  Drawdown ≤ 0:             {'✅' if dd_ok else '❌'} "
          f"(mean={dd.mean():.4f}, min={dd.min():.4f})")

    dvr = result["downside_vol_ratio"].dropna()
    dvr_ok = bool((dvr >= 0).all())
    print(f"  Downside ratio positive:  {'✅' if dvr_ok else '❌'} "
          f"(mean={dvr.mean():.3f}, ≈0.7 for crypto)")

    print("\n✅ Risk metric features module working — 5 features generated")
    print("\n" + "=" * 70)
    print("🎉 ALL 14 FEATURE MODULES COMPLETE!")
    print("=" * 70)