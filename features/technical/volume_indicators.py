"""
features/technical/volume_indicators.py

Module 7: Volume-price indicators (VWAP, MFI, CMF, force index).

10 features combining volume with price to track "smart money" flow.

Key concepts:
    - VWAP: institutional benchmark price (volume-weighted)
    - MFI: volume-weighted RSI (money flow momentum)
    - CMF: accumulation/distribution over rolling window
    - Force index: price change × volume (conviction)

All scale-invariant. CMF uses rolling (not cumulative) to avoid drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.logger import get_logger
import features.base.utils

logger = get_logger(__name__)

VOLUME_INDICATORS_VERSION = "1.0.0"

EPS = 1e-10


def _mfi(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    volume: np.ndarray, period: int = 14
) -> np.ndarray:
    """
    Money Flow Index (volume-weighted RSI).

    Uses typical price (H+L+C)/3 and raw money flow (typical × volume).
    Positive flow when typical price rises, negative when falls.
    """
    n = len(close)
    result = np.full(n, np.nan, dtype="float64")
    if n < period + 1:
        return result

    typical = (high + low + close) / 3.0
    raw_flow = typical * volume

    # Positive/negative flow based on typical price direction
    typ_change = np.diff(typical, prepend=typical[0])
    pos_flow = np.where(typ_change > 0, raw_flow, 0.0)
    neg_flow = np.where(typ_change < 0, raw_flow, 0.0)

    for i in range(period, n):
        pos_sum = float(np.sum(pos_flow[i - period + 1:i + 1]))
        neg_sum = float(np.sum(neg_flow[i - period + 1:i + 1]))
        mfr = pos_sum / (neg_sum + EPS)
        result[i] = 100.0 - 100.0 / (1.0 + mfr)

    return result


def compute_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 10 volume-price indicator features.

    Args:
        df: DataFrame with [high, low, close, volume, taker_buy_volume]

    Returns:
        df with 10 new feature columns.

    Features (all scale-invariant):
        1.  vwap_distance     — (close - VWAP20) / VWAP20
        2.  vwap_distance_60  — (close - VWAP60) / VWAP60
        3.  mfi_14            — Money Flow Index [0,100]
        4.  mfi_normalized    — (MFI-50)/50 [-1,1]
        5.  cmf_20            — Chaikin Money Flow (rolling, drift-free)
        6.  force_index_norm  — Price change × volume, normalized
        7.  ease_of_movement  — Price move per unit volume
        8.  volume_weighted_return — Return weighted by volume ratio
        9.  accumulation_ratio — Buy volume accumulation over 20 bars
        10. pvt_slope         — Price-Volume Trend slope (drift-free)
    """
    if df.empty:
        raise ValueError("Cannot compute features on empty DataFrame")

    required = ["high", "low", "close", "volume", "taker_buy_volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    idx = out.index

    high_arr = out["high"].to_numpy(dtype="float64")
    low_arr = out["low"].to_numpy(dtype="float64")
    close_arr = out["close"].to_numpy(dtype="float64")
    volume_arr = out["volume"].to_numpy(dtype="float64")

    close_series = out["close"]
    volume_series = out["volume"]

    # ============================================================
    # Features 1-2: VWAP distance (rolling VWAP)
    # ============================================================
    # Rolling VWAP = sum(price × vol) / sum(vol) over window
    typical = (out["high"] + out["low"] + out["close"]) / 3.0
    pv = typical * volume_series

    vwap_20 = (
        pv.rolling(20, min_periods=20).sum()
        / (volume_series.rolling(20, min_periods=20).sum() + EPS)
    )
    vwap_60 = (
        pv.rolling(60, min_periods=60).sum()
        / (volume_series.rolling(60, min_periods=60).sum() + EPS)
    )
    out["vwap_distance"] = (close_series - vwap_20) / (vwap_20 + EPS)
    out["vwap_distance_60"] = (close_series - vwap_60) / (vwap_60 + EPS)

    # ============================================================
    # Features 3-4: MFI (volume-weighted RSI)
    # ============================================================
    mfi_arr = _mfi(high_arr, low_arr, close_arr, volume_arr, 14)
    out["mfi_14"] = features.base.utils.series_from_array(mfi_arr, idx)
    out["mfi_normalized"] = features.base.utils.series_from_array((mfi_arr - 50.0) / 50.0, idx)

    # ============================================================
    # Feature 5: CMF (Chaikin Money Flow — rolling, NOT cumulative)
    # ============================================================
    # Money flow multiplier: where close is in bar range, weighted by vol
    hl_range = out["high"] - out["low"]
    mf_multiplier = (
        (out["close"] - out["low"]) - (out["high"] - out["close"])
    ) / (hl_range + EPS)
    mf_volume = mf_multiplier * volume_series
    # Rolling 20-bar sum (drift-free, unlike cumulative AD line)
    out["cmf_20"] = (
        mf_volume.rolling(20, min_periods=20).sum()
        / (volume_series.rolling(20, min_periods=20).sum() + EPS)
    )

    # ============================================================
    # Feature 6: Force index (normalized)
    # ============================================================
    # Price change × volume = conviction behind move
    price_change = close_series.pct_change()
    force_raw = price_change * volume_series
    # Normalize by rolling mean volume for scale-invariance
    vol_mean_20 = features.base.utils.safe_rolling_mean(volume_series, 20)
    out["force_index_norm"] = force_raw / (vol_mean_20 + EPS)

    # ============================================================
    # Feature 7: Ease of movement
    # ============================================================
    # How much price moves per unit of volume.
    # High = price moving easily (low resistance).
    mid_move = ((out["high"] + out["low"]) / 2.0).diff()
    box_ratio = volume_series / (hl_range + EPS)
    eom = mid_move / (box_ratio + EPS)
    # Normalize by close and smooth
    out["ease_of_movement"] = (eom / (close_series + EPS)).rolling(
        14, min_periods=14
    ).mean()

    # ============================================================
    # Feature 8: Volume-weighted return
    # ============================================================
    # Recent return weighted by how much volume confirmed it
    log_ret = pd.Series(
        np.log(close_series / close_series.shift(1)),
        index=idx, dtype="float64",
    )
    vol_ratio = volume_series / (vol_mean_20 + EPS)
    out["volume_weighted_return"] = log_ret * vol_ratio

    # ============================================================
    # Feature 9: Accumulation ratio (buy volume over 20 bars)
    # ============================================================
    taker_buy = out["taker_buy_volume"]
    buy_sum_20 = taker_buy.rolling(20, min_periods=20).sum()
    vol_sum_20 = volume_series.rolling(20, min_periods=20).sum()
    out["accumulation_ratio"] = buy_sum_20 / (vol_sum_20 + EPS)

    # ============================================================
    # Feature 10: PVT slope (Price-Volume Trend, drift-free)
    # ============================================================
    # PVT = cumulative (price_change_pct × volume) — but we use SLOPE
    pct_change = close_series.pct_change()
    pvt_increment = pct_change * volume_series
    pvt_cumulative = pvt_increment.cumsum()
    # Slope over 20 bars, normalized by rolling volume
    pvt_slope = (pvt_cumulative - pvt_cumulative.shift(20)) / 20.0
    out["pvt_slope"] = pvt_slope / (vol_mean_20 + EPS)

    return out


VOLUME_INDICATOR_NAMES = [
    "vwap_distance",
    "vwap_distance_60",
    "mfi_14",
    "mfi_normalized",
    "cmf_20",
    "force_index_norm",
    "ease_of_movement",
    "volume_weighted_return",
    "accumulation_ratio",
    "pvt_slope",
]


if __name__ == "__main__":
    from pathlib import Path
    import time

    print("=" * 70)
    print(f"Volume Indicators Smoke Test (v{VOLUME_INDICATORS_VERSION})")
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

    print("\nComputing 10 volume indicators...")
    t0 = time.time()
    result = compute_volume_indicators(df_test)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s ({len(df_test)/elapsed:,.0f} bars/sec)")

    print("\nFeatures created:")
    for fname in VOLUME_INDICATOR_NAMES:
        status = "✅" if fname in result.columns else "❌"
        print(f"  {status} {fname}")

    print("\nFeature statistics:")
    print("-" * 70)
    for fname in VOLUME_INDICATOR_NAMES:
        values = result[fname].dropna()
        if len(values) == 0:
            print(f"  {fname:24s} ALL NaN ❌")
            continue
        print(f"  {fname:24s} "
              f"mean={values.mean():+.5f}  "
              f"std={values.std():.5f}  "
              f"[{values.min():+.3f}, {values.max():+.3f}]")

    # Sanity checks
    print("\nSanity checks:")
    print("-" * 70)

    mfi = result["mfi_14"].dropna()
    mfi_ok = bool((mfi >= 0).all() and (mfi <= 100).all())
    print(f"  MFI in [0,100]:           {'✅' if mfi_ok else '❌'} "
          f"(mean={mfi.mean():.1f}, expected ≈50)")

    cmf = result["cmf_20"].dropna()
    cmf_ok = bool((cmf >= -1).all() and (cmf <= 1).all())
    print(f"  CMF in [-1,1]:            {'✅' if cmf_ok else '❌'} "
          f"(mean={cmf.mean():+.4f}, expected ≈0)")

    acc = result["accumulation_ratio"].dropna()
    acc_ok = bool((acc >= 0).all() and (acc <= 1).all())
    print(f"  Accumulation in [0,1]:    {'✅' if acc_ok else '❌'} "
          f"(mean={acc.mean():.3f}, expected ≈0.5)")

    vwap = result["vwap_distance"].dropna()
    print(f"  VWAP distance mean:       {vwap.mean():+.5f} "
          f"(expected ≈0, price oscillates around VWAP)")

    print("\n✅ Volume indicators module working — 10 features generated")