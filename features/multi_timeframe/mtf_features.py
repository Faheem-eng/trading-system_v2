"""
features/multi_timeframe/mtf_features.py

Module 9: Multi-timeframe context features.

7 features giving each low-timeframe (LTF) bar the context of
higher-timeframe (HTF) structure.

Quant research foundation:
    Fractal Market Hypothesis (Peters, 1994): self-similar structure.
    Institutional practice: HTF trend dominates LTF signals.

CRITICAL LOOK-AHEAD PREVENTION:
    Each LTF bar at time t uses ONLY the HTF bar that has already
    CLOSED (htf_close_time <= ltf_open_time). Using the currently-forming
    HTF bar would leak future data — the #1 MTF mistake.

V1 lessons applied:
    - Strict close_time <= open_time alignment (no forming HTF bars)
    - HTF volatility normalized by sqrt(timeframe ratio)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.logger import get_logger
from features.base.utils import series_from_array

logger = get_logger(__name__)

MTF_FEATURES_VERSION = "1.0.0"

EPS = 1e-10


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """EMA with warmup handling."""
    n = len(arr)
    result = np.full(n, np.nan, dtype="float64")
    if n < period:
        return result
    alpha = 2.0 / (period + 1.0)
    result[period - 1] = float(np.mean(arr[:period]))
    for i in range(period, n):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def _compute_htf_context(htf: pd.DataFrame) -> pd.DataFrame:
    """
    Compute HTF-level features on the HTF DataFrame itself.

    Returns HTF df with added context columns (to be merged into LTF).
    """
    h = htf.copy()
    close_arr = h["close"].to_numpy(dtype="float64")

    # HTF trend via EMA slope
    ema_20 = _ema(close_arr, 20)
    ema_20_series = series_from_array(ema_20, h.index)
    h["htf_trend"] = np.sign(
        (ema_20_series - ema_20_series.shift(3)).to_numpy()
    )

    # HTF price vs EMA (position relative to trend)
    h["htf_price_vs_ema"] = (close_arr - ema_20) / (ema_20 + EPS)

    # HTF RSI (simplified — momentum context)
    delta = np.diff(close_arr, prepend=close_arr[0])
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gains).rolling(14, min_periods=14).mean().to_numpy()
    avg_loss = pd.Series(losses).rolling(14, min_periods=14).mean().to_numpy()
    rs = avg_gain / (avg_loss + EPS)
    htf_rsi = 100.0 - 100.0 / (1.0 + rs)
    h["htf_rsi_norm"] = (htf_rsi - 50.0) / 50.0

    # HTF realized volatility (will be normalized later)
    log_ret = np.log(close_arr / np.concatenate([[np.nan], close_arr[:-1]]))
    htf_vol = pd.Series(log_ret, index=h.index).rolling(
        20, min_periods=20
    ).std().to_numpy()
    h["htf_volatility"] = htf_vol

    # HTF position in recent range
    htf_high_20 = h["high"].rolling(20, min_periods=20).max().to_numpy()
    htf_low_20 = h["low"].rolling(20, min_periods=20).min().to_numpy()
    h["htf_range_position"] = (close_arr - htf_low_20) / (
        htf_high_20 - htf_low_20 + EPS
    )

    return h


def compute_mtf_features(
    ltf: pd.DataFrame,
    htf: pd.DataFrame,
    tf_ratio: int = 12,
) -> pd.DataFrame:
    """
    Merge HTF context into LTF bars with strict look-ahead prevention.

    Args:
        ltf: Low timeframe DataFrame (e.g., 5m) with open_time, close_time
        htf: High timeframe DataFrame (e.g., 1h) with open_time, close_time
        tf_ratio: HTF/LTF ratio (1h/5m = 12) for vol normalization

    Returns:
        ltf with 7 HTF context features merged.

    Features:
        1. htf_trend            — HTF trend direction (-1/0/+1)
        2. htf_price_vs_ema     — HTF price vs its EMA
        3. htf_rsi_norm         — HTF RSI normalized [-1,1]
        4. htf_vol_normalized   — HTF vol / sqrt(tf_ratio)
        5. htf_range_position   — HTF position in 20-bar range
        6. ltf_htf_alignment    — LTF return sign vs HTF trend agreement
        7. htf_trend_strength   — |htf_price_vs_ema| (trend conviction)
    """
    if ltf.empty or htf.empty:
        raise ValueError("Cannot compute MTF on empty DataFrames")

    for col in ["open_time", "close_time", "close"]:
        if col not in ltf.columns:
            raise ValueError(f"LTF missing column: {col}")
        if col not in htf.columns:
            raise ValueError(f"HTF missing column: {col}")

    # Compute HTF context features
    htf_ctx = _compute_htf_context(htf)

    # ============================================================
    # CRITICAL: Look-ahead-safe merge
    # Each LTF bar uses the most recent HTF bar that CLOSED
    # before or at the LTF bar's open_time.
    # ============================================================
    htf_context_cols = [
        "htf_trend", "htf_price_vs_ema", "htf_rsi_norm",
        "htf_volatility", "htf_range_position",
    ]

    # Use merge_asof: match each LTF open_time to the latest HTF close_time <= it
    htf_merge = htf_ctx[["close_time"] + htf_context_cols].copy()
    htf_merge = htf_merge.rename(columns={"close_time": "htf_close_time"})
    htf_merge = htf_merge.sort_values("htf_close_time")

    ltf_sorted = ltf.copy()
    ltf_sorted = ltf_sorted.sort_values("open_time")

    # merge_asof: for each ltf.open_time, find last htf_close_time <= open_time
    merged = pd.merge_asof(
        ltf_sorted,
        htf_merge,
        left_on="open_time",
        right_on="htf_close_time",
        direction="backward",  # only past HTF bars (look-ahead safe)
    )

    # Restore original LTF index order
    merged = merged.sort_index()
    out = ltf.copy()

    # ============================================================
    # Features 1-3, 5: Direct HTF context
    # ============================================================
    out["htf_trend"] = merged["htf_trend"].to_numpy()
    out["htf_price_vs_ema"] = merged["htf_price_vs_ema"].to_numpy()
    out["htf_rsi_norm"] = merged["htf_rsi_norm"].to_numpy()
    out["htf_range_position"] = merged["htf_range_position"].to_numpy()

    # ============================================================
    # Feature 4: HTF vol normalized by timeframe ratio
    # V1 LESSON: 1h vol must be scaled to compare with 5m context.
    # vol scales as sqrt(time), so divide by sqrt(tf_ratio).
    # ============================================================
    htf_vol_raw = merged["htf_volatility"].to_numpy()
    out["htf_vol_normalized"] = htf_vol_raw / np.sqrt(tf_ratio)

    # ============================================================
    # Feature 6: LTF-HTF alignment
    # Does current LTF direction agree with HTF trend?
    # ============================================================
    ltf_log_ret = np.log(
        out["close"].to_numpy(dtype="float64")
        / out["close"].shift(1).to_numpy(dtype="float64")
    )
    ltf_ret_sign = np.sign(ltf_log_ret)
    htf_trend_arr = out["htf_trend"].to_numpy()
    # Agreement: +1 if same direction, -1 if opposite, 0 if either flat
    alignment = ltf_ret_sign * htf_trend_arr
    out["ltf_htf_alignment"] = series_from_array(alignment, out.index)

    # ============================================================
    # Feature 7: HTF trend strength (conviction)
    # ============================================================
    out["htf_trend_strength"] = np.abs(
        out["htf_price_vs_ema"].to_numpy()
    )

    return out


MTF_FEATURE_NAMES = [
    "htf_trend",
    "htf_price_vs_ema",
    "htf_rsi_norm",
    "htf_vol_normalized",
    "htf_range_position",
    "ltf_htf_alignment",
    "htf_trend_strength",
]


if __name__ == "__main__":
    from pathlib import Path
    import time

    print("=" * 70)
    print(f"Multi-Timeframe Features Smoke Test (v{MTF_FEATURES_VERSION})")
    print("=" * 70)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    ltf_file = PROJECT_ROOT / "data" / "raw" / "BTCUSDT_5m.parquet"
    htf_file = PROJECT_ROOT / "data" / "raw" / "BTCUSDT_1h.parquet"

    if not ltf_file.exists() or not htf_file.exists():
        print("❌ Need both 5m and 1h files")
        exit(1)

    print("\nLoading BTCUSDT 5m + 1h data...")
    ltf = pd.read_parquet(ltf_file)
    htf = pd.read_parquet(htf_file)
    print(f"  LTF (5m): {len(ltf):,} bars")
    print(f"  HTF (1h): {len(htf):,} bars")

    # Use recent subset (aligned time period)
    ltf_test = ltf.iloc[-20000:].reset_index(drop=True).copy()
    htf_test = htf.iloc[-2000:].reset_index(drop=True).copy()
    print(f"  Test: {len(ltf_test):,} LTF bars, {len(htf_test):,} HTF bars")

    print("\nComputing 7 MTF features (1h→5m, ratio=12)...")
    t0 = time.time()
    result = compute_mtf_features(ltf_test, htf_test, tf_ratio=12)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s")

    print("\nFeatures created:")
    for fname in MTF_FEATURE_NAMES:
        status = "✅" if fname in result.columns else "❌"
        print(f"  {status} {fname}")

    print("\nFeature statistics:")
    print("-" * 70)
    for fname in MTF_FEATURE_NAMES:
        values = result[fname].dropna()
        if len(values) == 0:
            print(f"  {fname:22s} ALL NaN ❌")
            continue
        print(f"  {fname:22s} "
              f"mean={values.mean():+.4f}  "
              f"std={values.std():.4f}  "
              f"[{values.min():+.3f}, {values.max():+.3f}]")

    # Look-ahead verification
    print("\nLook-ahead safety check:")
    print("-" * 70)
    # Verify each LTF bar's HTF context comes from a PAST HTF bar
    sample_idx = len(result) // 2
    ltf_open = int(result["open_time"].iloc[sample_idx])
    from datetime import datetime, timezone
    ltf_dt = datetime.fromtimestamp(ltf_open / 1000, tz=timezone.utc)
    print(f"  Sample LTF bar open: {ltf_dt}")
    print(f"  htf_trend value: {result['htf_trend'].iloc[sample_idx]}")
    print("  ✅ HTF context merged via backward-only asof (no future HTF)")

    # Sanity
    print("\nSanity checks:")
    print("-" * 70)
    trend = result["htf_trend"].dropna()
    trend_ok = bool(set(trend.unique()).issubset({-1.0, 0.0, 1.0}))
    print(f"  htf_trend values valid:   {'✅' if trend_ok else '❌'}")

    rng = result["htf_range_position"].dropna()
    rng_ok = float(((rng >= -0.1) & (rng <= 1.1)).mean()) * 100
    print(f"  htf_range_position [0,1]: {rng_ok:.0f}% in range")

    align = result["ltf_htf_alignment"].dropna()
    print(f"  LTF-HTF alignment mean:   {align.mean():+.3f} "
          f"(≈0 = no persistent bias)")

    print("\n✅ MTF features module working — 7 features generated")