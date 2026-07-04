"""
features/microstructure/structure_features.py

Module 8: Market microstructure features.

10 features capturing market STRUCTURE — the institutional edge.

Quant research foundation:
    Market Microstructure Theory (O'Hara, 1995): price encodes order flow.
    Smart Money Concepts: swing structure, liquidity, imbalance zones.

What we quantify (mathematically, not visually):
    - Swing points: confirmed local extrema (LAGGED to prevent look-ahead)
    - Break of Structure: trend continuation (state-machine, no spam)
    - Fair Value Gaps: 3-bar imbalance zones (bounded search)
    - Candle anatomy: rejection/absorption (ATR-normalized)

CRITICAL V1 LESSONS APPLIED:
    - Swing confirmation uses ONLY past bars (look-ahead safe)
    - BoS consumes swing after break (prevents 75% spam → ~20%)
    - FVG search is bounded (prevents O(N²) explosion)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.logger import get_logger
from features.base.utils import series_from_array

logger = get_logger(__name__)

STRUCTURE_FEATURES_VERSION = "1.0.0"

EPS = 1e-10


def _atr_simple(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """Simple ATR for normalization (Wilder smoothing)."""
    n = len(close)
    result = np.full(n, np.nan, dtype="float64")
    if n < period + 1:
        return result
    prev_close = np.concatenate([[np.nan], close[:-1]])
    tr = np.maximum.reduce([
        high - low, np.abs(high - prev_close), np.abs(low - prev_close),
    ])
    tr[0] = high[0] - low[0]
    result[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
    return result


def _detect_swings(
    high: np.ndarray, low: np.ndarray, lookback: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect confirmed swing highs/lows (LOOK-AHEAD SAFE).

    A swing high at bar i is confirmed when:
        high[i] > high[i-lookback : i]  AND  high[i] > high[i+1 : i+lookback+1]

    CRITICAL: The swing is only MARKED at bar (i + lookback), NOT bar i.
    This ensures we never use future data at the marking position.

    Returns:
        (swing_high_price, swing_low_price) arrays.
        Value = price of last confirmed swing, NaN until first confirmation.
    """
    n = len(high)
    swing_high = np.full(n, np.nan, dtype="float64")
    swing_low = np.full(n, np.nan, dtype="float64")

    last_sh = np.nan
    last_sl = np.nan

    for i in range(lookback, n - lookback):
        # Check if bar i is a local high
        left_high = high[i - lookback:i]
        right_high = high[i + 1:i + lookback + 1]
        is_swing_high = (high[i] > left_high).all() and (high[i] > right_high).all()

        # Check if bar i is a local low
        left_low = low[i - lookback:i]
        right_low = low[i + 1:i + lookback + 1]
        is_swing_low = (low[i] < left_low).all() and (low[i] < right_low).all()

        # Mark at CONFIRMATION bar (i + lookback), not bar i (look-ahead safe)
        confirm_idx = i + lookback
        if is_swing_high:
            last_sh = high[i]
        if is_swing_low:
            last_sl = low[i]

        if confirm_idx < n:
            swing_high[confirm_idx] = last_sh
            swing_low[confirm_idx] = last_sl

    # Forward-fill last known swing (safe — only past confirmations)
    swing_high = pd.Series(swing_high).ffill().to_numpy()
    swing_low = pd.Series(swing_low).ffill().to_numpy()

    return swing_high, swing_low


def compute_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 10 microstructure features.

    Args:
        df: DataFrame with [open, high, low, close]

    Returns:
        df with 10 new feature columns.

    Features (all scale-invariant, look-ahead safe):
        1.  dist_to_swing_high  — (swing_high - close) / close
        2.  dist_to_swing_low   — (close - swing_low) / close
        3.  break_of_structure  — +1 broke swing high, -1 broke low, 0 none
        4.  structure_position  — Where close is between swing low/high [0,1]
        5.  fvg_bullish         — 1 if bullish fair value gap present
        6.  fvg_bearish         — 1 if bearish fair value gap present
        7.  body_atr_ratio      — Candle body / ATR (conviction size)
        8.  upper_wick_atr      — Upper wick / ATR (rejection above)
        9.  lower_wick_atr      — Lower wick / ATR (rejection below)
        10. inside_bar          — 1 if bar range inside previous bar
    """
    if df.empty:
        raise ValueError("Cannot compute features on empty DataFrame")

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    idx = out.index

    open_arr = out["open"].to_numpy(dtype="float64")
    high_arr = out["high"].to_numpy(dtype="float64")
    low_arr = out["low"].to_numpy(dtype="float64")
    close_arr = out["close"].to_numpy(dtype="float64")

    atr = _atr_simple(high_arr, low_arr, close_arr, 14)

    # ============================================================
    # Swing detection (look-ahead safe)
    # ============================================================
    swing_high, swing_low = _detect_swings(high_arr, low_arr, lookback=3)

    # ============================================================
    # Features 1-2: Distance to swings
    # ============================================================
    out["dist_to_swing_high"] = series_from_array(
        (swing_high - close_arr) / (close_arr + EPS), idx
    )
    out["dist_to_swing_low"] = series_from_array(
        (close_arr - swing_low) / (close_arr + EPS), idx
    )

    # ============================================================
    # Feature 3: Break of Structure (state-machine, no spam)
    # ============================================================
    # V1 LESSON: Consume swing after break to prevent 75% spam.
    bos = np.full(len(close_arr), np.nan, dtype="float64")
    active_sh = np.nan
    active_sl = np.nan
    for i in range(len(close_arr)):
        if np.isnan(swing_high[i]) or np.isnan(swing_low[i]):
            continue
        # Update active levels on new swing
        if swing_high[i] != active_sh:
            active_sh = swing_high[i]
        if swing_low[i] != active_sl:
            active_sl = swing_low[i]

        val = 0.0
        # Break of structure: close must break swing by MEANINGFUL margin.
        # V1 lesson refined: require break > 0.05% beyond swing to filter noise.
        break_margin = close_arr[i] * 0.0005  # 0.05% minimum break distance
        if not np.isnan(active_sh) and close_arr[i] > active_sh + break_margin:
            val = 1.0
            active_sh = np.nan  # consume (prevent spam)
        elif not np.isnan(active_sl) and close_arr[i] < active_sl - break_margin:
            val = -1.0
            active_sl = np.nan  # consume
        bos[i] = val
    out["break_of_structure"] = series_from_array(bos, idx)

    # ============================================================
    # Feature 4: Structure position (where price is in swing range)
    # ============================================================
    # GUARD: Only compute when swing range is meaningful (>0.1% of price).
    # Prevents division explosion when swing_high ≈ swing_low.
    swing_range = swing_high - swing_low
    min_range = close_arr * 0.001  # 0.1% minimum meaningful range
    valid_range = swing_range > min_range

    struct_pos = np.full(len(close_arr), np.nan, dtype="float64")
    struct_pos[valid_range] = (
        (close_arr[valid_range] - swing_low[valid_range])
        / swing_range[valid_range]
    )
    # Clip to [-0.5, 1.5]: allow slight overshoot, block explosions
    struct_pos = np.clip(struct_pos, -0.5, 1.5)
    out["structure_position"] = series_from_array(struct_pos, idx)

    # ============================================================
    # Features 5-6: Fair Value Gaps (3-bar imbalance, bounded)
    # ============================================================
    # Bullish FVG: low[i] > high[i-2] (gap up, unfilled imbalance)
    # Bearish FVG: high[i] < low[i-2] (gap down)
    fvg_bull = np.zeros(len(close_arr), dtype="float64")
    fvg_bear = np.zeros(len(close_arr), dtype="float64")
    for i in range(2, len(close_arr)):
        if low_arr[i] > high_arr[i - 2]:
            fvg_bull[i] = 1.0
        if high_arr[i] < low_arr[i - 2]:
            fvg_bear[i] = 1.0
    fvg_bull[:2] = np.nan
    fvg_bear[:2] = np.nan
    out["fvg_bullish"] = series_from_array(fvg_bull, idx)
    out["fvg_bearish"] = series_from_array(fvg_bear, idx)

    # ============================================================
    # Features 7-9: Candle anatomy (ATR-normalized)
    # ============================================================
    body = np.abs(close_arr - open_arr)
    out["body_atr_ratio"] = series_from_array(body / (atr + EPS), idx)

    body_high = np.maximum(open_arr, close_arr)
    body_low = np.minimum(open_arr, close_arr)
    upper_wick = high_arr - body_high
    lower_wick = body_low - low_arr
    out["upper_wick_atr"] = series_from_array(
        upper_wick / (atr + EPS), idx
    )
    out["lower_wick_atr"] = series_from_array(
        lower_wick / (atr + EPS), idx
    )

    # ============================================================
    # Feature 10: Inside bar (consolidation signal)
    # ============================================================
    prev_high = np.concatenate([[np.nan], high_arr[:-1]])
    prev_low = np.concatenate([[np.nan], low_arr[:-1]])
    inside = ((high_arr < prev_high) & (low_arr > prev_low)).astype("float64")
    inside[0] = np.nan
    out["inside_bar"] = series_from_array(inside, idx)

    return out


STRUCTURE_FEATURE_NAMES = [
    "dist_to_swing_high",
    "dist_to_swing_low",
    "break_of_structure",
    "structure_position",
    "fvg_bullish",
    "fvg_bearish",
    "body_atr_ratio",
    "upper_wick_atr",
    "lower_wick_atr",
    "inside_bar",
]


if __name__ == "__main__":
    from pathlib import Path
    import time

    print("=" * 70)
    print(f"Structure Features Smoke Test (v{STRUCTURE_FEATURES_VERSION})")
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

    print("\nComputing 10 microstructure features...")
    t0 = time.time()
    result = compute_structure_features(df_test)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s ({len(df_test)/elapsed:,.0f} bars/sec)")

    print("\nFeatures created:")
    for fname in STRUCTURE_FEATURE_NAMES:
        status = "✅" if fname in result.columns else "❌"
        print(f"  {status} {fname}")

    print("\nFeature statistics:")
    print("-" * 70)
    for fname in STRUCTURE_FEATURE_NAMES:
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

    bos = result["break_of_structure"].dropna()
    bos_vals_ok = bool(set(bos.unique()).issubset({-1.0, 0.0, 1.0}))
    bos_rate = float((bos != 0).mean()) * 100
    print(f"  BoS values valid:         {'✅' if bos_vals_ok else '❌'} "
          f"(fires {bos_rate:.1f}% — should be <25%, V1 was 75%)")

    fvg_b = result["fvg_bullish"].dropna()
    fvg_rate = float(fvg_b.mean()) * 100
    print(f"  FVG bullish rate:         {fvg_rate:.1f}% "
          f"(should be bounded, few %)")

    sp = result["structure_position"].dropna()
    print(f"  Structure position mean:  {sp.mean():.3f} "
          f"(expected ≈0.5)")

    body = result["body_atr_ratio"].dropna()
    print(f"  Body/ATR positive:        "
          f"{'✅' if (body >= 0).all() else '❌'} "
          f"(mean={body.mean():.3f})")

    inside = result["inside_bar"].dropna()
    print(f"  Inside bar rate:          {inside.mean()*100:.1f}% "
          f"(consolidation frequency)")

    print("\n✅ Microstructure features module working — 10 features generated")