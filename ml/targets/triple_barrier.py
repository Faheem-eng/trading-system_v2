"""
ml/targets/triple_barrier.py

Triple Barrier Method for target labeling (López de Prado).

THE institutional standard for ML labels in quantitative trading.
Reference: "Advances in Financial Machine Learning" (2018), Chapter 3.

For each bar at time t, simulate a trade:
    - Upper barrier (profit target): entry + profit_mult × ATR
    - Lower barrier (stop loss):     entry - stop_mult × ATR
    - Time barrier (timeout):        t + max_holding_bars

Label = first barrier hit:
    +1: Upper hit first  (profit)
    -1: Lower hit first  (loss)
     0: Timeout (no clear direction)

CRITICAL DESIGN (anti-leakage):
    - Label at bar t uses ONLY future bars (t+1 onwards) — this is CORRECT
      (targets SHOULD use future; that's what we predict).
    - Last max_holding_bars get NaN (insufficient future to label).
    - ATR computed with past-only data (no look-ahead in barrier sizing).
    - Calibrated to 3×ATR (from V1: gives balanced ~40/40/20 distribution).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.logger import get_logger

logger = get_logger(__name__)

TRIPLE_BARRIER_VERSION = "1.0.0"

# Calibrated defaults (from V1 testing: 3×ATR gives balanced classes)
DEFAULT_PROFIT_MULT = 3.0
DEFAULT_STOP_MULT = 3.0
DEFAULT_MAX_HOLDING = 50
DEFAULT_ATR_PERIOD = 14

EPS = 1e-10


def _compute_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """
    ATR (Wilder). Uses ONLY past data — barrier at bar t sized by
    ATR computed from bars up to t (no look-ahead in sizing).
    """
    n = len(close)
    atr = np.full(n, np.nan, dtype="float64")
    if n < period + 1:
        return atr
    prev_close = np.concatenate([[np.nan], close[:-1]])
    tr = np.maximum.reduce([
        high - low, np.abs(high - prev_close), np.abs(low - prev_close),
    ])
    tr[0] = high[0] - low[0]
    atr[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _label_bar(
    entry_idx: int, close: np.ndarray, high: np.ndarray, low: np.ndarray,
    upper: float, lower: float, max_holding: int,
) -> tuple[float, int]:
    """
    Label one bar by scanning forward for first barrier hit.
    Returns (label, exit_offset).
    """
    n = len(close)
    end = min(entry_idx + max_holding, n - 1)
    for i in range(entry_idx + 1, end + 1):
        if high[i] >= upper:
            return 1.0, i - entry_idx
        if low[i] <= lower:
            return -1.0, i - entry_idx
    return 0.0, end - entry_idx


def compute_triple_barrier(
    df: pd.DataFrame,
    profit_mult: float = DEFAULT_PROFIT_MULT,
    stop_mult: float = DEFAULT_STOP_MULT,
    max_holding: int = DEFAULT_MAX_HOLDING,
    atr_period: int = DEFAULT_ATR_PERIOD,
) -> pd.DataFrame:
    """
    Compute triple barrier labels for the whole DataFrame.

    Args:
        df: DataFrame with [high, low, close]
        profit_mult, stop_mult: barrier sizes in ATR multiples
        max_holding: time barrier (bars)
        atr_period: ATR lookback

    Returns:
        df with columns:
            tb_label          — +1 / -1 / 0 / NaN
            tb_exit_offset    — bars until barrier hit
            tb_return         — actual log return at exit
    """
    if df.empty:
        raise ValueError("Empty DataFrame")
    required = ["high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    n = len(df)
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    close = df["close"].to_numpy(dtype="float64")

    atr = _compute_atr(high, low, close, atr_period)

    labels = np.full(n, np.nan, dtype="float64")
    exits = np.full(n, np.nan, dtype="float64")
    returns = np.full(n, np.nan, dtype="float64")

    start = atr_period
    end = n - max_holding  # last labelable bar

    for t in range(start, end):
        atr_t = atr[t]
        if np.isnan(atr_t) or atr_t <= 0:
            continue
        entry = close[t]
        upper = entry + profit_mult * atr_t
        lower = entry - stop_mult * atr_t
        label, exit_off = _label_bar(t, close, high, low, upper, lower, max_holding)
        labels[t] = label
        exits[t] = exit_off
        exit_price = close[t + exit_off]
        returns[t] = float(np.log(exit_price / entry))

    out = df.copy()
    out["tb_label"] = pd.Series(labels, index=df.index, dtype="float64")
    out["tb_exit_offset"] = pd.Series(exits, index=df.index, dtype="float64")
    out["tb_return"] = pd.Series(returns, index=df.index, dtype="float64")
    return out


def validate_labels(df: pd.DataFrame) -> dict[str, Any]:
    """Validate label distribution and sanity."""
    report: dict[str, Any] = {"issues": [], "stats": {}}
    labels = df["tb_label"].dropna()
    total = len(labels)
    if total == 0:
        report["issues"].append("No valid labels")
        return report

    profit = int((labels == 1).sum())
    loss = int((labels == -1).sum())
    timeout = int((labels == 0).sum())

    report["stats"]["total"] = total
    report["stats"]["profit_pct"] = profit / total * 100
    report["stats"]["loss_pct"] = loss / total * 100
    report["stats"]["timeout_pct"] = timeout / total * 100

    # Balance check: profit ≈ loss (symmetric barriers)
    imbalance = abs(profit - loss) / total * 100
    if imbalance > 15:
        report["issues"].append(
            f"Profit/loss imbalance {imbalance:.1f}% (possible bias)"
        )

    # Returns sanity
    profit_ret = df.loc[df["tb_label"] == 1, "tb_return"].mean()
    loss_ret = df.loc[df["tb_label"] == -1, "tb_return"].mean()
    report["stats"]["profit_avg_return"] = float(profit_ret)
    report["stats"]["loss_avg_return"] = float(loss_ret)

    if profit_ret <= 0:
        report["issues"].append("Profit trades have non-positive return")
    if loss_ret >= 0:
        report["issues"].append("Loss trades have non-negative return")

    return report


if __name__ == "__main__":
    from pathlib import Path
    import time

    print("=" * 70)
    print(f"Triple Barrier Labels Smoke Test (v{TRIPLE_BARRIER_VERSION})")
    print("=" * 70)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    test_file = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m.parquet"

    if not test_file.exists():
        print(f"❌ Feature file not found: {test_file}")
        print("   (Run Phase 2 feature engine first)")
        exit(1)

    print("\nLoading BTCUSDT features...")
    df = pd.read_parquet(test_file)
    df_test = df.iloc[-30000:].reset_index(drop=True).copy()
    print(f"  Test bars: {len(df_test):,}")

    print(f"\nBarrier settings: {DEFAULT_PROFIT_MULT}×ATR profit, "
          f"{DEFAULT_STOP_MULT}×ATR stop, {DEFAULT_MAX_HOLDING} bar timeout")
    print("\nComputing labels...")
    t0 = time.time()
    result = compute_triple_barrier(df_test)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    labels = result["tb_label"].dropna()
    total = len(labels)
    profit = int((labels == 1).sum())
    loss = int((labels == -1).sum())
    timeout = int((labels == 0).sum())

    print("\n" + "=" * 70)
    print("LABEL DISTRIBUTION")
    print("=" * 70)
    print(f"\nTotal labeled: {total:,}")
    print(f"  Profit (+1): {profit:,} ({profit/total*100:.1f}%)")
    print(f"  Loss (-1):   {loss:,} ({loss/total*100:.1f}%)")
    print(f"  Timeout (0): {timeout:,} ({timeout/total*100:.1f}%)")

    exits = result["tb_exit_offset"].dropna()
    print(f"\nExit timing: mean {exits.mean():.1f} bars, "
          f"median {exits.median():.0f} bars")

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    report = validate_labels(result)
    for key, val in report["stats"].items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")

    if report["issues"]:
        print("\n⚠  Issues:")
        for issue in report["issues"]:
            print(f"  - {issue}")
    else:
        print("\n✅ Labels balanced and sane")

    print("\n✅ Triple barrier module working")