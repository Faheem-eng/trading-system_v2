"""
risk/vol_targeting.py

Phase 5, Component 1: DOWNSIDE volatility-targeted position sizing.

VERSION HISTORY (design evolution, documented for future reference):
    v1.0.0 — Total-volatility targeting (Moreira & Muir 2017).
             Result: DD fixed (-49.1% → -25.2%) BUT edge destroyed
             (+145.96% → +21.10%, retention -37%).
             Root cause: crypto rallies are ALSO high-vol; total vol
             treated upside opportunity as risk. F2 evidence: XRP avg
             size was 36% during a +340% rally.
    v1.1.0 — DOWNSIDE-vol targeting (semi-deviation, Sortino logic).
             Penalizes ONLY negative-return volatility:
                 rally (high up-vol, low down-vol) → size stays HIGH
                 crash (down-vol explodes)         → size cuts fast
             This aligns the sizer with what the model already knows —
             its #1 feature is downside_vol_ratio.

THE PROBLEM THIS SOLVES:
    The regime overlay is profitable (+146% vs B&H +55%) and robust
    (3/3 threshold configs beat B&H), but worst drawdown -49.1% is
    psychologically untradeable for retail capital.

SIZING FORMULA:
    size_t = min(SIZE_CAP, TARGET_VOL / downside_vol_est_t)

    Calm/rising market → full size (capture rallies)
    Violent downside   → fractional size (survive crashes)

DOWNSIDE VOL ESTIMATE (crash-responsive, past-only):
    semi_dev = std of NEGATIVE returns within the rolling window
    vol_est  = max(semi_20bar, semi_5bar), annualized

    The 5-bar fast component reacts within ~20 hours of a downside
    spike; the 20-bar slow component provides stability. Taking the
    MAX shrinks fast and re-expands slowly — intentional asymmetry
    favoring survival. Windows with <2 negative returns yield NaN →
    conservative 0.25 size floor applies (unknown risk = small size).

PRE-DECLARED CONFIG (no parameter sweeping — validation discipline):
    TARGET_VOL = 0.20 annualized DOWNSIDE vol
                 (downside vol is typically ~2/3 of total vol, so this
                  is roughly equivalent to the v1.0 30% total target —
                  the TARGET is unchanged in spirit; only the RISK
                  DEFINITION changed, which is the structural fix)
    SIZE_CAP   = 1.00 (spot account, leverage impossible)
    Overlay thresholds unchanged: 0.55 in / 0.45 out.

PRE-REGISTERED SUCCESS CRITERIA (fixed before running):
    GO:      worst portfolio DD <= -30% AND edge retention >= 60%
             (edge = vs-B&H outperformance; unsized was +90.92%,
              so retained edge must be >= +54.55%)
    PARTIAL: DD <= -35% AND retention >= 40%
    FAIL:    neither → sizing chapter honest review;
             circuit-breaker-only path next

ANTI-LOOKAHEAD GUARANTEE:
    The vol estimate at bar i uses returns up to and including bar i
    only. It sizes the position held for the i -> i+1 return. The
    regime signal (proba) at bar i likewise decides the i -> i+1
    position. No same-bar information is used.

COST MODEL:
    Switch costs (0.125% per side) are charged on the TRADED FRACTION
    at entry/exit. Size drift while in-position is treated as costless
    (simplification: live trading would rebalance on a grid; this
    slightly understates costs — flagged for Phase 7 paper trading).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

from core.logger import get_logger
from ml.targets.triple_barrier import compute_triple_barrier
from ml.models.binary_1h import LGBM_PARAMS, _feature_cols
from backtest.backtest_4h import build_4h_features, _period_str

logger = get_logger(__name__)

VOL_TARGETING_VERSION = "1.1.0"

# ============================================================
# Pre-declared configuration (DO NOT sweep)
# ============================================================

TARGET_VOL = 0.20              # 20% annualized DOWNSIDE vol target
VOL_TARGETING_MODE = "downside"  # v1.1: semi-deviation (Sortino logic)
SIZE_CAP = 1.00                # spot: no leverage possible
SIZE_FLOOR_UNKNOWN = 0.25      # size when vol is unknown (warmup NaN)

TH_IN = 0.55                   # overlay entry threshold (unchanged)
TH_OUT = 0.45                  # overlay exit threshold (unchanged)
COST_PER_SIDE = 0.00125        # 0.125% per switch side

# Walk-forward protocol (identical to all prior tests)
N_FOLDS = 5
EMBARGO = 25                   # > 20-bar label window
MIN_TRAIN = 3500
TEST_SIZE = 900                # ~5 months of 4h bars per fold

BARS_PER_YEAR_4H = 6 * 365     # 2,190 four-hour bars per year
ANNUALIZE = float(np.sqrt(BARS_PER_YEAR_4H))

# Benchmark from the unsized overlay run (for edge-retention math)
UNSIZED_COMPOUND = 1.4596      # +145.96%
UNSIZED_EDGE_VS_BH = 0.9092    # +90.92% vs B&H

SYMBOLS = ["XRPUSDT", "SOLUSDT", "BTCUSDT", "MATICUSDT"]


# ============================================================
# Downside volatility estimation (semi-deviation)
# ============================================================


def _vol_estimate(log_ret: np.ndarray) -> np.ndarray:
    """
    Crash-responsive annualized DOWNSIDE volatility (semi-deviation).

    v1.1 STRUCTURAL CHANGE (pre-declared — not a parameter sweep):
        v1.0 used total volatility and destroyed rally capture
        (F2: avg size 36% during a +340% rally, edge retention -37%).
        Crypto up-moves are violent too; total vol treats opportunity
        as risk. Semi-deviation penalizes ONLY negative returns —
        the Sortino logic applied to position sizing.

    Computation:
        semi_dev[i] = std of NEGATIVE returns in the window ending at i
        vol_est[i]  = max(semi_20bar[i], semi_5bar[i]) * sqrt(bars/yr)

    Edge cases:
        - Window with <2 negative returns → NaN → caller applies the
          conservative SIZE_FLOOR_UNKNOWN (unknown risk = small size).
        - First 20 bars → NaN (insufficient history) → same floor.

    Look-ahead safety:
        The rolling window at bar i ends AT bar i (min_periods enforced,
        pandas rolling is trailing by default). No future data.

    Args:
        log_ret: per-bar log returns, length n, where log_ret[i] is the
                 return from bar i-1 to bar i (index 0 is NaN).

    Returns:
        Annualized downside vol estimate array, length n, float64.
    """
    s = pd.Series(log_ret)

    def _semi_dev(x: np.ndarray) -> float:
        neg = x[x < 0]
        if len(neg) < 2:
            return np.nan
        return float(np.std(neg))

    semi_slow = s.rolling(20, min_periods=20).apply(
        _semi_dev, raw=True
    ).to_numpy() * ANNUALIZE

    semi_fast = s.rolling(5, min_periods=5).apply(
        _semi_dev, raw=True
    ).to_numpy() * ANNUALIZE

    # fmax: element-wise max that propagates a number over NaN,
    # NaN only where BOTH are NaN (early warmup)
    est = np.fmax(semi_slow, semi_fast)
    return np.asarray(est, dtype="float64")


def _size_from_vol(vol_est_i: float) -> float:
    """
    Convert a downside-vol estimate into a position size.

        size = min(SIZE_CAP, TARGET_VOL / vol_est)

    Unknown vol (NaN) → conservative floor.

    Args:
        vol_est_i: annualized downside vol estimate at the current bar.

    Returns:
        Position size in [0, SIZE_CAP].
    """
    if np.isnan(vol_est_i):
        return SIZE_FLOOR_UNKNOWN
    return float(min(SIZE_CAP, TARGET_VOL / max(vol_est_i, 1e-9)))


# ============================================================
# Sized overlay simulation
# ============================================================


def _sized_overlay_sim(
    proba: np.ndarray, close: np.ndarray,
) -> dict[str, float]:
    """
    Regime overlay simulation WITH downside-vol-targeted sizing.

    Per-bar mechanics (bar i decides exposure for the i -> i+1 return):
        1. REGIME GATE (hysteresis, unchanged from validated overlay):
             enter when proba[i] > TH_IN, exit when proba[i] < TH_OUT.
        2. SIZE (only while in-position):
             size_t = min(SIZE_CAP, TARGET_VOL / downside_vol_est[i])
        3. PnL for the bar = size_t * log_ret[i]
           (the un-invested fraction sits in cash earning 0).
        4. COSTS charged on the traded fraction at entry and exit:
             log(1 - COST_PER_SIDE * size_at_switch).

    Args:
        proba: P(up) per bar from the fold's model (length n).
        close: close prices for the fold's test window (length n).

    Returns:
        Dict with total return, max drawdown, switch count,
        time-in-market fraction, and average size while invested.
    """
    n = len(close)
    log_ret = np.diff(np.log(close))              # ret[i] = bar i -> i+1

    # Return history aligned so vol_est[i] uses returns up to bar i:
    # ret_hist[i] = return from bar i-1 to bar i (past-only at i)
    ret_hist = np.concatenate([[np.nan], log_ret])
    vol_est = _vol_estimate(ret_hist)

    in_pos = False
    cur_size = 0.0
    equity_log: list[float] = [0.0]
    switches = 0
    bars_in = 0
    sizes_when_in: list[float] = []

    for i in range(n - 1):
        # ---- Step 1: regime gate ----
        if not in_pos and proba[i] > TH_IN:
            in_pos = True
            switches += 1
            cur_size = _size_from_vol(vol_est[i])
            # entry cost on traded fraction, then earn sized return
            step = (
                np.log(1 - COST_PER_SIDE * cur_size)
                + cur_size * log_ret[i]
            )
            equity_log.append(equity_log[-1] + step)
            bars_in += 1
            sizes_when_in.append(cur_size)
            continue

        if in_pos and proba[i] < TH_OUT:
            in_pos = False
            switches += 1
            equity_log.append(
                equity_log[-1] + np.log(1 - COST_PER_SIDE * cur_size)
            )
            cur_size = 0.0
            continue

        if in_pos:
            # ---- Step 2: refresh size (costless drift approximation) ----
            cur_size = _size_from_vol(vol_est[i])
            equity_log.append(equity_log[-1] + cur_size * log_ret[i])
            bars_in += 1
            sizes_when_in.append(cur_size)
        else:
            equity_log.append(equity_log[-1])

    eq = np.exp(np.array(equity_log, dtype="float64"))
    peak = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / peak).min())

    return {
        "total": float(eq[-1] - 1.0),
        "max_dd": max_dd,
        "switches": float(switches),
        "time_in_mkt": bars_in / max(n - 1, 1),
        "avg_size": float(np.mean(sizes_when_in)) if sizes_when_in else 0.0,
    }


# ============================================================
# Walk-forward per symbol
# ============================================================


def vol_targeted_symbol(symbol: str) -> Optional[dict[str, Any]]:
    """
    Walk-forward downside-vol-targeted overlay for one symbol.

    Protocol identical to every prior test (5 folds, 900-bar tests,
    25-bar embargo, model retrained per fold, zero per-fold tuning).

    Args:
        symbol: e.g. "BTCUSDT" (4h raw data must exist).

    Returns:
        Dict with per-fold results, or None if data insufficient.
    """
    df = build_4h_features(symbol)
    if df is None or len(df) < MIN_TRAIN + TEST_SIZE:
        return None

    df = compute_triple_barrier(df, profit_mult=3.0, stop_mult=3.0,
                                max_holding=20)
    fcols = _feature_cols(df)
    n = len(df)
    usable_end = n - 25

    folds: list[dict[str, Any]] = []

    for k in range(N_FOLDS):
        test_end = usable_end - (N_FOLDS - 1 - k) * TEST_SIZE
        test_start = test_end - TEST_SIZE
        train_end = test_start - EMBARGO
        if train_end < MIN_TRAIN:
            continue

        tr = df.iloc[:train_end]
        te = df.iloc[test_start:test_end].reset_index(drop=True)

        trl = tr[tr["tb_label"].isin([-1.0, 1.0])]
        X_tr = trl[fcols]
        y_tr = (trl["tb_label"] == 1.0).astype("int64")
        if len(X_tr) < 1500 or len(te) < 300:
            continue

        model = lgb.train(
            LGBM_PARAMS, lgb.Dataset(X_tr, label=y_tr),
            num_boost_round=100,
            callbacks=[lgb.log_evaluation(period=0)],
        )
        proba = np.asarray(model.predict(te[fcols]), dtype="float64")
        close = np.asarray(te["close"].to_numpy(), dtype="float64")

        sim = _sized_overlay_sim(proba, close)
        sim["fold"] = k + 1
        sim["period"] = _period_str(te) # type: ignore
        sim["bh"] = float(close[-1] / close[0] - 1.0)
        folds.append(sim)

    if not folds:
        return None
    return {"symbol": symbol, "folds": folds}


# ============================================================
# Main: run + report + pre-registered verdict
# ============================================================


if __name__ == "__main__":
    import time

    print("=" * 78)
    print(f"PHASE 5.1 — DOWNSIDE-VOL-TARGETED OVERLAY "
          f"(v{VOL_TARGETING_VERSION})")
    print("=" * 78)
    print(f"\nMode: {VOL_TARGETING_MODE.upper()} (semi-deviation, "
          f"Sortino logic)")
    print(f"Target DOWNSIDE vol: {TARGET_VOL*100:.0f}% ann. | "
          f"cap {SIZE_CAP*100:.0f}% | floor {SIZE_FLOOR_UNKNOWN*100:.0f}% "
          f"on unknown vol")
    print(f"vol_est = max(semi_20bar, semi_5bar) | overlay {TH_IN}/{TH_OUT} "
          f"unchanged")
    print("\nBenchmarks: unsized overlay +145.96% (DD -49.1%) | "
          "v1.0 total-vol +21.10% (DD -25.2%)")
    print("Pre-registered GO: DD <= -30% AND edge retention >= 60% "
          "(>= +54.55% vs B&H)\n")

    t0 = time.time()
    all_results: list[dict[str, Any]] = []

    for i, sym in enumerate(SYMBOLS, 1):
        print(f"[{i}/{len(SYMBOLS)}] {sym} ...", flush=True)
        r = vol_targeted_symbol(sym)
        if r is None:
            print("   skipped\n")
            continue
        all_results.append(r)
        for f in r["folds"]:
            print(f"   F{f['fold']} [{f['period']}] "
                  f"sys {f['total']*100:+8.2f}% (DD {f['max_dd']*100:5.1f}%) | "
                  f"avg size {f['avg_size']*100:3.0f}% | "
                  f"B&H {f['bh']*100:+8.2f}%")
        print()

    # ---- Portfolio aggregation (equal-weight across symbols) ----
    print("=" * 78)
    print("PORTFOLIO BY FOLD (equal-weight)")
    print("=" * 78)
    pf: dict[int, dict[str, list[float]]] = {}
    for r in all_results:
        for f in r["folds"]:
            d = pf.setdefault(int(f["fold"]),
                              {"sys": [], "bh": [], "dd": []})
            d["sys"].append(f["total"])
            d["bh"].append(f["bh"])
            d["dd"].append(f["max_dd"])

    print(f"\n{'Fold':>5s} {'SysNet':>9s} {'SysDD':>8s} {'B&H':>9s} "
          f"{'Alpha':>9s}")
    print("-" * 55)
    sys_f: list[float] = []
    bh_f: list[float] = []
    dd_all: list[float] = []
    for k in sorted(pf.keys()):
        s = float(np.mean(pf[k]["sys"]))
        b = float(np.mean(pf[k]["bh"]))
        d = float(np.min(pf[k]["dd"]))
        sys_f.append(s)
        bh_f.append(b)
        dd_all.append(d)
        print(f"{k:>5d} {s*100:>+8.2f}% {d*100:>7.1f}% {b*100:>+8.2f}% "
              f"{(s-b)*100:>+8.2f}%")

    sys_comp = float(np.exp(np.sum([np.log(1 + x) for x in sys_f])) - 1)
    bh_comp = float(np.exp(np.sum([np.log(1 + x) for x in bh_f])) - 1)
    worst_dd = float(np.min(dd_all))

    print("-" * 55)
    print(f"{'ALL':>5s} {sys_comp*100:>+8.2f}% {worst_dd*100:>7.1f}% "
          f"{bh_comp*100:>+8.2f}%")

    # ---- Pre-registered verdict ----
    new_edge = sys_comp - bh_comp
    edge_retention = (
        new_edge / UNSIZED_EDGE_VS_BH if UNSIZED_EDGE_VS_BH > 0 else 0.0
    )

    print("\n" + "=" * 78)
    print("VERDICT (pre-registered)")
    print("=" * 78)
    print(f"\n  Compound:        {sys_comp*100:+.2f}% "
          f"(unsized +145.96% | v1.0 total-vol +21.10%)")
    print(f"  Worst DD:        {worst_dd*100:.1f}% "
          f"(unsized -49.1% | v1.0 -25.2%)")
    print(f"  Edge vs B&H:     {new_edge*100:+.2f}% "
          f"(retention {edge_retention*100:.0f}% of +90.92%)")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")

    if worst_dd >= -0.30 and edge_retention >= 0.60:
        print("\n  🎯 GO — survivable drawdown WITH edge intact.")
        print("     Downside-vol sizing preserved rallies, cut crashes.")
        print("     Next: Component 3 (circuit breakers) + final assembly.")
    elif worst_dd >= -0.35 and edge_retention >= 0.40:
        print("\n  ⚠️  PARTIAL — close to target. Next: circuit breakers,")
        print("     then re-measure the combined effect.")
    else:
        print("\n  🔬 INSUFFICIENT — semi-deviation sizing also fails the")
        print("     retention bar. Honest review: sizing chapter closes;")
        print("     circuit-breaker-only path (raw overlay + kill-switch).")
    print("=" * 78)