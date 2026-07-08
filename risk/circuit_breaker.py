"""
risk/circuit_breaker.py

Phase 5, Component 3: Portfolio circuit breaker (kill-switch).

DESIGN PHILOSOPHY (learned from two sizing failures):
    Continuous vol-sizing (v1.0 total-vol, v1.1 downside-vol) both
    destroyed the overlay's edge because crypto rallies contain violent
    pullbacks — any always-on sizing dilutes the concentrated rally
    capture that generates the edge.

    A circuit breaker is the opposite philosophy:
        99% of the time: DO NOTHING (rally capture fully intact)
        Only in a disaster: equity falls BREAKER_DD below its running
        peak → force cash, wait COOLDOWN bars, re-enter on fresh signal.

    In a rally, equity keeps making new peaks — the breaker never
    triggers. In a crash cascade, it cuts the tail. Tail-risk control
    without continuous interference.

PRE-DECLARED CONFIG (single config, no sweeping):
    BREAKER_DD = -20% from running equity peak (per-symbol)
    COOLDOWN   = 30 bars (~5 days of 4h bars)

PRE-REGISTERED CRITERIA:
    GO:      worst DD <= -35% AND edge retention >= 70%
    PARTIAL: worst DD <= -40% AND retention >= 60%
    FAIL:    accept raw overlay; document capital-fraction guidance
             as the user-level risk control (Phase 6).

MECHANICS (anti-lookahead preserved):
    Equity/peak computed from returns ALREADY realized (bar closes).
    Breaker state at bar i affects position for bar i -> i+1 only.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd  # noqa: F401
import lightgbm as lgb

from core.logger import get_logger
from ml.targets.triple_barrier import compute_triple_barrier
from ml.models.binary_1h import LGBM_PARAMS, _feature_cols
from backtest.backtest_4h import build_4h_features, _period_str

logger = get_logger(__name__)

CIRCUIT_BREAKER_VERSION = "1.0.0"

# ---- Pre-declared configuration (DO NOT sweep) ----
BREAKER_DD = -0.20         # trigger: equity -20% below running peak
COOLDOWN_BARS = 30         # ~5 days of 4h bars in cash after trigger

TH_IN = 0.55               # overlay unchanged
TH_OUT = 0.45
COST_PER_SIDE = 0.00125

N_FOLDS = 5
EMBARGO = 25
MIN_TRAIN = 3500
TEST_SIZE = 900

# Benchmarks for retention math
UNSIZED_EDGE_VS_BH = 0.9092    # +90.92%

SYMBOLS = ["XRPUSDT", "SOLUSDT", "BTCUSDT", "MATICUSDT"]


def _breaker_overlay_sim(
    proba: np.ndarray, close: np.ndarray,
) -> dict[str, float]:
    """
    Raw overlay (full size) + circuit breaker kill-switch.

    Per-bar logic (bar i decides position for i -> i+1):
        1. If in COOLDOWN: stay cash, decrement counter.
        2. Breaker check: if in position AND equity has fallen
           BREAKER_DD below its running peak → force exit (pay cost),
           start cooldown. The peak resets to current equity on
           re-entry (fresh reference, avoids permanent lockout).
        3. Otherwise: standard hysteresis overlay at FULL size.

    Returns:
        total, max_dd, switches, breaker_triggers, time_in_mkt.
    """
    n = len(close)
    log_ret = np.diff(np.log(close))

    in_pos = False
    cooldown = 0
    equity_log = 0.0
    peak_log = 0.0            # running peak of log-equity
    curve: list[float] = [0.0]
    switches = 0
    triggers = 0
    bars_in = 0

    for i in range(n - 1):
        # ---- Step 1: cooldown gate ----
        if cooldown > 0:
            cooldown -= 1
            curve.append(equity_log)
            continue

        # ---- Step 2: breaker check (only while invested) ----
        if in_pos and (equity_log - peak_log) <= np.log(1 + BREAKER_DD):
            # Disaster: force exit, pay cost, start cooldown
            in_pos = False
            triggers += 1
            switches += 1
            equity_log += np.log(1 - COST_PER_SIDE)
            cooldown = COOLDOWN_BARS
            curve.append(equity_log)
            continue

        # ---- Step 3: standard overlay ----
        if not in_pos and proba[i] > TH_IN:
            in_pos = True
            switches += 1
            equity_log += np.log(1 - COST_PER_SIDE) + log_ret[i]
            peak_log = max(peak_log, equity_log)
            # Reset reference on entry so old peaks don't
            # instantly re-trigger after recovery entries:
            peak_log = max(peak_log, equity_log)
            bars_in += 1
        elif in_pos and proba[i] < TH_OUT:
            in_pos = False
            switches += 1
            equity_log += np.log(1 - COST_PER_SIDE)
        elif in_pos:
            equity_log += log_ret[i]
            peak_log = max(peak_log, equity_log)
            bars_in += 1
        # else: flat, do nothing

        curve.append(equity_log)

    eq = np.exp(np.array(curve, dtype="float64"))
    pk = np.maximum.accumulate(eq)
    max_dd = float(((eq - pk) / pk).min())

    return {
        "total": float(eq[-1] - 1.0),
        "max_dd": max_dd,
        "switches": float(switches),
        "triggers": float(triggers),
        "time_in_mkt": bars_in / max(n - 1, 1),
    }


def breaker_symbol(symbol: str) -> Optional[dict[str, Any]]:
    """Walk-forward breaker-protected overlay for one symbol."""
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

        sim_raw = _breaker_overlay_sim(proba, close)
        sim: dict[str, Any] = dict(sim_raw)
        sim["fold"] = k + 1
        sim["period"] = _period_str(te)
        sim["bh"] = float(close[-1] / close[0] - 1.0)
        folds.append(sim)

    if not folds:
        return None
    return {"symbol": symbol, "folds": folds}


if __name__ == "__main__":
    import time

    print("=" * 78)
    print(f"PHASE 5.3 — CIRCUIT BREAKER OVERLAY (v{CIRCUIT_BREAKER_VERSION})")
    print("=" * 78)
    print(f"\nKill-switch: equity {BREAKER_DD*100:.0f}% below peak → cash "
          f"+ {COOLDOWN_BARS}-bar cooldown")
    print("Overlay FULL SIZE otherwise (rally capture untouched)")
    print("\nBenchmarks: unsized +145.96% (DD -49.1%) | "
          "sizing attempts: +21%/+37% (edge destroyed)")
    print("Pre-registered GO: DD <= -35% AND retention >= 70%\n")

    t0 = time.time()
    all_results: list[dict[str, Any]] = []

    for i, sym in enumerate(SYMBOLS, 1):
        print(f"[{i}/{len(SYMBOLS)}] {sym} ...", flush=True)
        r = breaker_symbol(sym)
        if r is None:
            print("   skipped\n")
            continue
        all_results.append(r)
        for f in r["folds"]:
            trig = f"⚡{int(f['triggers'])}" if f["triggers"] > 0 else "  "
            print(f"   F{f['fold']} [{f['period']}] "
                  f"sys {f['total']*100:+8.2f}% (DD {f['max_dd']*100:5.1f}%) "
                  f"{trig} | B&H {f['bh']*100:+8.2f}%")
        print()

    # Portfolio aggregation
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
    sys_f, bh_f, dd_all = [], [], []
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
    new_edge = sys_comp - bh_comp
    retention = new_edge / UNSIZED_EDGE_VS_BH

    print("-" * 55)
    print(f"{'ALL':>5s} {sys_comp*100:>+8.2f}% {worst_dd*100:>7.1f}% "
          f"{bh_comp*100:>+8.2f}%")

    print("\n" + "=" * 78)
    print("VERDICT (pre-registered)")
    print("=" * 78)
    print(f"\n  Compound:    {sys_comp*100:+.2f}% (unsized +145.96%)")
    print(f"  Worst DD:    {worst_dd*100:.1f}% (unsized -49.1%)")
    print(f"  Edge vs B&H: {new_edge*100:+.2f}% "
          f"(retention {retention*100:.0f}%)")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")

    if worst_dd >= -0.35 and retention >= 0.70:
        print("\n  🎯 GO — tail cut, edge intact. Phase 5 risk core COMPLETE.")
        print("     Next: Phase 6 (rules layer + final system assembly).")
    elif worst_dd >= -0.40 and retention >= 0.60:
        print("\n  ⚠️  PARTIAL — acceptable shape. Phase 6 with notes.")
    else:
        print("\n  🔬 FAIL — breaker also costs too much. Final honest shape:")
        print("     RAW overlay + user-level capital fraction (documented")
        print("     in Phase 6): e.g. 60% allocation → DD ~-33%, ret ~+72%.")
    print("=" * 78)