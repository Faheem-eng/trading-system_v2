"""
backtest/regime_overlay.py

REGIME OVERLAY strategy — the pivot the data designed.

Five tests showed a consistent pattern: the model knows REGIMES
(crash-protection alpha in every bear fold: +10 to +31%) but fixed
triple-barrier exits waste it (profit-caps in rallies, churn in chop).

New structure: market-timing overlay on holding —
    LONG while P(up) > 0.55 (stay in, no profit cap)
    CASH when P(up) < 0.45 (hysteresis band avoids churn)
    Cost 0.125% per side, charged on each switch.

Anti-lookahead: signal from bar i's close decides position for
bar i+1's return.

Pre-registered (ONE config, no sweeping):
    GO:      overlay compound > B&H compound
    RISK-GO: compound >= 60% of B&H AND worst DD <= half of B&H DD
    NO-GO:   neither → timing chapter closed, honest report
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

REGIME_OVERLAY_VERSION = "1.0.0"

TH_IN = 0.55
TH_OUT = 0.45
COST_PER_SIDE = 0.00125
N_FOLDS = 5
EMBARGO = 25
MIN_TRAIN = 3500
TEST_SIZE = 900


def _overlay_sim(proba: np.ndarray, close: np.ndarray) -> dict[str, Any]:
    """
    Hysteresis overlay simulation.
    Signal at bar i close → position for bar i→i+1 return.
    """
    n = len(close)
    log_ret = np.diff(np.log(close))  # length n-1, ret[i] = close i→i+1

    in_pos = False
    equity_log = [0.0]
    switches = 0
    bars_in = 0

    for i in range(n - 1):
        # Decide position from signal at bar i
        if not in_pos and proba[i] > TH_IN:
            in_pos = True
            switches += 1
            equity_log.append(equity_log[-1] + np.log(1 - COST_PER_SIDE))
            equity_log[-1] += log_ret[i]
            bars_in += 1
            continue
        if in_pos and proba[i] < TH_OUT:
            in_pos = False
            switches += 1
            equity_log.append(equity_log[-1] + np.log(1 - COST_PER_SIDE))
            continue
        if in_pos:
            equity_log.append(equity_log[-1] + log_ret[i])
            bars_in += 1
        else:
            equity_log.append(equity_log[-1])

    eq = np.exp(np.array(equity_log, dtype="float64"))
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min())
    total = float(eq[-1] - 1.0)

    # B&H comparison on same bars
    bh_eq = close / close[0]
    bh_peak = np.maximum.accumulate(bh_eq)
    bh_dd = float(((bh_eq - bh_peak) / bh_peak).min())
    bh_total = float(close[-1] / close[0] - 1.0)

    return {
        "total": total, "max_dd": dd, "switches": switches,
        "time_in_mkt": bars_in / max(n - 1, 1),
        "bh_total": bh_total, "bh_dd": bh_dd,
    }


def overlay_symbol(symbol: str) -> Optional[dict[str, Any]]:
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

        sim = _overlay_sim(proba, close)
        sim["fold"] = k + 1
        sim["period"] = _period_str(te)
        folds.append(sim)

    if not folds:
        return None
    return {"symbol": symbol, "folds": folds}


if __name__ == "__main__":
    import time

    SYMBOLS = ["XRPUSDT", "SOLUSDT", "BTCUSDT", "MATICUSDT"]

    print("=" * 78)
    print(f"REGIME OVERLAY BACKTEST (v{REGIME_OVERLAY_VERSION})")
    print("=" * 78)
    print(f"\nLong while P(up)>{TH_IN}, cash when <{TH_OUT} (hysteresis)")
    print(f"Cost {COST_PER_SIDE*100:.3f}%/side per switch | 4h | walk-forward")
    print("GO: beats B&H | RISK-GO: ≥60% of B&H return at ≤50% of B&H DD\n")

    t0 = time.time()
    all_results = []

    for i, sym in enumerate(SYMBOLS, 1):
        print(f"[{i}/{len(SYMBOLS)}] {sym} ...", flush=True)
        r = overlay_symbol(sym)
        if r is None:
            print("   skipped\n")
            continue
        all_results.append(r)
        for f in r["folds"]:
            beat = "✅" if f["total"] > f["bh_total"] else "—"
            print(f"   F{f['fold']} [{f['period']}] "
                  f"sys {f['total']*100:+8.2f}% (DD {f['max_dd']*100:5.1f}%) | "
                  f"B&H {f['bh_total']*100:+8.2f}% (DD {f['bh_dd']*100:5.1f}%) | "
                  f"sw {f['switches']:3d} | in-mkt {f['time_in_mkt']*100:3.0f}% {beat}")
        print()

    # Portfolio aggregate
    print("=" * 78)
    print("PORTFOLIO BY FOLD (equal-weight)")
    print("=" * 78)
    pf: dict[int, dict[str, list[float]]] = {}
    for r in all_results:
        for f in r["folds"]:
            d = pf.setdefault(f["fold"], {"sys": [], "bh": [],
                                          "sdd": [], "bdd": []})
            d["sys"].append(f["total"])
            d["bh"].append(f["bh_total"])
            d["sdd"].append(f["max_dd"])
            d["bdd"].append(f["bh_dd"])

    print(f"\n{'Fold':>5s} {'SysNet':>9s} {'SysDD':>8s} {'B&H':>9s} "
          f"{'BhDD':>8s} {'Alpha':>9s}")
    print("-" * 60)
    sys_f, bh_f, sys_dd_all, bh_dd_all = [], [], [], []
    for k in sorted(pf.keys()):
        s = float(np.mean(pf[k]["sys"]))
        b = float(np.mean(pf[k]["bh"]))
        sdd = float(np.min(pf[k]["sdd"]))
        bdd = float(np.min(pf[k]["bdd"]))
        sys_f.append(s); bh_f.append(b)  # noqa: E702
        sys_dd_all.append(sdd); bh_dd_all.append(bdd)  # noqa: E702
        print(f"{k:>5d} {s*100:>+8.2f}% {sdd*100:>7.1f}% {b*100:>+8.2f}% "
              f"{bdd*100:>7.1f}% {(s-b)*100:>+8.2f}%")

    sys_comp = float(np.exp(np.sum([np.log(1+x) for x in sys_f])) - 1)
    bh_comp = float(np.exp(np.sum([np.log(1+x) for x in bh_f])) - 1)
    worst_sys_dd = float(np.min(sys_dd_all))
    worst_bh_dd = float(np.min(bh_dd_all))

    print("-" * 60)
    print(f"{'ALL':>5s} {sys_comp*100:>+8.2f}% {worst_sys_dd*100:>7.1f}% "
          f"{bh_comp*100:>+8.2f}% {worst_bh_dd*100:>7.1f}%")

    print("\n" + "=" * 78)
    print("FINAL VERDICT (pre-registered)")
    print("=" * 78)
    print(f"\n  Overlay: {sys_comp*100:+.2f}% (worst DD {worst_sys_dd*100:.1f}%)")
    print(f"  B&H:     {bh_comp*100:+.2f}% (worst DD {worst_bh_dd*100:.1f}%)")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")

    risk_go = (sys_comp >= 0.6 * bh_comp if bh_comp > 0 else sys_comp > 0) \
              and (worst_sys_dd >= 0.5 * worst_bh_dd)  # DDs negative: >= means smaller loss

    if sys_comp > bh_comp:
        print("\n  🎯 GO — overlay BEATS buy-and-hold. Phase 5 next.")
    elif risk_go:
        print("\n  ✅ RISK-GO — captures ≥60% of B&H at ≤half the drawdown.")
        print("     Usable as a defensive allocation strategy. Phase 5 next.")
    else:
        print("\n  🔬 NO-GO — timing chapter closed. Honest final report next.")
    print("=" * 78)