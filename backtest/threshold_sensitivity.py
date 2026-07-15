"""
backtest/threshold_sensitivity.py

Threshold sensitivity analysis for the regime overlay strategy.

PURPOSE (robustness validation, NOT optimization):
    The overlay's +146% vs B&H +55% was achieved with ONE pre-declared
    config (enter 0.55 / exit 0.45). If that result only exists at those
    exact numbers, it is a "parameter cliff" — the classic signature of
    overfitting — and would likely collapse in live trading.

    A REAL edge is a hill, not a spike: nearby parameters should also
    be profitable (perhaps less, but positive).

METHOD:
    Re-run the identical walk-forward overlay simulation with 3 configs:
        LOOSE : enter 0.50 / exit 0.40  (more time in market)
        BASE  : enter 0.55 / exit 0.45  (the tested config)
        TIGHT : enter 0.60 / exit 0.50  (only strong conviction)
    Everything else identical: same folds, same models, same costs.
    Models are trained ONCE per fold and reused across configs
    (thresholds only affect the simulation layer, not the model).

PRE-REGISTERED INTERPRETATION (fixed before running):
    ROBUST : all 3 configs positive compound AND >=2 beat or land
             within 20% of B&H  → proceed to Phase 5 with confidence
    PARTIAL: base + one neighbor positive → Phase 5 with conservative
             sizing assumptions
    FRAGILE: only base works → red flag; re-examine before Phase 5

This is validation. We will NOT switch to whichever config looks best —
that would be the exact overfitting this test exists to detect.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import lightgbm as lgb

from core.logger import get_logger
from ml.targets.triple_barrier import compute_triple_barrier
from ml.models.binary_1h import LGBM_PARAMS, _feature_cols
from backtest.backtest_4h import build_4h_features

logger = get_logger(__name__)

SENSITIVITY_VERSION = "1.0.0"

# The three configs under test (pre-declared)
CONFIGS: list[dict[str, float | str]] = [
    {"name": "LOOSE", "th_in": 0.50, "th_out": 0.40},
    {"name": "BASE",  "th_in": 0.55, "th_out": 0.45},
    {"name": "TIGHT", "th_in": 0.60, "th_out": 0.50},
]

COST_PER_SIDE = 0.00125     # 0.125% per switch side (same as overlay test)
N_FOLDS = 5
EMBARGO = 25                # > 20-bar label window
MIN_TRAIN = 3500
TEST_SIZE = 900             # ~5 months of 4h bars per fold

SYMBOLS = ["XRPUSDT", "SOLUSDT", "BTCUSDT", "MATICUSDT"]


def _overlay_sim(
    proba: np.ndarray, close: np.ndarray,
    th_in: float, th_out: float,
) -> dict[str, float]:
    """
    Hysteresis overlay simulation (identical mechanics to regime_overlay).

    Anti-lookahead: the signal computed at bar i's close decides the
    position held for the i→i+1 return. No same-bar action.

    Args:
        proba:  P(up) per bar from the fold's model.
        close:  Close prices for the fold's test window.
        th_in:  Enter long when proba crosses above this.
        th_out: Exit to cash when proba crosses below this.

    Returns:
        total return, max drawdown, switch count, time-in-market.
    """
    n = len(close)
    log_ret = np.diff(np.log(close))     # ret[i] = bar i → i+1

    in_pos = False
    equity_log: list[float] = [0.0]
    switches = 0
    bars_in = 0

    for i in range(n - 1):
        if not in_pos and proba[i] > th_in:
            in_pos = True
            switches += 1
            # Pay entry cost, then earn this bar's return
            equity_log.append(
                equity_log[-1] + np.log(1 - COST_PER_SIDE) + log_ret[i]
            )
            bars_in += 1
        elif in_pos and proba[i] < th_out:
            in_pos = False
            switches += 1
            # Pay exit cost, hold cash this bar
            equity_log.append(equity_log[-1] + np.log(1 - COST_PER_SIDE))
        elif in_pos:
            equity_log.append(equity_log[-1] + log_ret[i])
            bars_in += 1
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
    }


def sensitivity_symbol(symbol: str) -> Optional[dict[str, Any]]:
    """
    Run all 3 threshold configs on one symbol's walk-forward folds.

    Efficiency note: features are built once and the model is trained
    once per fold; each config only re-runs the cheap simulation layer.
    This also guarantees all configs see IDENTICAL model outputs —
    the only variable is the threshold pair (clean experiment).
    """
    df = build_4h_features(symbol)
    if df is None or len(df) < MIN_TRAIN + TEST_SIZE:
        return None

    df = compute_triple_barrier(df, profit_mult=3.0, stop_mult=3.0,
                                max_holding=20)
    fcols = _feature_cols(df)

    n = len(df)
    usable_end = n - 25

    # results[config_name] = list of per-fold dicts
    results: dict[str, list[dict[str, Any]]] = {
        str(c["name"]): [] for c in CONFIGS
    }
    bh_folds: list[dict[str, Any]] = []

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

        # Train ONCE per fold — shared across all threshold configs
        model = lgb.train(
            LGBM_PARAMS, lgb.Dataset(X_tr, label=y_tr),
            num_boost_round=100,
            callbacks=[lgb.log_evaluation(period=0)],
        )
        proba = np.asarray(model.predict(te[fcols]), dtype="float64")
        close = np.asarray(te["close"].to_numpy(), dtype="float64")

        bh_folds.append({
            "fold": k + 1,
            "bh": float(close[-1] / close[0] - 1.0),
        })

        for cfg in CONFIGS:
            sim = _overlay_sim(
                proba, close,
                th_in=float(cfg["th_in"]), th_out=float(cfg["th_out"]),
            )
            sim["fold"] = k + 1
            results[str(cfg["name"])].append(sim)

    if not bh_folds:
        return None
    return {"symbol": symbol, "results": results, "bh": bh_folds}


if __name__ == "__main__":
    # cp1252-safe console: this research script prints emoji/symbols; a run
    # must never crash on cosmetics (Windows default console is cp1252).
    import sys as _sys
    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import time

    print("=" * 78)
    print(f"THRESHOLD SENSITIVITY ANALYSIS (v{SENSITIVITY_VERSION})")
    print("=" * 78)
    print("\nConfigs: LOOSE 0.50/0.40 | BASE 0.55/0.45 | TIGHT 0.60/0.50")
    print("Same folds, same models — ONLY thresholds vary (clean experiment)")
    print("\nPre-registered: ROBUST = all 3 positive, ≥2 beat/near B&H")
    print("                FRAGILE = only BASE works → red flag\n")

    t0 = time.time()
    all_symbol_results: list[dict[str, Any]] = []

    for i, sym in enumerate(SYMBOLS, 1):
        print(f"[{i}/{len(SYMBOLS)}] {sym} (features + 5 folds × 3 configs)...",
              flush=True)
        r = sensitivity_symbol(sym)
        if r is None:
            print("   skipped\n")
            continue
        all_symbol_results.append(r)
        print("   done")
    print()

    # ================================================================
    # Aggregate: portfolio compound per config
    # ================================================================
    print("=" * 78)
    print("PORTFOLIO COMPOUND PER CONFIG (equal-weight, all folds)")
    print("=" * 78)

    # Build per-config, per-fold portfolio means
    config_summary: list[dict[str, Any]] = []

    # B&H portfolio per fold
    bh_by_fold: dict[int, list[float]] = {}
    for r in all_symbol_results:
        for b in r["bh"]:
            bh_by_fold.setdefault(b["fold"], []).append(b["bh"])
    bh_fold_means = [float(np.mean(v)) for _, v in sorted(bh_by_fold.items())]
    bh_comp = float(np.exp(np.sum([np.log(1 + x) for x in bh_fold_means])) - 1)

    print(f"\n{'Config':>7s} {'Compound':>10s} {'WorstDD':>9s} "
          f"{'AvgSwitches':>12s} {'vs B&H':>9s}")
    print("-" * 78)

    for cfg in CONFIGS:
        name = str(cfg["name"])
        by_fold: dict[int, list[float]] = {}
        dds: list[float] = []
        sws: list[float] = []
        for r in all_symbol_results:
            for f in r["results"][name]:
                by_fold.setdefault(int(f["fold"]), []).append(f["total"])
                dds.append(f["max_dd"])
                sws.append(f["switches"])
        fold_means = [float(np.mean(v)) for _, v in sorted(by_fold.items())]
        comp = float(np.exp(np.sum([np.log(1 + x) for x in fold_means])) - 1)
        worst_dd = float(np.min(dds)) if dds else 0.0
        avg_sw = float(np.mean(sws)) if sws else 0.0

        config_summary.append({
            "name": name, "compound": comp, "worst_dd": worst_dd,
            "fold_means": fold_means,
        })
        vs = comp - bh_comp
        print(f"{name:>7s} {comp*100:>+9.2f}% {worst_dd*100:>8.1f}% "
              f"{avg_sw:>12.1f} {vs*100:>+8.2f}%")

    print("-" * 78)
    print(f"{'B&H':>7s} {bh_comp*100:>+9.2f}%")

    # Per-fold detail per config
    print("\n" + "=" * 78)
    print("PER-FOLD PORTFOLIO RETURNS (config comparison)")
    print("=" * 78)
    header = f"\n{'Fold':>5s}"
    for cfg in config_summary:
        header += f" {cfg['name']:>10s}"
    header += f" {'B&H':>10s}"
    print(header)
    print("-" * 60)
    n_folds_found = len(config_summary[0]["fold_means"]) if config_summary else 0
    for fi in range(n_folds_found):
        row = f"{fi+1:>5d}"
        for cfg in config_summary:
            row += f" {cfg['fold_means'][fi]*100:>+9.2f}%"
        row += f" {bh_fold_means[fi]*100:>+9.2f}%"
        print(row)

    # ================================================================
    # Pre-registered verdict
    # ================================================================
    print("\n" + "=" * 78)
    print("VERDICT (pre-registered)")
    print("=" * 78)

    n_positive = sum(1 for c in config_summary if c["compound"] > 0)
    n_beat_or_near = sum(
        1 for c in config_summary
        if c["compound"] > bh_comp or c["compound"] >= 0.8 * bh_comp
    )

    print(f"\n  Configs positive:      {n_positive}/3")
    print(f"  Configs beat/near B&H: {n_beat_or_near}/3")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")

    if n_positive == 3 and n_beat_or_near >= 2:
        print("\n  🎯 ROBUST — the edge is a hill, not a spike.")
        print("     Nearby thresholds also work. Proceed to Phase 5")
        print("     with the BASE config (0.55/0.45) as planned.")
    elif n_positive >= 2:
        print("\n  ⚠️  PARTIALLY ROBUST — edge exists in a threshold")
        print("     neighborhood but weakens at one extreme. Proceed to")
        print("     Phase 5 with conservative sizing assumptions.")
    else:
        print("\n  🚨 FRAGILE — parameter cliff detected. The +146% was")
        print("     likely threshold luck. HALT before Phase 5; re-examine.")
    print("=" * 78)