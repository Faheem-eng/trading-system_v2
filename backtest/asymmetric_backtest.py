"""
backtest/asymmetric_backtest.py

Walk-forward backtest with ASYMMETRIC barriers (4xATR profit / 2xATR stop).

WHY (diagnosis from symmetric failure):
    Symmetric 3x/3x barriers: 50-55% WR x symmetric payoff - 0.25% cost
    = guaranteed negative drift. Structural math trap, not a model failure.

    Asymmetric 2:1 reward-risk changes the math:
        breakeven WR = 2/(4+2) = 33.3% (+costs ~36%)
        model's proven WR (42-55%) > 36% → positive expectancy possible.

DISCIPLINE (anti-overfitting):
    ONE pre-declared config only: 4xATR profit, 2xATR stop, 30-bar hold.
    No config sweeping. If this fails → pivot to 4h, no cherry-picking.

    Same protocol as symmetric walk-forward: threshold 0.58 fixed,
    long-only, non-overlapping, 0.25% costs, 5 folds, zero tuning.

Pre-registered verdict:
    GO:       3+/5 portfolio folds positive AND compound > B&H
    MARGINAL: compound positive but < B&H → regime filter next
    NO-GO:    compound negative → 4h pivot
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

from core.logger import get_logger
from ml.targets.triple_barrier import compute_triple_barrier
from ml.models.binary_1h import LGBM_PARAMS, _feature_cols

logger = get_logger(__name__)

ASYM_BACKTEST_VERSION = "1.0.0"

# PRE-DECLARED asymmetric config (the only one we test)
PROFIT_MULT = 4.0
STOP_MULT = 2.0
MAX_HOLDING = 30          # slightly longer to let 4xATR targets get hit

ENTRY_THRESHOLD = 0.58    # unchanged from symmetric test
ROUND_TRIP_COST = 0.0025
N_FOLDS = 5
EMBARGO = 35              # > 30-bar holding (embargo must exceed target window)
MIN_TRAIN = 8000
TEST_SIZE = 2700
BARS_PER_YEAR_1H = 24 * 365


def _period_str(df: pd.DataFrame) -> str:
    t0 = datetime.fromtimestamp(
        int(df["open_time"].min()) / 1000, tz=timezone.utc).strftime("%Y-%m")
    t1 = datetime.fromtimestamp(
        int(df["open_time"].max()) / 1000, tz=timezone.utc).strftime("%Y-%m")
    return f"{t0}→{t1}"


def _simulate(proba: np.ndarray, tb_return: np.ndarray,
              tb_exit: np.ndarray) -> list[float]:
    """Long-only, non-overlapping simulation. Returns net log-returns."""
    n = len(proba)
    rets: list[float] = []
    i = 0
    while i < n:
        if proba[i] > ENTRY_THRESHOLD and not np.isnan(tb_return[i]):
            rets.append(float(tb_return[i]) - ROUND_TRIP_COST)
            off = int(tb_exit[i]) if not np.isnan(tb_exit[i]) else MAX_HOLDING
            i += max(off, 1)
        else:
            i += 1
    return rets


def asym_backtest_symbol(
    symbol: str,
    features_dir: str = "data/features_1h",
) -> Optional[dict[str, Any]]:
    """Walk-forward asymmetric backtest for one symbol."""
    feature_file = Path(features_dir) / f"{symbol}_1h.parquet"
    if not feature_file.exists():
        return None

    df = pd.read_parquet(feature_file)
    # ASYMMETRIC labels: 4xATR profit, 2xATR stop
    df = compute_triple_barrier(df, profit_mult=PROFIT_MULT,
                                stop_mult=STOP_MULT,
                                max_holding=MAX_HOLDING)
    fcols = _feature_cols(df)

    # Label distribution (Risk 1 check)
    labels = df["tb_label"].dropna()
    dist = {
        "profit": float((labels == 1).mean()),
        "loss": float((labels == -1).mean()),
        "timeout": float((labels == 0).mean()),
    }

    n = len(df)
    usable_end = n - MAX_HOLDING - 5

    folds: list[dict[str, Any]] = []
    for k in range(N_FOLDS):
        test_end = usable_end - (N_FOLDS - 1 - k) * TEST_SIZE
        test_start = test_end - TEST_SIZE
        train_end = test_start - EMBARGO
        if train_end < MIN_TRAIN:
            continue

        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end].reset_index(drop=True)

        tr = train_df[train_df["tb_label"].isin([-1.0, 1.0])]
        X_tr = tr[fcols]
        y_tr = (tr["tb_label"] == 1.0).astype("int64")
        if len(X_tr) < 3000:
            continue

        # Class imbalance handling (asymmetric barriers skew classes)
        pos_frac = float(y_tr.mean())
        params = dict(LGBM_PARAMS)
        if pos_frac < 0.45:
            params["scale_pos_weight"] = (1 - pos_frac) / max(pos_frac, 1e-6)

        model = lgb.train(
            params, lgb.Dataset(X_tr, label=y_tr),
            num_boost_round=100,
            callbacks=[lgb.log_evaluation(period=0)],
        )

        proba = np.asarray(model.predict(test_df[fcols]), dtype="float64")
        tb_ret = np.asarray(test_df["tb_return"].to_numpy(), dtype="float64")
        tb_ex = np.asarray(test_df["tb_exit_offset"].to_numpy(), dtype="float64")
        close = np.asarray(test_df["close"].to_numpy(), dtype="float64")

        rets = _simulate(proba, tb_ret, tb_ex)
        bh = float(close[-1] / close[0] - 1.0)

        if len(rets) >= 3:
            arr = np.array(rets, dtype="float64")
            total = float(np.exp(np.sum(arr)) - 1.0)
            wr = float((arr > 0).mean())
            expectancy = float(np.mean(arr))
        else:
            total, wr, expectancy = 0.0, 0.0, 0.0

        folds.append({
            "fold": k + 1, "period": _period_str(test_df),
            "n_trades": len(rets), "total": total,
            "win_rate": wr, "expectancy": expectancy,
            "bh": bh, "beat_bh": total > bh,
        })

    if not folds:
        return None
    return {"symbol": symbol, "folds": folds, "label_dist": dist}


if __name__ == "__main__":
    import time

    STABLE_SYMBOLS = ["XRPUSDT", "SOLUSDT", "BTCUSDT", "MATICUSDT"]

    print("=" * 78)
    print(f"ASYMMETRIC WALK-FORWARD BACKTEST (v{ASYM_BACKTEST_VERSION})")
    print("=" * 78)
    print(f"\nBarriers: {PROFIT_MULT}xATR profit / {STOP_MULT}xATR stop "
          f"(2:1 R:R) | hold {MAX_HOLDING} bars")
    print(f"Breakeven WR: ~{STOP_MULT/(PROFIT_MULT+STOP_MULT)*100:.0f}% "
          f"(+costs ≈ 36%) | model's proven WR: 42-55%")
    print(f"Threshold {ENTRY_THRESHOLD} fixed | costs {ROUND_TRIP_COST*100:.2f}% | "
          f"long-only | ONE config, no sweeping\n")

    t0 = time.time()
    all_results = []

    for i, sym in enumerate(STABLE_SYMBOLS, 1):
        print(f"[{i}/{len(STABLE_SYMBOLS)}] {sym} ...", flush=True)
        r = asym_backtest_symbol(sym)
        if r is None:
            print("   skipped\n")
            continue
        all_results.append(r)
        d = r["label_dist"]
        print(f"   labels: profit {d['profit']*100:.0f}% / "
              f"loss {d['loss']*100:.0f}% / timeout {d['timeout']*100:.0f}%")
        for f in r["folds"]:
            marker = "✅" if f["total"] > 0 else "❌"
            print(f"   F{f['fold']} [{f['period']}] "
                  f"{f['n_trades']:3d} tr | net {f['total']*100:+7.2f}% | "
                  f"WR {f['win_rate']*100:4.1f}% | "
                  f"E[net/tr] {f['expectancy']*100:+.3f}% | "
                  f"B&H {f['bh']*100:+7.2f}% | {marker}")
        print()

    # Portfolio by fold
    print("=" * 78)
    print("PORTFOLIO BY FOLD (equal-weight)")
    print("=" * 78)
    pf: dict[int, list[float]] = {}
    bf: dict[int, list[float]] = {}
    for r in all_results:
        for f in r["folds"]:
            pf.setdefault(f["fold"], []).append(f["total"])
            bf.setdefault(f["fold"], []).append(f["bh"])

    print(f"\n{'Fold':>5s} {'SysNet':>9s} {'B&H':>9s} {'Alpha':>9s}")
    print("-" * 50)
    sys_folds, bh_folds = [], []
    for k in sorted(pf.keys()):
        s = float(np.mean(pf[k]))
        b = float(np.mean(bf[k]))
        sys_folds.append(s)
        bh_folds.append(b)
        print(f"{k:>5d} {s*100:>+8.2f}% {b*100:>+8.2f}% {(s-b)*100:>+8.2f}%")

    sys_comp = float(np.exp(np.sum([np.log(1+x) for x in sys_folds])) - 1)
    bh_comp = float(np.exp(np.sum([np.log(1+x) for x in bh_folds])) - 1)
    n_pos = sum(1 for x in sys_folds if x > 0)

    print("-" * 50)
    print(f"{'ALL':>5s} {sys_comp*100:>+8.2f}% {bh_comp*100:>+8.2f}% "
          f"{(sys_comp-bh_comp)*100:>+8.2f}%")

    print("\n" + "=" * 78)
    print("VERDICT (pre-registered — vs symmetric: sys -19.35%, B&H -2.99%)")
    print("=" * 78)
    print(f"\n  Positive folds: {n_pos}/{len(sys_folds)}")
    print(f"  System: {sys_comp*100:+.2f}% | B&H: {bh_comp*100:+.2f}%")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")

    if n_pos >= 3 and sys_comp > bh_comp:
        print("\n  🎯 GO — asymmetric structure fixed the math.")
        print("     Next: Phase 5 (risk mgmt, sizing, regime filter).")
    elif sys_comp > 0:
        print("\n  ⚠️  MARGINAL — positive but weak. Regime filter next.")
    else:
        print("\n  🔬 NO-GO — structure fix insufficient at 1h.")
        print("     Pivot: 4h timeframe (pre-registered path).")
    print("=" * 78)