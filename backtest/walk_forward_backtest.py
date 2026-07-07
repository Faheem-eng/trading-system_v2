"""
backtest/walk_forward_backtest.py

Walk-forward LONG-ONLY backtest across 5 rolling periods (all regimes).

The single-period backtest (May-Nov 2025 = bear) showed system flat
(+0.6%) vs market -19% — alpha exists but absolute return was weak.
One bear window cannot judge a long-only system. This tests ALL regimes.

Protocol per fold:
    Train [0..cut] → embargo 25 bars → simulate trades on next 2,700 bars.
    Entry threshold FIXED at 0.58 (chosen before this test; tuning on
    walk-forward results would be backtest overfitting).
    Long-only, non-overlapping, 0.25% round-trip costs, barrier exits.

Pre-registered verdict:
    GO:     3+/5 folds net-positive AND aggregate beats buy-and-hold
    REGIME: profitable in non-bear folds only → add regime filter (Phase 5/6)
    NO-GO:  negative across regimes → pivot to 4h timeframe
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

WF_BACKTEST_VERSION = "1.0.0"

ENTRY_THRESHOLD = 0.58        # fixed pre-test (no tuning on results)
ROUND_TRIP_COST = 0.0025
N_FOLDS = 5
EMBARGO = 25
MIN_TRAIN = 8000
TEST_SIZE = 2700
BARS_PER_YEAR_1H = 24 * 365


def _period_str(df: pd.DataFrame) -> str:
    t0 = datetime.fromtimestamp(
        int(df["open_time"].min()) / 1000, tz=timezone.utc).strftime("%Y-%m")
    t1 = datetime.fromtimestamp(
        int(df["open_time"].max()) / 1000, tz=timezone.utc).strftime("%Y-%m")
    return f"{t0}→{t1}"


def _simulate_long_only(
    proba: np.ndarray, tb_return: np.ndarray, tb_exit: np.ndarray,
) -> list[float]:
    """Non-overlapping long-only trade simulation. Returns net returns."""
    n = len(proba)
    net_rets: list[float] = []
    i = 0
    while i < n:
        if proba[i] > ENTRY_THRESHOLD and not np.isnan(tb_return[i]):
            net_rets.append(float(tb_return[i]) - ROUND_TRIP_COST)
            exit_off = int(tb_exit[i]) if not np.isnan(tb_exit[i]) else 20
            i += max(exit_off, 1)
        else:
            i += 1
    return net_rets


def wf_backtest_symbol(
    symbol: str,
    features_dir: str = "data/features_1h",
) -> Optional[dict[str, Any]]:
    """Walk-forward backtest for one symbol. Returns per-fold results."""
    feature_file = Path(features_dir) / f"{symbol}_1h.parquet"
    if not feature_file.exists():
        return None

    df = pd.read_parquet(feature_file)
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

        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end].reset_index(drop=True)

        tr = train_df[train_df["tb_label"].isin([-1.0, 1.0])]
        X_tr = tr[fcols]
        y_tr = (tr["tb_label"] == 1.0).astype("int64")
        if len(X_tr) < 3000:
            continue

        model = lgb.train(
            LGBM_PARAMS, lgb.Dataset(X_tr, label=y_tr),
            num_boost_round=100,
            callbacks=[lgb.log_evaluation(period=0)],
        )

        proba = np.asarray(model.predict(test_df[fcols]), dtype="float64")
        tb_ret = np.asarray(test_df["tb_return"].to_numpy(), dtype="float64")
        tb_ex = np.asarray(test_df["tb_exit_offset"].to_numpy(), dtype="float64")
        close = np.asarray(test_df["close"].to_numpy(), dtype="float64")

        net_rets = _simulate_long_only(proba, tb_ret, tb_ex)

        bh = float(close[-1] / close[0] - 1.0)

        if len(net_rets) >= 3:
            arr = np.array(net_rets, dtype="float64")
            total = float(np.exp(np.sum(arr)) - 1.0)
            wr = float((arr > 0).mean())
            # Fold-level annualized Sharpe (approx)
            years = TEST_SIZE / BARS_PER_YEAR_1H
            tpy = len(arr) / max(years, 1e-9)
            sd = float(np.std(arr, ddof=1))
            sharpe = (float(np.mean(arr)) / sd) * float(np.sqrt(tpy)) if sd > 0 else 0.0
        else:
            total, wr, sharpe = 0.0, 0.0, 0.0

        folds.append({
            "fold": k + 1,
            "period": _period_str(test_df),
            "n_trades": len(net_rets),
            "total": total,
            "win_rate": wr,
            "sharpe": sharpe,
            "bh": bh,
            "beat_bh": total > bh,
        })

    if not folds:
        return None
    return {"symbol": symbol, "folds": folds}


if __name__ == "__main__":
    import time

    STABLE_SYMBOLS = ["XRPUSDT", "SOLUSDT", "BTCUSDT", "MATICUSDT"]

    print("=" * 78)
    print(f"WALK-FORWARD LONG-ONLY BACKTEST (v{WF_BACKTEST_VERSION})")
    print("=" * 78)
    print(f"\nThreshold FIXED: >{ENTRY_THRESHOLD} | Costs: {ROUND_TRIP_COST*100:.2f}% RT")
    print(f"{N_FOLDS} folds × {TEST_SIZE} bars | long-only | non-overlapping\n")

    t0 = time.time()
    all_results = []

    for i, sym in enumerate(STABLE_SYMBOLS, 1):
        print(f"[{i}/{len(STABLE_SYMBOLS)}] {sym} ...", flush=True)
        r = wf_backtest_symbol(sym)
        if r is None:
            print("   skipped\n")
            continue
        all_results.append(r)
        for f in r["folds"]:
            marker = "✅" if f["total"] > 0 else "❌"
            beat = "beat B&H" if f["beat_bh"] else "lost to B&H"
            print(f"   F{f['fold']} [{f['period']}] "
                  f"{f['n_trades']:3d} trades | net {f['total']*100:+7.2f}% | "
                  f"WR {f['win_rate']*100:4.1f}% | B&H {f['bh']*100:+7.2f}% | "
                  f"{marker} ({beat})")
        print()

    # Aggregate per symbol
    print("=" * 78)
    print("PER-SYMBOL AGGREGATE (across all folds)")
    print("=" * 78)
    print(f"\n{'Symbol':11s} {'Folds+':>7s} {'TotalNet':>10s} {'TotalB&H':>10s} "
          f"{'BeatB&H':>8s}")
    print("-" * 78)

    portfolio_by_fold: dict[int, list[float]] = {}
    bh_by_fold: dict[int, list[float]] = {}

    for r in all_results:
        pos_folds = sum(1 for f in r["folds"] if f["total"] > 0)
        total_net = float(np.exp(np.sum([np.log(1 + f["total"]) for f in r["folds"]])) - 1)
        total_bh = float(np.exp(np.sum([np.log(1 + f["bh"]) for f in r["folds"]])) - 1)
        beats = sum(1 for f in r["folds"] if f["beat_bh"])
        print(f"{r['symbol']:11s} {pos_folds}/{len(r['folds']):>5d} "
              f"{total_net*100:>+9.2f}% {total_bh*100:>+9.2f}% "
              f"{beats}/{len(r['folds']):>6d}")
        for f in r["folds"]:
            portfolio_by_fold.setdefault(f["fold"], []).append(f["total"])
            bh_by_fold.setdefault(f["fold"], []).append(f["bh"])

    # Portfolio per fold (equal weight)
    print("\n" + "=" * 78)
    print("PORTFOLIO BY FOLD (equal-weight across symbols)")
    print("=" * 78)
    print(f"\n{'Fold':>5s} {'SysNet':>9s} {'B&H':>9s} {'Alpha':>9s}")
    print("-" * 50)
    fold_sys_totals = []
    fold_bh_totals = []
    for k in sorted(portfolio_by_fold.keys()):
        sys_mean = float(np.mean(portfolio_by_fold[k]))
        bh_mean = float(np.mean(bh_by_fold[k]))
        fold_sys_totals.append(sys_mean)
        fold_bh_totals.append(bh_mean)
        print(f"{k:>5d} {sys_mean*100:>+8.2f}% {bh_mean*100:>+8.2f}% "
              f"{(sys_mean-bh_mean)*100:>+8.2f}%")

    n_pos_folds = sum(1 for x in fold_sys_totals if x > 0)
    sys_compound = float(np.exp(np.sum([np.log(1 + x) for x in fold_sys_totals])) - 1)
    bh_compound = float(np.exp(np.sum([np.log(1 + x) for x in fold_bh_totals])) - 1)

    print("-" * 50)
    print(f"{'ALL':>5s} {sys_compound*100:>+8.2f}% {bh_compound*100:>+8.2f}% "
          f"{(sys_compound-bh_compound)*100:>+8.2f}%")

    print("\n" + "=" * 78)
    print("VERDICT (pre-registered)")
    print("=" * 78)
    print(f"\n  Profitable folds (portfolio): {n_pos_folds}/{len(fold_sys_totals)}")
    print(f"  System compound: {sys_compound*100:+.2f}% | "
          f"B&H compound: {bh_compound*100:+.2f}%")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")

    if n_pos_folds >= 3 and sys_compound > bh_compound:
        print("\n  🎯 GO — profitable across regimes AND beats buy-hold.")
        print("     Next: Phase 5 risk management + position sizing.")
    elif n_pos_folds >= 2 and sys_compound > bh_compound:
        print("\n  ⚠️  REGIME-DEPENDENT — alpha real but absolute returns")
        print("     concentrated in specific regimes. Next: regime filter")
        print("     (trade only when vol_regime/trend favorable) in Phase 5/6.")
    else:
        print("\n  🔬 NO-GO on 1h long-only — pivot to 4h (bigger moves vs")
        print("     fixed costs) or maker-fee execution.")
    print("=" * 78)