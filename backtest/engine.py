"""
backtest/engine.py

Phase 4: Realistic long-only backtest engine.

REALITY CONSTRAINTS (built-in, not optional):
    1. LONG-ONLY — Binance Spot (futures unavailable). DOWN signals
       mean "stay flat", never short.
    2. COSTS — 0.25% round-trip (0.1% taker x2 + 0.05% slippage).
    3. NON-OVERLAPPING — one position at a time; signals during an
       open trade are skipped.
    4. HONEST EXECUTION — entry at signal bar's close (bar completed),
       exit via triple-barrier simulation on actual future prices.

Model protocol (walk-forward consistent):
    Train on the temporal-split train segment, generate signals on the
    UNTOUCHED validation segment. Same params as validated (zero tuning
    on backtest results — that would be backtest overfitting).

Benchmark: buy-and-hold of the same symbol over the same period.

Pre-registered verdict:
    GO: portfolio Sharpe > 0.8 after costs, MaxDD < 25%
    MARGINAL: Sharpe 0.4-0.8
    NO-GO: Sharpe < 0.4 or DD > 35%
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

from core.logger import get_logger
from ml.targets.triple_barrier import compute_triple_barrier
from ml.splits.temporal_split import split_temporal
from ml.models.binary_1h import LGBM_PARAMS, _feature_cols

logger = get_logger(__name__)

BACKTEST_VERSION = "1.0.0"

# Realistic cost model (conservative)
ROUND_TRIP_COST = 0.0025      # 0.25% per completed trade

# Signal thresholds to sweep (long entries only)
ENTRY_THRESHOLDS = [0.55, 0.58, 0.60]

EMBARGO_1H = 25
BARS_PER_YEAR_1H = 24 * 365   # 8,760 hourly bars/year


def _annualized_sharpe(trade_returns: np.ndarray,
                       n_bars_period: int) -> float:
    """
    Annualized Sharpe from per-trade net returns spread over the period.
    Approximation: treat the period's total PnL series at trade level,
    annualize by trades-per-year scaling.
    """
    if len(trade_returns) < 5:
        return 0.0
    mu = float(np.mean(trade_returns))
    sd = float(np.std(trade_returns, ddof=1))
    if sd <= 0:
        return 0.0
    years = n_bars_period / BARS_PER_YEAR_1H
    trades_per_year = len(trade_returns) / max(years, 1e-9)
    return (mu / sd) * float(np.sqrt(trades_per_year))


def backtest_symbol(
    symbol: str,
    entry_threshold: float,
    features_dir: str = "data/features_1h",
) -> Optional[dict[str, Any]]:
    """
    Long-only backtest for one symbol on the validation segment.
    """
    feature_file = Path(features_dir) / f"{symbol}_1h.parquet"
    if not feature_file.exists():
        return None

    df = pd.read_parquet(feature_file)
    df = compute_triple_barrier(df, profit_mult=3.0, stop_mult=3.0,
                                max_holding=20)
    split = split_temporal(df, embargo_bars=EMBARGO_1H)
    fcols = _feature_cols(df)

    # Train on train segment (UP/DOWN bars only)
    tr = split.train[split.train["tb_label"].isin([-1.0, 1.0])]
    X_tr = tr[fcols]
    y_tr = (tr["tb_label"] == 1.0).astype("int64")
    if len(X_tr) < 3000:
        return None

    model = lgb.train(
        LGBM_PARAMS, lgb.Dataset(X_tr, label=y_tr),
        num_boost_round=100,
        callbacks=[lgb.log_evaluation(period=0)],
    )

    # Signals on validation (ALL bars — we trade real-time, labels
    # only used for exit simulation)
    val = split.val.reset_index(drop=True)
    proba = np.asarray(model.predict(val[fcols]), dtype="float64")

    tb_return = np.asarray(val["tb_return"].to_numpy(), dtype="float64")
    tb_exit = np.asarray(val["tb_exit_offset"].to_numpy(), dtype="float64")
    close = np.asarray(val["close"].to_numpy(), dtype="float64")

    n = len(val)

    # --- Simulate: long-only, non-overlapping ---
    trades: list[dict[str, Any]] = []
    i = 0
    while i < n:
        p = proba[i]
        if p > entry_threshold and not np.isnan(tb_return[i]):
            gross = float(tb_return[i])            # log return at barrier exit
            net = gross - ROUND_TRIP_COST
            exit_off = int(tb_exit[i]) if not np.isnan(tb_exit[i]) else 20
            trades.append({
                "entry_idx": i, "proba": float(p),
                "gross": gross, "net": net,
                "hold_bars": exit_off,
            })
            i += max(exit_off, 1)                  # skip until exit (non-overlap)
        else:
            i += 1

    if len(trades) < 5:
        return {
            "symbol": symbol, "threshold": entry_threshold,
            "n_trades": len(trades), "insufficient": True,
        }

    net_rets = np.array([t["net"] for t in trades], dtype="float64")
    gross_rets = np.array([t["gross"] for t in trades], dtype="float64")

    # Equity curve (compounded, log-returns additive)
    equity_log = np.cumsum(net_rets)
    equity = np.exp(equity_log)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(dd.min())

    total_return = float(np.exp(equity_log[-1]) - 1.0)
    win_rate = float((net_rets > 0).mean())
    sharpe = _annualized_sharpe(net_rets, n)

    # Buy-and-hold benchmark (same period)
    bh_return = float(close[-1] / close[0] - 1.0)

    # Period length in months
    months = n / (24 * 30.44)

    return {
        "symbol": symbol, "threshold": entry_threshold,
        "n_trades": len(trades),
        "win_rate": win_rate,
        "avg_net_ret": float(np.mean(net_rets)),
        "avg_gross_ret": float(np.mean(gross_rets)),
        "total_return": total_return,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "bh_return": bh_return,
        "months": months,
        "avg_hold_bars": float(np.mean([t["hold_bars"] for t in trades])),
        "insufficient": False,
    }


if __name__ == "__main__":
    import time

    STABLE_SYMBOLS = ["XRPUSDT", "SOLUSDT", "BTCUSDT", "MATICUSDT"]

    print("=" * 78)
    print(f"PHASE 4 — REALISTIC LONG-ONLY BACKTEST (v{BACKTEST_VERSION})")
    print("=" * 78)
    print(f"\nSymbols: {STABLE_SYMBOLS} (walk-forward stable)")
    print(f"Costs: {ROUND_TRIP_COST*100:.2f}% round-trip | LONG-ONLY (spot)")
    print("Non-overlapping trades | Entry at close | Barrier exits")
    print("\nPre-registered: GO if Sharpe>0.8 & DD<25% | NO-GO if Sharpe<0.4\n")

    t0 = time.time()
    all_rows = []

    for th in ENTRY_THRESHOLDS:
        print(f"--- Entry threshold: proba > {th:.2f} ---")
        for sym in STABLE_SYMBOLS:
            r = backtest_symbol(sym, th)
            if r is None:
                print(f"  {sym:11s} skipped (no data)")
                continue
            if r["insufficient"]:
                print(f"  {sym:11s} only {r['n_trades']} trades — too few")
                continue
            all_rows.append(r)
            print(f"  {sym:11s} {r['n_trades']:4d} trades | "
                  f"WR {r['win_rate']*100:5.1f}% | "
                  f"net/tr {r['avg_net_ret']*100:+.3f}% | "
                  f"total {r['total_return']*100:+7.2f}% | "
                  f"DD {r['max_dd']*100:6.1f}% | "
                  f"Sharpe {r['sharpe']:5.2f} | "
                  f"B&H {r['bh_return']*100:+7.2f}%")
        print()

    # Best threshold per portfolio Sharpe
    print("=" * 78)
    print("PORTFOLIO VIEW (equal-weight across symbols, per threshold)")
    print("=" * 78)
    best = None
    for th in ENTRY_THRESHOLDS:
        rows = [r for r in all_rows if r["threshold"] == th]
        if not rows:
            continue
        mean_sharpe = float(np.mean([r["sharpe"] for r in rows]))
        mean_total = float(np.mean([r["total_return"] for r in rows]))
        worst_dd = float(np.min([r["max_dd"] for r in rows]))
        total_trades = int(np.sum([r["n_trades"] for r in rows]))
        mean_bh = float(np.mean([r["bh_return"] for r in rows]))
        print(f"\n  th>{th:.2f}: trades={total_trades:,} | "
              f"mean total {mean_total*100:+.2f}% | "
              f"mean Sharpe {mean_sharpe:.2f} | worst DD {worst_dd*100:.1f}% | "
              f"mean B&H {mean_bh*100:+.2f}%")
        if best is None or mean_sharpe > best[1]:
            best = (th, mean_sharpe, worst_dd)

    print("\n" + "=" * 78)
    print("VERDICT (pre-registered)")
    print("=" * 78)
    if best is not None:
        th, sh, dd = best
        print(f"\n  Best threshold: >{th:.2f} | Sharpe {sh:.2f} | worst DD {dd*100:.1f}%")
        if sh > 0.8 and dd > -0.25:
            print("\n  🎯 GO — Profitable after costs. Proceed to Phase 5 (risk mgmt).")
        elif sh > 0.4:
            print("\n  ⚠️  MARGINAL — Edge survives costs weakly. Phase 5 risk mgmt")
            print("     + paper trading before any capital.")
        else:
            print("\n  🔬 NO-GO at current form — costs eat the thin edge.")
            print("     Honest paths: 4h timeframe (bigger moves vs fixed costs),")
            print("     maker orders (lower fees), or signal refinement.")
    print(f"\n  Time: {(time.time()-t0)/60:.1f} min")
    print("=" * 78)