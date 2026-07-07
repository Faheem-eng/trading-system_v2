"""
ml/models/multi_symbol_validation.py

THE GO/NO-GO TEST: does the 1h edge generalize across symbols?

BTC showed IC 0.03-0.045, val 53% (stable across 2 regularizations).
One symbol can be luck. 8+ of 12 symbols cannot.

Method:
    For each symbol: train binary 1h model (same config as BTC v2),
    measure IC + val accuracy + high-conf precision.

Verdict (pre-registered):
    8-12 symbols with IC>0.02 & val>51.5% → SYSTEMATIC EDGE (GO)
    5-7  symbols                          → PARTIAL (selective GO)
    0-4  symbols                          → BTC was luck (honest NO-GO)

Note: thresholds slightly relaxed vs single-symbol (IC 0.02 vs 0.03)
because we're testing CONSISTENCY across many symbols — a weaker but
universal signal beats a strong single-symbol fluke.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.logger import get_logger
from ml.models.binary_1h import prepare_1h, train_eval_1h

logger = get_logger(__name__)

MULTI_SYMBOL_VERSION = "1.0.0"

VALIDATION_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT",
    "AVAXUSDT", "LINKUSDT", "DOTUSDT", "MATICUSDT", "LTCUSDT", "ATOMUSDT",
]

# Pre-registered thresholds for "edge on this symbol"
IC_THRESHOLD = 0.02
VAL_ACC_THRESHOLD = 0.515


if __name__ == "__main__":
    import time

    print("=" * 78)
    print(f"MULTI-SYMBOL VALIDATION — GO/NO-GO Test (v{MULTI_SYMBOL_VERSION})")
    print("=" * 78)
    print(f"\nSymbols: {len(VALIDATION_SYMBOLS)}")
    print(f"Edge criteria per symbol: IC > {IC_THRESHOLD}, "
          f"val_acc > {VAL_ACC_THRESHOLD*100:.1f}%")
    print("\nGO: 8+, PARTIAL: 5-7, NO-GO: 0-4\n")

    results: list[dict[str, Any]] = []
    t_start = time.time()

    for i, symbol in enumerate(VALIDATION_SYMBOLS, 1):
        print(f"[{i:2d}/{len(VALIDATION_SYMBOLS)}] {symbol:12s} ...",
              end=" ", flush=True)
        t0 = time.time()
        try:
            data = prepare_1h(symbol)
            if data is None or len(data["X_train"]) < 3000:
                print("skipped (insufficient data)")
                continue
            r = train_eval_1h(data)
            results.append({
                "symbol": symbol,
                "val_acc": r["val_acc"],
                "ic": r["ic"],
                "gap": r["overfit_gap"],
                "best_iter": r["best_iter"],
                "prec_up": r["prec_up"],
                "prec_down": r["prec_down"],
                "n_hc": r["n_hc_up"] + r["n_hc_down"],
            })
            print(f"IC={r['ic']:+.4f} val={r['val_acc']*100:.1f}% "
                  f"({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"ERROR: {e}")

    total_time = time.time() - t_start

    # Results table
    print("\n" + "=" * 78)
    print("RESULTS (sorted by IC)")
    print("=" * 78)
    print(f"\n{'Symbol':12s} {'ValAcc':>7s} {'IC':>8s} {'Gap':>6s} "
          f"{'Iter':>5s} {'BestPrec':>9s} {'#HC':>5s} {'Edge?':>6s}")
    print("-" * 78)

    results_sorted = sorted(results, key=lambda x: x["ic"], reverse=True)
    n_edge = 0
    for r in results_sorted:
        has_edge = r["ic"] > IC_THRESHOLD and r["val_acc"] > VAL_ACC_THRESHOLD
        if has_edge:
            n_edge += 1
        best_prec = max(r["prec_up"], r["prec_down"])
        print(f"{r['symbol']:12s} {r['val_acc']*100:>6.1f}% {r['ic']:>+8.4f} "
              f"{r['gap']*100:>5.1f}% {r['best_iter']:>5d} "
              f"{best_prec*100:>8.1f}% {r['n_hc']:>5,} "
              f"{'✅' if has_edge else '—':>6s}")

    # Aggregate stats
    ics = [r["ic"] for r in results]
    vals = [r["val_acc"] for r in results]
    mean_ic = float(np.mean(ics)) if ics else 0.0
    mean_val = float(np.mean(vals)) if vals else 0.0

    print("-" * 78)
    print(f"{'MEAN':12s} {mean_val*100:>6.1f}% {mean_ic:>+8.4f}")

    # Verdict
    print("\n" + "=" * 78)
    print("VERDICT (pre-registered)")
    print("=" * 78)
    n_total = len(results)
    print(f"\n  Symbols with edge: {n_edge}/{n_total}")
    print(f"  Mean IC:           {mean_ic:+.4f}")
    print(f"  Mean val acc:      {mean_val*100:.1f}%")
    print(f"  Total time:        {total_time/60:.1f} min")

    if n_edge >= 8:
        print("\n  🎯 GO — SYSTEMATIC EDGE CONFIRMED across the market.")
        print("     Next: meta-labeling + Phase 4 backtest.")
    elif n_edge >= 5:
        print("\n  ⚠️  PARTIAL — edge exists on subset. Trade only those symbols.")
        print("     Next: meta-labeling on edge symbols.")
    else:
        print("\n  🔬 NO-GO — BTC result likely luck. Honest pivot needed.")
        print("     Next: mean-reversion strategy (matches Hurst 0.4).")
    print("=" * 78)