"""
ml/validation/walk_forward.py

Purged walk-forward validation — is the 1h edge TIME-STABLE?

The single-split test showed IC 0.03-0.05 on 6/12 symbols, but that
used ONE validation period (May-Nov 2025). This tests the edge across
MULTIPLE rolling periods spanning different market regimes.

Method (expanding-window walk-forward, López de Prado Ch. 11-12):
    Fold k: train on [start .. cut_k], embargo 25 bars,
            test on next ~2,700 bars (~3.7 months of 1h).
    Same hyperparameters every fold (NO per-fold tuning — that would
    be validation overfitting).

Pre-registered verdict (per symbol):
    4-5/5 folds IC>0 → TIME-STABLE edge
    3/5             → borderline
    0-2/5           → single-period luck (honest)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

from core.logger import get_logger
from ml.targets.triple_barrier import compute_triple_barrier
from ml.models.binary_1h import LGBM_PARAMS, _feature_cols

logger = get_logger(__name__)

WALK_FORWARD_VERSION = "1.0.0"

N_FOLDS = 5
EMBARGO = 25            # > 20-bar target holding
MIN_TRAIN = 8000        # minimum bars to train a fold
TEST_SIZE = 2700        # ~3.7 months of 1h bars per fold


def walk_forward_symbol(
    symbol: str,
    features_dir: str = "data/features_1h",
) -> Optional[dict[str, Any]]:
    """Run walk-forward validation for one symbol. Returns fold metrics."""
    feature_file = Path(features_dir) / f"{symbol}_1h.parquet"
    if not feature_file.exists():
        return None

    df = pd.read_parquet(feature_file)
    df = compute_triple_barrier(df, profit_mult=3.0, stop_mult=3.0,
                                max_holding=20)
    fcols = _feature_cols(df)

    # Usable region: labeled bars only (last 20 bars have NaN labels)
    n = len(df)
    usable_end = n - 25  # margin for label window

    # Fold boundaries: last N_FOLDS test windows, expanding train
    fold_results: list[dict[str, Any]] = []

    for k in range(N_FOLDS):
        # Test window k (from oldest fold to newest)
        test_end = usable_end - (N_FOLDS - 1 - k) * TEST_SIZE
        test_start = test_end - TEST_SIZE
        train_end = test_start - EMBARGO

        if train_end < MIN_TRAIN:
            continue  # not enough history for this fold

        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]

        def _prep(sdf: pd.DataFrame):
            mask = sdf["tb_label"].isin([-1.0, 1.0])
            sub = sdf[mask]
            X = sub[fcols]
            y = (sub["tb_label"] == 1.0).astype("int64")
            ret = np.asarray(sub["tb_return"].to_numpy(), dtype="float64")
            return X, y, ret

        X_tr, y_tr, _ = _prep(train_df)
        X_te, y_te, te_ret = _prep(test_df)

        if len(X_tr) < 3000 or len(X_te) < 500:
            continue

        # Train (SAME params every fold — no tuning)
        train_set = lgb.Dataset(X_tr, label=y_tr)
        model = lgb.train(
            LGBM_PARAMS, train_set, num_boost_round=100,
            callbacks=[lgb.log_evaluation(period=0)],
        )

        te_proba = np.asarray(model.predict(X_te), dtype="float64")
        te_pred = (te_proba > 0.5).astype("int64")
        y_te_arr = y_te.to_numpy()

        acc = float((te_pred == y_te_arr).mean())
        valid = ~np.isnan(te_ret)
        if int(valid.sum()) > 10:
            corr = spearmanr(te_proba[valid], te_ret[valid])
            ic = float(corr.statistic)  # type: ignore[attr-defined]
            ic = ic if not np.isnan(ic) else 0.0
        else:
            ic = 0.0

        # Time range for reporting
        from datetime import datetime, timezone
        t_start = datetime.fromtimestamp(
            int(test_df["open_time"].min()) / 1000, tz=timezone.utc
        ).strftime("%Y-%m")
        t_end = datetime.fromtimestamp(
            int(test_df["open_time"].max()) / 1000, tz=timezone.utc
        ).strftime("%Y-%m")

        fold_results.append({
            "fold": k + 1, "period": f"{t_start}→{t_end}",
            "train_n": len(X_tr), "test_n": len(X_te),
            "acc": acc, "ic": ic,
        })

    if not fold_results:
        return None

    ics = [f["ic"] for f in fold_results]
    return {
        "symbol": symbol,
        "folds": fold_results,
        "n_positive_ic": sum(1 for x in ics if x > 0),
        "n_folds": len(fold_results),
        "mean_ic": float(np.mean(ics)),
        "min_ic": float(np.min(ics)),
    }


if __name__ == "__main__":
    import time

    EDGE_SYMBOLS = ["XRPUSDT", "ADAUSDT", "AVAXUSDT",
                    "SOLUSDT", "BTCUSDT", "MATICUSDT"]

    print("=" * 78)
    print(f"PURGED WALK-FORWARD VALIDATION (v{WALK_FORWARD_VERSION})")
    print("=" * 78)
    print(f"\n{N_FOLDS} folds × {TEST_SIZE} bars (~3.7 months each), "
          f"embargo {EMBARGO} bars")
    print("Same params every fold (zero per-fold tuning)\n")

    all_results = []
    t0 = time.time()

    for i, sym in enumerate(EDGE_SYMBOLS, 1):
        print(f"[{i}/{len(EDGE_SYMBOLS)}] {sym:12s} ...", end=" ", flush=True)
        r = walk_forward_symbol(sym)
        if r is None:
            print("skipped")
            continue
        all_results.append(r)
        print(f"IC>0 in {r['n_positive_ic']}/{r['n_folds']} folds "
              f"(mean IC {r['mean_ic']:+.4f})")

    # Detailed table
    print("\n" + "=" * 78)
    print("PER-FOLD DETAIL")
    print("=" * 78)
    for r in all_results:
        print(f"\n{r['symbol']}:")
        for f in r["folds"]:
            marker = "✅" if f["ic"] > 0 else "❌"
            print(f"  Fold {f['fold']} [{f['period']}]  "
                  f"acc={f['acc']*100:5.1f}%  IC={f['ic']:+.4f}  {marker}")

    # Summary
    print("\n" + "=" * 78)
    print("SUMMARY (pre-registered: 4-5/5 stable, 3/5 borderline, 0-2/5 luck)")
    print("=" * 78)
    print(f"\n{'Symbol':12s} {'IC>0':>7s} {'MeanIC':>9s} {'MinIC':>9s} {'Verdict':>12s}")
    print("-" * 78)

    n_stable = 0
    for r in all_results:
        frac = f"{r['n_positive_ic']}/{r['n_folds']}"
        if r["n_positive_ic"] >= 4:
            verdict = "STABLE ✅"
            n_stable += 1
        elif r["n_positive_ic"] == 3:
            verdict = "borderline"
        else:
            verdict = "unstable ❌"
        print(f"{r['symbol']:12s} {frac:>7s} {r['mean_ic']:>+9.4f} "
              f"{r['min_ic']:>+9.4f} {verdict:>12s}")

    print(f"\nTime: {(time.time()-t0)/60:.1f} min")

    print("\n" + "=" * 78)
    print("FINAL VERDICT")
    print("=" * 78)
    print(f"\n  Time-stable symbols: {n_stable}/{len(all_results)}")
    if n_stable >= 4:
        print("\n  🎯 EDGE IS TIME-STABLE — proceed to Phase 4 backtest")
        print("     (realistic costs + confidence-based position sizing)")
    elif n_stable >= 2:
        print("\n  ⚠️  PARTIAL STABILITY — trade only stable symbols in Phase 4")
    else:
        print("\n  🔬 UNSTABLE — earlier result was period-specific. Honest")
        print("     pivot: 4h timeframe or mean-reversion approach.")
    print("=" * 78)