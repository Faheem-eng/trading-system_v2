"""
ml/models/meta_labeling.py

Meta-labeling (López de Prado, "Advances in Financial ML", Ch. 3.6).

Two-model architecture:
    PRIMARY model: predicts direction (UP vs DOWN) — already validated.
    META model:    predicts whether the primary's prediction is CORRECT.

Trade only when meta-confidence is high → fewer trades, higher win rate.
Multi-symbol validation already showed high-conf bars hit 58-78% precision.
This makes that systematic.

CRITICAL ANTI-LEAKAGE DESIGN:
    The meta model must NOT be trained on primary predictions made on
    the primary's own training data (primary is overconfident there).
    Solution: chronological sub-split of train —
        Primary trains on first 70% of train.
        Meta trains on last 30% of train (out-of-sample for primary).
    Validation remains untouched until final evaluation.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd  # noqa: F401
import lightgbm as lgb

from core.logger import get_logger
from ml.models.binary_1h import prepare_1h, LGBM_PARAMS

logger = get_logger(__name__)

META_LABELING_VERSION = "1.0.0"

# Meta model: small and simple (its job is easier than primary's)
META_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "max_depth": 3,
    "num_leaves": 7,
    "min_child_samples": 100,
    "learning_rate": 0.03,
    "reg_alpha": 0.5,
    "reg_lambda": 0.5,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

PRIMARY_FRACTION = 0.70  # of train → primary; rest → meta
META_THRESHOLDS = [0.55, 0.60, 0.65]


def run_meta_labeling(symbol: str) -> Optional[dict[str, Any]]:
    """
    Full meta-labeling pipeline for one symbol.

    Returns evaluation dict or None if data insufficient.
    """
    data = prepare_1h(symbol)
    if data is None or len(data["X_train"]) < 5000:
        return None

    X_train_full = data["X_train"].reset_index(drop=True)
    y_train_full = data["y_train"].reset_index(drop=True)
    X_val = data["X_val"].reset_index(drop=True)
    y_val = data["y_val"].reset_index(drop=True).to_numpy()
    val_ret = np.asarray(
        data["val_returns"].reset_index(drop=True).to_numpy(),
        dtype="float64",
    )

    # ============================================================
    # Chronological sub-split (anti-leakage for meta)
    # ============================================================
    n = len(X_train_full)
    cut = int(n * PRIMARY_FRACTION)
    X_prim, y_prim = X_train_full.iloc[:cut], y_train_full.iloc[:cut]
    X_meta, y_meta_dir = X_train_full.iloc[cut:], y_train_full.iloc[cut:]

    # ============================================================
    # Train PRIMARY on first 70%
    # ============================================================
    prim_set = lgb.Dataset(X_prim, label=y_prim)
    primary = lgb.train(
        LGBM_PARAMS, prim_set, num_boost_round=150,
        callbacks=[lgb.log_evaluation(period=0)],
    )

    # Primary predictions on meta segment (OUT-OF-SAMPLE for primary)
    meta_proba = np.asarray(primary.predict(X_meta), dtype="float64")
    meta_pred_dir = (meta_proba > 0.5).astype("int64")

    # ============================================================
    # META LABELS: was primary correct on the meta segment?
    # ============================================================
    y_meta = (meta_pred_dir == y_meta_dir.to_numpy()).astype("int64")

    # Meta features = original features + primary confidence
    X_meta_aug = X_meta.copy()
    X_meta_aug["primary_proba"] = meta_proba
    X_meta_aug["primary_conf"] = np.abs(meta_proba - 0.5) * 2.0

    meta_set = lgb.Dataset(X_meta_aug, label=y_meta)
    meta_model = lgb.train(
        META_PARAMS, meta_set, num_boost_round=100,
        callbacks=[lgb.log_evaluation(period=0)],
    )

    # ============================================================
    # VALIDATION: primary + meta together (untouched data)
    # ============================================================
    val_proba = np.asarray(primary.predict(X_val), dtype="float64")
    val_pred_dir = (val_proba > 0.5).astype("int64")

    X_val_aug = X_val.copy()
    X_val_aug["primary_proba"] = val_proba
    X_val_aug["primary_conf"] = np.abs(val_proba - 0.5) * 2.0
    meta_conf = np.asarray(meta_model.predict(X_val_aug), dtype="float64")

    # Baseline (no filter): all primary predictions
    base_correct = (val_pred_dir == y_val)
    base_wr = float(base_correct.mean())

    # Directional return of following primary's signal
    # (long if UP predicted → +ret; short if DOWN → -ret)
    signed_ret = np.where(val_pred_dir == 1, val_ret, -val_ret)

    thresholds_report = []
    for th in META_THRESHOLDS:
        mask = meta_conf > th
        n_trades = int(mask.sum())
        if n_trades >= 10:
            wr = float(base_correct[mask].mean())
            avg_ret = float(np.nanmean(signed_ret[mask]))
        else:
            wr, avg_ret = 0.0, 0.0
        thresholds_report.append({
            "threshold": th, "n_trades": n_trades,
            "win_rate": wr, "avg_return": avg_ret,
        })

    return {
        "symbol": symbol,
        "n_val": len(y_val),
        "base_win_rate": base_wr,
        "base_avg_return": float(np.nanmean(signed_ret)),
        "thresholds": thresholds_report,
    }


if __name__ == "__main__":
    import time

    EDGE_SYMBOLS = ["XRPUSDT", "ADAUSDT", "AVAXUSDT",
                    "SOLUSDT", "BTCUSDT", "MATICUSDT"]

    print("=" * 78)
    print(f"META-LABELING — Edge Extraction (v{META_LABELING_VERSION})")
    print("=" * 78)
    print(f"\nSymbols: {len(EDGE_SYMBOLS)} (validated edge symbols)")
    print("Anti-leakage: primary on 70% train, meta on last 30% (chrono)")
    print("Round-trip cost reference: ~0.20-0.25% (spot taker fees+slippage)\n")

    all_results = []
    t0 = time.time()

    for i, sym in enumerate(EDGE_SYMBOLS, 1):
        print(f"[{i}/{len(EDGE_SYMBOLS)}] {sym:12s} ...", end=" ", flush=True)
        r = run_meta_labeling(sym)
        if r is None:
            print("skipped")
            continue
        all_results.append(r)
        print(f"base WR={r['base_win_rate']*100:.1f}% "
              f"({r['n_val']:,} val bars)")

    # Report
    print("\n" + "=" * 78)
    print("META-LABELING RESULTS (validation set — win rate & avg return/trade)")
    print("=" * 78)
    print(f"\n{'Symbol':11s} {'Base WR':>8s} | " +
          " | ".join([f"th>{t:.2f}: WR / #tr / ret" for t in META_THRESHOLDS]))
    print("-" * 78)

    for r in all_results:
        row = f"{r['symbol']:11s} {r['base_win_rate']*100:>7.1f}% |"
        for t in r["thresholds"]:
            if t["n_trades"] >= 10:
                row += (f" {t['win_rate']*100:5.1f}%/{t['n_trades']:4d}/"
                        f"{t['avg_return']*100:+.3f}% |")
            else:
                row += f"   —  /{t['n_trades']:4d}/   —    |"
        print(row)

    # Aggregate: pooled trades at each threshold
    print("-" * 78)
    print("\nPOOLED (all symbols combined):")
    for i, th in enumerate(META_THRESHOLDS):
        total_tr = sum(r["thresholds"][i]["n_trades"] for r in all_results)
        wrs, rets = [], []
        for r in all_results:
            t = r["thresholds"][i]
            if t["n_trades"] >= 10:
                wrs.append((t["win_rate"], t["n_trades"]))
                rets.append((t["avg_return"], t["n_trades"]))
        if wrs:
            pooled_wr = sum(w * n for w, n in wrs) / sum(n for _, n in wrs)
            pooled_ret = sum(a * n for a, n in rets) / sum(n for _, n in rets)
        else:
            pooled_wr, pooled_ret = 0.0, 0.0
        print(f"  meta>{th:.2f}: {total_tr:5,} trades | "
              f"WR {pooled_wr*100:.1f}% | avg ret/trade {pooled_ret*100:+.3f}%")

    print(f"\nTime: {(time.time()-t0)/60:.1f} min")

    # Verdict
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("""
  Interpretation guide:
    WR > 55% + avg ret/trade > +0.30% → clears costs (~0.25%) = TRADEABLE
    WR 52-55% or ret 0.10-0.30%       → marginal, needs Phase 4 cost test
    WR ~50% or ret < 0.10%            → meta not adding value here
""")
    print("=" * 78)