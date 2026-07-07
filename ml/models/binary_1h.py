"""
ml/models/binary_1h.py

Binary LightGBM on 1h features — v2 with STRONGER regularization.

v1 found edge (IC 0.0446, val 53.2%) but overfit gap was 11.9%.
v2 reduces overfit to confirm the edge is REAL (survives regularization),
not a memorization artifact.

If val_acc stays >=53% and IC stays >=0.03 with gap <7% → EDGE CONFIRMED.
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
from ml.splits.temporal_split import split_temporal

logger = get_logger(__name__)

BINARY_1H_VERSION = "2.0.0"

# STRONGER regularization (v2 — reduce 11.9% overfit gap)
LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "max_depth": 4,
    "num_leaves": 8,             # was 15 → simpler
    "min_child_samples": 150,    # was 100 → more per leaf
    "learning_rate": 0.02,       # was 0.03 → slower
    "reg_alpha": 0.6,            # was 0.3 → stronger L1
    "reg_lambda": 0.6,           # was 0.3 → stronger L2
    "feature_fraction": 0.6,     # was 0.75 → more bagging
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

N_ESTIMATORS = 500
EARLY_STOPPING = 50

# 1h embargo: target holding is 20 bars, use 25-bar embargo (>20)
EMBARGO_1H = 25


def _feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {
        "open_time", "close_time", "open", "high", "low", "close",
        "volume", "quote_volume", "count", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore",
        "tb_label", "tb_exit_offset", "tb_return",
    }
    return [c for c in df.columns if c not in exclude]


def prepare_1h(
    symbol: str,
    features_dir: str = "data/features_1h",
    max_holding: int = 20,
) -> Optional[dict[str, Any]]:
    """Load 1h features, labels (20-bar barrier), split (25-bar embargo)."""
    feature_file = Path(features_dir) / f"{symbol}_1h.parquet"
    if not feature_file.exists():
        logger.warning(f"{symbol}: 1h feature file missing")
        return None

    df = pd.read_parquet(feature_file)
    df = compute_triple_barrier(df, profit_mult=3.0, stop_mult=3.0,
                                max_holding=max_holding)
    # Embargo tuned for 1h (25 > 20-bar holding)
    split = split_temporal(df, embargo_bars=EMBARGO_1H)
    fcols = _feature_cols(df)

    def _prep(sdf: pd.DataFrame):
        mask = sdf["tb_label"].isin([-1.0, 1.0])
        sub = sdf[mask].copy()
        X = sub[fcols]
        y = (sub["tb_label"] == 1.0).astype("int64")
        ret = sub["tb_return"]
        return X, y, ret

    X_train, y_train, _ = _prep(split.train)
    X_val, y_val, val_ret = _prep(split.val)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "feature_cols": fcols, "val_returns": val_ret,
    }


def train_eval_1h(data: dict[str, Any]) -> dict[str, Any]:
    train_set = lgb.Dataset(data["X_train"], label=data["y_train"])
    val_set = lgb.Dataset(data["X_val"], label=data["y_val"],
                          reference=train_set)
    model = lgb.train(
        LGBM_PARAMS, train_set, num_boost_round=N_ESTIMATORS,
        valid_sets=[train_set, val_set], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING),
                   lgb.log_evaluation(period=0)],
    )

    y_train = data["y_train"].to_numpy()
    y_val = data["y_val"].to_numpy()
    train_proba = np.asarray(model.predict(data["X_train"]), dtype="float64")
    val_proba = np.asarray(model.predict(data["X_val"]), dtype="float64")

    train_acc = float(((train_proba > 0.5).astype("int64") == y_train).mean())
    val_acc = float(((val_proba > 0.5).astype("int64") == y_val).mean())

    val_ret = np.asarray(data["val_returns"].to_numpy(), dtype="float64")
    valid = ~np.isnan(val_ret)
    if int(valid.sum()) > 10:
        corr = spearmanr(val_proba[valid], val_ret[valid])
        ic = float(corr.statistic)  # type: ignore[attr-defined]
        ic = ic if not np.isnan(ic) else 0.0
    else:
        ic = 0.0

    hc_up = val_proba > 0.58
    hc_down = val_proba < 0.42
    prec_up = float((y_val[hc_up] == 1).mean()) if int(hc_up.sum()) > 10 else 0.0
    prec_down = float((y_val[hc_down] == 0).mean()) if int(hc_down.sum()) > 10 else 0.0

    importance = model.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({
        "feature": list(data["feature_cols"]),
        "importance": list(importance),
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return {
        "best_iter": model.best_iteration,
        "train_acc": train_acc, "val_acc": val_acc,
        "overfit_gap": train_acc - val_acc, "ic": ic,
        "prec_up": prec_up, "n_hc_up": int(hc_up.sum()),
        "prec_down": prec_down, "n_hc_down": int(hc_down.sum()),
        "importance": imp_df,
    }


if __name__ == "__main__":
    import time

    print("=" * 70)
    print(f"Binary 1H Model v2 — Stronger Regularization (v{BINARY_1H_VERSION})")
    print("=" * 70)

    symbol = "BTCUSDT"
    print(f"\nSymbol: {symbol} | 1h | Stronger reg (num_leaves 15→8, reg 0.3→0.6)")
    print("Goal: confirm edge survives — val>=53%, IC>=0.03, gap<7%")

    print("\nPreparing...")
    t0 = time.time()
    data = prepare_1h(symbol)
    if data is None:
        print("❌ Failed")
        exit(1)
    print(f"  Train: {len(data['X_train']):,}, Val: {len(data['X_val']):,}")

    print("\nTraining...")
    r = train_eval_1h(data)
    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("RESULTS v2 (vs v1: gap was 11.9%, IC 0.0446, val 53.2%)")
    print("=" * 70)
    print(f"\n  Best iteration: {r['best_iter']}")
    print(f"  Train accuracy: {r['train_acc']*100:.1f}%")
    print(f"  Val accuracy:   {r['val_acc']*100:.1f}%")
    print(f"  Overfit gap:    {r['overfit_gap']*100:.1f}% "
          f"(was 11.9% — {'IMPROVED' if r['overfit_gap'] < 0.119 else 'same'})")
    print(f"  IC:             {r['ic']:+.4f} (was +0.0446)")

    print("\n  High-conf precision:")
    print(f"    UP  (>0.58): {r['prec_up']*100:.1f}% on {r['n_hc_up']:,}")
    print(f"    DOWN(<0.42): {r['prec_down']*100:.1f}% on {r['n_hc_down']:,}")

    print("\n" + "=" * 70)
    print("TOP 12 FEATURES")
    print("=" * 70)
    for pos, (_, row) in enumerate(r["importance"].head(12).iterrows(), 1):
        print(f"  {pos:2d}. {row['feature']:28s} {row['importance']:,.0f}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    ic, va, gap = r["ic"], r["val_acc"], r["overfit_gap"]
    print(f"\n  IC >= 0.03:       {'✅' if ic >= 0.03 else '❌'} ({ic:+.4f})")
    print(f"  Val acc >= 53%:   {'✅' if va >= 0.53 else '❌'} ({va*100:.1f}%)")
    print(f"  Overfit gap < 7%: {'✅' if gap < 0.07 else '❌'} ({gap*100:.1f}%)")
    print(f"\n  Time: {elapsed:.1f}s")

    if ic >= 0.03 and va >= 0.53 and gap < 0.07:
        print("\n  🎯 EDGE CONFIRMED — survives regularization. REAL signal.")
        print("     Next: multi-symbol validation.")
    elif ic >= 0.03 and va >= 0.52:
        print("\n  ✅ EDGE HOLDS — signal persists. Minor tuning left.")
    else:
        print("\n  ⚠️  Edge weakened under regularization — investigate.")
    print("=" * 70)