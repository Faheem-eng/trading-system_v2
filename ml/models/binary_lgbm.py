"""
ml/models/binary_lgbm.py

Binary LightGBM model (UP vs DOWN) with stronger regularization.

Improvement over baseline (3-class):
    1. BINARY classification — drop FLAT bars, predict only UP vs DOWN.
       Cleaner directional signal (López de Prado approach).
    2. STRONGER regularization — reduce the 10.8% overfit gap seen
       in the baseline (shallower trees, more min-samples, more penalty).

Rationale for binary:
    FLAT bars (|move| < barrier) are ambiguous. Including them forces
    the model to learn a fuzzy boundary. Dropping them lets the model
    focus purely on directional conviction — which is what we trade on.

Success criteria (same pre-registered bar):
    GO:    IC > 0.03, val accuracy > 52%, overfit gap < 5%
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

BINARY_LGBM_VERSION = "1.0.0"


# ============================================================
# Stronger anti-overfit params (each change justified)
# ============================================================

LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",       # UP vs DOWN
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "max_depth": 3,              # was 5 → shallower = less overfit
    "num_leaves": 7,             # was 31 → 2^3-1, matches depth 3
    "min_child_samples": 300,    # was 100 → more samples per leaf
    "learning_rate": 0.03,       # was 0.05 → slower, more careful
    "reg_alpha": 0.5,            # was 0.1 → stronger L1
    "reg_lambda": 0.5,           # was 0.1 → stronger L2
    "feature_fraction": 0.7,     # was 0.8 → more feature bagging
    "bagging_fraction": 0.7,     # was 0.8 → more data bagging
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

N_ESTIMATORS = 400
EARLY_STOPPING_ROUNDS = 40


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns (exclude OHLCV, time, target columns)."""
    exclude = {
        "open_time", "close_time", "open", "high", "low", "close",
        "volume", "quote_volume", "count", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore",
        "tb_label", "tb_exit_offset", "tb_return",
    }
    return [c for c in df.columns if c not in exclude]


def prepare_binary_data(
    symbol: str,
    features_dir: str = "data/features",
) -> Optional[dict[str, Any]]:
    """
    Load features, compute labels, DROP FLAT, split temporally.

    Binary mapping:
        -1 (down) → 0
        +1 (up)   → 1
         0 (flat) → DROPPED
    """
    feature_file = Path(features_dir) / f"{symbol}_5m.parquet"
    if not feature_file.exists():
        logger.warning(f"{symbol}: feature file missing")
        return None

    df = pd.read_parquet(feature_file)
    df = compute_triple_barrier(df)
    split = split_temporal(df)
    feature_cols = _get_feature_columns(df)

    def _prep(split_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        # Keep only UP (+1) and DOWN (-1), drop FLAT (0) and NaN
        mask = split_df["tb_label"].isin([-1.0, 1.0])
        subset = split_df[mask].copy()
        X = subset[feature_cols]
        # Map -1→0, +1→1
        y = (subset["tb_label"] == 1.0).astype("int64")
        returns = subset["tb_return"]
        return X, y, returns

    X_train, y_train, train_ret = _prep(split.train)
    X_val, y_val, val_ret = _prep(split.val)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "feature_cols": feature_cols,
        "train_returns": train_ret, "val_returns": val_ret,
    }


def train_binary(data: dict[str, Any]) -> dict[str, Any]:
    """Train binary LightGBM with early stopping."""
    train_set = lgb.Dataset(data["X_train"], label=data["y_train"])
    val_set = lgb.Dataset(
        data["X_val"], label=data["y_val"], reference=train_set
    )

    model = lgb.train(
        LGBM_PARAMS,
        train_set,
        num_boost_round=N_ESTIMATORS,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=0),
        ],
    )
    return {"model": model, "best_iteration": model.best_iteration}


def evaluate_binary(model: Any, data: dict[str, Any]) -> dict[str, Any]:
    """Honest binary evaluation: accuracy, IC, precision."""
    X_train = data["X_train"]
    y_train = data["y_train"].to_numpy()
    X_val = data["X_val"]
    y_val = data["y_val"].to_numpy()

    # Predicted probabilities of UP
    train_proba = np.asarray(model.predict(X_train), dtype="float64")
    val_proba = np.asarray(model.predict(X_val), dtype="float64")

    train_pred = (train_proba > 0.5).astype("int64")
    val_pred = (val_proba > 0.5).astype("int64")

    train_acc = float((train_pred == y_train).mean())
    val_acc = float((val_pred == y_val).mean())

    # Information Coefficient: P(up) vs actual return
    val_returns = np.asarray(data["val_returns"].to_numpy(), dtype="float64")
    valid = ~np.isnan(val_returns)
    if int(valid.sum()) > 10:
        corr = spearmanr(val_proba[valid], val_returns[valid])
        ic_val = float(corr.statistic)  # type: ignore[attr-defined]
        ic = ic_val if not np.isnan(ic_val) else 0.0
    else:
        ic = 0.0

    # Precision at high confidence (>0.6 prob → predict UP)
    high_conf_up = val_proba > 0.6
    if int(high_conf_up.sum()) > 10:
        prec_up = float((y_val[high_conf_up] == 1).mean())
        n_high_up = int(high_conf_up.sum())
    else:
        prec_up = 0.0
        n_high_up = 0

    high_conf_down = val_proba < 0.4
    if int(high_conf_down.sum()) > 10:
        prec_down = float((y_val[high_conf_down] == 0).mean())
        n_high_down = int(high_conf_down.sum())
    else:
        prec_down = 0.0
        n_high_down = 0

    return {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "overfit_gap": train_acc - val_acc,
        "information_coefficient": ic,
        "precision_up_highconf": prec_up,
        "n_highconf_up": n_high_up,
        "precision_down_highconf": prec_down,
        "n_highconf_down": n_high_down,
    }


def get_feature_importance(
    model: Any, feature_cols: list[str], top_n: int = 15
) -> pd.DataFrame:
    """Top-N features by gain."""
    importance = model.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({
        "feature": list(feature_cols),
        "importance": list(importance),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return imp_df.head(top_n)


if __name__ == "__main__":
    import time

    print("=" * 70)
    print(f"Binary LightGBM Model (v{BINARY_LGBM_VERSION})")
    print("=" * 70)

    symbol = "BTCUSDT"
    print(f"\nSymbol: {symbol}")
    print("Changes: BINARY (no FLAT) + stronger regularization")
    print("  max_depth: 5→3, min_child: 100→300, reg: 0.1→0.5")

    print("\nPreparing binary data...")
    t0 = time.time()
    data = prepare_binary_data(symbol)
    if data is None:
        print("❌ Data prep failed")
        exit(1)

    print(f"  Train: {len(data['X_train']):,} samples (FLAT dropped)")
    print(f"  Val:   {len(data['X_val']):,} samples")
    up_pct = float(data["y_train"].mean()) * 100
    print(f"  Train balance: UP={up_pct:.0f}% DOWN={100-up_pct:.0f}%")

    print("\nTraining (stronger regularization)...")
    result = train_binary(data)
    print(f"  Best iteration: {result['best_iteration']}")

    print("\nEvaluating...")
    metrics = evaluate_binary(result["model"], data)
    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n  Train accuracy:   {metrics['train_accuracy']*100:.1f}%")
    print(f"  Val accuracy:     {metrics['val_accuracy']*100:.1f}% (>52% = edge)")
    print(f"  Overfit gap:      {metrics['overfit_gap']*100:.1f}% (<5% healthy)")
    print(f"  IC:               {metrics['information_coefficient']:.4f} (>0.03 = edge)")

    print("\n  High-confidence precision (the meta-labeling preview):")
    print(f"    UP  (prob>0.6):  {metrics['precision_up_highconf']*100:.1f}% "
          f"correct on {metrics['n_highconf_up']:,} bars")
    print(f"    DOWN(prob<0.4):  {metrics['precision_down_highconf']*100:.1f}% "
          f"correct on {metrics['n_highconf_down']:,} bars")

    print("\n" + "=" * 70)
    print("TOP 15 FEATURES")
    print("=" * 70)
    imp = get_feature_importance(result["model"], data["feature_cols"])
    for pos, (_, row) in enumerate(imp.iterrows(), start=1):
        print(f"  {pos:2d}. {row['feature']:28s} {row['importance']:,.0f}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    ic = metrics["information_coefficient"]
    val_acc = metrics["val_accuracy"]
    gap = metrics["overfit_gap"]

    print(f"\n  IC > 0.03:          {'✅' if ic > 0.03 else '❌'} ({ic:.4f})")
    print(f"  Val acc > 52%:      {'✅' if val_acc > 0.52 else '❌'} ({val_acc*100:.1f}%)")
    print(f"  Overfit gap < 5%:   {'✅' if gap < 0.05 else '❌'} ({gap*100:.1f}%)")
    print(f"\n  Time: {elapsed:.1f}s")

    # Key insight: high-conf precision matters most for meta-labeling
    best_prec = max(metrics['precision_up_highconf'],
                    metrics['precision_down_highconf'])
    if ic > 0.03 and val_acc > 0.52 and gap < 0.05:
        print("\n  🎯 GO — Edge confirmed.")
    elif best_prec > 0.55:
        print(f"\n  ⚠️  META-LABELING PROMISE — High-conf precision "
              f"{best_prec*100:.0f}% suggests confidence filtering works.")
    else:
        print("\n  🔬 Still investigating — try meta-labeling or longer horizon.")
    print("=" * 70)