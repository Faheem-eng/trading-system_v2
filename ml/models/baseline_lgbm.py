"""
ml/models/baseline_lgbm.py

Baseline LightGBM model for directional prediction.

This is the MOMENT OF TRUTH: do our 116 features contain real edge?

Design philosophy (anti-overfitting is the priority):
    - Heavy regularization (shallow trees, min samples, L1/L2 penalties)
    - Early stopping on validation (stop before memorizing noise)
    - Honest metrics (Information Coefficient, per-class precision)
    - Pre-registered success criteria (no moving goalposts)

Success criteria (FIXED before seeing results):
    GO:    IC > 0.03, directional accuracy > 52%, consistent train/val
    NO-GO: IC < 0.02, or train >> val (overfitting), or single-symbol only

Why LightGBM:
    - Fast on tabular data (our features are tabular)
    - Handles missing values natively
    - Feature importance built-in (interpretability)
    - Industry standard for quant ML baselines
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

BASELINE_LGBM_VERSION = "1.0.0"


# ============================================================
# Anti-overfitting hyperparameters (each justified)
# ============================================================

LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,              # UP / FLAT / DOWN
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "max_depth": 5,              # shallow = less overfit
    "num_leaves": 31,            # limited branches
    "min_child_samples": 100,    # min samples per leaf (anti-noise)
    "learning_rate": 0.05,       # slow learning (anti-overfit)
    "reg_alpha": 0.1,            # L1 penalty
    "reg_lambda": 0.1,           # L2 penalty
    "feature_fraction": 0.8,     # use 80% features per tree (bagging)
    "bagging_fraction": 0.8,     # use 80% data per iteration
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

N_ESTIMATORS = 300
EARLY_STOPPING_ROUNDS = 30


def _map_labels(tb_label: pd.Series) -> pd.Series:
    """
    Map triple barrier labels to LightGBM class indices.
        -1 (loss/down) → 0
         0 (flat)      → 1
        +1 (profit/up) → 2
    """
    mapping = {-1.0: 0, 0.0: 1, 1.0: 2}
    return tb_label.map(mapping)


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Identify feature columns (exclude OHLCV, time, and target columns).
    """
    exclude = {
        "open_time", "close_time", "open", "high", "low", "close",
        "volume", "quote_volume", "count", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore",
        "tb_label", "tb_exit_offset", "tb_return",
    }
    return [c for c in df.columns if c not in exclude]


def prepare_data(
    symbol: str,
    features_dir: str = "data/features",
) -> Optional[dict[str, Any]]:
    """
    Load features, compute labels, split temporally.

    Returns dict with train/val X,y and feature names, or None.
    """
    feature_file = Path(features_dir) / f"{symbol}_5m.parquet"
    if not feature_file.exists():
        logger.warning(f"{symbol}: feature file missing")
        return None

    df = pd.read_parquet(feature_file)

    # Compute triple barrier labels
    df = compute_triple_barrier(df)

    # Temporal split (test locked — we use train/val here)
    split = split_temporal(df)

    feature_cols = _get_feature_columns(df)

    # Prepare train
    train = split.train.dropna(subset=["tb_label"])
    X_train = train[feature_cols]
    y_train = _map_labels(train["tb_label"])

    # Prepare val
    val = split.val.dropna(subset=["tb_label"])
    X_val = val[feature_cols]
    y_val = _map_labels(val["tb_label"])

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "feature_cols": feature_cols,
        "train_returns": train["tb_return"],
        "val_returns": val["tb_return"],
    }


def train_model(data: dict[str, Any]) -> dict[str, Any]:
    """
    Train LightGBM with early stopping. Returns model + metrics.
    """
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    model = lgb.train(
        LGBM_PARAMS,
        train_set,
        num_boost_round=N_ESTIMATORS,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=0),  # silent
        ],
    )

    return {"model": model, "best_iteration": model.best_iteration}


def evaluate_model(
    model: Any, data: dict[str, Any]
) -> dict[str, Any]:
    """
    Honest evaluation: IC, directional accuracy, per-class metrics.
    """
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    # Predictions (probabilities per class)
    train_proba = model.predict(X_train)
    val_proba = model.predict(X_val)

    # Predicted class = argmax
    train_pred = np.argmax(train_proba, axis=1)
    val_pred = np.argmax(val_proba, axis=1)

    # Directional accuracy (excluding flat class for direction)
    def directional_accuracy(y_true, y_pred):
        # Only consider bars where truth is UP(2) or DOWN(0)
        directional = (y_true == 0) | (y_true == 2)
        if directional.sum() == 0:
            return 0.0
        correct = (y_true[directional] == y_pred[directional])
        return float(correct.mean())

    train_dir_acc = directional_accuracy(y_train.to_numpy(), train_pred)
    val_dir_acc = directional_accuracy(y_val.to_numpy(), val_pred)

    # Information Coefficient: correlation of predicted "up-ness" with actual returns
    # up_score = P(up) - P(down)
    val_up_score = val_proba[:, 2] - val_proba[:, 0]
    val_returns = data["val_returns"].to_numpy()
    # Align (val_returns already matches val rows)
    valid = ~np.isnan(val_returns)
    if int(valid.sum()) > 10:
        corr_result = spearmanr(
            np.asarray(val_up_score[valid], dtype="float64"),
            np.asarray(val_returns[valid], dtype="float64"),
        )
        ic_val = float(corr_result.statistic)  # type: ignore[attr-defined]
        ic = ic_val if not np.isnan(ic_val) else 0.0
    else:
        ic = 0.0

    # Per-class accuracy
    train_acc = float((train_pred == y_train.to_numpy()).mean())
    val_acc = float((val_pred == y_val.to_numpy()).mean())

    return {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "train_dir_accuracy": train_dir_acc,
        "val_dir_accuracy": val_dir_acc,
        "information_coefficient": ic,
        "overfit_gap": train_acc - val_acc,
    }


def get_feature_importance(
    model: Any, feature_cols: list[str], top_n: int = 15
) -> pd.DataFrame:
    """Return top-N most important features."""
    importance = model.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({
        "feature": list(feature_cols),
        "importance": list(importance),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return imp_df.head(top_n)


if __name__ == "__main__":
    import time

    print("=" * 70)
    print(f"Baseline LightGBM Model (v{BASELINE_LGBM_VERSION})")
    print("=" * 70)

    symbol = "BTCUSDT"
    print(f"\nSymbol: {symbol}")
    print(f"Anti-overfit params: max_depth={LGBM_PARAMS['max_depth']}, "
          f"min_child={LGBM_PARAMS['min_child_samples']}, "
          f"lr={LGBM_PARAMS['learning_rate']}")

    # Prepare
    print("\nPreparing data (features + labels + temporal split)...")
    t0 = time.time()
    data = prepare_data(symbol)
    if data is None:
        print("❌ Data preparation failed")
        exit(1)
    print(f"  Train: {len(data['X_train']):,} samples")
    print(f"  Val:   {len(data['X_val']):,} samples")
    print(f"  Features: {len(data['feature_cols'])}")

    # Class distribution
    train_dist = data["y_train"].value_counts(normalize=True).sort_index()
    print(f"  Train class balance: "
          f"DOWN={train_dist.get(0, 0)*100:.0f}% "
          f"FLAT={train_dist.get(1, 0)*100:.0f}% "
          f"UP={train_dist.get(2, 0)*100:.0f}%")

    # Train
    print("\nTraining LightGBM (early stopping on val)...")
    result = train_model(data)
    print(f"  Best iteration: {result['best_iteration']} "
          f"(early stopped before {N_ESTIMATORS})")

    # Evaluate
    print("\nEvaluating (honest metrics)...")
    metrics = evaluate_model(result["model"], data)
    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("RESULTS (Moment of Truth)")
    print("=" * 70)
    print(f"\n  Train accuracy:        {metrics['train_accuracy']*100:.1f}%")
    print(f"  Val accuracy:          {metrics['val_accuracy']*100:.1f}%")
    print(f"  Overfit gap:           {metrics['overfit_gap']*100:.1f}% "
          f"(train - val; <5% is healthy)")
    print(f"\n  Train dir. accuracy:   {metrics['train_dir_accuracy']*100:.1f}%")
    print(f"  Val dir. accuracy:     {metrics['val_dir_accuracy']*100:.1f}% "
          f"(>52% = edge)")
    print(f"\n  Information Coefficient: {metrics['information_coefficient']:.4f} "
          f"(>0.03 = edge)")

    # Feature importance
    print("\n" + "=" * 70)
    print("TOP 15 FEATURES (what the model relies on)")
    print("=" * 70)
    imp = get_feature_importance(result["model"], data["feature_cols"])
    for pos, (_, row) in enumerate(imp.iterrows(), start=1):
        print(f"  {pos:2d}. {row['feature']:28s} {row['importance']:,.0f}")

    # Verdict (pre-registered criteria)
    print("\n" + "=" * 70)
    print("VERDICT (pre-registered criteria)")
    print("=" * 70)
    ic = metrics["information_coefficient"]
    val_dir = metrics["val_dir_accuracy"]
    gap = metrics["overfit_gap"]

    edge_ic = ic > 0.03
    edge_dir = val_dir > 0.52
    not_overfit = gap < 0.05

    print(f"\n  IC > 0.03:              {'✅' if edge_ic else '❌'} ({ic:.4f})")
    print(f"  Val dir. acc > 52%:     {'✅' if edge_dir else '❌'} ({val_dir*100:.1f}%)")
    print(f"  Overfit gap < 5%:       {'✅' if not_overfit else '❌'} ({gap*100:.1f}%)")

    print(f"\n  Total time: {elapsed:.1f}s")

    if edge_ic and edge_dir and not_overfit:
        print("\n  🎯 GO — Edge detected. Features contain signal.")
    elif not_overfit and (edge_ic or edge_dir):
        print("\n  ⚠️  WEAK EDGE — Some signal, needs improvement.")
    else:
        print("\n  🔬 NO CLEAR EDGE YET — Honest result. Needs investigation.")
    print("=" * 70)