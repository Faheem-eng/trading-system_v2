"""
ml/models/horizon_scan.py

Multi-horizon / multi-timeframe edge scan.

Purpose:
    The 5m + 50-bar baseline showed near-random direction (IC ~0.02).
    Hurst = 0.4 confirmed 5m is mean-reverting/noisy.
    This scans DIFFERENT timeframes and horizons to find where
    a directional edge actually exists.

Method:
    For each (timeframe, barrier) config:
        1. Load features for that timeframe
        2. Compute triple-barrier labels with that horizon
        3. Binary model (UP vs DOWN) with balanced regularization
        4. Measure IC + val accuracy + high-conf precision
    Report which config has the strongest signal.

Balanced regularization (depth=4):
    depth=5 overfit (gap 10.8%), depth=3 too simple (best_iter=1).
    depth=4 is the middle — enough capacity, controlled overfit.

This is honest, data-driven edge discovery — not hope.
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

HORIZON_SCAN_VERSION = "1.0.0"


# Balanced regularization (between overfit depth=5 and too-simple depth=3)
LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "max_depth": 4,              # balanced
    "num_leaves": 15,            # 2^4-1
    "min_child_samples": 200,    # between 100 and 300
    "learning_rate": 0.03,
    "reg_alpha": 0.3,            # between 0.1 and 0.5
    "reg_lambda": 0.3,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.75,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

N_ESTIMATORS = 400
EARLY_STOPPING = 40


def _feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {
        "open_time", "close_time", "open", "high", "low", "close",
        "volume", "quote_volume", "count", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore",
        "tb_label", "tb_exit_offset", "tb_return",
    }
    return [c for c in df.columns if c not in exclude]


def scan_config(
    symbol: str,
    timeframe: str,
    profit_mult: float,
    stop_mult: float,
    max_holding: int,
    features_dir: str = "data/features",
) -> Optional[dict[str, Any]]:
    """
    Test one (timeframe, barrier) configuration.

    Returns metrics dict, or None if data missing.
    """
    # Note: features only exist for 5m currently.
    # For 1h, we compute features on the fly would be needed — but
    # for this scan we use the 5m feature file and vary the HORIZON,
    # plus we resample to test 1h separately.
    feature_file = Path(features_dir) / f"{symbol}_5m.parquet"
    if not feature_file.exists():
        return None

    df = pd.read_parquet(feature_file)

    # For 1h timeframe: resample the 5m features to hourly (take every 12th bar)
    # This approximates 1h sampling while keeping the computed features.
    if timeframe == "1h":
        df = df.iloc[::12].reset_index(drop=True).copy()

    # Compute labels with this horizon
    df = compute_triple_barrier(
        df,
        profit_mult=profit_mult,
        stop_mult=stop_mult,
        max_holding=max_holding,
    )

    split = split_temporal(df)
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

    if len(X_train) < 1000 or len(X_val) < 200:
        return None  # insufficient data

    # Train
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    model = lgb.train(
        LGBM_PARAMS, train_set, num_boost_round=N_ESTIMATORS,
        valid_sets=[val_set], valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING),
                   lgb.log_evaluation(period=0)],
    )

    # Evaluate
    y_val_arr = y_val.to_numpy()
    val_proba = np.asarray(model.predict(X_val), dtype="float64")
    val_pred = (val_proba > 0.5).astype("int64")
    val_acc = float((val_pred == y_val_arr).mean())

    train_proba = np.asarray(model.predict(X_train), dtype="float64")
    train_pred = (train_proba > 0.5).astype("int64")
    train_acc = float((train_pred == y_train.to_numpy()).mean())

    val_ret_arr = np.asarray(val_ret.to_numpy(), dtype="float64")
    valid = ~np.isnan(val_ret_arr)
    if int(valid.sum()) > 10:
        corr = spearmanr(val_proba[valid], val_ret_arr[valid])
        ic = float(corr.statistic)  # type: ignore[attr-defined]
        ic = ic if not np.isnan(ic) else 0.0
    else:
        ic = 0.0

    # High-confidence precision
    hc_up = val_proba > 0.58
    hc_down = val_proba < 0.42
    prec_up = float((y_val_arr[hc_up] == 1).mean()) if int(hc_up.sum()) > 10 else 0.0
    prec_down = float((y_val_arr[hc_down] == 0).mean()) if int(hc_down.sum()) > 10 else 0.0
    n_hc = int(hc_up.sum()) + int(hc_down.sum())

    return {
        "timeframe": timeframe,
        "barrier": f"{profit_mult}x/{max_holding}bar",
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "best_iter": model.best_iteration,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "overfit_gap": train_acc - val_acc,
        "ic": ic,
        "prec_up": prec_up,
        "prec_down": prec_down,
        "n_highconf": n_hc,
    }


if __name__ == "__main__":
    import time

    print("=" * 78)
    print(f"MULTI-HORIZON EDGE SCAN (v{HORIZON_SCAN_VERSION})")
    print("=" * 78)
    print("\nFinding WHERE directional edge exists (data-driven, honest).")
    print("Balanced regularization: max_depth=4\n")

    symbol = "BTCUSDT"

    # Configs to test: (timeframe, profit_mult, stop_mult, max_holding)
    configs = [
        ("5m", 3.0, 3.0, 50),    # baseline (reference)
        ("5m", 3.0, 3.0, 100),   # 5m longer hold (~8 hrs)
        ("5m", 4.0, 4.0, 150),   # 5m even longer, wider barrier
        ("1h", 3.0, 3.0, 20),    # 1h medium (~20 hrs)
        ("1h", 3.0, 3.0, 50),    # 1h longer (~2 days)
    ]

    results = []
    for tf, pm, sm, mh in configs:
        label = f"{tf} {pm}x {mh}bar"
        print(f"Testing {label:20s} ...", end=" ", flush=True)
        t0 = time.time()
        r = scan_config(symbol, tf, pm, sm, mh)
        if r is None:
            print("skipped (insufficient data)")
            continue
        results.append(r)
        print(f"IC={r['ic']:+.4f} val_acc={r['val_acc']*100:.1f}% "
              f"({time.time()-t0:.0f}s)")

    # Summary table
    print("\n" + "=" * 78)
    print("RESULTS (sorted by IC — higher = more edge)")
    print("=" * 78)
    print(f"\n{'Config':22s} {'ValAcc':>7s} {'Gap':>6s} {'IC':>8s} "
          f"{'HC-Prec':>8s} {'#HC':>6s}")
    print("-" * 78)

    results_sorted = sorted(results, key=lambda x: x["ic"], reverse=True)
    for r in results_sorted:
        best_prec = max(r["prec_up"], r["prec_down"])
        cfg = f"{r['timeframe']} {r['barrier']}"
        print(f"{cfg:22s} {r['val_acc']*100:>6.1f}% "
              f"{r['overfit_gap']*100:>5.1f}% {r['ic']:>+8.4f} "
              f"{best_prec*100:>7.1f}% {r['n_highconf']:>6,}")

    # Verdict
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if results_sorted:
        best = results_sorted[0]
        best_prec = max(best["prec_up"], best["prec_down"])
        print(f"\n  Best config: {best['timeframe']} {best['barrier']}")
        print(f"    IC:            {best['ic']:+.4f}")
        print(f"    Val accuracy:  {best['val_acc']*100:.1f}%")
        print(f"    High-conf prec: {best_prec*100:.1f}% on {best['n_highconf']:,} bars")

        if best["ic"] > 0.03 or best["val_acc"] > 0.53:
            print("\n  🎯 EDGE FOUND at this config — worth pursuing.")
        elif best_prec > 0.55 and best["n_highconf"] > 100:
            print(f"\n  ⚠️  META-LABELING PROMISE — high-conf trades show "
                  f"{best_prec*100:.0f}% precision.")
        else:
            print("\n  🔬 Edge still weak everywhere — deeper analysis needed.")
            print("     (This is honest — some markets are just efficient.)")
    print("=" * 78)