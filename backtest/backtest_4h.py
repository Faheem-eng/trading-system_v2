"""
backtest/backtest_4h.py

FINAL directional test: 4h timeframe walk-forward backtest.

Pre-registered as the last directional attempt at retail taker costs.
Rationale: 4h moves are larger → 0.25% round-trip cost is a smaller
fraction of the average move. Data: ~9,600 bars (4.5 years).

Honest context going in:
    1h diagnosis showed the model learned a RISK signal (top features:
    downside_vol_ratio, var_95, vol_clustering), giving crash-protection
    alpha but weak long-entry alpha. 4h may inherit this.

Pipeline (in-memory, no intermediate files):
    1. Load 4h raw data (4 stable symbols + BTC for cross-asset)
    2. Resample 4h→1D (proper OHLCV aggregation, completed bars) for HTF
    3. Compute all 14 feature modules on 4h bars
    4. Triple barrier 3xATR/3xATR, 20-bar hold (~3.3 days)
    5. Walk-forward: 5 folds, threshold 0.58, long-only, 0.25% costs

Pre-registered verdict:
    GO:    3+/5 portfolio folds positive AND compound > B&H → Phase 5
    NO-GO: directional trading closed at these costs → pivot to the
           system's PROVEN strength (risk/regime overlay strategy)
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

from features.base.price_features import compute_price_features
from features.base.volume_features import compute_volume_features
from features.base.time_features import compute_time_features
from features.technical.momentum_features import compute_momentum_features
from features.technical.trend_features import compute_trend_features
from features.technical.volatility_features import compute_volatility_features
from features.technical.volume_indicators import compute_volume_indicators
from features.microstructure.structure_features import compute_structure_features
from features.multi_timeframe.mtf_features import compute_mtf_features
from features.cross_asset.cross_asset_features import compute_cross_asset_features
from features.regime.regime_features import compute_regime_features
from features.alpha_factors.alpha_features import compute_alpha_features
from features.microstructure_quality.quality_features import compute_quality_features
from features.risk_metrics.risk_features import compute_risk_features

logger = get_logger(__name__)

BACKTEST_4H_VERSION = "1.0.0"

ENTRY_THRESHOLD = 0.58
ROUND_TRIP_COST = 0.0025
N_FOLDS = 5
EMBARGO = 25            # > 20-bar holding
MIN_TRAIN = 3500
TEST_SIZE = 900         # ~5 months of 4h bars
WARMUP = 300
BARS_PER_YEAR_4H = 6 * 365


def _resample_4h_to_1d(df4h: pd.DataFrame) -> pd.DataFrame:
    """
    Proper OHLCV resample 4h→1D (leakage-safe: aggregates completed bars;
    daily close_time = last 4h bar's close_time of that day).
    """
    d = df4h.copy()
    d["dt"] = pd.to_datetime(d["open_time"], unit="ms", utc=True)
    d = d.set_index("dt")
    agg = d.resample("1D").agg(
        open_time=("open_time", "first"),
        close_time=("close_time", "last"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        quote_volume=("quote_volume", "sum"),
        taker_buy_volume=("taker_buy_volume", "sum"),
    ).dropna().reset_index(drop=True)
    return agg


def build_4h_features(symbol: str, raw_dir: str = "data/raw",
                      btc_symbol: str = "BTCUSDT") -> Optional[pd.DataFrame]:
    """Compute all features on 4h bars (in-memory)."""
    raw = Path(raw_dir)
    f_sym = raw / f"{symbol}_4h.parquet"
    f_btc = raw / f"{btc_symbol}_4h.parquet"
    if not f_sym.exists() or not f_btc.exists():
        return None

    df = pd.read_parquet(f_sym)
    btc = pd.read_parquet(f_btc)
    htf_daily = _resample_4h_to_1d(df)

    df = compute_price_features(df)
    df = compute_volume_features(df)
    df = compute_time_features(df)
    df = compute_momentum_features(df)
    df = compute_trend_features(df)
    df = compute_volatility_features(df)
    df = compute_volume_indicators(df)
    df = compute_structure_features(df)
    df = compute_regime_features(df)
    df = compute_alpha_features(df)
    df = compute_quality_features(df)
    df = compute_risk_features(df)
    df = compute_mtf_features(df, htf_daily, tf_ratio=6)   # 1D/4h = 6
    df = compute_cross_asset_features(df, btc)

    return df.iloc[WARMUP:].reset_index(drop=True)


def _period_str(df: pd.DataFrame) -> str:
    t0 = datetime.fromtimestamp(
        int(df["open_time"].min()) / 1000, tz=timezone.utc).strftime("%Y-%m")
    t1 = datetime.fromtimestamp(
        int(df["open_time"].max()) / 1000, tz=timezone.utc).strftime("%Y-%m")
    return f"{t0}→{t1}"


def _simulate(proba: np.ndarray, tb_return: np.ndarray,
              tb_exit: np.ndarray) -> list[float]:
    n = len(proba)
    rets: list[float] = []
    i = 0
    while i < n:
        if proba[i] > ENTRY_THRESHOLD and not np.isnan(tb_return[i]):
            rets.append(float(tb_return[i]) - ROUND_TRIP_COST)
            off = int(tb_exit[i]) if not np.isnan(tb_exit[i]) else 20
            i += max(off, 1)
        else:
            i += 1
    return rets


def backtest_4h_symbol(symbol: str) -> Optional[dict[str, Any]]:
    df = build_4h_features(symbol)
    if df is None or len(df) < MIN_TRAIN + TEST_SIZE:
        return None

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
        if len(X_tr) < 1500 or len(test_df) < 300:
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

        rets = _simulate(proba, tb_ret, tb_ex)
        bh = float(close[-1] / close[0] - 1.0)

        if len(rets) >= 3:
            arr = np.array(rets, dtype="float64")
            total = float(np.exp(np.sum(arr)) - 1.0)
            wr = float((arr > 0).mean())
            exp_tr = float(np.mean(arr))
        else:
            total, wr, exp_tr = 0.0, 0.0, 0.0

        folds.append({
            "fold": k + 1, "period": _period_str(test_df),
            "n_trades": len(rets), "total": total, "win_rate": wr,
            "expectancy": exp_tr, "bh": bh,
        })

    if not folds:
        return None
    return {"symbol": symbol, "folds": folds}


if __name__ == "__main__":
    import time

    SYMBOLS = ["XRPUSDT", "SOLUSDT", "BTCUSDT", "MATICUSDT"]

    print("=" * 78)
    print(f"4H WALK-FORWARD BACKTEST — FINAL DIRECTIONAL TEST "
          f"(v{BACKTEST_4H_VERSION})")
    print("=" * 78)
    print(f"\nBarriers 3x/3x ATR, 20-bar (~3.3 days) | threshold "
          f"{ENTRY_THRESHOLD} | costs {ROUND_TRIP_COST*100:.2f}%")
    print("Pre-registered: GO = 3+/5 folds positive AND beats B&H")
    print("NO-GO = directional closed → pivot to risk/regime overlay\n")

    t0 = time.time()
    all_results = []

    for i, sym in enumerate(SYMBOLS, 1):
        print(f"[{i}/{len(SYMBOLS)}] {sym} (computing 4h features + backtest)...",
              flush=True)
        r = backtest_4h_symbol(sym)
        if r is None:
            print("   skipped (insufficient data)\n")
            continue
        all_results.append(r)
        for f in r["folds"]:
            marker = "✅" if f["total"] > 0 else "❌"
            print(f"   F{f['fold']} [{f['period']}] {f['n_trades']:3d} tr | "
                  f"net {f['total']*100:+7.2f}% | WR {f['win_rate']*100:4.1f}% | "
                  f"E[/tr] {f['expectancy']*100:+.3f}% | "
                  f"B&H {f['bh']*100:+7.2f}% | {marker}")
        print()

    # Portfolio
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

    if sys_folds:
        sys_comp = float(np.exp(np.sum([np.log(1+x) for x in sys_folds])) - 1)
        bh_comp = float(np.exp(np.sum([np.log(1+x) for x in bh_folds])) - 1)
        n_pos = sum(1 for x in sys_folds if x > 0)
        print("-" * 50)
        print(f"{'ALL':>5s} {sys_comp*100:>+8.2f}% {bh_comp*100:>+8.2f}% "
              f"{(sys_comp-bh_comp)*100:>+8.2f}%")

        print("\n" + "=" * 78)
        print("FINAL VERDICT (pre-registered)")
        print("=" * 78)
        print(f"\n  Positive folds: {n_pos}/{len(sys_folds)}")
        print(f"  System: {sys_comp*100:+.2f}% | B&H: {bh_comp*100:+.2f}%")
        print(f"  Time: {(time.time()-t0)/60:.1f} min")

        if n_pos >= 3 and sys_comp > bh_comp:
            print("\n  🎯 GO — 4h directional works. Phase 5 next.")
        else:
            print("\n  🔬 NO-GO — directional trading at taker costs: CLOSED.")
            print("     Honest pivot to system's PROVEN strength:")
            print("     risk/regime overlay (crash-protection alpha was real")
            print("     in every test: +28-31% alpha in crash folds).")
    print("=" * 78)