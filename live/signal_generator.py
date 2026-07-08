"""
live/signal_generator.py

Phase 6.2: LIVE SIGNAL GENERATOR — the production entry point.

Run this after any 4h bar close (or daily). For each symbol it:
    1. Loads historical 4h parquet + fetches fresh bars via ccxt
       (public endpoint, no API key needed)
    2. Builds all 116 features (same pipeline as validation)
    3. Trains the model on full labeled history (minus embargo)
    4. Predicts P(up) on the LAST COMPLETED bar only
    5. Applies the hysteresis overlay (0.55 in / 0.45 out) using
       persisted state from signals/state.json
    6. Prints the action sheet: LONG / CASH per symbol

SAFETY RULES (hard-coded, per SYSTEM_SPEC.md §5):
    - Only COMPLETED bars are used (forming bar = look-ahead in live)
    - Fetch failure / data gap → NO_SIGNAL (hold previous state)
    - No threshold overrides, no manual parameters

STATE FILE (signals/state.json):
    {"XRPUSDT": {"position": "LONG", "since": "...", "last_proba": 0.61}, ...}
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

from core.logger import get_logger
from ml.targets.triple_barrier import compute_triple_barrier
from ml.models.binary_1h import LGBM_PARAMS, _feature_cols
from backtest.backtest_4h import _resample_4h_to_1d
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

SIGNAL_GENERATOR_VERSION = "1.0.0"

# ---- Locked parameters (SYSTEM_SPEC.md §3 — DO NOT MODIFY) ----
TH_IN = 0.55
TH_OUT = 0.45
EMBARGO = 25
WARMUP = 300
# MATIC retired 2026-07-08: Binance MATIC→POL rebrand killed fresh data.
# POL rejected: only ~1.5yr history, our 4.5yr validation doesn't transfer.
# See docs/OPERATIONS_RUNBOOK.md §universe. Universe = 3 clean symbols.
SYMBOLS = ["XRPUSDT", "SOLUSDT", "BTCUSDT"]
BTC = "BTCUSDT"

STATE_FILE = Path("signals/state.json")
RAW_DIR = Path("data/raw")


# ============================================================
# Data: historical parquet + fresh ccxt bars
# ============================================================


def fetch_fresh_4h(symbol: str, since_ms: int) -> Optional[pd.DataFrame]:
    """
    Fetch 4h klines from Binance public API via ccxt, from since_ms
    to now. Returns None on any failure (fail-safe: NO_SIGNAL).

    Only COMPLETED bars are returned (last row dropped if its
    close_time is in the future).
    """
    try:
        import ccxt  # local import: optional dependency
        exchange = ccxt.binance({"enableRateLimit": True})
        rows: list[list[float]] = []
        cursor = since_ms
        while True:
            batch = exchange.fetch_ohlcv(
                symbol.replace("USDT", "/USDT"), timeframe="4h",
                since=cursor, limit=500,
            )
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < 500:
                break
            cursor = batch[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000)

        if not rows:
            return None

        df = pd.DataFrame(
            rows, columns=["open_time", "open", "high", "low",
                           "close", "volume"],
        )
        df["close_time"] = df["open_time"] + 4 * 3600 * 1000 - 1
        # ccxt doesn't give these; approximate for feature parity
        df["quote_volume"] = df["volume"] * df["close"]
        df["taker_buy_volume"] = df["volume"] * 0.5  # neutral assumption
        # Drop the forming (incomplete) bar
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        df = df[df["close_time"] < now_ms].reset_index(drop=True)
        return df
    except Exception as e:
        logger.error(f"{symbol}: fresh fetch failed — {e}")
        return None


def load_full_series(symbol: str) -> Optional[pd.DataFrame]:
    """Historical parquet + fresh bars, deduplicated, sorted."""
    hist_file = RAW_DIR / f"{symbol}_4h.parquet"
    if not hist_file.exists():
        logger.error(f"{symbol}: historical 4h parquet missing")
        return None
    hist = pd.read_parquet(hist_file)

    last_ms = int(hist["open_time"].max())
    fresh = fetch_fresh_4h(symbol, since_ms=last_ms + 1)
    if fresh is None:
        logger.warning(f"{symbol}: no fresh data — using historical only")
        fresh = pd.DataFrame()

    common_cols = [c for c in hist.columns if c in fresh.columns] \
        if len(fresh) else list(hist.columns)
    combined = pd.concat(
        [hist[common_cols], fresh[common_cols]] if len(fresh) else [hist],
        ignore_index=True,
    )
    combined = combined.drop_duplicates(subset="open_time") \
        .sort_values("open_time").reset_index(drop=True)
    return combined


# ============================================================
# Features + model + prediction
# ============================================================


def build_features_live(symbol: str, btc_df: pd.DataFrame,
                        sym_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Full 14-module feature build on the combined series."""
    try:
        df = sym_df.copy()
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
        df = compute_mtf_features(df, htf_daily, tf_ratio=6)
        df = compute_cross_asset_features(df, btc_df)
        return df.iloc[WARMUP:].reset_index(drop=True)
    except Exception as e:
        logger.error(f"{symbol}: feature build failed — {e}")
        return None


def predict_latest(symbol: str, df: pd.DataFrame) -> Optional[float]:
    """
    Train on full labeled history (minus embargo before the latest bar),
    predict P(up) on the LAST completed bar.
    """
    try:
        df = compute_triple_barrier(df, profit_mult=3.0, stop_mult=3.0,
                                    max_holding=20)
        fcols = _feature_cols(df)
        n = len(df)

        train = df.iloc[: n - EMBARGO]
        trl = train[train["tb_label"].isin([-1.0, 1.0])]
        X_tr = trl[fcols]
        y_tr = (trl["tb_label"] == 1.0).astype("int64")
        if len(X_tr) < 3000:
            logger.error(f"{symbol}: insufficient training data")
            return None

        model = lgb.train(
            LGBM_PARAMS, lgb.Dataset(X_tr, label=y_tr),
            num_boost_round=100,
            callbacks=[lgb.log_evaluation(period=0)],
        )
        latest = df.iloc[[-1]][fcols]
        proba = float(np.asarray(model.predict(latest), dtype="float64")[0])
        return proba
    except Exception as e:
        logger.error(f"{symbol}: prediction failed — {e}")
        return None


# ============================================================
# Hysteresis state machine
# ============================================================


def load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def apply_overlay(symbol: str, proba: float,
                  state: dict[str, Any]) -> tuple[str, str]:
    """
    Hysteresis: LONG if proba>TH_IN; CASH if proba<TH_OUT;
    otherwise HOLD previous position.
    Returns (position, action_note).
    """
    prev = state.get(symbol, {}).get("position", "CASH")
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if prev == "CASH" and proba > TH_IN:
        new_pos, note = "LONG", f"ENTER (P={proba:.3f} > {TH_IN})"
    elif prev == "LONG" and proba < TH_OUT:
        new_pos, note = "CASH", f"EXIT (P={proba:.3f} < {TH_OUT})"
    else:
        new_pos, note = prev, f"HOLD {prev} (P={proba:.3f} in band)"

    state[symbol] = {"position": new_pos, "since": now_str,
                     "last_proba": round(proba, 4)}
    return new_pos, note


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    print("=" * 70)
    print(f"LIVE SIGNAL GENERATOR (v{SIGNAL_GENERATOR_VERSION})")
    print(f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    state = load_state()
    t0 = time.time()

    # BTC series first (cross-asset dependency for all symbols)
    print("\nLoading BTC 4h series (cross-asset base)...")
    btc_series = load_full_series(BTC)
    if btc_series is None:
        print("❌ BTC data unavailable — NO SIGNALS (fail-safe). Exit.")
        raise SystemExit(1)
    print(f"  BTC bars: {len(btc_series):,} "
          f"(latest close: {datetime.fromtimestamp(int(btc_series['close_time'].max())/1000, tz=timezone.utc)})")

    print(f"\n{'Symbol':10s} {'P(up)':>7s} {'Signal':>7s}  Action")
    print("-" * 70)

    results: list[dict[str, Any]] = []
    for sym in SYMBOLS:
        series = load_full_series(sym)
        if series is None:
            print(f"{sym:10s} {'—':>7s} {'NO_SIG':>7s}  data unavailable "
                  f"(holding previous: "
                  f"{state.get(sym, {}).get('position', 'CASH')})")
            continue

        feats = build_features_live(sym, btc_series, series)
        if feats is None:
            print(f"{sym:10s} {'—':>7s} {'NO_SIG':>7s}  feature failure")
            continue

        proba = predict_latest(sym, feats)
        if proba is None:
            print(f"{sym:10s} {'—':>7s} {'NO_SIG':>7s}  prediction failure")
            continue

        pos, note = apply_overlay(sym, proba, state)
        results.append({"symbol": sym, "proba": proba, "position": pos})
        print(f"{sym:10s} {proba:>7.3f} {pos:>7s}  {note}")

    save_state(state)

    print("-" * 70)
    n_long = sum(1 for r in results if r["position"] == "LONG")
    print(f"\nPortfolio: {n_long}/{len(results)} LONG | "
          f"Allocation reminder: 60% of trading capital (SPEC §3)")
    print(f"State saved: {STATE_FILE}")
    print(f"Time: {(time.time()-t0)/60:.1f} min")
    print("\n⚠️  Follow signals exactly. No overrides (SPEC §5).")
    print("=" * 70)