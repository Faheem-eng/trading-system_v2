"""
live/signal_generator.py

Phase 6.2: LIVE SIGNAL GENERATOR — the production entry point.

Run this after any 4h bar close (or via Task Scheduler every 4 hours).
For each symbol it:
    1. Loads historical 4h parquet + fetches fresh bars via ccxt
       (public endpoint, no API key needed)
    2. Builds all 116 features (same pipeline as validation)
    3. Trains the model on full labeled history (minus embargo)
    4. Predicts P(up) on the LAST COMPLETED bar only
    5. Applies the hysteresis overlay (0.55 in / 0.45 out) using
       persisted state from signals/state.json
    6. Auto-logs any position CHANGE to signals/paper_log.csv
    7. Prints the action sheet: LONG / CASH per symbol

SAFETY RULES (hard-coded, per docs/SYSTEM_SPEC.md sec.5):
    - Only COMPLETED bars are used (forming bar = look-ahead in live)
    - Fetch failure / data gap → NO_SIGNAL (hold previous state)
    - No threshold overrides, no manual parameters

ENCODING DISCIPLINE:
    All console output is ASCII-only. Windows console/file encoding
    (cp1252) cannot write emoji or the section symbol, and a scheduled
    run would crash on the final print. Production scripts must never
    fail on cosmetics.

UNIVERSE (5 symbols, walk-forward validated — SPEC v1.2.0):
    XRPUSDT, SOLUSDT, BTCUSDT (original, Phase 3-4)
    BNBUSDT, ADAUSDT (added 2026-07-11 after overlay walk-forward:
    each beats B&H in 3/5 folds with materially lower drawdowns;
    both were in the original 12 pre-registered validation symbols)
    MATICUSDT retired 2026-07-08 (Binance MATIC->POL rebrand killed
    fresh data; POL rejected: only ~1.5yr history, our 4.5yr
    validation does not transfer). See docs/OPERATIONS_RUNBOOK.md.

STATE FILE (signals/state.json):
    {"XRPUSDT": {"position": "CASH", "since": "...", "last_proba": 0.47}, ...}
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
from live.paper_logger import log_trade # type: ignore

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

SIGNAL_GENERATOR_VERSION = "1.1.0"

# ============================================================
# Locked parameters (SYSTEM_SPEC.md sec.3 — DO NOT MODIFY)
# ============================================================

TH_IN = 0.55                # enter LONG when P(up) crosses above
TH_OUT = 0.45               # exit to CASH when P(up) crosses below
EMBARGO = 25                # bars excluded before the prediction bar
WARMUP = 300                # feature warmup bars dropped

# Universe expanded 2026-07-11: BNB + ADA added after 4h walk-forward
# validation (4/5 folds IC>0) + overlay backtest (beats B&H).
# LINK tested and REJECTED (overlay -45.9% vs B&H -46.5% — both
# negative, no genuine value-add). AVAX/ETH/DOT/ATOM: not stable.
SYMBOLS = ["XRPUSDT", "SOLUSDT", "BTCUSDT", "BNBUSDT", "ADAUSDT"]
BTC = "BTCUSDT"             # cross-asset reference

STATE_FILE = Path("signals/state.json")
RAW_DIR = Path("data/raw")


# ============================================================
# Data: historical parquet + fresh ccxt bars
# ============================================================


def fetch_fresh_4h(symbol: str, since_ms: int) -> Optional[pd.DataFrame]:
    """
    Fetch 4h klines from Binance public API via ccxt, from since_ms to now.

    Only COMPLETED bars are returned (a bar whose close_time is still in
    the future is dropped — using a forming bar would be look-ahead bias
    in live trading, the exact thing we spent Phase 2-3 eliminating).

    Args:
        symbol:   e.g. "BTCUSDT"
        since_ms: fetch bars with open_time >= this Unix ms timestamp

    Returns:
        DataFrame of fresh bars, or None on any failure (fail-safe).
    """
    try:
        import ccxt  # optional dependency, imported lazily
        exchange = ccxt.binance({"enableRateLimit": True})
        rows: list[list[float]] = []
        cursor = since_ms

        while True:
            batch = exchange.fetch_ohlcv(
                symbol.replace("USDT", "/USDT"),
                timeframe="4h", since=cursor, limit=500,
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
            rows,
            columns=["open_time", "open", "high", "low", "close", "volume"],
        )
        df["close_time"] = df["open_time"] + 4 * 3600 * 1000 - 1
        # ccxt OHLCV lacks these; approximate for feature parity
        df["quote_volume"] = df["volume"] * df["close"]
        df["taker_buy_volume"] = df["volume"] * 0.5   # neutral assumption

        # Drop the forming (incomplete) bar
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        df = df[df["close_time"] < now_ms].reset_index(drop=True)
        return df

    except Exception as e:
        logger.error(f"{symbol}: fresh fetch failed - {e}")
        return None


def load_full_series(symbol: str) -> Optional[pd.DataFrame]:
    """
    Historical parquet + fresh ccxt bars, deduplicated and sorted.

    If the fresh fetch fails, we fall back to historical data and log a
    warning — the signal will be stale but the run will not crash
    (fail-safe: stale signal + warning beats no signal + exception).
    """
    hist_file = RAW_DIR / f"{symbol}_4h.parquet"
    if not hist_file.exists():
        logger.error(f"{symbol}: historical 4h parquet missing")
        return None

    hist = pd.read_parquet(hist_file)
    last_ms = int(hist["open_time"].max())
    fresh = fetch_fresh_4h(symbol, since_ms=last_ms + 1)

    if fresh is None or len(fresh) == 0:
        # Distinguish "no NEW bars yet" (normal when historical data
        # is already current) from a genuine fetch failure upstream.
        last_dt = datetime.fromtimestamp(last_ms / 1000, tz=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
        if age_hours < 8:
            logger.info(f"{symbol}: historical data already current "
                        f"(last bar {age_hours:.1f}h old) - no new bars")
        else:
            logger.warning(f"{symbol}: no fresh data and historical is "
                           f"{age_hours:.1f}h stale - investigate")
        fresh = pd.DataFrame()

    if len(fresh):
        common_cols = [c for c in hist.columns if c in fresh.columns]
        combined = pd.concat(
            [hist[common_cols], fresh[common_cols]], ignore_index=True
        )
    else:
        combined = hist.copy()

    combined = (
        combined.drop_duplicates(subset="open_time")
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    return combined


# ============================================================
# Features + model + prediction
# ============================================================


def build_features_live(symbol: str, btc_df: pd.DataFrame,
                        sym_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Full 14-module feature build on the combined 4h series.

    Identical module order to backtest/backtest_4h.py — any divergence
    here would silently invalidate the live signal versus what was
    validated. Never reorder or skip a module.
    """
    try:
        df = sym_df.copy()
        htf_daily = _resample_4h_to_1d(df)      # 1D HTF context

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
        df = compute_cross_asset_features(df, btc_df)

        return df.iloc[WARMUP:].reset_index(drop=True)

    except Exception as e:
        logger.error(f"{symbol}: feature build failed - {e}")
        return None


def predict_latest(symbol: str, df: pd.DataFrame) -> Optional[float]:
    """
    Train on full labeled history (minus embargo), predict the last bar.

    The embargo excludes the final EMBARGO bars from training because
    their triple-barrier labels overlap the prediction bar's future —
    the same leakage discipline used in every validation run.

    Returns:
        P(up) for the last completed bar, or None on failure.
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
            logger.error(f"{symbol}: insufficient training data "
                         f"({len(X_tr)} rows)")
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
        logger.error(f"{symbol}: prediction failed - {e}")
        return None


# ============================================================
# Hysteresis state machine
# ============================================================


def load_state() -> dict[str, Any]:
    """Load persisted positions. Empty dict on first run."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict[str, Any]) -> None:
    """Persist positions so hysteresis survives across runs."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def apply_overlay(symbol: str, proba: float,
                  state: dict[str, Any]) -> tuple[str, str]:
    """
    Hysteresis overlay: LONG above TH_IN, CASH below TH_OUT,
    otherwise HOLD the previous position.

    The dead band between TH_OUT and TH_IN is what prevents churn —
    without it, a probability oscillating around 0.5 would generate
    a switch (and a cost) on nearly every bar.

    Returns:
        (new_position, human-readable action note)
    """
    prev = state.get(symbol, {}).get("position", "CASH")
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if prev == "CASH" and proba > TH_IN:
        new_pos, note = "LONG", f"ENTER (P={proba:.3f} > {TH_IN})"
    elif prev == "LONG" and proba < TH_OUT:
        new_pos, note = "CASH", f"EXIT (P={proba:.3f} < {TH_OUT})"
    else:
        new_pos, note = prev, f"HOLD {prev} (P={proba:.3f} in band)"

    state[symbol] = {
        "position": new_pos,
        "since": now_str,
        "last_proba": round(proba, 4),
    }
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

    # BTC series first (cross-asset dependency for every symbol)
    print("\nLoading BTC 4h series (cross-asset base)...")
    btc_series = load_full_series(BTC)
    if btc_series is None:
        print("[X] BTC data unavailable - NO SIGNALS (fail-safe). Exit.")
        raise SystemExit(1)

    btc_last = datetime.fromtimestamp(
        int(btc_series["close_time"].max()) / 1000, tz=timezone.utc
    )
    print(f"  BTC bars: {len(btc_series):,} (latest close: {btc_last})")

    print(f"\n{'Symbol':10s} {'P(up)':>7s} {'Signal':>7s}  Action")
    print("-" * 70)

    results: list[dict[str, Any]] = []

    for sym in SYMBOLS:
        series = load_full_series(sym)
        if series is None:
            held = state.get(sym, {}).get("position", "CASH")
            print(f"{sym:10s} {'-':>7s} {'NO_SIG':>7s}  data unavailable "
                  f"(holding previous: {held})")
            continue

        feats = build_features_live(sym, btc_series, series)
        if feats is None:
            print(f"{sym:10s} {'-':>7s} {'NO_SIG':>7s}  feature failure")
            continue

        proba = predict_latest(sym, feats)
        if proba is None:
            print(f"{sym:10s} {'-':>7s} {'NO_SIG':>7s}  prediction failure")
            continue

        prev_pos = state.get(sym, {}).get("position", "CASH")
        pos, note = apply_overlay(sym, proba, state)
        results.append({"symbol": sym, "proba": proba, "position": pos})
        print(f"{sym:10s} {proba:>7.3f} {pos:>7s}  {note}")

        # Auto paper-log on position CHANGE only (Phase 7.1)
        if pos != prev_pos:
            action = "ENTER_LONG" if pos == "LONG" else "EXIT_LONG"
            trade = log_trade(sym, action, proba)
            price_str = trade["price"] if trade["price"] else "MANUAL-FILL"
            rt = f" | round-trip: {trade['net_return']}" \
                if trade["net_return"] else ""
            print(f"{'':10s} [LOG] {action} @ {price_str}{rt}")

    save_state(state)

    print("-" * 70)
    n_long = sum(1 for r in results if r["position"] == "LONG")
    print(f"\nPortfolio: {n_long}/{len(results)} LONG | "
          f"Allocation reminder: 60% of trading capital (SPEC sec.3)")
    print(f"State saved: {STATE_FILE}")
    print(f"Time: {(time.time() - t0) / 60:.1f} min")
    print("\n[!] Follow signals exactly. No overrides (SPEC sec.5).")
    print("=" * 70)