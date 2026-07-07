"""
features/computation/feature_engine_1h.py

Feature engine for 1-HOUR timeframe.

Why a separate 1h engine:
    The 5m directional model showed no edge (Hurst 0.4 = noisy/mean-reverting).
    The horizon scan hinted 1h has more signal (53.5% val acc even with
    improperly resampled features). This computes PROPER 1h features:
    real 1h rolling windows, real 1h context — not resampled 5m.

Pipeline per symbol (1h):
    1. Load 1h (LTF) + 4h (HTF for MTF) + BTC 1h (cross-asset)
    2. Run all 14 feature modules on 1h bars
    3. Merge features, drop warmup
    4. Save to data/features_1h/

HTF ratio: 4h / 1h = 4 (for MTF volatility normalization)
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from core.logger import get_logger

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

FEATURE_ENGINE_1H_VERSION = "1.0.0"

WARMUP_BARS = 300      # same warmup (rolling windows)
HTF_RATIO_1H = 4       # 4h / 1h = 4


def compute_all_features_1h(
    symbol: str,
    raw_dir: str = "data/raw",
    btc_symbol: str = "BTCUSDT",
) -> Optional[pd.DataFrame]:
    """Compute all features on 1h bars for one symbol."""
    raw_path = Path(raw_dir)

    ltf_file = raw_path / f"{symbol}_1h.parquet"      # 1h as base
    htf_file = raw_path / f"{symbol}_4h.parquet"      # 4h as higher TF
    btc_file = raw_path / f"{btc_symbol}_1h.parquet"  # BTC 1h

    if not ltf_file.exists():
        logger.warning(f"{symbol}: 1h file missing")
        return None
    if not htf_file.exists():
        logger.warning(f"{symbol}: 4h file missing")
        return None
    if not btc_file.exists():
        logger.warning(f"{btc_symbol}: BTC 1h file missing")
        return None

    ltf = pd.read_parquet(ltf_file)
    htf = pd.read_parquet(htf_file)
    btc = pd.read_parquet(btc_file)

    df = ltf.copy()

    # Single-timeframe modules
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

    # MTF: 4h context into 1h (ratio 4)
    df = compute_mtf_features(df, htf, tf_ratio=HTF_RATIO_1H)

    # Cross-asset: BTC 1h
    df = compute_cross_asset_features(df, btc)

    # Drop warmup
    df = df.iloc[WARMUP_BARS:].reset_index(drop=True)

    del ltf, htf, btc
    gc.collect()
    return df


def process_symbol_1h(
    symbol: str,
    raw_dir: str = "data/raw",
    output_dir: str = "data/features_1h",
    btc_symbol: str = "BTCUSDT",
) -> dict[str, Any]:
    """Compute + save 1h features for one symbol."""
    t0 = time.time()
    result: dict[str, Any] = {
        "symbol": symbol, "status": "unknown",
        "bars": 0, "elapsed_s": 0.0,
    }
    try:
        df = compute_all_features_1h(symbol, raw_dir, btc_symbol)
        if df is None:
            result["status"] = "skipped"
            result["elapsed_s"] = time.time() - t0
            return result

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / f"{symbol}_1h.parquet"
        df.to_parquet(out_file, compression="snappy", index=False)

        result["status"] = "ok"
        result["bars"] = len(df)
        result["elapsed_s"] = time.time() - t0
        del df
        gc.collect()
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["elapsed_s"] = time.time() - t0
        logger.error(f"{symbol}: {e}")
    return result


if __name__ == "__main__":
    print("=" * 70)
    print(f"Feature Engine 1H Smoke Test (v{FEATURE_ENGINE_1H_VERSION})")
    print("=" * 70)

    symbol = "BTCUSDT"
    print(f"\nProcessing {symbol} on 1h timeframe...")
    print("  Base: 1h, HTF: 4h (ratio 4), Cross-asset: BTC 1h")

    result = process_symbol_1h(symbol)

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"\n  Status: {result['status']}")
    print(f"  Bars:   {result['bars']:,}")
    print(f"  Time:   {result['elapsed_s']:.1f}s")

    if result["status"] == "ok":
        out_file = Path("data/features_1h") / f"{symbol}_1h.parquet"
        saved = pd.read_parquet(out_file)
        print(f"\n  Saved: {out_file}")
        print(f"  Shape: {saved.shape}")

        # Feature NaN check
        feature_cols = [c for c in saved.columns if c not in {
            "open_time", "close_time", "open", "high", "low", "close",
            "volume", "quote_volume", "count", "taker_buy_volume",
            "taker_buy_quote_volume",
        }]
        nan_pct = saved[feature_cols].isnull().sum().sum() / (
            len(saved) * len(feature_cols)) * 100
        print(f"  Features: {len(feature_cols)}")
        print(f"  NaN: {nan_pct:.3f}%")
        print("\n✅ 1h feature engine working")
    else:
        print(f"\n❌ {result.get('error', 'failed')}")