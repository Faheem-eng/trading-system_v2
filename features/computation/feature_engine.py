"""
features/computation/engine.py

Feature computation engine — orchestrates all 14 feature modules.

Combines 111 features from 14 modules into a single DataFrame per symbol,
then saves to the feature store (Parquet).

Data requirements per symbol:
    - 5m data (primary, all single-TF modules)
    - 1h data (for multi-timeframe module)
    - BTC 5m data (for cross-asset module)

Critical design:
    - Feature name uniqueness verified (no silent collisions)
    - Index preserved across all merges (length assertion)
    - Warmup NaN tracked and documented
    - Memory-efficient (one symbol at a time, float32 storage)
    - BTC cross-asset self-reference handled
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from core.logger import get_logger

# Import all 14 feature modules
from features.base.price_features import (
    compute_price_features, PRICE_FEATURE_NAMES,
)
from features.base.volume_features import (
    compute_volume_features, VOLUME_FEATURE_NAMES,
)
from features.base.time_features import (
    compute_time_features, TIME_FEATURE_NAMES,
)
from features.technical.momentum_features import (
    compute_momentum_features, MOMENTUM_FEATURE_NAMES,
)
from features.technical.trend_features import (
    compute_trend_features, TREND_FEATURE_NAMES,
)
from features.technical.volatility_features import (
    compute_volatility_features, VOLATILITY_FEATURE_NAMES,
)
from features.technical.volume_indicators import (
    compute_volume_indicators, VOLUME_INDICATOR_NAMES,
)
from features.microstructure.structure_features import (
    compute_structure_features, STRUCTURE_FEATURE_NAMES,
)
from features.multi_timeframe.mtf_features import (
    compute_mtf_features, MTF_FEATURE_NAMES,
)
from features.cross_asset.cross_asset_features import (
    compute_cross_asset_features, CROSS_ASSET_FEATURE_NAMES,
)
from features.regime.regime_features import (
    compute_regime_features, REGIME_FEATURE_NAMES,
)
from features.alpha_factors.alpha_features import (
    compute_alpha_features, ALPHA_FEATURE_NAMES,
)
from features.microstructure_quality.quality_features import (
    compute_quality_features, QUALITY_FEATURE_NAMES,
)
from features.risk_metrics.risk_features import (
    compute_risk_features, RISK_FEATURE_NAMES,
)

logger = get_logger(__name__)

ENGINE_VERSION = "1.0.0"

# All feature names combined (for validation)
ALL_FEATURE_NAMES = (
    PRICE_FEATURE_NAMES + VOLUME_FEATURE_NAMES + TIME_FEATURE_NAMES
    + MOMENTUM_FEATURE_NAMES + TREND_FEATURE_NAMES
    + VOLATILITY_FEATURE_NAMES + VOLUME_INDICATOR_NAMES
    + STRUCTURE_FEATURE_NAMES + MTF_FEATURE_NAMES
    + CROSS_ASSET_FEATURE_NAMES + REGIME_FEATURE_NAMES
    + ALPHA_FEATURE_NAMES + QUALITY_FEATURE_NAMES
    + RISK_FEATURE_NAMES
)

# OHLCV columns to keep in output
OHLCV_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "taker_buy_volume",
]

# Max warmup across all modules (regime Hurst=100, MA50, etc.)
MAX_WARMUP_BARS = 200


def verify_feature_uniqueness() -> None:
    """Assert no duplicate feature names across modules (fail-fast)."""
    seen: dict[str, int] = {}
    for name in ALL_FEATURE_NAMES:
        seen[name] = seen.get(name, 0) + 1
    duplicates = {k: v for k, v in seen.items() if v > 1}
    if duplicates:
        raise ValueError(f"Duplicate feature names: {duplicates}")


def compute_all_features(
    df_5m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_btc_5m: pd.DataFrame,
    is_btc: bool = False,
) -> pd.DataFrame:
    """
    Run all 14 modules on one symbol and merge into single DataFrame.

    Args:
        df_5m: Symbol's 5m data
        df_1h: Symbol's 1h data (for MTF)
        df_btc_5m: BTC 5m data (for cross-asset)
        is_btc: True if this symbol IS BTC (cross-asset self-ref handling)

    Returns:
        DataFrame with OHLCV + 111 features.
    """
    if df_5m.empty:
        raise ValueError("Empty 5m DataFrame")

    n_input = len(df_5m)
    out = df_5m.copy()

    # --- Single-timeframe modules (run on 5m) ---
    out = compute_price_features(out)
    out = compute_volume_features(out)
    out = compute_time_features(out)
    out = compute_momentum_features(out)
    out = compute_trend_features(out)
    out = compute_volatility_features(out)
    out = compute_volume_indicators(out)
    out = compute_structure_features(out)
    out = compute_regime_features(out)
    out = compute_alpha_features(out)
    out = compute_quality_features(out)
    out = compute_risk_features(out)

    # --- Multi-timeframe module (5m + 1h) ---
    out = compute_mtf_features(out, df_1h, tf_ratio=12)

    # --- Cross-asset module (5m + BTC 5m) ---
    if is_btc:
        # BTC vs itself: correlation=1, beta=1, residual=0
        for fname in CROSS_ASSET_FEATURE_NAMES:
            if "correlation" in fname or "beta" in fname:
                out[fname] = 1.0
            else:
                out[fname] = 0.0
    else:
        out = compute_cross_asset_features(out, df_btc_5m)

    # --- Verify integrity ---
    if len(out) != n_input:
        raise ValueError(
            f"Row count changed: {n_input} -> {len(out)} (index misalignment!)"
        )

    # Verify all features present
    missing = [f for f in ALL_FEATURE_NAMES if f not in out.columns]
    if missing:
        raise ValueError(f"Missing features after computation: {missing}")

    return out


def process_symbol(
    symbol: str,
    raw_dir: str = "data/raw",
    output_dir: str = "data/features",
    trim_warmup: bool = True,
) -> dict[str, Any]:
    """
    Full pipeline for one symbol: load, compute, save.

    Args:
        symbol: e.g. "ETHUSDT"
        raw_dir: Directory with raw Parquet files
        output_dir: Where to save feature Parquet
        trim_warmup: If True, drop first MAX_WARMUP_BARS rows (NaN warmup)

    Returns:
        Result dict with status, feature count, timing.
    """
    result: dict[str, Any] = {
        "symbol": symbol,
        "status": "unknown",
        "n_features": 0,
        "n_bars_out": 0,
        "elapsed_s": 0.0,
    }
    t0 = time.time()

    raw_path = Path(raw_dir)
    file_5m = raw_path / f"{symbol}_5m.parquet"
    file_1h = raw_path / f"{symbol}_1h.parquet"
    file_btc = raw_path / "BTCUSDT_5m.parquet"

    # Check files exist
    for f in [file_5m, file_1h, file_btc]:
        if not f.exists():
            result["status"] = "missing_file"
            result["error"] = f"Missing: {f.name}"
            return result

    try:
        df_5m = pd.read_parquet(file_5m)
        df_1h = pd.read_parquet(file_1h)
        df_btc = pd.read_parquet(file_btc)

        is_btc = symbol == "BTCUSDT"
        features = compute_all_features(df_5m, df_1h, df_btc, is_btc=is_btc)

        # Keep OHLCV + features only
        keep_cols = [c for c in OHLCV_COLUMNS if c in features.columns]
        keep_cols += [f for f in ALL_FEATURE_NAMES if f in features.columns]
        features = features[keep_cols]

        # Trim warmup period (rows with cascading NaN)
        if trim_warmup and len(features) > MAX_WARMUP_BARS:
            features = features.iloc[MAX_WARMUP_BARS:].reset_index(drop=True)

        # Convert feature columns to float32 (memory efficiency)
        for f in ALL_FEATURE_NAMES:
            if f in features.columns:
                features[f] = features[f].astype("float32")

        # Save
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / f"{symbol}_5m.parquet"
        features.to_parquet(out_file, compression="snappy", index=False)

        result["status"] = "ok"
        result["n_features"] = len(ALL_FEATURE_NAMES)
        result["n_bars_out"] = len(features)
        result["elapsed_s"] = time.time() - t0

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["elapsed_s"] = time.time() - t0

    return result


if __name__ == "__main__":
    print("=" * 70)
    print(f"Feature Computation Engine Smoke Test (v{ENGINE_VERSION})")
    print("=" * 70)

    # Verify feature uniqueness first
    print("\nVerifying feature name uniqueness...")
    verify_feature_uniqueness()
    print(f"  ✅ {len(ALL_FEATURE_NAMES)} unique feature names")

    # Test on ETHUSDT (has BTC cross-asset)
    test_symbol = "ETHUSDT"
    print(f"\nProcessing {test_symbol} (full 14-module pipeline)...")
    print("  This computes ALL 111 features on full dataset.")
    print("  Expect ~30-90 seconds...\n")

    result = process_symbol(test_symbol)

    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"\nSymbol:      {result['symbol']}")
    print(f"Status:      {result['status']}")

    if result["status"] == "ok":
        print(f"Features:    {result['n_features']}")
        print(f"Bars output: {result['n_bars_out']:,}")
        print(f"Time:        {result['elapsed_s']:.1f}s")

        # Load and verify saved file
        saved = pd.read_parquet(f"data/features/{test_symbol}_5m.parquet")
        print("\nSaved file verification:")
        print(f"  Shape: {saved.shape}")
        print(f"  Columns: {len(saved.columns)} "
              f"({len(OHLCV_COLUMNS)} OHLCV + {len(ALL_FEATURE_NAMES)} features)")

        # NaN check (should be minimal after warmup trim)
        total_cells = saved.shape[0] * len(ALL_FEATURE_NAMES)
        nan_cells = int(saved[ALL_FEATURE_NAMES].isnull().sum().sum())
        nan_pct = nan_cells / total_cells * 100
        print(f"  Feature NaN: {nan_cells:,} / {total_cells:,} "
              f"({nan_pct:.2f}%)")

        # Sample feature values
        print("\nSample features (last row):")
        last = saved.iloc[-1]
        for fname in ["log_return_1", "rsi_14", "macd_norm",
                      "btc_correlation_60", "hurst_100", "rolling_sharpe_60"]:
            if fname in saved.columns:
                print(f"  {fname:22s} = {last[fname]:+.5f}")

        print(f"\n✅ Engine working — {test_symbol} fully processed & saved")
    else:
        print(f"Error:       {result.get('error', 'unknown')}")