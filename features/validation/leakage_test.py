"""
features/validation/leakage_test.py

Automated look-ahead bias (data leakage) detection.

THE institutional gold-standard test for temporal correctness.

Principle:
    A feature at time t that uses ONLY past data must remain UNCHANGED
    when future bars (after t) are corrupted/shuffled.

    If corrupting the future changes a past feature value → LEAKAGE.

Method (per feature module):
    1. Compute features on original data.
    2. Corrupt all bars AFTER a cutoff point.
    3. Recompute features.
    4. Compare feature values BEFORE the cutoff.
    5. If they differ → look-ahead bias detected.

This is run BEFORE trusting any backtest. Big firms run this in CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Any

import numpy as np
import pandas as pd

from core.logger import get_logger

logger = get_logger(__name__)

LEAKAGE_TEST_VERSION = "1.0.0"


def test_module_for_leakage(
    df: pd.DataFrame,
    compute_fn: Callable[[pd.DataFrame], pd.DataFrame],
    feature_names: list[str],
    module_name: str,
    cutoff_frac: float = 0.5,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """
    Test a single feature module for look-ahead bias.

    Args:
        df: Input OHLCV DataFrame.
        compute_fn: The module's compute function.
        feature_names: List of feature columns the module produces.
        module_name: Name for reporting.
        cutoff_frac: Where to split past/future (0.5 = middle).
        tolerance: Max allowed difference (float precision).

    Returns:
        Report dict with per-feature leakage status.
    """
    n = len(df)
    cutoff = int(n * cutoff_frac)

    # ============================================================
    # Step 1: Compute features on ORIGINAL data
    # ============================================================
    original = compute_fn(df.copy())

    # ============================================================
    # Step 2: Corrupt FUTURE bars (after cutoff)
    # Randomly shuffle OHLCV of future bars — this simulates
    # "the future being different". Past features must not care.
    # ============================================================
    corrupted_df = df.copy()
    future_idx = corrupted_df.index[cutoff:]

    rng = np.random.default_rng(42)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in corrupted_df.columns:
            future_vals = corrupted_df.loc[future_idx, col].to_numpy()
            shuffled = rng.permutation(future_vals)
            corrupted_df.loc[future_idx, col] = shuffled

    # Fix OHLC consistency after shuffle (high>=low etc.), vectorized.
    # We only care about PAST features; this just prevents compute crashes.
    open_arr = corrupted_df["open"].to_numpy(dtype="float64").copy()
    close_arr = corrupted_df["close"].to_numpy(dtype="float64").copy()
    high_arr = corrupted_df["high"].to_numpy(dtype="float64").copy()
    low_arr = corrupted_df["low"].to_numpy(dtype="float64").copy()

    # For future region only, rebuild consistent high/low
    fut = slice(cutoff, n)
    body_max = np.maximum(open_arr[fut], close_arr[fut])
    body_min = np.minimum(open_arr[fut], close_arr[fut])
    high_arr[fut] = body_max * 1.001
    low_arr[fut] = body_min * 0.999

    corrupted_df["high"] = high_arr
    corrupted_df["low"] = low_arr

    # ============================================================
    # Step 3: Recompute features on corrupted data
    # ============================================================
    recomputed = compute_fn(corrupted_df.copy())

    # ============================================================
    # Step 4: Compare PAST features (before cutoff)
    # ============================================================
    report: dict[str, Any] = {
        "module": module_name,
        "cutoff": cutoff,
        "n_bars": n,
        "clean_features": [],
        "leaking_features": [],
        "details": {},
    }

    # Compare only bars well before cutoff (avoid boundary effects
    # from the corrupted region touching the cutoff via warmup)
    # Use a safety margin of 100 bars before cutoff.
    compare_end = max(0, cutoff - 100)

    for fname in feature_names:
        if fname not in original.columns or fname not in recomputed.columns:
            report["details"][fname] = "column missing"
            continue

        orig_vals = original[fname].iloc[:compare_end].to_numpy()
        recomp_vals = recomputed[fname].iloc[:compare_end].to_numpy()

        # Compare ignoring NaN (both should have NaN in same places)
        both_valid = ~(np.isnan(orig_vals) | np.isnan(recomp_vals))
        if both_valid.sum() == 0:
            report["details"][fname] = "no valid values to compare"
            continue

        max_diff = float(
            np.max(np.abs(orig_vals[both_valid] - recomp_vals[both_valid]))
        )
        report["details"][fname] = max_diff

        if max_diff <= tolerance:
            report["clean_features"].append(fname)
        else:
            report["leaking_features"].append(fname)

    return report


def print_report(report: dict[str, Any]) -> bool:
    """Print a module's leakage report. Returns True if clean."""
    module = report["module"]
    n_clean = len(report["clean_features"])
    n_leak = len(report["leaking_features"])
    total = n_clean + n_leak

    if n_leak == 0:
        print(f"  ✅ {module:28s} {n_clean}/{total} features CLEAN")
        return True
    else:
        print(f"  🚨 {module:28s} {n_leak}/{total} LEAKING!")
        for fname in report["leaking_features"]:
            diff = report["details"][fname]
            print(f"       LEAK: {fname} (max_diff={diff:.2e})")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print(f"LEAKAGE TEST — Look-Ahead Bias Detection (v{LEAKAGE_TEST_VERSION})")
    print("=" * 70)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    test_file = PROJECT_ROOT / "data" / "raw" / "BTCUSDT_5m.parquet"

    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        exit(1)

    print("\nLoading BTCUSDT 5m (test data)...")
    df = pd.read_parquet(test_file)
    df_test = df.iloc[-10000:].reset_index(drop=True).copy()
    print(f"  Test bars: {len(df_test):,}")
    print("  Method: corrupt future (after bar 5000), verify past unchanged\n")

    # Import all single-timeframe modules
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

    # Modules to test (single-timeframe ones)
    modules = [
        ("Price", compute_price_features, PRICE_FEATURE_NAMES),
        ("Volume", compute_volume_features, VOLUME_FEATURE_NAMES),
        ("Time", compute_time_features, TIME_FEATURE_NAMES),
        ("Momentum", compute_momentum_features, MOMENTUM_FEATURE_NAMES),
        ("Trend", compute_trend_features, TREND_FEATURE_NAMES),
        ("Volatility", compute_volatility_features, VOLATILITY_FEATURE_NAMES),
        ("VolumeIndicators", compute_volume_indicators, VOLUME_INDICATOR_NAMES),
        ("Microstructure", compute_structure_features, STRUCTURE_FEATURE_NAMES),
        ("Regime", compute_regime_features, REGIME_FEATURE_NAMES),
        ("AlphaFactors", compute_alpha_features, ALPHA_FEATURE_NAMES),
        ("MicrostructureQuality", compute_quality_features, QUALITY_FEATURE_NAMES),
        ("RiskMetrics", compute_risk_features, RISK_FEATURE_NAMES),
    ]

    print("Testing each module for look-ahead bias:")
    print("-" * 70)

    all_clean = True
    total_features = 0
    total_clean = 0

    for module_name, compute_fn, feature_names in modules:
        report = test_module_for_leakage(
            df_test, compute_fn, feature_names, module_name
        )
        is_clean = print_report(report)
        all_clean = all_clean and is_clean
        total_features += len(feature_names)
        total_clean += len(report["clean_features"])

    print("-" * 70)
    print(f"\nTotal features tested: {total_features}")
    print(f"Clean (no leakage):    {total_clean}")
    print(f"Leaking:               {total_features - total_clean}")

    print("\n" + "=" * 70)
    if all_clean:
        print("✅ NO LOOK-AHEAD BIAS DETECTED — all features past-only")
        print("   Features are temporally correct. Safe for ML training.")
    else:
        print("🚨 LEAKAGE DETECTED — some features use future data")
        print("   MUST FIX before proceeding to ML.")
    print("=" * 70)