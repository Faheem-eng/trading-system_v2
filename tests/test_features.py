"""
tests/test_features.py

Regression tests for the feature engineering pipeline.

WHY THESE PROPERTIES MATTER:
    116 features are computed in a fixed order across 14 modules.
    Silent bugs here (wrong column count, NaN explosion, scale
    dependence) would corrupt the model's input without any error.

DESIGN: Uses the same module-by-module pipeline as
    backtest/backtest_4h.py:build_4h_features (the validated path),
    applied to synthetic OHLCV from conftest. Avoids depending on
    feature_engine.compute_all_features whose signature may vary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
from backtest.backtest_4h import _resample_4h_to_1d
from ml.models.binary_1h import _feature_cols

EXPECTED_FEATURE_COUNT = 116
MAX_NAN_FRACTION = 0.10     # at most 10% NaN after warmup
WARMUP = 300


def _featurize(df: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    """
    Full 14-module feature pipeline (identical order to build_4h_features).
    Any reordering here would catch regressions from accidental shuffles.
    """
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
    df = compute_cross_asset_features(df, btc)

    return df.iloc[WARMUP:].reset_index(drop=True)


def test_feature_count(ohlcv: pd.DataFrame,
                       ohlcv_btc: pd.DataFrame) -> None:
    """
    Exactly EXPECTED_FEATURE_COUNT feature columns must be present.
    A missing module or duplicate drop silently shrinks the input
    to the model without raising an error.
    """
    df = _featurize(ohlcv.copy(), ohlcv_btc)
    fcols = _feature_cols(df)
    assert len(fcols) == EXPECTED_FEATURE_COUNT, (
        f"expected {EXPECTED_FEATURE_COUNT} features, got {len(fcols)}"
    )


def test_no_inf_values(ohlcv: pd.DataFrame,
                       ohlcv_btc: pd.DataFrame) -> None:
    """
    Inf values propagate through LightGBM silently and corrupt every
    prediction. This pins the structure_position range-guard fix
    (the -2037 explosion caught in Phase 2).
    """
    df = _featurize(ohlcv.copy(), ohlcv_btc)
    fcols = _feature_cols(df)
    inf_cols = [
        c for c in fcols
        if np.isinf(df[c].to_numpy()).any()
    ]
    assert not inf_cols, f"Inf found in features: {inf_cols}"


def test_nan_fraction_acceptable(ohlcv: pd.DataFrame,
                                 ohlcv_btc: pd.DataFrame) -> None:
    """
    After warmup, NaN fraction per feature must stay low.
    High NaN means a rolling window is too wide or a module failed
    silently — LightGBM imputes NaN internally so there is no error.
    """
    df = _featurize(ohlcv.copy(), ohlcv_btc)
    fcols = _feature_cols(df)
    bad = [
        (col, float(df[col].isna().mean()))
        for col in fcols
        if float(df[col].isna().mean()) > MAX_NAN_FRACTION
    ]
    assert not bad, (
        f"High NaN fraction (>{MAX_NAN_FRACTION*100:.0f}%): "
        + ", ".join(f"{c}:{f:.2f}" for c, f in bad)
    )


def test_scale_invariance(ohlcv: pd.DataFrame,
                          ohlcv_scaled: pd.DataFrame,
                          ohlcv_btc: pd.DataFrame) -> None:
    """
    Normalized features (returns, RSI, z-scores, ATR-ratios) must be
    IDENTICAL whether price is $100 or $1000.

    KNOWN EXCEPTIONS (intentionally scale-dependent — not bugs):
        ease_of_movement: uses raw price range / volume (not normalized)
        amihud_illiquidity: |return| / dollar_volume — dollar_volume
            scales with price by design (market impact measure)

    These are MICROSTRUCTURE features where the price level IS the
    signal (impact per dollar traded). They are kept in the model
    because the model is always trained and predicted at the same
    price scale. They are excluded from this invariance test.

    Tolerance 1e-4 allows for float rounding across 14 modules.
    """
    # Features that are intentionally scale-dependent (by design)
    SCALE_DEPENDENT_EXCEPTIONS = {
        "ease_of_movement",
        "amihud_illiquidity",
    }

    df1 = _featurize(ohlcv.copy(), ohlcv_btc)
    df2 = _featurize(ohlcv_scaled.copy(), ohlcv_btc)
    fcols = _feature_cols(df1)

    failed: list[tuple[str, float]] = []
    for col in fcols:
        if col in SCALE_DEPENDENT_EXCEPTIONS:
            continue   # intentionally scale-dependent, skip
        a = df1[col].to_numpy(dtype="float64")
        b = df2[col].to_numpy(dtype="float64")
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() == 0:
            continue
        max_diff = float(np.abs(a[mask] - b[mask]).max())
        if max_diff > 1e-4:     # generous tolerance for float ops
            failed.append((col, max_diff))

    if failed:
        details = "\n".join(
            f"  {c}: max_diff={d:.2e}" for c, d in failed
        )
        pytest.fail(
            f"Unexpected scale-dependent features "
            f"(exceptions: {SCALE_DEPENDENT_EXCEPTIONS}):\n{details}"
        )