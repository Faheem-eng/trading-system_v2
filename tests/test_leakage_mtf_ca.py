"""
tests/test_leakage_mtf_ca.py

Corrupt-future leakage tests for the two multi-DataFrame feature modules.

WHY THIS FILE EXISTS:
    features/validation/leakage_test.py covers the 12 single-DataFrame
    modules (103 features) but cannot exercise the two modules that take
    a second DataFrame — multi_timeframe (7 features, needs an HTF frame)
    and cross_asset (6 features, needs the BTC frame). Those 13 features
    are exactly the difference between "116 in the model" and "103 tested",
    and resampling/alignment code is historically the #1 leak source in
    this project (Phase 2 caught an iloc[::12] resampling bug).

METHOD (same gold standard as leakage_test.py):
    A feature at bar t built from past data only must remain UNCHANGED
    when every bar AFTER a cutoff is replaced with random garbage —
    including the reference frame (HTF is re-resampled from the corrupted
    series; BTC future is corrupted independently). Any pre-cutoff change
    means future data is flowing backwards: look-ahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.backtest_4h import _resample_4h_to_1d
from features.multi_timeframe.mtf_features import (
    MTF_FEATURE_NAMES,
    compute_mtf_features,
)
from features.cross_asset.cross_asset_features import (
    CROSS_ASSET_FEATURE_NAMES,
    compute_cross_asset_features,
)

CUTOFF_FRAC = 0.5
TOLERANCE = 1e-8


def _corrupt_after(df: pd.DataFrame, cutoff: int,
                   seed: int = 7) -> pd.DataFrame:
    """Replace every bar after `cutoff` with random garbage (OHLC-valid)."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    m = len(d) - cutoff
    base = float(d["close"].iloc[cutoff - 1])
    fake_close = base * np.exp(np.cumsum(rng.normal(0, 0.05, m)))
    fake_open = np.concatenate([[base], fake_close[:-1]])
    spread = np.abs(rng.normal(0, 0.02, m)) * fake_close
    idx = d.index[cutoff:]
    d.loc[idx, "close"] = fake_close
    d.loc[idx, "open"] = fake_open
    d.loc[idx, "high"] = np.maximum(fake_open, fake_close) + spread
    d.loc[idx, "low"] = np.minimum(fake_open, fake_close) - spread
    for col in ("volume", "quote_volume", "taker_buy_volume"):
        if col in d.columns:
            d.loc[idx, col] = rng.lognormal(10, 1, m)
    return d


def _assert_past_invariant(base: pd.DataFrame, corrupted: pd.DataFrame,
                           names: list[str], cutoff: int) -> None:
    """Pre-cutoff feature values must be bit-for-bit stable (tolerance eps)."""
    leaking: list[str] = []
    for f in names:
        a = base[f].to_numpy(dtype="float64")[:cutoff]
        b = corrupted[f].to_numpy(dtype="float64")[:cutoff]
        both_nan = np.isnan(a) & np.isnan(b)
        nan_mismatch = np.isnan(a) != np.isnan(b)
        with np.errstate(invalid="ignore"):
            val_diff = np.abs(a - b) > TOLERANCE
        if (nan_mismatch | (~both_nan & val_diff)).any():
            leaking.append(f)
    assert not leaking, f"look-ahead bias (past changed): {leaking}"


def test_mtf_features_past_invariant(ohlcv: pd.DataFrame) -> None:
    """
    MTF features must not change in the past when the future is corrupted.
    The HTF frame is re-resampled FROM the corrupted series — exactly how
    the live path builds it (build_features_live / build_4h_features).
    """
    df = ohlcv.copy()
    cutoff = int(len(df) * CUTOFF_FRAC)

    base = compute_mtf_features(df, _resample_4h_to_1d(df), tf_ratio=6)
    df_c = _corrupt_after(df, cutoff)
    corr = compute_mtf_features(df_c, _resample_4h_to_1d(df_c), tf_ratio=6)

    _assert_past_invariant(base, corr, MTF_FEATURE_NAMES, cutoff)


def test_cross_asset_features_past_invariant(
        ohlcv: pd.DataFrame, ohlcv_btc: pd.DataFrame) -> None:
    """
    Cross-asset features must not change in the past when the future of
    BOTH the asset and the BTC reference is corrupted (independently —
    different seeds — so a leak through either frame is caught).
    """
    df, btc = ohlcv.copy(), ohlcv_btc.copy()
    cutoff = int(len(df) * CUTOFF_FRAC)

    base = compute_cross_asset_features(df, btc)
    corr = compute_cross_asset_features(
        _corrupt_after(df, cutoff, seed=11),
        _corrupt_after(btc, cutoff, seed=13),
    )

    _assert_past_invariant(base, corr, CROSS_ASSET_FEATURE_NAMES, cutoff)
