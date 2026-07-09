"""
tests/conftest.py

Shared fixtures for the regression test suite.

DESIGN PRINCIPLE — HERMETIC TESTS:
    Tests must never depend on parquet files, network access, or the
    state of data/. They run on synthetic OHLCV built here, so
    `pytest` works on a fresh clone, offline, in 30 seconds.

The synthetic series is deliberately REALISTIC (geometric random walk
with volatility clustering and a mild uptrend) so that feature modules
exercise their real code paths — flat or constant prices would hide
division-by-zero and degenerate-window bugs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

RNG_SEED = 42
N_BARS = 2000          # enough for 300-bar warmup + rolling windows
BAR_MS = 4 * 3600 * 1000   # 4h in milliseconds


def _synthetic_ohlcv(n: int = N_BARS, seed: int = RNG_SEED,
                     start_price: float = 100.0) -> pd.DataFrame:
    """
    Geometric random walk with volatility clustering (GARCH-like).

    Produces the exact column contract the feature modules expect:
        open_time, close_time, open, high, low, close, volume,
        quote_volume, taker_buy_volume
    """
    rng = np.random.default_rng(seed)

    # Volatility clustering: vol follows its own AR(1) process
    vol = np.zeros(n, dtype="float64")
    vol[0] = 0.01
    for i in range(1, n):
        vol[i] = 0.9 * vol[i - 1] + 0.1 * abs(rng.normal(0, 0.015))
    vol = np.clip(vol, 0.002, 0.08)

    drift = 0.0002                       # mild uptrend
    log_ret = drift + rng.normal(0, 1, n) * vol
    close = start_price * np.exp(np.cumsum(log_ret))

    # Intrabar range proportional to that bar's volatility
    spread = close * vol * 1.5
    high = close + np.abs(rng.normal(0, 1, n)) * spread
    low = close - np.abs(rng.normal(0, 1, n)) * spread
    open_ = np.concatenate([[start_price], close[:-1]])

    # Guarantee OHLC ordering invariants
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])

    volume = rng.lognormal(mean=10.0, sigma=0.5, size=n)
    open_time = np.arange(n, dtype="int64") * BAR_MS + 1_600_000_000_000

    return pd.DataFrame({
        "open_time": open_time,
        "close_time": open_time + BAR_MS - 1,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "quote_volume": volume * close,
        "taker_buy_volume": volume * 0.5,
    })


@pytest.fixture(scope="session")
def ohlcv() -> pd.DataFrame:
    """Primary synthetic 4h series (the 'symbol under test')."""
    return _synthetic_ohlcv()


@pytest.fixture(scope="session")
def ohlcv_btc() -> pd.DataFrame:
    """Second correlated series, used as the BTC cross-asset reference."""
    return _synthetic_ohlcv(seed=RNG_SEED + 1, start_price=30000.0)


@pytest.fixture(scope="session")
def ohlcv_scaled() -> pd.DataFrame:
    """
    Same path as `ohlcv` but with all prices multiplied by 10.

    Used to assert SCALE INVARIANCE: normalized features (returns,
    RSI, z-scores) must be identical regardless of price level. A
    feature that changes when the price is denominated differently
    is leaking absolute price into the model.
    """
    df = _synthetic_ohlcv()
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col] * 10.0
    df["quote_volume"] = df["volume"] * df["close"]
    return df