"""
features/base/time_features.py

Module 3: Time-based features.

8 features capturing temporal patterns in 24/7 crypto markets.

Key insight:
    Crypto trades 24/7 but activity is NOT uniform:
    - US session (14:30-21:00 UTC): highest volume, trends
    - Asia session (00:00-08:00 UTC): different dynamics
    - Weekends: lower liquidity, more manipulation-prone
    - Hour-of-day seasonality is real and exploitable

Cyclical encoding (sin/cos):
    Hour 23 and hour 0 are ADJACENT in time but 23 apart numerically.
    Sin/cos encoding preserves cyclical adjacency for ML models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.logger import get_logger
from features.base.utils import series_from_array

logger = get_logger(__name__)

TIME_FEATURES_VERSION = "1.0.0"


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 8 time-based features from open_time.

    Args:
        df: DataFrame with open_time column (Unix ms)

    Returns:
        df with 8 new feature columns.

    Features:
        1. hour_sin          — sin(2π × hour/24)   cyclical hour
        2. hour_cos          — cos(2π × hour/24)   cyclical hour
        3. dow_sin           — sin(2π × dow/7)     cyclical day-of-week
        4. dow_cos           — cos(2π × dow/7)     cyclical day-of-week
        5. is_weekend        — 1 if Sat/Sun (UTC)
        6. is_us_session     — 1 if 14:30-21:00 UTC (NYSE overlap)
        7. is_asia_session   — 1 if 00:00-08:00 UTC
        8. is_month_end      — 1 if last 2 days of month (rebalancing flows)
    """
    if df.empty:
        raise ValueError("Cannot compute features on empty DataFrame")

    if "open_time" not in df.columns:
        raise ValueError("Missing required column: open_time")

    out = df.copy()

    # Convert Unix ms to datetime (UTC)
    dt = pd.to_datetime(out["open_time"], unit="ms", utc=True)

    hour = dt.dt.hour.to_numpy(dtype="float64")
    minute = dt.dt.minute.to_numpy(dtype="float64")
    dow = dt.dt.dayofweek.to_numpy(dtype="float64")  # 0=Mon, 6=Sun
    day = dt.dt.day.to_numpy(dtype="float64")
    days_in_month = dt.dt.days_in_month.to_numpy(dtype="float64")
    idx = out.index

    # Fractional hour (e.g., 14:30 → 14.5) for smoother cyclical encoding
    frac_hour = hour + minute / 60.0

    # Features 1-2: Cyclical hour encoding
    out["hour_sin"] = series_from_array(
        np.sin(2 * np.pi * frac_hour / 24.0), idx
    )
    out["hour_cos"] = series_from_array(
        np.cos(2 * np.pi * frac_hour / 24.0), idx
    )

    # Features 3-4: Cyclical day-of-week encoding
    out["dow_sin"] = series_from_array(
        np.sin(2 * np.pi * dow / 7.0), idx
    )
    out["dow_cos"] = series_from_array(
        np.cos(2 * np.pi * dow / 7.0), idx
    )

    # Feature 5: Weekend indicator (Sat=5, Sun=6)
    out["is_weekend"] = series_from_array(
        (dow >= 5).astype("float64"), idx
    )

    # Feature 6: US session (14:30 - 21:00 UTC — NYSE hours overlap)
    us_session = ((frac_hour >= 14.5) & (frac_hour < 21.0)).astype("float64")
    out["is_us_session"] = series_from_array(us_session, idx)

    # Feature 7: Asia session (00:00 - 08:00 UTC)
    asia_session = ((frac_hour >= 0.0) & (frac_hour < 8.0)).astype("float64")
    out["is_asia_session"] = series_from_array(asia_session, idx)

    # Feature 8: Month-end (last 2 days — institutional rebalancing)
    month_end = (day >= (days_in_month - 1)).astype("float64")
    out["is_month_end"] = series_from_array(month_end, idx)

    return out


TIME_FEATURE_NAMES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_us_session",
    "is_asia_session",
    "is_month_end",
]


if __name__ == "__main__":
    from pathlib import Path
    import time

    print("=" * 70)
    print(f"Time Features Smoke Test (v{TIME_FEATURES_VERSION})")
    print("=" * 70)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    test_file = PROJECT_ROOT / "data" / "raw" / "BTCUSDT_5m.parquet"

    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        exit(1)

    print("\nLoading BTCUSDT 5m data...")
    df = pd.read_parquet(test_file)
    df_test = df.iloc[-10000:].reset_index(drop=True).copy()
    print(f"  Test subset: {len(df_test):,} bars")

    print("\nComputing 8 time features...")
    t0 = time.time()
    result = compute_time_features(df_test)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s ({len(df_test)/elapsed:,.0f} bars/sec)")

    print("\nFeatures created:")
    for fname in TIME_FEATURE_NAMES:
        status = "✅" if fname in result.columns else "❌"
        print(f"  {status} {fname}")

    print("\nFeature statistics:")
    print("-" * 70)
    for fname in TIME_FEATURE_NAMES:
        values = result[fname].dropna()
        print(f"  {fname:18s} "
              f"mean={values.mean():+.4f}  "
              f"[{values.min():+.3f}, {values.max():+.3f}]")

    # Sanity checks
    print("\nSanity checks:")
    print("-" * 70)

    # sin² + cos² = 1 (cyclical encoding correctness)
    hour_check = (
        result["hour_sin"] ** 2 + result["hour_cos"] ** 2
    ).round(6)
    hour_ok = bool((hour_check == 1.0).all())
    print(f"  hour sin²+cos²=1:     {'✅' if hour_ok else '❌'}")

    dow_check = (
        result["dow_sin"] ** 2 + result["dow_cos"] ** 2
    ).round(6)
    dow_ok = bool((dow_check == 1.0).all())
    print(f"  dow sin²+cos²=1:      {'✅' if dow_ok else '❌'}")

    # Weekend ~2/7 = 28.6% of bars
    wk = result["is_weekend"].mean()
    print(f"  weekend fraction:     {wk*100:.1f}% (expected ≈28.6%)")

    # US session = 6.5/24 = 27.1%
    us = result["is_us_session"].mean()
    print(f"  US session fraction:  {us*100:.1f}% (expected ≈27.1%)")

    # Asia session = 8/24 = 33.3%
    asia = result["is_asia_session"].mean()
    print(f"  Asia session fraction: {asia*100:.1f}% (expected ≈33.3%)")

    # NaN check — time features should have ZERO NaN
    total_nan = sum(
        int(result[f].isnull().sum()) for f in TIME_FEATURE_NAMES
    )
    print(f"  Total NaN:            {total_nan} (expected 0)")

    print("\n✅ Time features module working — 8 features generated")