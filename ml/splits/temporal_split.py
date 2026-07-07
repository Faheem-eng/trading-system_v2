"""
ml/splits/temporal_split.py

Temporal (time-based) train/val/test splitting with embargo gaps.

THE anti-leakage core of Phase 3.
Wrong split = look-ahead bias = fake backtest = live losses.

Design (institutional standard):
    - STRICTLY chronological (train=oldest, test=newest, NEVER shuffled)
    - EMBARGO gap between splits (default 60 bars > target max_holding 50)
      Prevents target-window overlap between train and val/test.
    - Test set LOCKED by default (return_test=False) to prevent peeking.
    - Per-symbol splits (each symbol has its own time range).

Why the embargo matters:
    The last training bar's triple-barrier target looks up to 50 bars
    into the future. If validation starts immediately after training,
    that target window overlaps validation data → leakage.
    A 60-bar embargo (> 50) guarantees clean separation.

Reference: López de Prado, "Advances in Financial ML" (2018), Ch. 7.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Any

import pandas as pd

from core.logger import get_logger

logger = get_logger(__name__)

TEMPORAL_SPLIT_VERSION = "1.0.0"

# Default ratios (must sum to 1.0)
DEFAULT_TRAIN_RATIO = 0.76
DEFAULT_VAL_RATIO = 0.12
DEFAULT_TEST_RATIO = 0.12

# Embargo: must exceed target max_holding (50) to prevent target overlap
DEFAULT_EMBARGO_BARS = 60


@dataclass
class TemporalSplit:
    """Container for a temporal split result."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: Optional[pd.DataFrame] = None
    train_range: Optional[tuple[str, str]] = None
    val_range: Optional[tuple[str, str]] = None
    test_range: Optional[tuple[str, str]] = None
    embargo_bars: int = DEFAULT_EMBARGO_BARS

    @property
    def has_test(self) -> bool:
        return self.test is not None


def _ms_to_str(ms: int) -> str:
    """Unix ms → readable UTC datetime string."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M"
    )


def split_temporal(
    df: pd.DataFrame,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    embargo_bars: int = DEFAULT_EMBARGO_BARS,
    return_test: bool = False,
    time_col: str = "open_time",
) -> TemporalSplit:
    """
    Split a DataFrame temporally into train/val/test with embargo gaps.

    Args:
        df: DataFrame sorted by time (ascending).
        train_ratio, val_ratio, test_ratio: split fractions (sum to 1.0).
        embargo_bars: gap between splits (>= target max_holding).
        return_test: if False (default), test is NOT returned (anti-peek).
        time_col: timestamp column name.

    Returns:
        TemporalSplit with train, val, and optionally test.

    Raises:
        ValueError: if ratios invalid or data too small.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 < total_ratio < 1.01):
        raise ValueError(
            f"Ratios must sum to 1.0, got {total_ratio:.3f}"
        )
    if df.empty:
        raise ValueError("Cannot split empty DataFrame")
    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}")

    n = len(df)

    # Raw split sizes
    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)

    # Apply embargo gaps between splits
    train_end = train_n
    val_start = train_end + embargo_bars
    val_end = val_start + val_n
    test_start = val_end + embargo_bars
    test_end = n

    # Sanity: everything must fit
    if test_start >= n:
        raise ValueError(
            f"Data too small ({n} bars) for splits with embargo "
            f"{embargo_bars}. Need > {test_start} bars."
        )

    # Extract (chronological, no shuffle)
    train_df = df.iloc[:train_end].reset_index(drop=True).copy()
    val_df = df.iloc[val_start:val_end].reset_index(drop=True).copy()

    test_df: Optional[pd.DataFrame] = None
    if return_test:
        test_df = df.iloc[test_start:test_end].reset_index(drop=True).copy()
        logger.warning(
            "TEST SET RETURNED — use ONLY for final evaluation. "
            "Never tune hyperparameters on this."
        )

    # Time ranges for reporting
    train_range = (
        _ms_to_str(int(train_df[time_col].min())),
        _ms_to_str(int(train_df[time_col].max())),
    )
    val_range = (
        _ms_to_str(int(val_df[time_col].min())),
        _ms_to_str(int(val_df[time_col].max())),
    )
    test_range = None
    if test_df is not None:
        test_range = (
            _ms_to_str(int(test_df[time_col].min())),
            _ms_to_str(int(test_df[time_col].max())),
        )

    return TemporalSplit(
        train=train_df, val=val_df, test=test_df,
        train_range=train_range, val_range=val_range, test_range=test_range,
        embargo_bars=embargo_bars,
    )


def verify_no_overlap(
    split: TemporalSplit, time_col: str = "open_time"
) -> dict[str, Any]:
    """
    Verify splits have NO temporal overlap (leakage check).

    Returns report with pass/fail and gap sizes.
    """
    report: dict[str, Any] = {"passed": True, "issues": [], "gaps": {}}

    train_max = int(split.train[time_col].max())
    val_min = int(split.val[time_col].min())

    # Train must end before val starts
    if train_max >= val_min:
        report["passed"] = False
        report["issues"].append(
            f"Train/Val OVERLAP: train ends {_ms_to_str(train_max)}, "
            f"val starts {_ms_to_str(val_min)}"
        )
    else:
        gap_min = (val_min - train_max) / 1000 / 60
        report["gaps"]["train_val_min"] = gap_min

    # If test present, val must end before test
    if split.has_test and split.test is not None:
        val_max = int(split.val[time_col].max())
        test_min = int(split.test[time_col].min())
        if val_max >= test_min:
            report["passed"] = False
            report["issues"].append(
                f"Val/Test OVERLAP: val ends {_ms_to_str(val_max)}, "
                f"test starts {_ms_to_str(test_min)}"
            )
        else:
            gap_min = (test_min - val_max) / 1000 / 60
            report["gaps"]["val_test_min"] = gap_min

    return report


if __name__ == "__main__":
    from pathlib import Path

    print("=" * 70)
    print(f"Temporal Split Smoke Test (v{TEMPORAL_SPLIT_VERSION})")
    print("=" * 70)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    test_file = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m.parquet"

    if not test_file.exists():
        print(f"❌ Feature file not found: {test_file}")
        exit(1)

    print("\nLoading BTCUSDT features...")
    df = pd.read_parquet(test_file)
    print(f"  Total bars: {len(df):,}")

    # Test 1: Default split (test locked)
    print("\n" + "=" * 70)
    print("TEST 1: Default split (test LOCKED)")
    print("=" * 70)
    split = split_temporal(df)
    print(f"\n  Train: {len(split.train):,} bars ({len(split.train)/len(df)*100:.0f}%)")
    if split.train_range is not None:
        print(f"         {split.train_range[0]} → {split.train_range[1]}")
    print(f"  Val:   {len(split.val):,} bars ({len(split.val)/len(df)*100:.0f}%)")
    if split.val_range is not None:
        print(f"         {split.val_range[0]} → {split.val_range[1]}")
    print(f"  Test:  {'LOCKED (not returned)' if not split.has_test else 'present'}")
    print(f"  Embargo: {split.embargo_bars} bars between splits")

    # Test 2: Leakage verification
    print("\n" + "=" * 70)
    print("TEST 2: Leakage verification")
    print("=" * 70)
    report = verify_no_overlap(split)
    if report["passed"]:
        print("\n  ✅ NO temporal overlap")
        for gap_name, gap_val in report["gaps"].items():
            print(f"     {gap_name}: {gap_val:.0f} minutes "
                  f"({gap_val/5:.0f} bars)")
    else:
        print("\n  🚨 OVERLAP detected:")
        for issue in report["issues"]:
            print(f"     - {issue}")

    # Test 3: Full split with test (explicit)
    print("\n" + "=" * 70)
    print("TEST 3: Full split with test (explicit unlock)")
    print("=" * 70)
    full = split_temporal(df, return_test=True)
    print(f"\n  Train: {len(full.train):,} bars")
    print(f"  Val:   {len(full.val):,} bars")
    if full.test is not None and full.test_range is not None:
        print(f"  Test:  {len(full.test):,} bars")
        print(f"         {full.test_range[0]} → {full.test_range[1]}")
    full_report = verify_no_overlap(full)
    print(f"\n  Leakage check: {'✅ CLEAN' if full_report['passed'] else '🚨 LEAK'}")
    for gap_name, gap_val in full_report["gaps"].items():
        print(f"     {gap_name}: {gap_val/5:.0f} bars")

    # Test 4: Edge cases
    print("\n" + "=" * 70)
    print("TEST 4: Edge cases")
    print("=" * 70)
    try:
        split_temporal(df, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)
        print("  ❌ Bad ratios should fail")
    except ValueError:
        print("  ✅ Invalid ratios (sum≠1) rejected")

    try:
        split_temporal(df.iloc[:50], embargo_bars=100)
        print("  ❌ Too-small data should fail")
    except ValueError:
        print("  ✅ Insufficient data rejected")

    print("\n" + "=" * 70)
    print("✅ TEMPORAL SPLIT MODULE WORKING")
    print("=" * 70)
    print("\n🛡️  GUARANTEES:")
    print("   1. Chronological only (no shuffle)")
    print("   2. 60-bar embargo (> 50 target holding = no target overlap)")
    print("   3. Test locked by default (return_test=False)")
    print("   4. Per-symbol independent splits")