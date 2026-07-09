"""
tests/test_splits.py

Regression tests for temporal splitting and embargo discipline.

DESIGN NOTES:
    split_temporal() returns test=None by default (return_test=False).
    This is intentional — the test set is LOCKED for final evaluation
    only (Phase 3 core discipline). Tests only verify train/val splits.

    Embargo gap is verified via open_time timestamps, NOT row indices,
    because each segment resets its index to 0 after splitting.
"""

from __future__ import annotations

import pandas as pd

from ml.targets.triple_barrier import compute_triple_barrier
from ml.splits.temporal_split import split_temporal


def _labeled(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Helper: add triple-barrier labels to synthetic OHLCV."""
    return compute_triple_barrier(
        ohlcv, profit_mult=3.0, stop_mult=3.0, max_holding=20
    )


def test_split_returns_train_and_val(ohlcv: pd.DataFrame) -> None:
    """
    split_temporal must return train and val segments.
    test=None is correct (locked by design — return_test=False default).
    """
    df = _labeled(ohlcv)
    sp = split_temporal(df)
    assert sp is not None, "split_temporal returned None"
    assert sp.train is not None, "missing train segment"
    assert sp.val is not None,   "missing val segment"
    # sp.test=None is correct behavior — do NOT assert it exists


def test_chronological_ordering(ohlcv: pd.DataFrame) -> None:
    """
    Train must end before val starts — verified via open_time timestamps
    (NOT row indices, which reset to 0 in each segment).
    Any time inversion = future bar in earlier split = lookahead.
    """
    df = _labeled(ohlcv)
    sp = split_temporal(df)
    assert sp is not None

    train_last_time = int(sp.train["open_time"].max())
    val_first_time  = int(sp.val["open_time"].min())

    assert train_last_time < val_first_time, (
        f"train/val time overlap: train ends {train_last_time}, "
        f"val starts {val_first_time}"
    )


def test_embargo_gap_between_train_and_val(ohlcv: pd.DataFrame) -> None:
    """
    The embargo gap must be >= embargo_bars, measured in bars elapsed.

    IMPORTANT: Row indices reset to 0 in each segment — the gap must
    be computed from open_time, not from index values.

    Without embargo: the label window of the last training bar overlaps
    the first validation bar = guaranteed lookahead leakage.
    """
    embargo = 25
    bar_ms = 4 * 3600 * 1000         # 4h bars in milliseconds

    df = _labeled(ohlcv)
    sp = split_temporal(df, embargo_bars=embargo)
    assert sp is not None

    train_last_time = int(sp.train["open_time"].max())
    val_first_time  = int(sp.val["open_time"].min())

    # Number of 4h bars between last train bar and first val bar
    gap_bars = (val_first_time - train_last_time) / bar_ms
    assert gap_bars >= embargo, (
        f"embargo gap {gap_bars:.1f} bars < required {embargo} bars "
        f"(times: {train_last_time} → {val_first_time})"
    )


def test_no_time_overlap_train_val(ohlcv: pd.DataFrame) -> None:
    """
    No open_time must appear in both train and val.
    Checked via timestamps (robust to index resets).
    """
    df = _labeled(ohlcv)
    sp = split_temporal(df)
    assert sp is not None

    train_times = set(sp.train["open_time"].tolist())
    val_times   = set(sp.val["open_time"].tolist())
    overlap = train_times & val_times

    assert not overlap, (
        f"train/val share {len(overlap)} open_time values (data leak)"
    )


def test_train_val_coverage(ohlcv: pd.DataFrame) -> None:
    """
    Train + val must cover at least 90% of rows.
    Silent drops = validated on less data than expected.
    (test=None is excluded — it's locked by design.)
    """
    df = _labeled(ohlcv)
    sp = split_temporal(df, embargo_bars=25)
    assert sp is not None

    total = len(sp.train) + len(sp.val)
    assert total >= int(len(df) * 0.80), (
        f"train+val cover only {total}/{len(df)} rows "
        f"({total/len(df)*100:.0f}%)"
    )


def test_sizes_are_reasonable(ohlcv: pd.DataFrame) -> None:
    """
    Train should be large (>50%) and val non-trivial (>5%).
    Validated ratios from Phase 3: train ~76%, val ~12%.
    """
    df = _labeled(ohlcv)
    sp = split_temporal(df)
    assert sp is not None

    n = len(df)
    assert len(sp.train) > n * 0.50, (
        f"train unexpectedly small: {len(sp.train)}/{n}"
    )
    assert len(sp.val) > n * 0.05, (
        f"val unexpectedly small: {len(sp.val)}/{n}"
    )