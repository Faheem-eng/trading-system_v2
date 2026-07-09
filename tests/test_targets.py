"""
tests/test_targets.py

Regression tests for the triple-barrier labeling module.

WHY THESE INVARIANTS:
    The labels ARE the supervision signal. A silent bug here
    (lookahead, wrong sign, degenerate classes) would poison every
    downstream result while every metric still "looks fine".
"""

from __future__ import annotations

import numpy as np  # noqa: F401
import pandas as pd

from ml.targets.triple_barrier import compute_triple_barrier


def test_labels_have_expected_schema(ohlcv: pd.DataFrame) -> None:
    """Labeling must add exactly the columns the pipeline consumes."""
    out = compute_triple_barrier(ohlcv, profit_mult=3.0, stop_mult=3.0,
                                 max_holding=20)
    for col in ["tb_label", "tb_return", "tb_exit_offset"]:
        assert col in out.columns, f"missing label column: {col}"
    assert len(out) == len(ohlcv), "labeling must not drop or add rows"


def test_label_values_are_valid(ohlcv: pd.DataFrame) -> None:
    """Labels must be in {-1, 0, 1} (loss / timeout / profit) or NaN."""
    out = compute_triple_barrier(ohlcv, profit_mult=3.0, stop_mult=3.0,
                                 max_holding=20)
    labels = out["tb_label"].dropna().unique()
    assert set(labels).issubset({-1.0, 0.0, 1.0}), \
        f"unexpected label values: {labels}"


def test_final_bars_are_unlabeled(ohlcv: pd.DataFrame) -> None:
    """
    The last `max_holding` bars cannot be labeled — their outcome
    window extends past the data. If they ARE labeled, the module is
    peeking or padding, both of which are lookahead bugs.
    """
    max_hold = 20
    out = compute_triple_barrier(ohlcv, profit_mult=3.0, stop_mult=3.0,
                                 max_holding=max_hold)
    tail = out["tb_label"].iloc[-max_hold:]
    assert tail.isna().all(), \
        "final bars must be NaN (their future window is unavailable)"


def test_symmetric_barriers_give_balanced_classes(ohlcv: pd.DataFrame) -> None:
    """
    With symmetric 3x/3x barriers on a near-random walk, profit and
    loss classes should be roughly balanced. A large skew signals a
    sign error or an asymmetric barrier computation.
    """
    out = compute_triple_barrier(ohlcv, profit_mult=3.0, stop_mult=3.0,
                                 max_holding=20)
    labels = out["tb_label"].dropna()
    profit = float((labels == 1.0).mean())
    loss = float((labels == -1.0).mean())
    assert abs(profit - loss) < 0.20, \
        f"symmetric barriers gave skewed classes: {profit:.2f}/{loss:.2f}"


def test_asymmetric_barriers_shift_hit_probability(ohlcv: pd.DataFrame) -> None:
    """
    A far profit target (4x) and near stop (2x) must produce FEWER
    profit hits than symmetric barriers. This is the exact effect that
    killed the asymmetric backtest — the test locks in that we measure
    it correctly rather than assuming free expectancy.
    """
    sym = compute_triple_barrier(ohlcv, profit_mult=3.0, stop_mult=3.0,
                                 max_holding=30)
    asym = compute_triple_barrier(ohlcv, profit_mult=4.0, stop_mult=2.0,
                                  max_holding=30)
    sym_profit = float((sym["tb_label"].dropna() == 1.0).mean())
    asym_profit = float((asym["tb_label"].dropna() == 1.0).mean())
    assert asym_profit < sym_profit, \
        "a farther target must hit less often (hit-probability law)"


def test_exit_offset_within_holding_window(ohlcv: pd.DataFrame) -> None:
    """Exit offsets must lie in [1, max_holding] — never zero, never beyond."""
    max_hold = 20
    out = compute_triple_barrier(ohlcv, profit_mult=3.0, stop_mult=3.0,
                                 max_holding=max_hold)
    offs = out["tb_exit_offset"].dropna().to_numpy()
    assert offs.min() >= 1, "exit offset must be at least 1 bar ahead"
    assert offs.max() <= max_hold, "exit offset exceeds holding window"


def test_labeling_is_past_invariant(ohlcv: pd.DataFrame) -> None:
    """
    LOOKAHEAD GUARD: corrupt the last 100 bars' prices. Labels on bars
    far in the past (before the corrupted region minus the holding
    window) must be UNCHANGED. If they shift, the labeler is reading
    data it shouldn't.
    """
    max_hold = 20
    clean = compute_triple_barrier(ohlcv, profit_mult=3.0, stop_mult=3.0,
                                   max_holding=max_hold)

    corrupted = ohlcv.copy()
    tail = corrupted.index[-100:]
    for col in ["open", "high", "low", "close"]:
        corrupted.loc[tail, col] = corrupted.loc[tail, col] * 5.0

    dirty = compute_triple_barrier(corrupted, profit_mult=3.0, stop_mult=3.0,
                                   max_holding=max_hold)

    safe_end = len(ohlcv) - 100 - max_hold - 1
    a = clean["tb_label"].iloc[:safe_end]
    b = dirty["tb_label"].iloc[:safe_end]
    pd.testing.assert_series_equal(a, b, check_names=False)