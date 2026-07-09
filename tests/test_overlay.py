"""
tests/test_overlay.py

Regression tests for the hysteresis state machine — the component
that actually decides whether real capital is exposed.

WHY THIS MATTERS MOST:
    Every backtest number (+145.96%, DD -49.1%) assumed this exact
    behavior. If the live state machine drifts from the simulated one,
    the validation is void. These tests pin the contract.
"""

from __future__ import annotations

from typing import Any

from live.signal_generator import apply_overlay, TH_IN, TH_OUT


def _fresh_state() -> dict[str, Any]:
    return {}


def test_thresholds_match_locked_spec() -> None:
    """SPEC sec.3 locks these. A silent change invalidates the backtest."""
    assert TH_IN == 0.55
    assert TH_OUT == 0.45


def test_enters_long_above_entry_threshold() -> None:
    state = _fresh_state()
    pos, _ = apply_overlay("TESTUSDT", 0.60, state)
    assert pos == "LONG"


def test_stays_cash_inside_band_from_cash() -> None:
    """0.50 is inside the dead band: from CASH, no entry."""
    state = _fresh_state()
    pos, _ = apply_overlay("TESTUSDT", 0.50, state)
    assert pos == "CASH"


def test_holds_long_inside_band() -> None:
    """
    The whole point of hysteresis: once LONG, a probability that
    merely softens (0.50) must NOT trigger an exit. This is what
    kept the live SOL position through its 0.568 dip.
    """
    state = _fresh_state()
    apply_overlay("TESTUSDT", 0.60, state)     # enter
    pos, _ = apply_overlay("TESTUSDT", 0.50, state)
    assert pos == "LONG"


def test_exits_long_below_exit_threshold() -> None:
    state = _fresh_state()
    apply_overlay("TESTUSDT", 0.60, state)     # enter
    pos, _ = apply_overlay("TESTUSDT", 0.40, state)
    assert pos == "CASH"


def test_boundaries_are_strict_inequalities() -> None:
    """
    Exactly 0.55 must NOT enter; exactly 0.45 must NOT exit.
    Backtests used strict `>` and `<`; the live code must agree
    or the two diverge on ties.
    """
    state = _fresh_state()
    pos, _ = apply_overlay("A", TH_IN, state)
    assert pos == "CASH", "proba == TH_IN must not enter"

    state2 = _fresh_state()
    apply_overlay("B", 0.90, state2)
    pos2, _ = apply_overlay("B", TH_OUT, state2)
    assert pos2 == "LONG", "proba == TH_OUT must not exit"


def test_state_persists_per_symbol_independently() -> None:
    """One symbol's regime must never contaminate another's."""
    state = _fresh_state()
    apply_overlay("XRPUSDT", 0.70, state)      # LONG
    apply_overlay("BTCUSDT", 0.30, state)      # CASH
    assert state["XRPUSDT"]["position"] == "LONG"
    assert state["BTCUSDT"]["position"] == "CASH"


def test_state_records_probability() -> None:
    """The audit trail needs the deciding probability, rounded."""
    state = _fresh_state()
    apply_overlay("TESTUSDT", 0.7081234, state)
    assert state["TESTUSDT"]["last_proba"] == 0.7081


def test_no_churn_on_oscillation_inside_band() -> None:
    """
    A probability oscillating inside the band must produce ZERO
    switches. Without hysteresis this pattern would pay costs on
    every bar — the failure mode the dead band exists to prevent.
    """
    state = _fresh_state()
    apply_overlay("TESTUSDT", 0.60, state)     # enter once
    for p in [0.52, 0.48, 0.51, 0.46, 0.54, 0.47]:
        pos, _ = apply_overlay("TESTUSDT", p, state)
        assert pos == "LONG", f"spurious exit at P={p}"