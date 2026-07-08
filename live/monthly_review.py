"""
live/monthly_review.py

Phase 7.2: Monthly paper-trading review — the Phase 8 gate tool.

PURPOSE:
    Answers ONE question objectively: "Is live behavior consistent
    with what the backtest promised?" This is the pre-registered gate
    between paper trading and real capital (RUNBOOK §6).

WHAT IT DOES:
    1. Reads signals/paper_log.csv (auto-logged trades)
    2. Reconstructs per-symbol round-trips and open positions
       (open positions marked-to-market via ccxt current price)
    3. Computes monthly portfolio return (equal-weight, 60% allocation
       applied for the "total capital" view)
    4. Compares against the BACKTEST ENVELOPE (pre-registered):
           Walk-forward fold monthlies ranged ~-8% to +45% (strategy
           level). Envelope with margin: [-12%, +50%] per month.
    5. Verdict:
           ON-TRACK: inside envelope
           WATCH:    1st or 2nd consecutive month outside
           HALT:     3rd consecutive month outside (SPEC §7 kill rule)

STATE:
    signals/review_state.json tracks consecutive out-of-envelope
    months across runs (the HALT counter must persist).

USAGE:
    python -m live.monthly_review              (current month)
    python -m live.monthly_review 2026-07      (specific month)
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from core.logger import get_logger

logger = get_logger(__name__)

MONTHLY_REVIEW_VERSION = "1.0.0"

LOG_FILE = Path("signals/paper_log.csv")
REVIEW_STATE = Path("signals/review_state.json")

# Pre-registered backtest envelope (strategy-level monthly, with margin)
ENVELOPE_LOW = -0.12
ENVELOPE_HIGH = 0.50

ALLOCATION = 0.60          # capital fraction (SPEC §3)
COST_PER_SIDE = 0.00125
SYMBOLS = ["XRPUSDT", "SOLUSDT", "BTCUSDT"]


def _fetch_price(symbol: str) -> Optional[float]:
    """Current price for marking open positions. None on failure."""
    try:
        import ccxt
        ex = ccxt.binance({"enableRateLimit": True})
        t = ex.fetch_ticker(symbol.replace("USDT", "/USDT"))
        last = t.get("last")
        return float(last) if last is not None else None
    except Exception as e:
        logger.warning(f"{symbol}: mark price fetch failed — {e}")
        return None


def load_trades() -> list[dict[str, str]]:
    """All logged trades, chronological."""
    if not LOG_FILE.exists():
        return []
    with LOG_FILE.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def month_pnl(trades: list[dict[str, str]],
              month: str) -> dict[str, Any]:
    """
    Per-symbol P&L for the given month ("YYYY-MM").

    Method:
        - Closed round-trips whose EXIT falls in the month:
          use the logged net_return.
        - Position OPEN at month end (entry logged, no exit yet):
          mark-to-market at current price minus one-side cost
          (unrealized, flagged separately).

    Returns dict: symbol -> {"realized": [...], "unrealized": float|None,
                             "entry_price": float|None}
    """
    result: dict[str, Any] = {
        s: {"realized": [], "unrealized": None, "entry_price": None}
        for s in SYMBOLS
    }

    open_entry: dict[str, Optional[float]] = {s: None for s in SYMBOLS}

    for row in trades:
        # Robust access: .get() + strip — survives header variations,
        # stray spaces, or manually edited rows (never crash on a log row)
        sym = (row.get("symbol") or "").strip()
        if sym not in result:
            continue
        action = (row.get("action") or "").strip()
        price_raw = (row.get("price") or "").strip()
        try:
            price = float(price_raw) if price_raw else None
        except ValueError:
            price = None

        if action == "ENTER_LONG":
            open_entry[sym] = price
        elif action == "EXIT_LONG":
            row_date = (row.get("date") or "").strip()
            net_raw = (row.get("net_return") or "").strip()
            if row_date.startswith(month) and net_raw:
                try:
                    result[sym]["realized"].append(float(net_raw))
                except ValueError:
                    pass
            open_entry[sym] = None

    # Mark open positions
    for sym, entry in open_entry.items():
        if entry is not None and entry > 0:
            mark = _fetch_price(sym)
            result[sym]["entry_price"] = entry
            if mark is not None:
                result[sym]["unrealized"] = (
                    mark / entry - 1.0 - COST_PER_SIDE
                )
    return result


def load_review_state() -> dict[str, Any]:
    if REVIEW_STATE.exists():
        return json.loads(REVIEW_STATE.read_text())
    return {"consecutive_outside": 0, "history": []}


def save_review_state(state: dict[str, Any]) -> None:
    REVIEW_STATE.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_STATE.write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    month = sys.argv[1] if len(sys.argv) > 1 else \
        datetime.now(timezone.utc).strftime("%Y-%m")

    print("=" * 70)
    print(f"MONTHLY PAPER-TRADING REVIEW (v{MONTHLY_REVIEW_VERSION})")
    print(f"Month: {month} | Envelope: "
          f"[{ENVELOPE_LOW*100:.0f}%, {ENVELOPE_HIGH*100:.0f}%]/month")
    print("=" * 70)

    trades = load_trades()
    if not trades:
        print("\n  No trades logged yet — nothing to review.")
        print("  (Signals holding steady = zero trades is VALID behavior.)")
        raise SystemExit(0)

    pnl = month_pnl(trades, month)

    print(f"\n{'Symbol':10s} {'Realized':>12s} {'Unrealized':>12s}")
    print("-" * 70)
    symbol_returns: list[float] = []
    for sym in SYMBOLS:
        d = pnl[sym]
        realized = float(np.sum(d["realized"])) if d["realized"] else 0.0
        unreal = d["unrealized"]
        total_sym = realized + (unreal if unreal is not None else 0.0)
        symbol_returns.append(total_sym)
        ur_str = f"{unreal*100:+.2f}%" if unreal is not None else "—"
        note = f" (open @ {d['entry_price']})" if d["entry_price"] else ""
        print(f"{sym:10s} {realized*100:>+11.2f}% {ur_str:>12s}{note}")

    strategy_ret = float(np.mean(symbol_returns))
    capital_ret = strategy_ret * ALLOCATION

    print("-" * 70)
    print(f"\n  Strategy return (month):      {strategy_ret*100:+.2f}%")
    print(f"  Total-capital return (60%):   {capital_ret*100:+.2f}%")

    # Envelope check + persistent counter
    inside = ENVELOPE_LOW <= strategy_ret <= ENVELOPE_HIGH
    state = load_review_state()
    if inside:
        state["consecutive_outside"] = 0
    else:
        state["consecutive_outside"] = int(
            state.get("consecutive_outside", 0)) + 1
    state["history"].append({
        "month": month, "strategy_return": round(strategy_ret, 4),
        "inside_envelope": inside,
    })
    save_review_state(state)

    consec = state["consecutive_outside"]
    print(f"\n  Inside envelope:              {'YES ✅' if inside else 'NO ⚠️'}")
    print(f"  Consecutive months outside:   {consec}")

    print("\n" + "=" * 70)
    print("VERDICT (pre-registered — RUNBOOK §4, SPEC §7)")
    print("=" * 70)
    if inside:
        print("\n  ✅ ON-TRACK — live behavior consistent with backtest.")
    elif consec < 3:
        print(f"\n  ⚠️  WATCH — month {consec}/3 outside envelope."
              f" Continue paper trading, investigate drivers.")
    else:
        print("\n  🚨 HALT — 3 consecutive months outside envelope."
              " STOP, full re-validation required before continuing"
              " (walk_forward.py re-run, regime analysis).")
    print("=" * 70)