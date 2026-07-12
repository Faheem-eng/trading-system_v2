"""
live/portfolio_status.py

Portfolio dashboard — one command shows everything:
    - Each position: entry price, current price, P&L (%, USDT)
    - Days held, signal probability journey
    - Portfolio totals: deployed, cash, total P&L

NOTE ON TP/SL:
    This system does NOT use price-based take-profit/stop-loss.
    Phase 4-5 proved fixed barriers destroy the edge (rally caps,
    churn). Exits are REGIME-based: P(up) < 0.45 -> exit to cash.
    The "stop" is the model changing its mind, not a price level.

Run: python -m live.portfolio_status
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _load_env() -> None:
    env_path = Path(__file__).parent.parent / ".env"
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=True)
    except ImportError:
        pass


_load_env()

STATE_FILE = Path("signals/state.json")
PAPER_LOG  = Path("signals/paper_log.csv")

SYMBOLS = ["XRPUSDT", "SOLUSDT", "BTCUSDT", "BNBUSDT", "ADAUSDT"]

TOTAL_CAPITAL       = float(os.environ.get("TOTAL_CAPITAL_USDT", "10000"))
ALLOCATION_FRACTION = 0.60
PER_SYMBOL_FRACTION = ALLOCATION_FRACTION / len(SYMBOLS)
COST_PER_SIDE       = 0.00125

TH_OUT = 0.45   # regime exit threshold (the "dynamic stop")


def _fetch_price(symbol: str) -> Optional[float]:
    """Live price via ccxt public ticker."""
    try:
        import ccxt
        ex = ccxt.binance({"enableRateLimit": True})
        t = ex.fetch_ticker(symbol.replace("USDT", "/USDT"))
        last = t.get("last")
        return float(last) if last is not None else None
    except Exception:
        return None


def _open_entries() -> dict[str, dict[str, Any]]:
    """
    Reconstruct open positions from paper_log.csv.
    An ENTER without a later EXIT = open position.
    Returns {symbol: {"entry_price": float, "entry_date": str,
                      "entry_proba": float}}
    """
    entries: dict[str, dict[str, Any]] = {}
    if not PAPER_LOG.exists():
        return entries
    with PAPER_LOG.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sym    = (row.get("symbol") or "").strip()
            action = (row.get("action") or "").strip()
            if action == "ENTER_LONG":
                price_raw = (row.get("price") or "").strip()
                try:
                    price = float(price_raw) if price_raw else None
                except ValueError:
                    price = None
                entries[sym] = {
                    "entry_price": price,
                    "entry_date": (row.get("date") or "").strip(),
                    "entry_proba": float(row.get("proba") or 0),
                }
            elif action == "EXIT_LONG" and sym in entries:
                del entries[sym]
    return entries


def _realized_pnl() -> dict[str, float]:
    """Sum of closed round-trip net returns per symbol."""
    realized: dict[str, float] = {s: 0.0 for s in SYMBOLS}
    if not PAPER_LOG.exists():
        return realized
    with PAPER_LOG.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sym = (row.get("symbol") or "").strip()
            net = (row.get("net_return") or "").strip()
            if sym in realized and net:
                try:
                    realized[sym] += float(net)
                except ValueError:
                    pass
    return realized


if __name__ == "__main__":
    now = datetime.now(timezone.utc)
    print("=" * 78)
    print("PORTFOLIO STATUS DASHBOARD")
    print(f"Time: {now.strftime('%Y-%m-%d %H:%M UTC')} | "
          f"Capital: {TOTAL_CAPITAL:.0f} USDT | "
          f"Per-symbol: {TOTAL_CAPITAL * PER_SYMBOL_FRACTION:.0f} USDT")
    print("=" * 78)

    # Load state + entries
    state: dict[str, Any] = {}
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
    entries  = _open_entries()
    realized = _realized_pnl()

    print(f"\n{'Symbol':9s} {'Pos':>5s} {'P(up)':>6s} {'Entry':>10s} "
          f"{'Now':>10s} {'P&L%':>7s} {'P&L USDT':>9s} {'Days':>5s}")
    print("-" * 78)

    total_unrealized = 0.0
    total_deployed   = 0.0
    n_long = 0

    for sym in SYMBOLS:
        s     = state.get(sym, {})
        pos   = s.get("position", "CASH")
        proba = float(s.get("last_proba", 0))

        if pos == "LONG" and sym in entries:
            n_long += 1
            e = entries[sym]
            entry_price = e["entry_price"]
            entry_date  = e["entry_date"]

            # Days held
            try:
                ed = datetime.strptime(entry_date, "%Y-%m-%d") \
                    .replace(tzinfo=timezone.utc)
                days = (now - ed).days
            except ValueError:
                days = 0

            current = _fetch_price(sym)
            size_usdt = TOTAL_CAPITAL * PER_SYMBOL_FRACTION
            total_deployed += size_usdt

            if entry_price and current:
                gross = current / entry_price - 1.0
                net   = gross - COST_PER_SIDE   # entry cost paid
                pnl_usdt = size_usdt * net
                total_unrealized += pnl_usdt
                print(f"{sym:9s} {pos:>5s} {proba:>6.3f} "
                      f"{entry_price:>10.4f} {current:>10.4f} "
                      f"{net*100:>+6.2f}% {pnl_usdt:>+8.2f} {days:>5d}")
            else:
                print(f"{sym:9s} {pos:>5s} {proba:>6.3f} "
                      f"{'?':>10s} {'?':>10s} {'—':>7s} {'—':>9s} "
                      f"{days:>5d}")
        else:
            print(f"{sym:9s} {pos:>5s} {proba:>6.3f} "
                  f"{'—':>10s} {'—':>10s} {'—':>7s} {'—':>9s} {'—':>5s}")

    # ---- Portfolio summary ----
    cash = TOTAL_CAPITAL - total_deployed
    total_realized_usdt = sum(
        realized[s] * TOTAL_CAPITAL * PER_SYMBOL_FRACTION
        for s in SYMBOLS
    )
    portfolio_value = TOTAL_CAPITAL + total_unrealized + total_realized_usdt

    print("-" * 78)
    print(f"\n  Positions:        {n_long}/{len(SYMBOLS)} LONG")
    print(f"  Deployed:         {total_deployed:>10.2f} USDT "
          f"({total_deployed/TOTAL_CAPITAL*100:.0f}%)")
    print(f"  Cash:             {cash:>10.2f} USDT")
    print(f"  Unrealized P&L:   {total_unrealized:>+10.2f} USDT")
    print(f"  Realized P&L:     {total_realized_usdt:>+10.2f} USDT")
    print(f"  Portfolio value:  {portfolio_value:>10.2f} USDT "
          f"({(portfolio_value/TOTAL_CAPITAL-1)*100:+.2f}%)")

    print("\n" + "-" * 78)
    print("  EXIT LOGIC (no price SL/TP — regime-based by design):")
    print(f"  Position exits when P(up) < {TH_OUT} — the model's regime")
    print("  call IS the stop. Fixed price barriers destroyed the edge")
    print("  in Phase 4-5 testing (rally caps + churn). See SPEC sec.5.")
    print("=" * 78)