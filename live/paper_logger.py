"""
live/paper_logger.py

Phase 7.1: Automatic paper-trade logging.

PROBLEM THIS SOLVES:
    Manual logging (open Binance, find price, edit CSV) is error-prone
    and easy to forget. Missed or wrong entries corrupt the monthly
    review, which is the Phase 8 gate. Automation = data integrity.

WHAT IT DOES:
    Called by signal_generator on every run. When a symbol's position
    CHANGES (ENTER_LONG / EXIT_LONG), it:
        1. Fetches the current market price via ccxt (public ticker)
        2. Appends a row to signals/paper_log.csv
        3. On EXIT: computes the round-trip return versus the matching
           entry, net of assumed 0.125%/side costs (instant feedback)

    HOLD bars are NOT logged. The log records TRADES only; the monthly
    review reconstructs the equity curve from entries and exits.

FAIL-SAFE:
    A price-fetch failure logs the row with an empty price and a
    PRICE_FETCH_FAILED note (fill manually later). Logging must never
    crash the signal run — the signal itself is the critical output.

ENCODING: ASCII-only output (Windows cp1252 safe, scheduler safe).

CSV SCHEMA (append-only, never rewrite history):
    date,time_utc,symbol,action,price,proba,net_return,notes
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from core.logger import get_logger

logger = get_logger(__name__)

PAPER_LOGGER_VERSION = "1.0.0"

LOG_FILE = Path("signals/paper_log.csv")
COST_PER_SIDE = 0.00125     # same assumption used in every backtest

CSV_HEADER = ["date", "time_utc", "symbol", "action", "price",
              "proba", "net_return", "notes"]


def _fetch_price(symbol: str) -> Optional[float]:
    """
    Current market price via ccxt public ticker.

    Args:
        symbol: e.g. "SOLUSDT"

    Returns:
        Last traded price, or None on any failure (fail-safe).
    """
    try:
        import ccxt  # lazy import: optional dependency
        exchange = ccxt.binance({"enableRateLimit": True})
        ticker = exchange.fetch_ticker(symbol.replace("USDT", "/USDT"))
        last = ticker.get("last")
        return float(last) if last is not None else None
    except Exception as e:
        logger.warning(f"{symbol}: price fetch failed - {e}")
        return None


def _ensure_log_exists() -> None:
    """Create the log file with its header if it does not exist."""
    if not LOG_FILE.exists():
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)


def _last_entry_price(symbol: str) -> Optional[float]:
    """
    Find the most recent ENTER_LONG price for this symbol that has no
    matching EXIT yet (used to compute the round-trip return on exit).

    Walks the log chronologically: an ENTER sets the open price, an
    EXIT clears it. Whatever remains at the end is the open position.

    Returns:
        Open entry price, or None if flat / price was never filled.
    """
    if not LOG_FILE.exists():
        return None

    entry_price: Optional[float] = None
    with LOG_FILE.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("symbol") or "").strip() != symbol:
                continue
            action = (row.get("action") or "").strip()
            if action == "ENTER_LONG":
                price_raw = (row.get("price") or "").strip()
                try:
                    entry_price = float(price_raw) if price_raw else None
                except ValueError:
                    entry_price = None
            elif action == "EXIT_LONG":
                entry_price = None      # round-trip closed
    return entry_price


def log_trade(symbol: str, action: str, proba: float,
              notes: str = "") -> dict[str, Any]:
    """
    Log a position change with an auto-fetched market price.

    Args:
        symbol: e.g. "SOLUSDT"
        action: "ENTER_LONG" or "EXIT_LONG"
        proba:  model P(up) at the decision bar
        notes:  optional free-text context

    Returns:
        Dict with the logged fields, for the caller's console output:
        {"symbol", "action", "price", "net_return"}
    """
    _ensure_log_exists()
    now = datetime.now(timezone.utc)
    price = _fetch_price(symbol)

    net_return_str = ""
    if action == "EXIT_LONG" and price is not None:
        entry = _last_entry_price(symbol)
        if entry is not None and entry > 0:
            gross = price / entry - 1.0
            net = gross - 2 * COST_PER_SIDE     # entry + exit costs
            net_return_str = f"{net:+.4f}"

    note_final = notes
    if price is None:
        note_final = (notes + " PRICE_FETCH_FAILED").strip()

    row = [
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M"),
        symbol,
        action,
        f"{price:.6f}" if price is not None else "",
        f"{proba:.4f}",
        net_return_str,
        note_final,
    ]
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

    price_disp = f"{price:.6f}" if price is not None else "N/A"
    logger.info(f"Paper log: {symbol} {action} @ {price_disp} "
                f"(P={proba:.3f})")

    return {
        "symbol": symbol,
        "action": action,
        "price": price,
        "net_return": net_return_str,
    }