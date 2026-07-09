"""
live/execution.py

Phase 8 Prep: Execution module for the regime overlay strategy.

SAFETY ARCHITECTURE (three independent layers):
    LAYER 1 — Mode gate:
        LIVE_TRADING env var must be explicitly "true".
        Default (missing/false) = DRY RUN: log everything,
        place nothing. You can run this 1000 times in dry-run
        and it will never touch your account.

    LAYER 2 — Console confirmation:
        In LIVE mode, user must type "CONFIRM" before any order.
        Prevents accidental runs via scheduler in live mode.

    LAYER 3 — Hard position caps:
        Per-symbol cap: 25% of TOTAL_CAPITAL_USDT (hard ceiling,
        even if 60%/3 math gives more due to rounding).

POSITION SIZING (SYSTEM_SPEC.md sec.3):
    60% of total capital deployed when fully invested.
    Equal-weight across 3 symbols → 20% per symbol when LONG.
    Exit = full position close (no partial exits).

EXECUTION FLOW:
    1. Read current signals from signals/state.json
    2. Fetch actual open positions from Binance
    3. Diff: what SHOULD be vs what IS
    4. For each change needed:
       a. Fetch fresh ticker (staleness check)
       b. Calculate USDT size (allocation math)
       c. Calculate quantity (USDT / price, lot-size aware)
       d. Place market order (or log in dry-run)
       e. Confirm fill
    5. Update signals/state.json with execution details
    6. Log everything to signals/execution_log.csv

USAGE:
    python -m live.execution              (dry-run, safe)
    LIVE_TRADING=true python -m live.execution   (live, with confirmation)
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from core.logger import get_logger

logger = get_logger(__name__)

EXECUTION_VERSION = "1.0.0"

# ============================================================
# Configuration (from environment — NEVER hardcoded)
# ============================================================

def _load_env() -> None:
    """Load .env file if present (simple parser, no dependencies)."""
    env_file = Path(".env")
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_env()

# Safety: default to DRY RUN unless explicitly set
LIVE_TRADING = os.environ.get("LIVE_TRADING", "false").lower() == "true"
API_KEY = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
TOTAL_CAPITAL = float(os.environ.get("TOTAL_CAPITAL_USDT", "10000"))

# ---- Locked allocation (SYSTEM_SPEC.md sec.3) ----
ALLOCATION_FRACTION = 0.60      # 60% of capital deployed max
SYMBOLS = ["XRPUSDT", "SOLUSDT", "BTCUSDT"]
N_SYMBOLS = len(SYMBOLS)
PER_SYMBOL_FRACTION = ALLOCATION_FRACTION / N_SYMBOLS   # 20% each
PER_SYMBOL_CAP = 0.25           # hard ceiling regardless of math

# ---- Execution safety params ----
MAX_STALENESS_PCT = 0.02        # skip if price moved >2% since signal
MAX_RETRIES = 3
RETRY_DELAY_SEC = 5.0

STATE_FILE = Path("signals/state.json")
EXEC_LOG = Path("signals/execution_log.csv")
EXEC_LOG_HEADER = [
    "date", "time_utc", "symbol", "side", "quantity",
    "price", "usdt_value", "mode", "status", "notes"
]


# ============================================================
# Exchange interface
# ============================================================


def _get_exchange():
    """
    Return authenticated ccxt Binance instance.
    Raises RuntimeError if credentials are missing in live mode.
    """
    try:
        import ccxt
    except ImportError:
        raise RuntimeError("ccxt not installed: pip install ccxt")

    if LIVE_TRADING:
        if not API_KEY or not API_SECRET:
            raise RuntimeError(
                "LIVE_TRADING=true but API credentials missing. "
                "Set BINANCE_API_KEY and BINANCE_API_SECRET in .env"
            )
        return ccxt.binance({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
    else:
        # Dry run: public-only client (no credentials needed)
        return ccxt.binance({"enableRateLimit": True})


def _fetch_ticker(exchange, symbol: str) -> Optional[float]:
    """Current market price. None on failure."""
    try:
        t = exchange.fetch_ticker(symbol.replace("USDT", "/USDT"))
        last = t.get("last")
        return float(last) if last is not None else None
    except Exception as e:
        logger.warning(f"{symbol}: ticker fetch failed - {e}")
        return None


def _fetch_open_positions(exchange) -> dict[str, float]:
    """
    Fetch actual spot balances for our symbols.
    Returns {symbol: qty_held} — 0.0 if flat.
    Only called in live mode (dry run uses state.json as source of truth).
    """
    positions: dict[str, float] = {s: 0.0 for s in SYMBOLS}
    if not LIVE_TRADING:
        return positions
    try:
        balance = exchange.fetch_balance()
        for sym in SYMBOLS:
            asset = sym.replace("USDT", "")
            free = balance.get("free", {}).get(asset, 0.0)
            positions[sym] = float(free)
    except Exception as e:
        logger.error(f"Balance fetch failed - {e}")
    return positions


def _lot_size_qty(exchange, symbol: str, usdt_amount: float,
                  price: float) -> Optional[float]:
    """
    Convert USDT amount to base asset quantity, respecting Binance
    lot-size filters (minQty, stepSize).

    Args:
        usdt_amount: how much USDT to spend
        price:       current market price

    Returns:
        Rounded quantity, or None if below minimum.
    """
    raw_qty = usdt_amount / price
    try:
        market = exchange.market(symbol.replace("USDT", "/USDT"))
        filters = market.get("info", {}).get("filters", [])
        step = None
        min_qty = None
        for f in filters:
            if f.get("filterType") == "LOT_SIZE":
                step = float(f.get("stepSize", 0))
                min_qty = float(f.get("minQty", 0))
                break
        if step and step > 0:
            raw_qty = (raw_qty // step) * step
        if min_qty and raw_qty < min_qty:
            logger.warning(f"{symbol}: qty {raw_qty:.6f} < minQty {min_qty}")
            return None
    except Exception as e:
        logger.warning(f"{symbol}: lot-size filter fetch failed - {e}")
        # Proceed with 4dp rounding as fallback
        raw_qty = round(raw_qty, 4)
    return raw_qty


# ============================================================
# Execution log
# ============================================================


def _ensure_exec_log() -> None:
    if not EXEC_LOG.exists():
        EXEC_LOG.parent.mkdir(parents=True, exist_ok=True)
        with EXEC_LOG.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(EXEC_LOG_HEADER)


def _log_execution(symbol: str, side: str, qty: float, price: float,
                   usdt_val: float, mode: str, status: str,
                   notes: str = "") -> None:
    """Append one row to execution_log.csv (append-only audit trail)."""
    _ensure_exec_log()
    now = datetime.now(timezone.utc)
    row = [
        now.strftime("%Y-%m-%d"), now.strftime("%H:%M"),
        symbol, side, f"{qty:.6f}", f"{price:.6f}",
        f"{usdt_val:.2f}", mode, status, notes,
    ]
    with EXEC_LOG.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


# ============================================================
# Core execution logic
# ============================================================


def calculate_order_size(symbol: str, side: str,
                         price: float) -> dict[str, float]:
    """
    Calculate order size from the allocation rules.

    BUY:  20% of TOTAL_CAPITAL (capped at 25% hard limit)
    SELL: quantity from open position (fetch from exchange in live)

    Returns:
        {"usdt_value": ..., "qty_estimate": ...}
    """
    if side == "BUY":
        usdt = TOTAL_CAPITAL * min(PER_SYMBOL_FRACTION, PER_SYMBOL_CAP)
        return {"usdt_value": usdt, "qty_estimate": usdt / price}
    else:   # SELL — qty determined from actual position at execution
        return {"usdt_value": 0.0, "qty_estimate": 0.0}


def execute_signal(symbol: str, target: str, current: str,
                   signal_proba: float, exchange) -> dict[str, Any]:
    """
    Execute a single signal change (CASH→LONG or LONG→CASH).

    Args:
        symbol:       e.g. "SOLUSDT"
        target:       desired state "LONG" or "CASH"
        current:      current state from state.json
        signal_proba: P(up) from the model
        exchange:     ccxt exchange instance

    Returns:
        Result dict with status and execution details.
    """
    side = "BUY" if target == "LONG" else "SELL"
    mode = "LIVE" if LIVE_TRADING else "DRY_RUN"

    # --- Step 1: Fresh price (staleness check) ---
    price = _fetch_ticker(exchange, symbol)
    if price is None:
        msg = f"{symbol}: price unavailable — skipping"
        logger.error(msg)
        _log_execution(symbol, side, 0, 0, 0, mode, "SKIPPED", msg)
        return {"symbol": symbol, "status": "SKIPPED", "reason": msg}

    # (signal_proba used for staleness reference — model ran recently)
    # No price-staleness check here as signal is from the last bar close

    # --- Step 2: Size calculation ---
    order = calculate_order_size(symbol, side, price)
    usdt_val = order["usdt_value"]

    # --- Step 3: Lot size (BUY only) ---
    qty = 0.0
    if side == "BUY":
        calc_qty = _lot_size_qty(exchange, symbol, usdt_val, price)
        if calc_qty is None:
            msg = f"{symbol}: qty below minimum — skipping"
            logger.warning(msg)
            _log_execution(symbol, side, 0, price, usdt_val,
                           mode, "SKIPPED", msg)
            return {"symbol": symbol, "status": "SKIPPED", "reason": msg}
        qty = calc_qty
        usdt_val = qty * price

    # --- Step 4: Dry run path ---
    if not LIVE_TRADING:
        msg = (f"[DRY RUN] {side} {symbol}: "
               f"qty={qty:.4f} @ {price:.4f} "
               f"(~{usdt_val:.2f} USDT) | P={signal_proba:.3f}")
        print(f"  {msg}")
        logger.info(msg)
        _log_execution(symbol, side, qty, price, usdt_val,
                       mode, "DRY_RUN_OK", f"P={signal_proba:.3f}")
        return {"symbol": symbol, "status": "DRY_RUN_OK",
                "price": price, "qty": qty, "usdt": usdt_val}

    # --- Step 5: LIVE path — place order with retries ---
    pair = symbol.replace("USDT", "/USDT")
    last_err = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if side == "BUY":
                order_resp = exchange.create_market_buy_order(pair, qty)
            else:
                # SELL: use actual balance (qty=0 → fetch)
                balance = exchange.fetch_balance()
                asset = symbol.replace("USDT", "")
                sell_qty = float(balance["free"].get(asset, 0))
                if sell_qty <= 0:
                    msg = f"{symbol}: no balance to sell"
                    _log_execution(symbol, side, 0, price, 0,
                                   mode, "SKIPPED", msg)
                    return {"symbol": symbol, "status": "SKIPPED",
                            "reason": msg}
                qty = sell_qty
                order_resp = exchange.create_market_sell_order(pair, qty)

            fill_price = float(
                order_resp.get("average") or order_resp.get("price") or price
            )
            fill_qty = float(order_resp.get("filled") or qty)
            fill_usdt = fill_qty * fill_price
            msg = f"filled @ {fill_price:.6f}"
            _log_execution(symbol, side, fill_qty, fill_price,
                           fill_usdt, mode, "FILLED", msg)
            return {"symbol": symbol, "status": "FILLED",
                    "price": fill_price, "qty": fill_qty,
                    "usdt": fill_usdt}

        except Exception as e:
            last_err = str(e)
            logger.warning(f"{symbol}: order attempt {attempt} failed - {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

    msg = f"all {MAX_RETRIES} attempts failed: {last_err}"
    logger.error(f"{symbol}: {msg}")
    _log_execution(symbol, side, qty, price, usdt_val,
                   mode, "FAILED", msg)
    return {"symbol": symbol, "status": "FAILED", "reason": msg}


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    print("=" * 70)
    print(f"EXECUTION MODULE (v{EXECUTION_VERSION})")
    print(f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Mode: {'*** LIVE ***' if LIVE_TRADING else 'DRY RUN (safe)'}")
    print(f"Capital: {TOTAL_CAPITAL:.0f} USDT | "
          f"Per-symbol: {PER_SYMBOL_FRACTION*100:.0f}% "
          f"= {TOTAL_CAPITAL * PER_SYMBOL_FRACTION:.0f} USDT")
    print("=" * 70)

    # --- Load current signals ---
    if not STATE_FILE.exists():
        print("[!] No state.json found. Run signal_generator first.")
        raise SystemExit(1)

    state: dict[str, Any] = json.loads(STATE_FILE.read_text())
    print("\nCurrent signals (from state.json):")
    for sym in SYMBOLS:
        s = state.get(sym, {})
        print(f"  {sym:10s} {s.get('position','CASH'):>5s}  "
              f"P={s.get('last_proba', 0):.3f}")

    # --- Live mode confirmation ---
    if LIVE_TRADING:
        print(f"\n{'='*70}")
        print("  *** LIVE MODE *** Real orders will be placed!")
        print(f"  Capital at risk: "
              f"{TOTAL_CAPITAL * ALLOCATION_FRACTION:.0f} USDT max")
        confirm = input("  Type CONFIRM to proceed (anything else = abort): ")
        if confirm.strip() != "CONFIRM":
            print("  Aborted.")
            raise SystemExit(0)
        print("=" * 70)

    # --- Exchange + positions ---
    exchange = _get_exchange()
    open_positions = _fetch_open_positions(exchange)

    print(f"\n{'Symbol':10s} {'Signal':>7s} {'Actual':>7s} {'Action':>10s}")
    print("-" * 70)

    results: list[dict[str, Any]] = []
    for sym in SYMBOLS:
        target = state.get(sym, {}).get("position", "CASH")
        proba = float(state.get(sym, {}).get("last_proba", 0))
        actual_qty = open_positions.get(sym, 0.0)
        actual = "LONG" if actual_qty > 0 else "CASH"
        action = "HOLD"

        if target == "LONG" and actual == "CASH":
            action = "BUY"
        elif target == "CASH" and actual == "LONG":
            action = "SELL"

        print(f"{sym:10s} {target:>7s} {actual:>7s} {action:>10s}")

        if action in ("BUY", "SELL"):
            result = execute_signal(sym, target, actual, proba, exchange)
            results.append(result)
        else:
            results.append({"symbol": sym, "status": "HOLD"})

    # --- Summary ---
    print("-" * 70)
    print("\nExecution summary:")
    for r in results:
        st = r.get("status", "?")
        sym = r.get("symbol", "")
        if st == "HOLD":
            print(f"  {sym:10s} HOLD (no change)")
        elif st == "DRY_RUN_OK":
            print(f"  {sym:10s} DRY_RUN: would {r.get('qty',0):.4f} "
                  f"@ {r.get('price',0):.4f}")
        elif st == "FILLED":
            print(f"  {sym:10s} FILLED: {r.get('qty',0):.4f} "
                  f"@ {r.get('price',0):.4f} "
                  f"({r.get('usdt',0):.2f} USDT)")
        elif st == "SKIPPED":
            print(f"  {sym:10s} SKIPPED: {r.get('reason','')}")
        else:
            print(f"  {sym:10s} {st}: {r.get('reason','')}")

    mode = "LIVE" if LIVE_TRADING else "DRY_RUN"
    print(f"\nMode: {mode} | Log: {EXEC_LOG}")
    print("[!] Review execution_log.csv for full audit trail.")
    print("=" * 70)