"""
live/execution.py

Phase 8 Prep: Execution module for the regime overlay strategy.

SAFETY ARCHITECTURE (three independent layers):
    LAYER 1 — Mode gate (3 modes):
        DRY_RUN  (LIVE_TRADING=false/missing): log only, no exchange calls
        TESTNET  (LIVE_TRADING=testnet):       real API calls, fake money
        LIVE     (LIVE_TRADING=true):          real API calls, real money

    LAYER 2 — Console confirmation:
        TESTNET and LIVE both require typing "CONFIRM" before orders.

    LAYER 3 — Hard position caps:
        Per-symbol max: 25% of TOTAL_CAPITAL_USDT regardless of math.

TESTNET USAGE (recommended before going live):
    1. Get testnet API keys from https://testnet.binance.vision/
    2. Add to .env: LIVE_TRADING=testnet
    3. Run: python -m live.execution
    4. Verify orders appear on testnet.binance.vision dashboard

MODES SUMMARY:
    DRY_RUN → no exchange → no orders → just logs
    TESTNET → testnet exchange → fake orders → fake fills
    LIVE    → real exchange → real orders → real fills
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

EXECUTION_VERSION = "1.1.0"


# ============================================================
# Configuration (from environment — NEVER hardcoded)
# ============================================================

def _load_env() -> None:
    """Load .env file if present (simple key=value parser)."""
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

# Three-way mode: dry_run / testnet / live
_raw_mode = os.environ.get("LIVE_TRADING", "false").lower().strip()
DRY_RUN  = _raw_mode not in ("true", "testnet")
TESTNET  = _raw_mode == "testnet"
LIVE     = _raw_mode == "true"

# Convenience label for logging
if DRY_RUN:
    MODE_LABEL = "DRY_RUN"
elif TESTNET:
    MODE_LABEL = "TESTNET"
else:
    MODE_LABEL = "LIVE"

API_KEY      = os.environ.get("BINANCE_API_KEY", "")
API_SECRET   = os.environ.get("BINANCE_API_SECRET", "")
TOTAL_CAPITAL = float(os.environ.get("TOTAL_CAPITAL_USDT", "10000"))

# ---- Locked allocation (SYSTEM_SPEC.md sec.3) ----
ALLOCATION_FRACTION  = 0.60
SYMBOLS              = ["XRPUSDT", "SOLUSDT", "BTCUSDT"]
N_SYMBOLS            = len(SYMBOLS)
PER_SYMBOL_FRACTION  = ALLOCATION_FRACTION / N_SYMBOLS   # 0.20 each
PER_SYMBOL_CAP       = 0.25     # hard ceiling

# ---- Execution safety ----
MAX_RETRIES      = 3
RETRY_DELAY_SEC  = 5.0

STATE_FILE  = Path("signals/state.json")
EXEC_LOG    = Path("signals/execution_log.csv")
EXEC_LOG_HEADER = [
    "date", "time_utc", "symbol", "side", "quantity",
    "price", "usdt_value", "mode", "status", "notes"
]


# ============================================================
# Exchange interface
# ============================================================


def _get_exchange():
    """
    Return the appropriate ccxt exchange instance for the current mode.

    DRY_RUN : no credentials needed (public endpoints only)
    TESTNET : credentials required, pointed at testnet.binance.vision
    LIVE    : credentials required, pointed at real binance.com

    Raises RuntimeError if credentials are missing in TESTNET/LIVE mode.
    """
    try:
        import ccxt
    except ImportError:
        raise RuntimeError("ccxt not installed: pip install ccxt")

    if DRY_RUN:
        # Public-only client — safe to call _get_exchange() in dry-run
        # (ticker fetches are skipped anyway, but keeps code uniform)
        return ccxt.binance({"enableRateLimit": True})

    # Both TESTNET and LIVE need credentials
    if not API_KEY or not API_SECRET:
        raise RuntimeError(
            f"Mode is {MODE_LABEL} but API credentials are missing. "
            "Set BINANCE_API_KEY and BINANCE_API_SECRET in .env"
        )

    if TESTNET:
        # Binance Spot Testnet — same ccxt interface, different URLs
        exchange = ccxt.binance({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
            "urls": {
                "api": {
                    "public":  "https://testnet.binance.vision/api",
                    "private": "https://testnet.binance.vision/api",
                }
            },
        })
    else:  # LIVE
        exchange = ccxt.binance({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })

    return exchange


def _fetch_ticker(exchange, symbol: str) -> Optional[float]:
    """Current market price. None on failure (fail-safe)."""
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
    Works for both TESTNET and LIVE (same API endpoint, different server).
    Returns {symbol: qty_held} — 0.0 if flat.
    """
    positions: dict[str, float] = {s: 0.0 for s in SYMBOLS}
    if DRY_RUN:
        return positions
    try:
        balance = exchange.fetch_balance()
        for sym in SYMBOLS:
            asset = sym.replace("USDT", "")
            free = balance.get("free", {}).get(asset, 0.0)
            positions[sym] = float(free or 0.0)
    except Exception as e:
        logger.error(f"Balance fetch failed - {e}")
    return positions


def _lot_size_qty(exchange, symbol: str, usdt_amount: float,
                  price: float) -> Optional[float]:
    """
    Convert USDT amount to base-asset quantity, respecting Binance
    lot-size filters (minQty, stepSize).
    Works identically for TESTNET and LIVE.
    """
    raw_qty = usdt_amount / price
    try:
        market = exchange.market(symbol.replace("USDT", "/USDT"))
        filters = market.get("info", {}).get("filters", [])
        step = None
        min_qty = None
        for f in filters:
            if f.get("filterType") == "LOT_SIZE":
                step    = float(f.get("stepSize", 0))
                min_qty = float(f.get("minQty", 0))
                break
        if step and step > 0:
            raw_qty = (raw_qty // step) * step
        if min_qty and raw_qty < min_qty:
            logger.warning(f"{symbol}: qty {raw_qty:.6f} < minQty {min_qty}")
            return None
    except Exception as e:
        logger.warning(f"{symbol}: lot-size fetch failed - {e}")
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
                   usdt_val: float, status: str, notes: str = "") -> None:
    """Append one row to execution_log.csv (append-only audit trail)."""
    _ensure_exec_log()
    now = datetime.now(timezone.utc)
    row = [
        now.strftime("%Y-%m-%d"), now.strftime("%H:%M"),
        symbol, side, f"{qty:.6f}", f"{price:.6f}",
        f"{usdt_val:.2f}", MODE_LABEL, status, notes,
    ]
    with EXEC_LOG.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


# ============================================================
# Core execution logic
# ============================================================


def calculate_order_size(symbol: str, side: str,
                         price: float) -> dict[str, float]:
    """BUY: 20% of capital (hard-capped at 25%). SELL: from position."""
    if side == "BUY":
        usdt = TOTAL_CAPITAL * min(PER_SYMBOL_FRACTION, PER_SYMBOL_CAP)
        return {"usdt_value": usdt, "qty_estimate": usdt / price}
    return {"usdt_value": 0.0, "qty_estimate": 0.0}


def execute_signal(symbol: str, target: str, current: str,
                   signal_proba: float, exchange) -> dict[str, Any]:
    """
    Execute a single signal change.

    DRY_RUN:  log + print, no exchange calls.
    TESTNET:  real ccxt call to testnet server.
    LIVE:     real ccxt call to production server.
    """
    side = "BUY" if target == "LONG" else "SELL"

    # ---- DRY RUN path (no exchange needed) ----
    if DRY_RUN:
        # Use a dummy price for sizing display
        dummy_price = 1.0   # actual price not fetched in dry-run
        order = calculate_order_size(symbol, side, dummy_price)
        msg = (f"[DRY RUN] {side} {symbol}: "
               f"~{order['usdt_value']:.2f} USDT | P={signal_proba:.3f}")
        print(f"  {msg}")
        logger.info(msg)
        _log_execution(symbol, side, 0, 0, order["usdt_value"],
                       "DRY_RUN_OK", f"P={signal_proba:.3f}")
        return {"symbol": symbol, "status": "DRY_RUN_OK"}

    # ---- TESTNET / LIVE path ----
    price = _fetch_ticker(exchange, symbol)
    if price is None:
        msg = f"{symbol}: price unavailable — skipping"
        logger.error(msg)
        _log_execution(symbol, side, 0, 0, 0, "SKIPPED", msg)
        return {"symbol": symbol, "status": "SKIPPED", "reason": msg}

    order = calculate_order_size(symbol, side, price)
    usdt_val = order["usdt_value"]
    qty = 0.0

    if side == "BUY":
        calc_qty = _lot_size_qty(exchange, symbol, usdt_val, price)
        if calc_qty is None:
            msg = f"{symbol}: qty below minimum — skipping"
            _log_execution(symbol, side, 0, price, usdt_val, "SKIPPED", msg)
            return {"symbol": symbol, "status": "SKIPPED", "reason": msg}
        qty = calc_qty
        usdt_val = qty * price

    pair = symbol.replace("USDT", "/USDT")
    last_err = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if side == "BUY":
                resp = exchange.create_market_buy_order(pair, qty)
            else:
                balance = exchange.fetch_balance()
                asset = symbol.replace("USDT", "")
                sell_qty = float(balance["free"].get(asset, 0))
                if sell_qty <= 0:
                    msg = f"{symbol}: no balance to sell"
                    _log_execution(symbol, side, 0, price, 0, "SKIPPED", msg)
                    return {"symbol": symbol, "status": "SKIPPED",
                            "reason": msg}
                qty = sell_qty
                resp = exchange.create_market_sell_order(pair, qty)

            fill_price = float(
                resp.get("average") or resp.get("price") or price
            )
            fill_qty   = float(resp.get("filled") or qty)
            fill_usdt  = fill_qty * fill_price
            msg = f"filled @ {fill_price:.6f}"
            _log_execution(symbol, side, fill_qty, fill_price,
                           fill_usdt, "FILLED", msg)
            return {"symbol": symbol, "status": "FILLED",
                    "price": fill_price, "qty": fill_qty,
                    "usdt": fill_usdt}

        except Exception as e:
            last_err = str(e)
            logger.warning(f"{symbol}: attempt {attempt} failed - {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

    msg = f"all {MAX_RETRIES} attempts failed: {last_err}"
    logger.error(f"{symbol}: {msg}")
    _log_execution(symbol, side, qty, price, usdt_val, "FAILED", msg)
    return {"symbol": symbol, "status": "FAILED", "reason": msg}


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    print("=" * 70)
    print(f"EXECUTION MODULE (v{EXECUTION_VERSION})")
    print(f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Mode: {MODE_LABEL}")
    if TESTNET:
        print("  [TESTNET] Real order flow, FAKE money.")
        print("  Verify fills at: https://testnet.binance.vision/")
    elif LIVE:
        print("  [*** LIVE ***] Real orders, REAL money.")
    else:
        print("  [DRY RUN] No exchange calls. Safe to run anytime.")
    print(f"Capital: {TOTAL_CAPITAL:.0f} USDT | "
          f"Per-symbol: {PER_SYMBOL_FRACTION*100:.0f}% "
          f"= {TOTAL_CAPITAL * PER_SYMBOL_FRACTION:.0f} USDT")
    print("=" * 70)

    if not STATE_FILE.exists():
        print("[!] No state.json found. Run signal_generator first.")
        raise SystemExit(1)

    state: dict[str, Any] = json.loads(STATE_FILE.read_text())
    print("\nCurrent signals (from state.json):")
    for sym in SYMBOLS:
        s = state.get(sym, {})
        print(f"  {sym:10s} {s.get('position','CASH'):>5s}  "
              f"P={s.get('last_proba', 0):.3f}")

    # Confirmation for TESTNET and LIVE both
    if not DRY_RUN:
        print(f"\n{'='*70}")
        print(f"  Mode: {MODE_LABEL}")
        if LIVE:
            print(f"  Capital at risk: "
                  f"{TOTAL_CAPITAL * ALLOCATION_FRACTION:.0f} USDT (REAL)")
        else:
            print("  Capital at risk: FAKE (testnet)")
        confirm = input("  Type CONFIRM to proceed (anything else = abort): ")
        if confirm.strip() != "CONFIRM":
            print("  Aborted.")
            raise SystemExit(0)
        print("=" * 70)

    exchange = _get_exchange()
    open_positions = _fetch_open_positions(exchange)

    print(f"\n{'Symbol':10s} {'Signal':>7s} {'Actual':>7s} {'Action':>10s}")
    print("-" * 70)

    results: list[dict[str, Any]] = []
    for sym in SYMBOLS:
        target  = state.get(sym, {}).get("position", "CASH")
        proba   = float(state.get(sym, {}).get("last_proba", 0))
        actual  = "LONG" if open_positions.get(sym, 0.0) > 0 else "CASH"
        action  = "HOLD"

        if target == "LONG"  and actual == "CASH": action = "BUY"  # noqa: E701
        elif target == "CASH" and actual == "LONG": action = "SELL"  # noqa: E701

        print(f"{sym:10s} {target:>7s} {actual:>7s} {action:>10s}")

        if action in ("BUY", "SELL"):
            result = execute_signal(sym, target, actual, proba, exchange)
            results.append(result)
        else:
            results.append({"symbol": sym, "status": "HOLD"})

    print("-" * 70)
    print("\nExecution summary:")
    for r in results:
        st  = r.get("status", "?")
        sym = r.get("symbol", "")
        if st == "HOLD":
            print(f"  {sym:10s} HOLD")
        elif st == "DRY_RUN_OK":
            print(f"  {sym:10s} DRY_RUN logged")
        elif st == "FILLED":
            print(f"  {sym:10s} FILLED: {r.get('qty',0):.4f} "
                  f"@ {r.get('price',0):.4f} "
                  f"({r.get('usdt',0):.2f} USDT)")
        elif st == "SKIPPED":
            print(f"  {sym:10s} SKIPPED: {r.get('reason','')}")
        else:
            print(f"  {sym:10s} {st}: {r.get('reason','')}")

    print(f"\nMode: {MODE_LABEL} | Log: {EXEC_LOG}")
    if TESTNET:
        print("[!] Check fills: https://testnet.binance.vision/")
    print("=" * 70)