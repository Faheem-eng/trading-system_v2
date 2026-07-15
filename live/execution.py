"""
live/execution.py

Phase 8: Execution module for the regime overlay strategy.

SAFETY ARCHITECTURE (three independent layers):
    LAYER 1 — Mode gate (3 modes via LIVE_TRADING in .env):
        DRY_RUN  (false/missing): log only, no exchange calls
        TESTNET  (testnet):       real orders, FAKE money
                                  (Binance Spot Testnet via ccxt
                                   set_sandbox_mode — manual URL
                                   overrides silently fall through
                                   to production; never use them)
        LIVE     (true):          real orders, REAL money

    LAYER 2 — Console confirmation:
        TESTNET and LIVE require typing "CONFIRM" (exact, uppercase).

    LAYER 3 — Hard position caps:
        Per-symbol max: 25% of TOTAL_CAPITAL_USDT regardless of math.

UNIVERSE (v1.3.0 — expanded 2026-07-11):
    XRP, SOL, BTC (original, validated Phase 3-4)
    BNB, ADA (added after 4h walk-forward 4/5 folds IC>0
              + overlay backtest beats B&H)
    LINK tested and REJECTED (overlay -45.9% vs B&H -46.5%,
    both negative — no genuine value-add).

POSITION SIZING (SYSTEM_SPEC.md sec.3):
    60% of total capital deployed when fully invested.
    Equal-weight across 5 symbols -> 12% per symbol when LONG.
    Exit = full position close (no partial exits).

USAGE:
    python -m live.execution
    (mode controlled ONLY by LIVE_TRADING in .env)
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

EXECUTION_VERSION = "1.3.0"


# ============================================================
# Configuration (from .env — NEVER hardcoded)
# ============================================================


def _load_env() -> None:
    """
    Load .env with an absolute path anchored to the project root.
    A relative Path(".env") breaks when the working directory
    differs (e.g. Task Scheduler runs).
    Falls back to manual parsing if python-dotenv is missing.
    """
    env_path = Path(__file__).parent.parent / ".env"
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=True)
    except ImportError:
        if not env_path.exists():
            return
        for line in env_path.read_text(
                encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ[key.strip()] = val.strip().strip('"').strip("'")


_load_env()

# ---- Three-way mode gate ----
_raw_mode = os.environ.get("LIVE_TRADING", "false").lower().strip()
DRY_RUN = _raw_mode not in ("true", "testnet")
TESTNET = _raw_mode == "testnet"
LIVE    = _raw_mode == "true"

if DRY_RUN:
    MODE_LABEL = "DRY_RUN"
elif TESTNET:
    MODE_LABEL = "TESTNET"
else:
    MODE_LABEL = "LIVE"

API_KEY       = os.environ.get("BINANCE_API_KEY", "")
API_SECRET    = os.environ.get("BINANCE_API_SECRET", "")
TOTAL_CAPITAL = float(os.environ.get("TOTAL_CAPITAL_USDT", "10000"))

# ---- Locked allocation (SYSTEM_SPEC.md sec.3, v1.2.0) ----
ALLOCATION_FRACTION = 0.60
SYMBOLS = ["XRPUSDT", "SOLUSDT", "BTCUSDT", "BNBUSDT", "ADAUSDT"]
N_SYMBOLS           = len(SYMBOLS)
PER_SYMBOL_FRACTION = ALLOCATION_FRACTION / N_SYMBOLS   # 0.12 each
PER_SYMBOL_CAP      = 0.25      # hard ceiling, belt + suspenders

# ---- Execution safety ----
MAX_RETRIES     = 3
RETRY_DELAY_SEC = 5.0
DUST_THRESHOLD_USDT = 10.0      # holdings below this = flat
STALE_STATE_HOURS   = 6.0       # refuse to trade on signals older than this
                                # (4h bar cadence + grace for scheduler jitter)

STATE_FILE = Path("signals/state.json")
EXEC_LOG   = Path("signals/execution_log.csv")
EXEC_LOG_HEADER = [
    "date", "time_utc", "symbol", "side", "quantity",
    "price", "usdt_value", "mode", "status", "notes",
]


# ============================================================
# Exchange interface
# ============================================================


def _get_exchange():
    """
    Build the correctly-configured ccxt Binance instance.

    DRY_RUN : public-only client, no credentials.
    TESTNET : authenticated -> Binance Spot Testnet via
              ccxt set_sandbox_mode(True).
    LIVE    : authenticated -> production Binance Spot.

    Raises:
        RuntimeError if credentials missing in TESTNET/LIVE.
    """
    try:
        import ccxt
    except ImportError:
        raise RuntimeError("ccxt not installed: pip install ccxt")

    if DRY_RUN:
        return ccxt.binance({"enableRateLimit": True})

    if not API_KEY or not API_SECRET:
        raise RuntimeError(
            f"Mode is {MODE_LABEL} but API credentials missing. "
            "Set BINANCE_API_KEY and BINANCE_API_SECRET in .env"
        )

    exchange = ccxt.binance({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })

    if TESTNET:
        exchange.set_sandbox_mode(True)   # -> testnet.binance.vision

    return exchange


def _fetch_ticker(exchange, symbol: str) -> Optional[float]:
    """Current market price. None on failure (fail-safe: skip)."""
    try:
        t = exchange.fetch_ticker(symbol.replace("USDT", "/USDT"))
        last = t.get("last")
        return float(last) if last is not None else None
    except Exception as e:
        logger.warning(f"{symbol}: ticker fetch failed - {e}")
        return None


def _fetch_open_positions(exchange) -> dict[str, float]:
    """
    Actual spot balances -> {symbol: qty_held}.

    Dust filter: holdings worth < DUST_THRESHOLD_USDT are treated
    as flat. Testnet resets and lot-size rounding leave residues
    (0.001 SOL etc.) that would otherwise be misread as open LONGs.
    """
    positions: dict[str, float] = {s: 0.0 for s in SYMBOLS}
    if DRY_RUN:
        return positions
    try:
        balance = exchange.fetch_balance()
        for sym in SYMBOLS:
            asset = sym.replace("USDT", "")
            qty = float(balance.get("free", {}).get(asset, 0) or 0)
            if qty <= 0:
                continue
            price = _fetch_ticker(exchange, sym)
            value = qty * price if price else 0.0
            positions[sym] = qty if value >= DUST_THRESHOLD_USDT else 0.0
    except Exception as e:
        logger.error(f"Balance fetch failed - {e}")
    return positions


def _fetch_free_usdt(exchange) -> float:
    """
    Free (spendable) USDT balance. Returns TOTAL_CAPITAL in DRY_RUN so the
    dry path is unaffected; returns 0.0 on any failure so a BUY fails safe
    (below the dust floor -> skipped) rather than over-spending blind.
    """
    if DRY_RUN:
        return TOTAL_CAPITAL
    try:
        bal = exchange.fetch_balance()
        return float(bal.get("free", {}).get("USDT", 0) or 0)
    except Exception as e:
        logger.error(f"USDT balance fetch failed - {e}")
        return 0.0


def _lot_size_qty(exchange, symbol: str, usdt_amount: float,
                  price: float) -> Optional[float]:
    """
    Convert USDT amount into base-asset quantity, respecting the
    exchange LOT_SIZE filter (stepSize, minQty).

    Returns:
        Rounded quantity, or None if below the exchange minimum.
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
            logger.warning(
                f"{symbol}: qty {raw_qty:.6f} < minQty {min_qty}"
            )
            return None
    except Exception as e:
        logger.warning(f"{symbol}: lot-size fetch failed - {e}")
        raw_qty = round(raw_qty, 4)     # conservative fallback
    return raw_qty


# ============================================================
# Execution audit log
# ============================================================


def _ensure_exec_log() -> None:
    """Create the log file with header if missing."""
    if not EXEC_LOG.exists():
        EXEC_LOG.parent.mkdir(parents=True, exist_ok=True)
        with EXEC_LOG.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(EXEC_LOG_HEADER)


def _log_execution(symbol: str, side: str, qty: float, price: float,
                   usdt_val: float, status: str,
                   notes: str = "") -> None:
    """Append one row to execution_log.csv (append-only audit)."""
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
    """
    BUY:  12% of TOTAL_CAPITAL (60% / 5 symbols, capped at 25%).
    SELL: quantity comes from the actual position at execution.
    """
    if side == "BUY":
        usdt = TOTAL_CAPITAL * min(PER_SYMBOL_FRACTION, PER_SYMBOL_CAP)
        return {"usdt_value": usdt, "qty_estimate": usdt / price}
    return {"usdt_value": 0.0, "qty_estimate": 0.0}


def execute_signal(symbol: str, target: str, current: str,
                   signal_proba: float, exchange) -> dict[str, Any]:
    """
    Execute one signal change (CASH->LONG = BUY, LONG->CASH = SELL).

    DRY_RUN:  print + log only.
    TESTNET:  real ccxt order against the sandbox server.
    LIVE:     real ccxt order against production.
    """
    side = "BUY" if target == "LONG" else "SELL"

    # ---- DRY RUN path ----
    if DRY_RUN:
        order = calculate_order_size(symbol, side, 1.0)
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
        msg = f"{symbol}: price unavailable - skipping"
        logger.error(msg)
        _log_execution(symbol, side, 0, 0, 0, "SKIPPED", msg)
        return {"symbol": symbol, "status": "SKIPPED", "reason": msg}

    order = calculate_order_size(symbol, side, price)
    usdt_val = order["usdt_value"]
    qty = 0.0

    if side == "BUY":
        # Safety: never attempt to spend more USDT than is actually free.
        # The intended notional is TOTAL_CAPITAL-based (SPEC sec.3); this
        # only clamps DOWN to reality, it never sizes up.
        free_usdt = _fetch_free_usdt(exchange)
        if usdt_val > free_usdt:
            logger.warning(
                f"{symbol}: intended {usdt_val:.2f} USDT exceeds free "
                f"{free_usdt:.2f} - clamping to available balance")
            usdt_val = free_usdt * 0.995   # buffer for fees / price drift
        if usdt_val < DUST_THRESHOLD_USDT:
            msg = (f"{symbol}: free USDT {free_usdt:.2f} too low to open "
                   "position - skipping")
            _log_execution(symbol, side, 0, price, usdt_val, "SKIPPED", msg)
            return {"symbol": symbol, "status": "SKIPPED", "reason": msg}
        calc_qty = _lot_size_qty(exchange, symbol, usdt_val, price)
        if calc_qty is None:
            msg = f"{symbol}: qty below exchange minimum - skipping"
            _log_execution(symbol, side, 0, price, usdt_val,
                           "SKIPPED", msg)
            return {"symbol": symbol, "status": "SKIPPED",
                    "reason": msg}
        qty = calc_qty
        usdt_val = qty * price

    pair = symbol.replace("USDT", "/USDT")
    last_err = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if side == "BUY":
                resp = exchange.create_market_buy_order(pair, qty)
            else:
                # SELL: liquidate the actual balance
                balance = exchange.fetch_balance()
                asset = symbol.replace("USDT", "")
                sell_qty = float(balance["free"].get(asset, 0) or 0)
                if sell_qty <= 0:
                    msg = f"{symbol}: no balance to sell"
                    _log_execution(symbol, side, 0, price, 0,
                                   "SKIPPED", msg)
                    return {"symbol": symbol, "status": "SKIPPED",
                            "reason": msg}
                adj = _lot_size_qty(exchange, symbol,
                                    sell_qty * price, price)
                qty = adj if adj else sell_qty
                resp = exchange.create_market_sell_order(pair, qty)

            fill_price = float(
                resp.get("average") or resp.get("price") or price
            )
            fill_qty  = float(resp.get("filled") or qty)
            fill_usdt = fill_qty * fill_price
            _log_execution(symbol, side, fill_qty, fill_price,
                           fill_usdt, "FILLED",
                           f"filled @ {fill_price:.6f}")
            return {"symbol": symbol, "status": "FILLED",
                    "price": fill_price, "qty": fill_qty,
                    "usdt": fill_usdt}

        except Exception as e:
            last_err = str(e)
            logger.warning(f"{symbol}: attempt {attempt} failed - {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

    msg = f"all {MAX_RETRIES} attempts failed: {last_err[:100]}"
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
    elif LIVE:
        print("  [*** LIVE ***] Real orders, REAL money.")
    else:
        print("  [DRY RUN] No exchange calls. Safe to run anytime.")
    print(f"Capital: {TOTAL_CAPITAL:.0f} USDT | "
          f"Universe: {N_SYMBOLS} symbols | "
          f"Per-symbol: {PER_SYMBOL_FRACTION*100:.0f}% "
          f"= {TOTAL_CAPITAL * PER_SYMBOL_FRACTION:.0f} USDT")
    print("=" * 70)

    # ---- Load current signals ----
    if not STATE_FILE.exists():
        print("[!] No state.json found. Run signal_generator first.")
        raise SystemExit(1)

    state: dict[str, Any] = json.loads(STATE_FILE.read_text())

    # ---- State freshness guard (fail-safe: never trade on stale signals) ----
    state_age_h = (time.time() - STATE_FILE.stat().st_mtime) / 3600.0
    if state_age_h > STALE_STATE_HOURS:
        smsg = (f"state.json is {state_age_h:.1f}h old (> {STALE_STATE_HOURS}h) "
                "- signal_generator may not have run this cycle")
        logger.warning(smsg)
        if DRY_RUN:
            print(f"\n[!] WARNING: {smsg} (DRY_RUN: continuing)")
        else:
            print(f"\n[X] ABORT: {smsg}.\n    Refusing to trade on stale "
                  "signals. Run 'python -m live.signal_generator' first.")
            raise SystemExit(1)

    print("\nCurrent signals (from state.json):")
    for sym in SYMBOLS:
        s = state.get(sym, {})
        print(f"  {sym:10s} {s.get('position', 'CASH'):>5s}  "
              f"P={s.get('last_proba', 0):.3f}")

    # ---- Confirmation gate ----
    if not DRY_RUN:
        print(f"\n{'='*70}")
        print(f"  Mode: {MODE_LABEL}")
        if LIVE:
            print(f"  Capital at risk: "
                  f"{TOTAL_CAPITAL * ALLOCATION_FRACTION:.0f} USDT (REAL)")
        else:
            print("  Capital at risk: FAKE (testnet)")
        confirm = input(
            "  Type CONFIRM (uppercase) to proceed, anything else aborts: "
        )
        if confirm.strip() != "CONFIRM":
            print("  Aborted.")
            raise SystemExit(0)
        print("=" * 70)

    # ---- Diff and execute ----
    exchange = _get_exchange()
    open_positions = _fetch_open_positions(exchange)

    # ---- Equity sanity: warn if configured capital has drifted from reality ----
    # Sizing stays LOCKED to TOTAL_CAPITAL (SPEC sec.3); this only surfaces a
    # mismatch so the operator can update .env, rather than silently drifting.
    if not DRY_RUN:
        _free = _fetch_free_usdt(exchange)
        _held = 0.0
        for _s in SYMBOLS:
            _q = open_positions.get(_s, 0.0) or 0.0
            if _q > 0:
                _p = _fetch_ticker(exchange, _s) or 0.0
                _held += _q * _p
        _equity = _free + _held
        if _equity > 0 and abs(_equity - TOTAL_CAPITAL) / max(TOTAL_CAPITAL, 1.0) > 0.20:
            _note = (f"actual equity ~{_equity:.0f} USDT differs >20% from "
                     f"TOTAL_CAPITAL_USDT={TOTAL_CAPITAL:.0f}")
            print(f"[!] NOTE: {_note}. Update .env so sizing tracks real "
                  "capital (SPEC sec.3: 60% allocation).")
            logger.warning(_note)

    print(f"\n{'Symbol':10s} {'Signal':>7s} {'Actual':>7s} {'Action':>10s}")
    print("-" * 70)

    results: list[dict[str, Any]] = []
    for sym in SYMBOLS:
        target = state.get(sym, {}).get("position", "CASH")
        proba  = float(state.get(sym, {}).get("last_proba", 0))
        actual = "LONG" if open_positions.get(sym, 0.0) > 0 else "CASH"
        action = "HOLD"

        if target == "LONG" and actual == "CASH":
            action = "BUY"
        elif target == "CASH" and actual == "LONG":
            action = "SELL"

        print(f"{sym:10s} {target:>7s} {actual:>7s} {action:>10s}")

        if action in ("BUY", "SELL"):
            results.append(
                execute_signal(sym, target, actual, proba, exchange)
            )
        else:
            results.append({"symbol": sym, "status": "HOLD"})

    # ---- Summary ----
    print("-" * 70)
    print("\nExecution summary:")
    for r in results:
        st  = r.get("status", "?")
        sym = r.get("symbol", "")
        if st == "HOLD":
            print(f"  {sym:10s} HOLD (no change)")
        elif st == "DRY_RUN_OK":
            print(f"  {sym:10s} DRY_RUN logged")
        elif st == "FILLED":
            print(f"  {sym:10s} FILLED: {r.get('qty', 0):.4f} "
                  f"@ {r.get('price', 0):.4f} "
                  f"({r.get('usdt', 0):.2f} USDT)")
        elif st == "SKIPPED":
            print(f"  {sym:10s} SKIPPED: {r.get('reason', '')}")
        else:
            print(f"  {sym:10s} {st}: {r.get('reason', '')}")

    print(f"\nMode: {MODE_LABEL} | Log: {EXEC_LOG}")
    if TESTNET:
        print("[!] Verify: python -m live.testnet_check")
    print("=" * 70)