"""
live/testnet_check.py

Bybit Spot Testnet — balance + order check.

WHY BYBIT TESTNET (not Binance):
    Binance spot testnet requires GitHub login and had key-permission
    issues. Bybit testnet is simpler: email registration, instant API
    keys, auto-funded account.

    This validates the EXECUTION CODE (order placement, balance fetch,
    lot-size handling). Production will use Binance Spot.

Run: python -m live.testnet_check
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path


def _load_env() -> None:
    """Load .env using python-dotenv (path-safe)."""
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

API_KEY    = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
EXCHANGE   = os.environ.get("EXCHANGE", "bybit").lower()
MODE       = os.environ.get("LIVE_TRADING", "false").lower()

# Universe v1.2.0 (SPEC sec.1): 5 symbols since 2026-07-11
SYMBOLS = ["XRPUSDT", "SOLUSDT", "BTCUSDT", "BNBUSDT", "ADAUSDT"]


def _build_exchange(api_key: str, api_secret: str):
    """ccxt exchange for testnet (bybit primary, binance fallback)."""
    import ccxt
    if EXCHANGE == "bybit":
        ex = ccxt.bybit({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
    else:
        ex = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
    ex.set_sandbox_mode(True)
    return ex


if __name__ == "__main__":
    print("=" * 60)
    print(f"TESTNET ACCOUNT CHECK ({EXCHANGE.upper()} SPOT)")
    print(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Mode: {MODE}")
    print("=" * 60)

    if not API_KEY:
        print("\n[ERROR] API Key empty — check .env file")
        raise SystemExit(1)

    # Length diagnostic: Windows `echo` can silently truncate a
    # trailing digit before `>` (cmd reads `3>` as a redirect).
    # Bybit keys: ~18-24 chars, secrets: ~36. Binance: 64 each.
    print(f"\n[OK] API Key:    {API_KEY[:6]}...{API_KEY[-4:]} "
          f"(length: {len(API_KEY)})")
    print(f"[OK] API Secret: masked (length: {len(API_SECRET)})")
    if len(API_KEY) < 15 or len(API_SECRET) < 20:
        print("\n[WARNING] Key or secret looks TOO SHORT — likely")
        print("  truncated. Recreate .env in VS Code (not via echo).")

    exchange = _build_exchange(API_KEY, API_SECRET)

    # ---- Balances ----
    print("\n[1] TESTNET BALANCES:")
    print("-" * 40)
    try:
        balance = exchange.fetch_balance()
        assets = ["USDT", "XRP", "SOL", "BTC", "BNB", "USDC"]
        found = False
        for asset in assets:
            total = float(balance.get("total", {}).get(asset, 0) or 0)
            free  = float(balance.get("free",  {}).get(asset, 0) or 0)
            if total > 0:
                print(f"  {asset:6s}: {free:>15.6f} free "
                      f"/ {total:>15.6f} total")
                found = True
        if not found:
            print("  No balances found — check Assets on testnet.bybit.com")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ---- Recent orders ----
    print("\n[2] RECENT ORDERS (last 3 per symbol):")
    print("-" * 40)
    for sym in SYMBOLS:
        try:
            pair = sym.replace("USDT", "/USDT")
            orders = exchange.fetch_orders(pair, limit=3)
            if not orders:
                print(f"  {sym}: no orders yet")
                continue
            print(f"  {sym}:")
            for o in orders:
                dt = datetime.fromtimestamp(
                    o["timestamp"] / 1000, tz=timezone.utc # type: ignore
                ).strftime("%m-%d %H:%M")
                print(f"    {dt} | {o['side'].upper():4s} | "
                      f"qty={o.get('amount', 0):.4f} | "
                      f"status={o['status']}")
        except Exception as e:
            print(f"  {sym}: {e}")

    # ---- Open positions ----
    print("\n[3] OPEN POSITIONS:")
    print("-" * 40)
    try:
        balance = exchange.fetch_balance()
        found = False
        for sym in SYMBOLS:
            asset = sym.replace("USDT", "")
            qty = float(balance.get("free", {}).get(asset, 0) or 0)
            if qty > 0.0001:
                try:
                    ticker = exchange.fetch_ticker(
                        sym.replace("USDT", "/USDT")
                    )
                    price = float(ticker.get("last", 0) or 0)
                    print(f"  {sym}: {qty:.6f} @ {price:.4f} "
                          f"= {qty * price:.2f} USDT")
                except Exception:
                    print(f"  {sym}: {qty:.6f} (price unavailable)")
                found = True
        if not found:
            print("  No open positions (all CASH)")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("Bybit testnet: https://testnet.bybit.com")
    print("Check orders: Assets -> Order History")
    print("=" * 60)