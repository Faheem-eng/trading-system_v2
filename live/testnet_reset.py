"""
live/testnet_reset.py

One-time testnet cleanup: sell all pre-funded assets (SOL/BTC/XRP)
to USDT so the execution module starts from a clean CASH state.

The testnet pre-funds accounts with sample assets. Our execution
module reads actual balances as positions — pre-funded assets would
be misread as open LONGs. This resets to all-CASH.

Run once: python -m live.testnet_reset
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_env() -> None:
    env_path = Path(__file__).parent.parent / ".env"
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=True)
    except ImportError:
        pass


_load_env()

API_KEY    = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")

# Assets to liquidate to USDT (our trading universe)
ASSETS_TO_SELL = ["SOL", "BTC", "XRP"]


if __name__ == "__main__":
    import ccxt

    print("=" * 60)
    print("TESTNET RESET — sell pre-funded assets to USDT")
    print("=" * 60)

    exchange = ccxt.binance({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    exchange.set_sandbox_mode(True)

    balance = exchange.fetch_balance()

    for asset in ASSETS_TO_SELL:
        qty = float(balance.get("free", {}).get(asset, 0) or 0)
        if qty <= 0:
            print(f"  {asset}: nothing to sell")
            continue
        pair = f"{asset}/USDT"
        try:
            # Respect lot size
            market = exchange.market(pair)
            filters = market.get("info", {}).get("filters", [])
            for f in filters:
                if f.get("filterType") == "LOT_SIZE":
                    step = float(f.get("stepSize", 0))
                    if step > 0:
                        qty = (qty // step) * step
                    break
            if qty <= 0:
                print(f"  {asset}: qty too small after lot-size")
                continue
            resp = exchange.create_market_sell_order(pair, qty)
            fill = float(resp.get("average") or 0)
            print(f"  {asset}: SOLD {qty} @ ~{fill:.4f}")
        except Exception as e:
            print(f"  {asset}: FAILED - {str(e)[:80]}")

    # Final balance
    print("\nFinal balances:")
    balance = exchange.fetch_balance()
    for asset in ["USDT"] + ASSETS_TO_SELL:
        total = float(balance.get("total", {}).get(asset, 0) or 0)
        if total > 0.0001:
            print(f"  {asset}: {total:.4f}")

    print("\n[OK] Clean slate. Now run: python -m live.execution")
    print("=" * 60)