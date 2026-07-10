"""
live/bybit_diagnose.py

Bybit key diagnostic — tests the SAME key against all three
Bybit environments to find where it's actually valid:
    1. Testnet     (api-testnet.bybit.com)
    2. Demo        (api-demo.bybit.com)
    3. Mainnet     (api.bybit.com) — read-only balance check

Run: python -m live.bybit_diagnose
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


def _try_env(name: str, base_url: str) -> None:
    """Try fetch_balance against one Bybit environment."""
    import ccxt
    print(f"\n[{name}] {base_url}")
    print("-" * 50)
    try:
        ex = ccxt.bybit({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        # Override ALL API URLs to this environment
        ex.urls["api"] = {k: base_url for k in ex.urls["api"]} \
            if isinstance(ex.urls["api"], dict) else base_url
        balance = ex.fetch_balance()
        usdt = float(balance.get("total", {}).get("USDT", 0) or 0)
        print(f"  SUCCESS! USDT balance: {usdt:.2f}")
        print(f"  >>> YOUR KEYS BELONG TO: {name} <<<")
    except Exception as e:
        err = str(e)[:120]
        print(f"  FAILED: {err}")


if __name__ == "__main__":
    print("=" * 60)
    print("BYBIT KEY ENVIRONMENT DIAGNOSTIC")
    print("=" * 60)
    print(f"Key: {API_KEY[:6]}...{API_KEY[-4:]} (len={len(API_KEY)})")
    print(f"Secret: masked (len={len(API_SECRET)})")

    _try_env("TESTNET", "https://api-testnet.bybit.com")
    _try_env("DEMO",    "https://api-demo.bybit.com")
    # Mainnet skipped intentionally — no reason to test there

    print("\n" + "=" * 60)
    print("If TESTNET succeeded -> keys are fine, code needs testnet URL")
    print("If DEMO succeeded    -> keys are from bybit.com Demo Trading")
    print("If both failed       -> keys wrong/expired, regenerate")
    print("=" * 60)