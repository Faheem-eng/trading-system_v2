# Trading System V2 — Institutional Regime Overlay (Binance Spot)

**Status:** Phase 7 — Paper Trading (Live since 2026-07-08)**
**Universe:** XRPUSDT | SOLUSDT | BTCUSDT
**Type:** Long-only 4h regime-timing overlay

---

## What This System Does

A LightGBM model trained on 116 features detects market regimes
(bullish-hold vs risky-cash) on 4-hour bars. When P(up) > 0.55,
the system holds long; when P(up) < 0.45, it moves to cash.
Hysteresis prevents churn in the dead band between thresholds.

**Backtest result (walk-forward, 5 folds, 2024-2026):**
- System: +145.96% | Buy-and-hold: +55.04% | Worst DD: -49.1%
- Robust across 3 threshold configs (0.50/0.60 also profitable)
- Capital allocation: 60% deployed, 40% cash (risk control)

---

## Quick Start

```cmd
REM Daily signal (or runs automatically every 4h via Task Scheduler)
python -m live.signal_generator

REM Monthly paper-trading review (run on 1st of each month)
python -m live.monthly_review

REM Health check (run after any code change)
python -m pytest tests/ -v
```

---

## Project Structure