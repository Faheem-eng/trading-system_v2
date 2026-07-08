# TRADING SYSTEM V2 — FINAL SYSTEM SPECIFICATION
**Universe:** XRPUSDT, SOLUSDT, BTCUSDT (Binance Spot)
(MATICUSDT retired 2026-07-08 — POL rebrand, see OPERATIONS_RUNBOOK.md)
**Version:** 1.1.0 | **Status:** LOCKED (Phase 6) | **Date:** July 2026
Yeh document poore system ka single source of truth hai.
Koi bhi change ko is document mein version-bump ke saath aana hoga.

---

## 1. SYSTEM IDENTITY

**Naam:** Regime Overlay Strategy (4h Crypto Momentum Timing)
**Type:** Long-only market-timing overlay (NOT high-frequency, NOT per-signal trading)
**Universe:** XRPUSDT, SOLUSDT, BTCUSDT, MATICUSDT (Binance Spot)
**Timeframe:** 4h decision bars + 1D HTF context + BTC 4h cross-asset

## 2. THE EDGE (Kya, Kyun, Kaise Proven)

**Kya:** LightGBM model 116 features se market regime detect karta hai
(bullish-hold vs risky-cash). Model ka asli skill DIRECTION nahi,
RISK-REGIME detection hai (top features: downside_vol_ratio, var_95,
vol_clustering, htf context).

**Proof chain (har step pre-registered):**
1. IC 0.03-0.05, time-stable 4/5 walk-forward folds (4 symbols)
2. Overlay backtest: +145.96% vs B&H +55.04% (20 months, 5 regimes)
3. Robustness: 3/3 threshold configs (0.50/0.40, 0.55/0.45, 0.60/0.50)
   sab profitable, sab B&H se aage — edge is a hill, not a spike

## 3. FINAL PARAMETERS (LOCKED — tuning FORBIDDEN without re-validation)

| Parameter | Value | Source |
|---|---|---|
| Entry threshold | P(up) > 0.55 | Pre-declared, robustness-tested |
| Exit threshold | P(up) < 0.45 | Hysteresis (churn prevention) |
| Position size (in-strategy) | 100% | 3 sizing attempts FAILED (see §5) |
| **Capital allocation** | **60% of trading capital** | Phase 5 conclusion |
| Model | LightGBM binary, depth 4, leaves 8, reg 0.6 | binary_1h.py v2 params |
| Labels (training) | Triple barrier 3×ATR/3×ATR, 20-bar | López de Prado |
| Retraining | Every ~900 4h bars (~5 months) | Walk-forward cadence |
| Costs assumed | 0.125% per switch side | Spot taker + slippage |

## 4. EXPECTED BEHAVIOR (Backtest-Derived, NOT Guaranteed)

- **Per 20-month cycle (60% allocation):** ~+87% total capital return
- **Worst drawdown:** ~-30% of total capital
- **Character:** Rally capture (F2: +125% portfolio fold) + crash
  protection (bear folds alpha +11 to +19%) + chop survival
- **Weakness:** Mild-bull underperformance (F3: -15% alpha),
  edge concentrated in strong rallies

## 5. FORBIDDEN ACTIONS (3 Failures Se Proven)

| Action | Kyun Forbidden | Proof |
|---|---|---|
| Vol-targeted sizing (any flavor) | Edge -37%/-20% destroyed | v1.0, v1.1 |
| Circuit breakers / kill-switches | -20% DD crypto mein routine hai; retention -5% | v1.2 |
| Threshold tuning on live results | Backtest overfitting | Discipline |
| Manual override of signals | Rules > emotions | Phase 6 rule |
| Shorting | Spot account, impossible | Reality |
| Leverage | Cap 100%, DD math breaks | Reality |

**Risk control = capital allocation (60%), NOT in-strategy intervention.**

## 6. KNOWN RISKS (Honest Register)

1. **F2-dependence:** Bada hissa profit mega-rally capture se.
   Rally na aaye → flat/negative periods expected.
2. **Correlated universe:** 4 crypto symbols ≈ 2-3 independent bets.
3. **Recent regime (F5) weak:** 2026 bear mein -18% (B&H -34%).
4. **Backtest ≠ live:** Slippage, gaps, execution delays unmodeled.
   Phase 7 paper trading is the real exam.

## 7. OPERATIONAL PROTOCOL

1. Har 4h bar close ke baad: signal generator run karo (Phase 6.3 script)
2. Signal output follow karo — NO exceptions, NO overrides
3. Har ~5 months: model retrain (walk-forward protocol)
4. Monthly: live vs backtest expectation compare (Phase 7 harness)
5. Agar live 3 months consecutively backtest envelope se bahar →
   HALT, investigate, re-validate (pre-registered kill condition)