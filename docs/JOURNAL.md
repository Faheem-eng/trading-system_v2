# Project Journal — Trading System V2

Complete honest record of every major decision,
discovery, and failure. The anti-vibe-coding document.

---

## Phase 0 — Foundation (June 2026)

**Goal:** Institutional-grade scaffolding from day one.
**Key decisions:**
- uv for dependency management
- structlog for structured logging
- Type-safe config loader
- Git-first: every milestone committed before next phase

---

## Phase 1 — Data Collection

**Result:** 35M bars, 56 symbols, 4.5 years, Binance Spot.
**Key discovery:** MATIC/USDT → POL rebrand in Oct 2024 broke
fresh data pipeline. Flagged early, handled in Phase 6.
**Architecture:** DuckDB + Parquet. RAM-efficient for 8GB constraint.

---

## Phase 2 — Feature Engineering (116 Features)

**Result:** 14 modules, 116 features, LEAKAGE-PROVEN.
**Key bugs caught:**
- `structure_position` blew up to -2037 (division by near-zero
  swing range) → fixed with range-guard + clip[-0.5, 1.5]
- Duplicate feature name `momentum_quality` → renamed
- 1h resampling via iloc[::12] was WRONG (not OHLCV-aware)
  → replaced with proper resample().agg()
**Validation:** 103/103 features clean on leakage test
(corrupt future bars → past features UNCHANGED).

---

## Phase 3 — ML: The Edge Discovery Journey

**The honest path:**

| Test | Result | Lesson |
|---|---|---|
| 5m 3-class | val 45.7%, IC 0.015 | No edge, Hurst 0.40 |
| 5m binary | best_iter=1 (gave up) | Direction near-random |
| 1h binary v1 | IC 0.045, val 53.2% | BREAKTHROUGH |
| 1h binary v2 | IC 0.030, gap stable | Edge not overfit artifact |
| Multi-symbol | 6/12 symbols edge | XRP/SOL/BTC/MATIC confirmed |
| Meta-labeling | WR ~50%, costs not cleared | Binary filter adds no value |
| Walk-forward | 4/6 TIME-STABLE | Edge is real across regimes |

**Key insight:** Model learned RISK DETECTION, not direction.
Top features: downside_vol_ratio, var_95, vol_clustering, htf_rsi.
This insight shaped every subsequent design decision.

---

## Phase 4 — Backtesting (5 Structures Tested)

**Pre-registered verdict before each test.
All tests honest — no goalpost moving.**

| Structure | Result | Verdict |
|---|---|---|
| 1h symmetric barrier | -19.4% | NO-GO: cost math trap |
| 1h asymmetric 4x/2x | -17.8% | NO-GO: hit-prob cancelled payoff |
| 4h threshold barrier | +44.2% < B&H +55% | NO-GO: barriers cap rallies |
| 4h regime overlay | +145.96% > B&H +55% | GO |
| Threshold sensitivity | 3/3 configs beat B&H | ROBUST — hill not spike |

**Key learning:** Model = regime detector.
Barrier exits were WASTING the regime signal.
Hold-while-bullish = let the regime play out.

---

## Phase 5 — Risk Management (3 Failures → Capital Fraction)

**3 in-strategy interventions, all pre-registered, all honest:**

| Intervention | Compound | DD | Edge Retention |
|---|---|---|---|
| Raw overlay (baseline) | +145.96% | -49.1% | 100% |
| Total-vol sizing (v1.0) | +21.10% | -25.2% | -37% FAIL |
| Downside-vol sizing (v1.1) | +36.82% | -28.1% | -20% FAIL |
| Circuit breaker (v1.2) | +50.53% | -33.5% | -5% FAIL |

**Fundamental law discovered:** Crypto momentum overlay +
continuous vol-sizing = STRUCTURALLY incompatible.
Crypto rallies contain violent pullbacks — any sizing
that reacts to volatility dilutes concentrated rally capture.

**Solution:** 60% capital allocation (account level).
DD: -49% → -29.5%. Return: +87.6%. Edge: 100% retained.
Pure arithmetic. Zero regime-dependence.

---

## Phase 6 — System Assembly (July 2026)

**MATIC retirement:** Binance MATIC→POL rebrand (Oct 2024)
killed fresh data. POL rejected (only 1.5yr history, our
4.5yr validation doesn't transfer). Final universe: 3 symbols.

**First live signal:** 2026-07-08 03:40 UTC
- SOL: LONG (P=0.650) ← Paper trade entry #1
- XRP: CASH | BTC: CASH

**Signal generator key design decisions:**
- Historical parquet + ccxt fresh bars (no API key needed)
- Only COMPLETED bars (forming bar = live lookahead)
- ASCII-only output (Windows cp1252 scheduler-safe)
- Fail-safe: any failure = NO_SIGNAL + hold previous

---

## Phase 7 — Paper Trading (Ongoing from 2026-07-08)

**Infrastructure built:**
- Auto paper-logger (price fetch + round-trip returns)
- Monthly review tool (envelope gate + HALT counter)
- Task Scheduler (4h repeat, indefinitely)
- 26-test pytest suite (runs in 25 seconds)

**Open positions:** SOL LONG since 2026-07-08 @ 76.30
SOL P(up) journey: 0.650 → 0.583 → 0.568 → 0.708

**Phase 8 gate:** 2+ months paper + envelope pass + zero violations
Earliest go-live: ~September 2026 (if envelope conditions met)

---

## Recurring Lessons (Apply To All Future Projects)

1. **Pre-register before testing.** Results feel very
   different when the verdict is written first.

2. **Small samples lie.** 44-bar precision of 72% = noise.
   Only trust n>500 results.

3. **Failure IS the research.** 4 honest failures designed
   the 5th success. Vibe-coding would have stopped at the
   first "promising" result.

4. **Model learns what the features teach.** Risk features
   → risk model. Don't fight the model's nature; align the
   trade structure to it.

5. **Costs are not negotiable.** IC 0.03 at 0.25% cost =
   math is tight. Every design must clear this bar.

6. **Git is a safety net.** Three files accidentally deleted
   during the project. All recovered in seconds.