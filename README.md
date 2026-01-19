# Auction Bot — v0.2 (Deprecated / No Edge)

**Status: Experimental — Core Hypothesis Rejected**

This version represents the conclusion of an auction-failure research track on MES (Micro E-mini S&P 500).

Multiple controlled backtests rejected the core premise that auction failure + volume confirmation produces a tradable edge under realistic costs.

---

## What v0.2 Is

* A **fully instrumented intraday trading system**
* Designed to test auction-failure and structure-based hypotheses
* Includes:

  * Finite-state machine (FSM) with invariant enforcement
  * Risk kill-switch and capital protection
  * Detailed trade-density and diagnostics reporting
  * TIME_STOP exit logic
  * Structure-anchored stop logic (value-area based)

---

## What Was Learned

* Fixed, price-based stops were structurally invalid
* Structure-anchored stops materially reduced stop-loss damage
* TIME_STOP exits showed consistent positive contribution
* Strategy performance was dominated by a narrow time-of-day effect
* No robust, regime-independent edge was found

As a result, the **auction-failure hypothesis was rejected**.

---

## What v0.2 Is *Not*

* ❌ Not a profitable trading system
* ❌ Not a validated auction-failure strategy
* ❌ Not suitable for live trading

This code is released **for research and educational purposes only**.

---

## Guarantees (Still Valid)

* No double exits
* No capital leakage
* FSM invariants enforced
* Risk kill-switch enforced
* Diagnostics and stress tests fully passing

---

## Development Policy

**v0.2 is frozen. Do not modify.**

Further development continues in **v0.3**, which abandons the auction-failure premise and explores alternative intraday hypotheses.