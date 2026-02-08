# Intraday Regime Lab — v0.3 (Frozen)

This repository documents controlled intraday research on MES (Micro E-mini S&P 500).

## What was tested
- Time-of-day directional drift
- Fixed-horizon directional strategies
- Hour × holding-period regime maps
- Mean-reversion relative to hour open
- All tests evaluated net of realistic transaction costs

## Results
- Directional drift exists statistically but collapses after costs
- Mean reversion performs worse than directional after costs
- No intraday time-based regime in MES survives friction

## Conclusion
There is no standalone intraday edge in MES based on:
- time of day
- fixed holding periods
- simple directional or mean-reverting logic

This version is frozen as a **negative-results research release**.

## Status
- v0.3 frozen
- No further development in this repository
- New hypotheses continue in v0.4 (separate project)

This repo is released for educational and research purposes.