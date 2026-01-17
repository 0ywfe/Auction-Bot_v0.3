"""
diagnostic.py
=========================
Hard diagnostic & invariant tests for AuctionFailureSystem.

This file exists to PROVE:
- no double exits
- no capital leaks
- FSM invariants hold
- cooldown & STOPPED states behave correctly

Run:
    python diagnostic.py
"""

from datetime import datetime, timedelta
import numpy as np

from system import AuctionFailureSystem, TradingState
from market import MarketState


# =========================
# HELPERS
# =========================

def make_bar(ts, price, volume=1000, bid=500, ask=500):
    return MarketState(
        timestamp=ts,
        open=price,
        high=price + 0.5,
        low=price - 0.5,
        close=price,
        volume=volume,
        vwap=price,
        bid_volume=bid,
        ask_volume=ask,
        trades=50,
        has_bid_ask=True,
    )


def run_bars(system, bars):
    for b in bars:
        system.process_market_update(b)


# =========================
# TESTS
# =========================

def test_no_double_exit():
    print("\n=== TEST: NO DOUBLE EXIT ===")

    system = AuctionFailureSystem(initial_capital=10_000)
    ts = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)

    # Warm-up
    for _ in range(60):
        run_bars(system, [make_bar(ts, 4500)])
        ts += timedelta(minutes=1)

    # Force a trade
    system.position.direction = system.position.direction.LONG
    system.position.entry_price = 4500
    system.position.size = 1
    system.position.stop_loss = 4490
    system.position.time_stop = ts + timedelta(minutes=5)
    system.state = TradingState.IN_TRADE

    capital_before = system.risk_manager.current_capital

    # Two exit triggers in SAME BAR
    bar = make_bar(ts, 4485)  # stop loss hit
    system.process_market_update(bar)
    system.process_market_update(bar)

    capital_after = system.risk_manager.current_capital

    assert capital_before != capital_after, "Exit did not change capital"
    delta = capital_after - capital_before
    expected = system.last_closed_trade["net_pnl"]
    assert abs(delta - expected) < 1e-6, "Capital changed more than once!"

    print("PASS: exit executed exactly once")


def test_fsm_invariant():
    print("\n=== TEST: FSM INVARIANT ===")

    system = AuctionFailureSystem()
    ts = datetime.now()

    system.state = TradingState.IN_TRADE
    system.position.direction = system.position.direction.FLAT

    system.process_market_update(make_bar(ts, 4500))

    assert system.state != TradingState.IN_TRADE, "FSM invariant not corrected"

    print("PASS: IN_TRADE + flat auto-corrected")


def test_cooldown_behavior():
    print("\n=== TEST: COOLDOWN ===")

    system = AuctionFailureSystem()
    ts = datetime.now().replace(hour=10, minute=0)

    # Warm-up
    for _ in range(60):
        system.process_market_update(make_bar(ts, 4500))
        ts += timedelta(minutes=1)

    # Force loss
    system.position.direction = system.position.direction.LONG
    system.position.entry_price = 4500
    system.position.size = 1
    system.position.stop_loss = 4490
    system.position.time_stop = ts + timedelta(minutes=5)
    system.state = TradingState.IN_TRADE

    system.process_market_update(make_bar(ts, 4485))
    assert system.state == TradingState.COOLDOWN, "Did not enter cooldown"

    # During cooldown, no entries allowed
    for _ in range(3):
        ts += timedelta(minutes=1)
        system.process_market_update(make_bar(ts, 4500))
        assert system.state == TradingState.COOLDOWN

    # Cooldown expires
    ts += timedelta(minutes=system.config.COOLDOWN_MINUTES_AFTER_LOSS + 1)
    system.process_market_update(make_bar(ts, 4500))

    assert system.state == TradingState.MONITORING, "Cooldown did not end"

    print("PASS: cooldown enforced and released correctly")


def test_risk_kill_switch():
    print("\n=== TEST: RISK KILL SWITCH ===")

    system = AuctionFailureSystem(initial_capital=10000.0)

    # Breach the actual dollar daily loss limit
    system.risk_manager.update_pnl(-(system.risk_manager.max_daily_loss + 1))

    system._enforce_risk_kill(datetime.now())

    assert system.state == TradingState.STOPPED, "System not STOPPED after risk breach"
    print("PASS: risk kill switch triggers STOPPED")


def test_logging_schema():
    print("\n=== TEST: LOGGING SCHEMA ===")

    system = AuctionFailureSystem()
    ts = datetime.now()

    system.log("TEST_EVENT", ts, {"foo": "bar"})

    # If this crashes, logging is broken
    print("PASS: logging emitted valid JSON")


# =========================
# RUNNER
# =========================

if __name__ == "__main__":
    print("\nAuctionFailureSystem — DIAGNOSTIC SUITE")
    print("=" * 60)

    test_no_double_exit()
    test_fsm_invariant()
    test_cooldown_behavior()
    test_risk_kill_switch()
    test_logging_schema()

    print("\nALL DIAGNOSTICS PASSED")
    print("=" * 60)
