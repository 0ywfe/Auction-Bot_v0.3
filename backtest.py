# backtest.py
# V0.2 — Edge Validation Backtester (NO STRATEGY CHANGES)

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
import argparse

from market import MarketState
from system import AuctionFailureSystem
from config import Config


# =========================
# DATA LOADING
# =========================

def load_csv(path: str) -> List[MarketState]:
    df = pd.read_csv(path)

    # normalize column names
    df.columns = [c.lower() for c in df.columns]

    # map timestamp column
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif "ts_event" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ts_event"])
    else:
        raise ValueError(
            f"No usable timestamp column found. Columns: {df.columns.tolist()}"
        )

    market_states = []
    for _, r in df.iterrows():
        market_states.append(
            MarketState(
                timestamp=r["timestamp"],
                open=r["open"],
                high=r["high"],
                low=r["low"],
                close=r["close"],
                volume=int(r["volume"]),
                vwap=r.get(
                    "vwap",
                    (r["open"] + r["high"] + r["low"] + r["close"]) / 4,
                ),
                bid_volume=int(r.get("bid_volume", 0)),
                ask_volume=int(r.get("ask_volume", 0)),
                trades=int(r.get("trades", 0)),
                has_bid_ask=("bid_volume" in df.columns and "ask_volume" in df.columns),
            )
        )

    return market_states



# =========================
# BACKTEST CORE
# =========================

def run_backtest(
    market_states: List[MarketState],
    initial_capital: float = 10000.0,
) -> pd.DataFrame:

    system = AuctionFailureSystem(initial_capital=initial_capital)
    trades: List[Dict] = []

    for ms in market_states:
        system.process_market_update(ms)

        if system.last_closed_trade:
            trades.append(system.last_closed_trade.copy())
            system.last_closed_trade = None

    return pd.DataFrame(trades)


# =========================
# METRICS
# =========================

def compute_edge_metrics(trades: pd.DataFrame) -> Dict:
    if trades.empty:
        return {}

    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] <= 0]

    expectancy = trades["net_pnl"].mean()
    win_rate = len(wins) / len(trades)
    avg_win = wins["net_pnl"].mean() if not wins.empty else 0.0
    avg_loss = losses["net_pnl"].mean() if not losses.empty else 0.0

    gross_profit = wins["net_pnl"].sum()
    gross_loss = abs(losses["net_pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        "trades": len(trades),
        "expectancy": expectancy,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
    }


def print_metrics(metrics: Dict):
    print("\n=== EDGE METRICS (NET OF COSTS) ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:15s}: {v: .4f}")
        else:
            print(f"{k:15s}: {v}")


# =========================
# CONDITIONAL EXPECTANCY
# =========================

def print_conditional_tables(trades: pd.DataFrame):
    if trades.empty:
        return

    trades["hour"] = trades["entry_time"].dt.hour

    print("\n=== EXPECTANCY BY HOUR ===")
    print(trades.groupby("hour")["net_pnl"].mean().round(2))

    print("\n=== EXPECTANCY BY EXIT REASON ===")
    print(trades.groupby("exit_reason")["net_pnl"].mean().round(2))


# =========================
# STRESS TESTING
# =========================

def stress_test(
    market_states: List[MarketState],
    base_config: Config,
    initial_capital: float,
):
    print("\n=== PARAMETER STRESS TEST (±25%) ===")

    params = {
        "LOW_VOLUME_THRESHOLD": base_config.LOW_VOLUME_THRESHOLD,
        "HIGH_VOLUME_THRESHOLD": base_config.HIGH_VOLUME_THRESHOLD,
        "TIME_STOP_MINUTES": base_config.TIME_STOP_MINUTES,
    }

    rows = []

    for name, base in params.items():
        for mult in [0.75, 1.0, 1.25]:
            cfg = Config()
            setattr(cfg, name, base * mult)

            system = AuctionFailureSystem(initial_capital=initial_capital)
            system.config = cfg

            trades = []
            for ms in market_states:
                system.process_market_update(ms)
                if system.last_closed_trade:
                    trades.append(system.last_closed_trade.copy())
                    system.last_closed_trade = None

            df = pd.DataFrame(trades)
            expectancy = df["net_pnl"].mean() if not df.empty else 0.0

            rows.append({
                "parameter": name,
                "multiplier": mult,
                "expectancy": expectancy,
                "trades": len(df),
            })

    stress_df = pd.DataFrame(rows)
    print(stress_df.pivot(index="multiplier", columns="parameter", values="expectancy").round(2))


# =========================
# CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auction Failure Bot — V0.2 Backtest")
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV")
    parser.add_argument("--out", default="trades_v02.csv", help="Output trade CSV")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--stress", action="store_true")

    args = parser.parse_args()

    market_states = load_csv(args.csv)
    trades = run_backtest(market_states, args.capital)

    trades.to_csv(args.out, index=False)
    print(f"\nSaved trade dataset → {args.out}")

    metrics = compute_edge_metrics(trades)
    print_metrics(metrics)
    print_conditional_tables(trades)

    if args.stress:
        stress_test(market_states, Config(), args.capital)
