# backtest.py
# V0.2 — Edge Validation Backtester + Trade-Density Calibration

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import argparse

from market import MarketState
from system import AuctionFailureSystem
from config import Config
from datetime import timedelta

HOLD_WINDOWS = [15, 30, 45, 60, 90]
TEST_HOURS = [10, 11, 13, 14, 15]


def load_csv(path: str) -> List[MarketState]:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    elif "ts_event" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ts_event"], utc=True)
    else:
        raise ValueError(f"No usable timestamp column found. Columns: {df.columns.tolist()}")

    # UTC -> New York, KEEP tzinfo
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")

    # --- CRITICAL: FILTER TO A SINGLE FUTURES STREAM ---
    # Drop spreads like "MESU4-MESZ4"
    df["symbol"] = df["symbol"].astype(str)
    df = df[~df["symbol"].str.contains("-", regex=False)]

    # Keep only MES contract symbols like MESU4, MESZ4 (MES + letter + digit)
    df = df[df["symbol"].str.match(r"^MES[A-Z]\d$", na=False)]

    # One bar per timestamp: pick the contract with max volume (liquidity proxy)
    df = (
        df.sort_values(["timestamp", "volume"], ascending=[True, False])
          .groupby("timestamp", as_index=False)
          .first()
          .sort_values("timestamp")
    )

    market_states: List[MarketState] = []
    for _, r in df.iterrows():
        ts = r["timestamp"].to_pydatetime()  # tz-aware NY datetime
        market_states.append(
            MarketState(
                timestamp=ts,
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=int(r["volume"]),
                vwap=float(r.get("vwap", (r["open"] + r["high"] + r["low"] + r["close"]) / 4)),
                bid_volume=int(r.get("bid_volume", 0)),
                ask_volume=int(r.get("ask_volume", 0)),
                trades=int(r.get("trades", 0)),
                has_bid_ask=("bid_volume" in df.columns and "ask_volume" in df.columns),
            )
        )
    return market_states



def _run_system(market_states: List[MarketState], cfg: Config, capital: float) -> Tuple[pd.DataFrame, Dict]:
    system = AuctionFailureSystem(initial_capital=capital)
    system.config = cfg  # keep your existing pattern

    trades: List[Dict] = []
    for ms in market_states:
        system.process_market_update(ms)
        if system.last_closed_trade:
            trades.append(system.last_closed_trade.copy())
            system.last_closed_trade = None

    df = pd.DataFrame(trades)
    # normalize datetime columns if present
    for c in ("setup_time", "entry_time", "exit_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    return df, system.density


def run_directional_drift(market_states: List[MarketState], hold_minutes: int = 60) -> pd.DataFrame:
    trades = []

    last_hour = None
    in_trade = False
    entry_price = None
    entry_time = None
    entry_hour = None

    for ms in market_states:
        ts = ms.timestamp
        hour = ts.hour

        # only test RTH-style hours you already observed
        if hour not in [10, 11, 13, 14, 15]:
            continue

        # detect new hour
        hour_bucket = ts.replace(minute=0, second=0, microsecond=0)

        if not in_trade and hour_bucket != last_hour:
            # enter LONG at first bar of hour
            in_trade = True
            entry_price = ms.open
            entry_time = ts
            entry_hour = hour
            last_hour = hour_bucket
            continue

        if in_trade and ts >= entry_time + timedelta(minutes=hold_minutes):
            raw_pnl = (ms.close - entry_price) * 5
            cost = 10.0
            pnl = raw_pnl - cost

            trades.append({
                "entry_time": entry_time,
                "exit_time": ts,
                "hour": entry_hour,
                "net_pnl": pnl,
                "exit_reason": "TIME_STOP"
            })
            in_trade = False
            entry_price = None
            entry_time = None
            entry_hour = None

    return pd.DataFrame(trades)


def run_mean_reversion(market_states: List[MarketState], hold_minutes: int) -> pd.DataFrame:
    trades = []

    last_hour = None
    in_trade = False
    entry_price = None
    entry_time = None
    entry_hour = None
    direction = None
    hour_open = None

    for ms in market_states:
        ts = ms.timestamp
        hour = ts.hour

        if hour not in [10, 11, 13, 14, 15]:
            continue

        hour_bucket = ts.replace(minute=0, second=0, microsecond=0)

        if hour_bucket != last_hour:
            # reset for new hour
            hour_open = ms.open
            last_hour = hour_bucket
            in_trade = False

        # enter on first deviation from hour open
        if not in_trade and hour_open is not None:
            if ms.close > hour_open:
                direction = -1  # SHORT
            elif ms.close < hour_open:
                direction = 1   # LONG
            else:
                continue

            in_trade = True
            entry_price = ms.close
            entry_time = ts
            entry_hour = hour
            continue

        if in_trade and ts >= entry_time + timedelta(minutes=hold_minutes):
            raw_pnl = (ms.close - entry_price) * 5 * direction
            pnl = raw_pnl - 10.0  # costs
            trades.append({
                "entry_time": entry_time,
                "exit_time": ts,
                "hour": entry_hour,
                "hold_minutes": hold_minutes,
                "net_pnl": pnl,
                "exit_reason": "TIME_STOP"
            })
            in_trade = False
            entry_price = None
            entry_time = None
            entry_hour = None
            direction = None

    return pd.DataFrame(trades)


def build_mean_reversion_map(market_states: List[MarketState]) -> pd.DataFrame:
    rows = []

    for hold in HOLD_WINDOWS:
        trades = run_mean_reversion(market_states, hold_minutes=hold)
        if trades.empty:
            continue

        grouped = trades.groupby("hour")["net_pnl"].mean()
        for hour, exp in grouped.items():
            rows.append({
                "hour": hour,
                "hold_minutes": hold,
                "expectancy": round(exp, 2),
                "trades": int((trades["hour"] == hour).sum())
            })

    return pd.DataFrame(rows)



def build_regime_map(market_states: List[MarketState]) -> pd.DataFrame:
    rows = []

    for hold in HOLD_WINDOWS:
        trades = run_directional_drift(market_states, hold_minutes=hold)
        if trades.empty:
            continue

        grouped = trades.groupby("hour")["net_pnl"].mean()
        for hour, exp in grouped.items():
            rows.append({
                "hour": hour,
                "hold_minutes": hold,
                "expectancy": round(exp, 2),
                "trades": int((trades["hour"] == hour).sum())
            })

    return pd.DataFrame(rows)


def compute_edge_metrics(trades: pd.DataFrame) -> Dict:
    if trades.empty:
        return {}

    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] <= 0]

    expectancy = float(trades["net_pnl"].mean())
    win_rate = float(len(wins) / len(trades))
    avg_win = float(wins["net_pnl"].mean()) if not wins.empty else 0.0
    avg_loss = float(losses["net_pnl"].mean()) if not losses.empty else 0.0

    gross_profit = float(wins["net_pnl"].sum())
    gross_loss = float(abs(losses["net_pnl"].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        "trades": int(len(trades)),
        "expectancy": expectancy,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
    }


def print_metrics(metrics: Dict):
    print("\n=== EDGE METRICS (NET OF COSTS) ===")
    if not metrics:
        print("No completed trades.")
        return
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:15s}: {v: .4f}")
        else:
            print(f"{k:15s}: {v}")


def print_conditional_tables(trades: pd.DataFrame):
    if trades.empty or "entry_time" not in trades.columns:
        return

    trades = trades.dropna(subset=["entry_time"])
    if trades.empty:
        return

    trades["hour"] = trades["entry_time"].dt.hour

    print("\n=== EXPECTANCY BY HOUR ===")
    print(trades.groupby("hour")["net_pnl"].mean().round(2))

    print("\n=== EXPECTANCY BY EXIT REASON ===")
    print(trades.groupby("exit_reason")["net_pnl"].mean().round(2))


def print_density_report(density: Dict):
    print("\n=== TRADE-DENSITY REPORT ===")
    for k in [
        "bars_total",
        "bars_in_session",
        "bars_outside_session",
        "setups_detected",
        "setups_failed_volume_confirm",
        "setups_expired_time_window",
        "entries_blocked_by_session",
        "entries_blocked_by_risk",
    ]:
        if k in density:
            print(f"{k:30s}: {density[k]}")


def trades_per_year(trades_count: int, market_states: List[MarketState]) -> float:
    days = len({ms.timestamp.date() for ms in market_states})
    if days <= 0:
        return 0.0
    return trades_count / days * 252.0


def calibrate_density(market_states: List[MarketState], capital: float, target_trades_per_year: int = 100) -> Config:
    cfg = Config()

    print("\n=== CALIBRATION: BASELINE RUN ===")
    base_trades, base_density = _run_system(market_states, cfg, capital)
    base_tpy = trades_per_year(len(base_trades), market_states)
    print_density_report(base_density)
    print(f"baseline trades: {len(base_trades)} | trades/year≈ {base_tpy:.1f}")

    if base_tpy >= target_trades_per_year:
        print("Target met at baseline.")
        return cfg

    # Ordered, minimal relaxation (one knob at a time, small steps)
    if base_density.get("setups_detected", 0) == 0:
        steps = [("LOW_VOLUME_THRESHOLD", +0.05, 1.00)]
    elif base_density.get("setups_failed_volume_confirm", 0) > 0:
        steps = [("HIGH_VOLUME_THRESHOLD", -0.05, 0.90)]
    elif base_density.get("setups_expired_time_window", 0) > 0:
        steps = [("BASE_ACCEPTANCE_WINDOW_MINUTES", +2, 20)]
    else:
        steps = [
            ("HIGH_VOLUME_THRESHOLD", -0.05, 0.90),
            ("LOW_VOLUME_THRESHOLD", +0.05, 1.00),
            ("BASE_ACCEPTANCE_WINDOW_MINUTES", +2, 20),
        ]

    for param, delta, bound in steps:
        print(f"\n=== CALIBRATION: ADJUST {param} ===")
        for _ in range(1, 7):
            current = getattr(cfg, param)
            new_val = current + delta

            if delta > 0 and new_val > bound:
                break
            if delta < 0 and new_val < bound:
                break

            setattr(cfg, param, new_val)

            trades_df, density = _run_system(market_states, cfg, capital)
            tpy = trades_per_year(len(trades_df), market_states)

            print_density_report(density)
            print(f"{param}={getattr(cfg, param)} | trades: {len(trades_df)} | trades/year≈ {tpy:.1f}")

            if tpy >= target_trades_per_year:
                print("\nTarget trade density reached with minimal change.")
                return cfg

    print("\nCalibration could not reach target trade density with minimal allowed changes.")
    return cfg


def stress_test(market_states: List[MarketState], base_config: Config, initial_capital: float):
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
            cfg.LOW_VOLUME_THRESHOLD = base_config.LOW_VOLUME_THRESHOLD
            cfg.HIGH_VOLUME_THRESHOLD = base_config.HIGH_VOLUME_THRESHOLD
            cfg.TIME_STOP_MINUTES = base_config.TIME_STOP_MINUTES
            if hasattr(cfg, "BASE_ACCEPTANCE_WINDOW_MINUTES"):
                cfg.BASE_ACCEPTANCE_WINDOW_MINUTES = getattr(base_config, "BASE_ACCEPTANCE_WINDOW_MINUTES", 10)

            setattr(cfg, name, base * mult)

            df, _ = _run_system(market_states, cfg, initial_capital)
            expectancy = float(df["net_pnl"].mean()) if not df.empty else 0.0

            rows.append({"parameter": name, "multiplier": mult, "expectancy": expectancy, "trades": len(df)})

    stress_df = pd.DataFrame(rows)
    print(stress_df.pivot(index="multiplier", columns="parameter", values="expectancy").round(2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auction Failure Bot — V0.2 Backtest")
    parser.add_argument("--regime-map", action="store_true", help="Build hour × holding-period expectancy map")
    parser.add_argument("--directional", action="store_true", help="Run directional drift by hour experiment")
    parser.add_argument("--hold", type=int, default=45, help="Holding period in minutes for directional drift experiment")
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV")
    parser.add_argument("--out", default="trades_v02.csv", help="Output trade CSV")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--calibrate_density", action="store_true")
    parser.add_argument("--mean-reversion", action="store_true", help="Build hour × holding-period mean reversion map")


    args = parser.parse_args()


    market_states = load_csv(args.csv)


    if args.mean_reversion:
        df = build_mean_reversion_map(market_states)

        print("\n=== INTRADAY MEAN REVERSION REGIME MAP ===")
        print(df.pivot(index="hour",
                    columns="hold_minutes",
                    values="expectancy").fillna(0))

        print("\nTrades per cell:")
        print(df.pivot(index="hour",
                    columns="hold_minutes",
                    values="trades").fillna(0))

    exit(0)



    if args.regime_map:
        df = build_regime_map(market_states)

        print("\n=== INTRADAY DIRECTIONAL REGIME MAP ===")
        print(df.pivot(index="hour",
                    columns="hold_minutes",
                    values="expectancy").fillna(0))

        print("\nTrades per cell:")
        print(df.pivot(index="hour",
                    columns="hold_minutes",
                    values="trades").fillna(0))

    exit(0)

    if args.directional:
        trades = run_directional_drift(market_states, hold_minutes=args.hold)


        print("\n=== DIRECTIONAL DRIFT (LONG-ONLY, TIME EXIT) ===")
        if trades.empty:
            print("No trades generated.")
        else:
            print("\nExpectancy by hour:")
            print(trades.groupby("hour")["net_pnl"].mean().round(2))
            print("\nTrades by hour:")
            print(trades.groupby("hour").size())
            print(f"\nOverall expectancy: {trades['net_pnl'].mean():.2f}")

        exit(0)

    trades.to_csv(args.out, index=False)
    print(f"\nSaved trade dataset → {args.out}")

    print_density_report(density)

    metrics = compute_edge_metrics(trades)
    print_metrics(metrics)
    print_conditional_tables(trades)

    if args.stress:
        stress_test(market_states, cfg, args.capital)

    if trades.empty:
        print("\nVERDICT: edge untestable (no completed trades).")
    else:
        exp = metrics.get("expectancy", 0.0)
        if exp <= 0:
            print("\nVERDICT: edge rejected (expectancy <= 0 net of costs).")
        else:
            print("\nVERDICT: edge survives calibration (expectancy > 0 net of costs).")
