# risk.py - Risk management module

from typing import Tuple
from config import Config


class RiskManager:
    """Hard risk limits, no discretion"""

    def __init__(self, config: Config, initial_capital: float = 10000.0):
        self.config = config
        self.initial_capital = float(initial_capital)

        # Account
        self.current_capital = float(initial_capital)
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0

        # Limits
        # Config.MAX_DAILY_LOSS is a fraction (e.g. 0.02 = 2%)
        self.max_daily_loss = float(config.MAX_DAILY_LOSS) * self.initial_capital
        self.max_daily_trades = int(config.MAX_DAILY_TRADES)

        # Drawdown tracking
        self.peak_capital = float(initial_capital)
        self.current_drawdown = 0.0

        # Position limits
        self.max_position_size = int(config.MAX_POSITION_SIZE)

        # Hard stop flag (for STOPPED enforcement)
        self.hard_stop_triggered = False

    def update_pnl(self, pnl: float):
        """Update PnL and update hard-stop flag if breached."""
        pnl = float(pnl)

        self.daily_pnl += pnl
        self.current_capital += pnl
        self.daily_trades += 1

        # Peak + drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        if self.peak_capital > 0:
            self.current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        else:
            self.current_drawdown = 0.0

        # Consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Hard kill check (daily loss)
        if self.daily_pnl <= -self.max_daily_loss:
            self.hard_stop_triggered = True

    def is_hard_stopped(self) -> bool:
        return self.hard_stop_triggered

    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed"""

        # Hard stop wins
        if self.hard_stop_triggered:
            return False, "Daily loss limit reached"

        # Daily loss limit (redundant safety)
        if self.daily_pnl <= -self.max_daily_loss:
            return False, "Daily loss limit reached"

        # Daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return False, "Daily trade limit reached"

        # Consecutive losses
        if self.consecutive_losses >= self.config.MAX_CONSECUTIVE_LOSSES:
            return False, f"{self.config.MAX_CONSECUTIVE_LOSSES} consecutive losses"

        # Drawdown warnings / limits
        if self.current_drawdown >= 0.10:
            return False, f"Max drawdown reached: {self.current_drawdown:.1%}"

        if self.current_drawdown >= 0.05:
            return True, f"Drawdown warning: {self.current_drawdown:.1%}"

        return True, "OK"

    def get_position_size(self, stop_distance_points: float, instrument: str = "MES") -> int:
        """Calculate position size based on risk"""
        stop_distance_points = float(stop_distance_points)

        # Risk $ per trade
        risk_per_trade = float(self.config.RISK_PER_TRADE) * self.current_capital

        # $ per point
        dollars_per_point = 5.0 if instrument == "MES" else 2.0

        dollar_risk = stop_distance_points * dollars_per_point
        if dollar_risk <= 0:
            return 1

        contracts = int(risk_per_trade / dollar_risk)

        # Drawdown scaling
        if self.current_drawdown >= 0.08:
            contracts = max(1, contracts // 4)
        elif self.current_drawdown >= 0.05:
            contracts = max(1, contracts // 2)

        # Hard limits
        contracts = min(contracts, self.max_position_size)
        contracts = max(1, contracts)
        return contracts

    def reset_daily(self):
        """Reset daily counters (does NOT reset capital)."""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.hard_stop_triggered = False
