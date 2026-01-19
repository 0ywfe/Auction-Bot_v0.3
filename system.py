# system.py — Auction Failure Trading System
# V0.2: Trade-Density Calibration (instrumentation + pending setup tracking)

from datetime import datetime, timedelta
from typing import Optional, Dict, List
from enum import Enum, auto
from dataclasses import dataclass
import logging
import json
import numpy as np
from zoneinfo import ZoneInfo  # Python 3.9+

from config import Config
from market import MarketState, BarVolumeProfile
from risk import RiskManager


class TradingState(Enum):
    INITIALIZING = auto()
    NO_TRADE = auto()
    MONITORING = auto()
    IN_TRADE = auto()
    COOLDOWN = auto()
    STOPPED = auto()


class PositionDirection(Enum):
    LONG = auto()
    SHORT = auto()
    FLAT = auto()


@dataclass
class Position:
    direction: PositionDirection
    entry_price: float
    entry_time: datetime
    size: int
    stop_loss: float
    profit_targets: List[float]
    time_stop: datetime

    @property
    def is_active(self) -> bool:
        return self.direction != PositionDirection.FLAT


class AdaptiveParameters:
    def __init__(self, config: Config):
        self.config = config
        self.avg_volume = 0.0
        self.recent_volatility = config.VOLATILITY_BASELINE
        self.recent_range = 0.0

    def update(self, market_data: List[MarketState]):
        n = len(market_data)
        if n < 5:
            return

        vols = [m.volume for m in market_data[-min(30, n):]]
        self.avg_volume = float(np.mean(vols)) if vols else 0.0

        prices = [m.close for m in market_data[-min(30, n):]]
        self.recent_range = max(prices) - min(prices)

        if n < 21:
            return

        returns = []
        for i in range(n - 20, n):
            prev = market_data[i - 1].close
            curr = market_data[i].close
            if prev > 0:
                returns.append(curr / prev - 1)

        if returns:
            self.recent_volatility = np.std(returns) * np.sqrt(252)

    def acceptance_window_minutes(self) -> int:
        base = getattr(self.config, "BASE_ACCEPTANCE_WINDOW_MINUTES", 10)
        vol_ratio = max(0.5, min(2.0, self.recent_volatility / self.config.VOLATILITY_BASELINE))
        return int(base / vol_ratio)


class AuctionFailureSystem:
    def __init__(self, instrument: str = "MES", initial_capital: float = 10000.0):
        self.config = Config()
        self.instrument = instrument

        # Timezones: backtest feeds NY tz-aware timestamps; if naive, assume NY by default.
        self.exchange_tz = ZoneInfo(getattr(self.config, "EXCHANGE_TIMEZONE", "America/New_York"))
        self.data_tz = ZoneInfo(getattr(self.config, "DATA_TIMEZONE", "America/New_York"))

        self.state = TradingState.INITIALIZING
        self.position = Position(PositionDirection.FLAT, 0.0, datetime.utcnow(), 0, 0.0, [], datetime.utcnow())
        self.cooldown_until: Optional[datetime] = None

        self.market_data: List[MarketState] = []
        self.volume_profile = BarVolumeProfile(self.config.VOLUME_PROFILE_WINDOW_MINUTES)
        self.adaptive = AdaptiveParameters(self.config)

        self.risk_manager = RiskManager(self.config, initial_capital)

        self.last_closed_trade: Optional[Dict] = None
        self.pending_setup: Optional[Dict] = None

        self._current_setup_time: Optional[datetime] = None
        self._current_setup_type: Optional[str] = None

        self._mae = 0.0
        self._mfe = 0.0

        self.density = {
            "setups_detected": 0,
            "setups_failed_volume_confirm": 0,
            "setups_expired_time_window": 0,
            "entries_blocked_by_session": 0,
            "entries_blocked_by_risk": 0,
            "bars_total": 0,
            "bars_in_session": 0,
            "bars_outside_session": 0,
        }

        self.last_reset_date = None
        self._setup_logging()

    # -------------------------
    # TIME HELPERS
    # -------------------------

    def _to_exchange_time(self, ts: datetime) -> datetime:
        # If ts is naive, assume data_tz. Then convert to exchange_tz.
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=self.data_tz)
        return ts.astimezone(self.exchange_tz)

    @staticmethod
    def _in_window(t, start, end) -> bool:
        # Handles overnight windows too
        if start <= end:
            return start <= t < end
        return t >= start or t < end

    def _in_hours(self, ts: datetime) -> bool:
        # Normalize first
        ts = self._to_exchange_time(ts)
        t = ts.time()

        # Must be within the main session
        if not self._in_window(t, self.config.SESSION_START, self.config.SESSION_END):
            return False

        # Treat NO_TRADE_START/END as the *trade window* inside the session (matches your prior behavior)
        if not self._in_window(t, self.config.NO_TRADE_START, self.config.NO_TRADE_END):
            return False

        # Exclude lunch window
        if self._in_window(t, self.config.LUNCH_START, self.config.LUNCH_END):
            return False

        return True

    # -------------------------
    # LOGGING
    # -------------------------

    def _setup_logging(self):
        self.logger = logging.getLogger("auction_bot")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            fh = logging.FileHandler(f"trading_{datetime.utcnow().strftime('%Y%m%d')}.log")
            fh.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(fh)

    def log(self, event: str, market_time: Optional[datetime], data: Dict = None, level="INFO"):
        payload = {
            "event": event,
            "state": self.state.name,
            "wall_time": datetime.utcnow().isoformat(),
            "market_time": market_time.isoformat() if market_time else None,
            "position": {
                "direction": self.position.direction.name,
                "size": self.position.size,
                "entry_price": self.position.entry_price,
            },
            "risk": {
                "capital": self.risk_manager.current_capital,
                "daily_pnl": self.risk_manager.daily_pnl,
                "drawdown": self.risk_manager.current_drawdown,
            },
            "data": data or {},
        }
        getattr(self.logger, level.lower())(json.dumps(payload, default=str))

    # -------------------------
    # FSM HELPERS
    # -------------------------

    def _transition(self, new_state: TradingState, reason: str, market_time: datetime):
        old = self.state
        self.state = new_state
        self.log("STATE_CHANGE", market_time, {"from": old.name, "to": new_state.name, "reason": reason})

    def _enforce_invariants(self, market_time: datetime):
        if self.state == TradingState.IN_TRADE and not self.position.is_active:
            self.log("INVARIANT_VIOLATION", market_time, {"issue": "IN_TRADE but flat"}, level="ERROR")
            self._transition(TradingState.MONITORING, "Invariant correction", market_time)

    def _enforce_risk_kill(self, timestamp: datetime):
        if self.risk_manager.is_hard_stopped():
            if self.state != TradingState.STOPPED:
                self._transition(TradingState.STOPPED, "Risk limit breached", timestamp)

    # -------------------------
    # CORE LOOP
    # -------------------------

    def process_market_update(self, ms: MarketState):
        # Normalize to exchange time once
        mt = self._to_exchange_time(ms.timestamp)

        # ALWAYS allow daily reset even if STOPPED
        if self.last_reset_date != mt.date() and mt.time() >= self.config.SESSION_START:
            self.risk_manager.reset_daily()
            self.last_reset_date = mt.date()
            self.log("DAILY_RESET", mt)

            if self.state == TradingState.STOPPED:
                self._transition(
                    TradingState.MONITORING if self._in_hours(mt) else TradingState.NO_TRADE,
                    "Daily reset cleared STOPPED",
                    mt,
                )

        # risk kill check (for current day)
        self._enforce_risk_kill(mt)
        if self.state == TradingState.STOPPED:
            return

        # --- BAR ACCOUNTING (REQUIRED FOR DENSITY & CALIBRATION) ---
        self.density["bars_total"] += 1
        if self._in_hours(mt):
            self.density["bars_in_session"] += 1
        else:
            self.density["bars_outside_session"] += 1

        # Use a MarketState with normalized timestamp for the rest of the system
        ms = MarketState(
            timestamp=mt,
            open=ms.open,
            high=ms.high,
            low=ms.low,
            close=ms.close,
            volume=ms.volume,
            vwap=ms.vwap,
            bid_volume=ms.bid_volume,
            ask_volume=ms.ask_volume,
            trades=ms.trades,
            has_bid_ask=ms.has_bid_ask,
        )

        self.market_data.append(ms)
        self.market_data = self.market_data[-self.config.MAX_BARS_STORED:]

        self.volume_profile.update(ms.timestamp, ms.typical_price, ms.volume)
        self.adaptive.update(self.market_data)

        # Update MAE/MFE while in trade
        if self.state == TradingState.IN_TRADE and self.position.is_active:
            if self.position.direction == PositionDirection.LONG:
                self._mae = min(self._mae, ms.low - self.position.entry_price)
                self._mfe = max(self._mfe, ms.high - self.position.entry_price)
            else:
                self._mae = min(self._mae, self.position.entry_price - ms.high)
                self._mfe = max(self._mfe, self.position.entry_price - ms.low)

        # INIT
        if self.state == TradingState.INITIALIZING:
            if len(self.market_data) >= self.config.MIN_BARS_FOR_ANALYSIS:
                self._transition(
                    TradingState.MONITORING if self._in_hours(ms.timestamp) else TradingState.NO_TRADE,
                    "Warmup complete",
                    ms.timestamp,
                )
            return

        # NO_TRADE → MONITORING when trading window opens
        if self.state == TradingState.NO_TRADE:
            if self._in_hours(ms.timestamp):
                self._transition(TradingState.MONITORING, "Session opened", ms.timestamp)
            return

        # MONITORING
        if self.state == TradingState.MONITORING:
            if not self._in_hours(ms.timestamp):
                if self.pending_setup is not None:
                    self.density["entries_blocked_by_session"] += 1
                    self.pending_setup = None
                self._transition(TradingState.NO_TRADE, "Market closed", ms.timestamp)
                return

            can_trade, reason = self.risk_manager.can_trade()
            if not can_trade:
                self.density["entries_blocked_by_risk"] += 1
                self._transition(TradingState.STOPPED, reason, ms.timestamp)
                return

            if self.pending_setup is not None:
                age_min = (ms.timestamp - self.pending_setup["setup_time"]).total_seconds() / 60.0
                if age_min > self.adaptive.acceptance_window_minutes():
                    self.density["setups_expired_time_window"] += 1
                    if not self.pending_setup.get("saw_volume_confirm", False):
                        self.density["setups_failed_volume_confirm"] += 1
                    self.pending_setup = None
                    return

                if ms.volume > self.adaptive.avg_volume * self.config.HIGH_VOLUME_THRESHOLD:
                    self.pending_setup["saw_volume_confirm"] = True

                if self._confirm_setup(self.pending_setup, ms):
                    setup = self.pending_setup
                    self.pending_setup = None
                    self._enter_trade(setup, ms)
                return

            setup = self._detect_setup(ms)
            if setup:
                self.density["setups_detected"] += 1
                setup["saw_volume_confirm"] = False
                self.pending_setup = setup
            return

        # IN_TRADE
        if self.state == TradingState.IN_TRADE:
            reason = self._check_exit(ms)
            if reason:
                self._exit_trade(ms, reason)
            return

        # COOLDOWN
        if self.state == TradingState.COOLDOWN:
            if self.cooldown_until and ms.timestamp >= self.cooldown_until:
                self._transition(
                    TradingState.MONITORING if self._in_hours(ms.timestamp) else TradingState.NO_TRADE,
                    "Cooldown ended",
                    ms.timestamp,
                )
                self.cooldown_until = None
            return

    # -------------------------
    # STRATEGY LOGIC (FROZEN)
    # -------------------------

    def _detect_setup(self, ms: MarketState) -> Optional[Dict]:
        if len(self.market_data) < self.config.MIN_BARS_FOR_ANALYSIS:
            return None

        vlow, vhigh = self.volume_profile.get_value_area()
        if vlow == vhigh:
            return None

        if ms.close > vhigh and ms.volume < self.adaptive.avg_volume * self.config.LOW_VOLUME_THRESHOLD:
            return {"type": "UPSIDE_ATTEMPT", "setup_time": ms.timestamp}

        if ms.close < vlow and ms.volume < self.adaptive.avg_volume * self.config.LOW_VOLUME_THRESHOLD:
            return {"type": "DOWNSIDE_ATTEMPT", "setup_time": ms.timestamp}

        return None

    def _confirm_setup(self, setup: Dict, ms: MarketState) -> bool:
        if (ms.timestamp - setup["setup_time"]).total_seconds() / 60 > self.adaptive.acceptance_window_minutes():
            return False

        vlow, vhigh = self.volume_profile.get_value_area()
        if vlow == vhigh:
            return False

        if setup["type"] == "UPSIDE_ATTEMPT":
            return ms.close < vhigh   # price rejected above value and re-entered

        if setup["type"] == "DOWNSIDE_ATTEMPT":
            return ms.close > vlow   # price rejected below value and re-entered

        return False


    # -------------------------
    # EXECUTION (PAPER)
    # -------------------------

    def _enter_trade(self, setup: Dict, ms: MarketState):
        self._current_setup_time = setup.get("setup_time")
        self._current_setup_type = setup.get("type")

        direction = PositionDirection.SHORT if setup["type"] == "UPSIDE_ATTEMPT" else PositionDirection.LONG
        stop = ms.close + 10 if direction == PositionDirection.SHORT else ms.close - 10
        size = self.risk_manager.get_position_size(abs(ms.close - stop), self.instrument)

        self.position = Position(
            direction=direction,
            entry_price=ms.close,
            entry_time=ms.timestamp,
            size=size,
            stop_loss=stop,
            profit_targets=[],
            time_stop=ms.timestamp + timedelta(minutes=self.config.TIME_STOP_MINUTES),
        )

        self._mae = 0.0
        self._mfe = 0.0
        self._transition(TradingState.IN_TRADE, "Position entered", ms.timestamp)

    def _exit_trade(self, ms: MarketState, reason: str):
        if not self.position.is_active:
            self.log("EXIT_IGNORED_ALREADY_FLAT", ms.timestamp)
            return

        entry_price = self.position.entry_price
        entry_time = self.position.entry_time
        direction = self.position.direction
        size = self.position.size

        self.position = Position(PositionDirection.FLAT, 0.0, ms.timestamp, 0, 0.0, [], ms.timestamp)

        direction_mult = 1 if direction == PositionDirection.LONG else -1
        gross_pnl = (ms.close - entry_price) * direction_mult * size * 5

        commission = size * self.config.COMMISSION_PER_CONTRACT_PER_SIDE * 2
        slippage = size * self.config.SLIPPAGE_TICKS * self.config.TICK_SIZE * 5
        net_pnl = gross_pnl - commission - slippage

        self.risk_manager.update_pnl(net_pnl)

        mae_dollars = abs(min(0.0, self._mae)) * size * 5
        mfe_dollars = max(0.0, self._mfe) * size * 5

        self.last_closed_trade = {
            "setup_time": self._current_setup_time,
            "setup_type": self._current_setup_type,
            "entry_time": entry_time,
            "exit_time": ms.timestamp,
            "direction": direction.name,
            "size": size,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "commission": commission,
            "slippage": slippage,
            "mae": mae_dollars,
            "mfe": mfe_dollars,
            "time_in_trade_min": (ms.timestamp - entry_time).total_seconds() / 60.0,
            "exit_reason": reason,
        }

        if net_pnl <= 0:
            self.cooldown_until = ms.timestamp + timedelta(minutes=self.config.COOLDOWN_MINUTES_AFTER_LOSS)
            self._transition(TradingState.COOLDOWN, "Loss cooldown", ms.timestamp)
        else:
            self._transition(TradingState.MONITORING, "Trade closed", ms.timestamp)

    def _check_exit(self, ms: MarketState) -> Optional[str]:
        if self.position.direction == PositionDirection.LONG and ms.close <= self.position.stop_loss:
            return "STOP_LOSS"
        if self.position.direction == PositionDirection.SHORT and ms.close >= self.position.stop_loss:
            return "STOP_LOSS"
        if ms.timestamp >= self.position.time_stop:
            return "TIME_STOP"
        return None
