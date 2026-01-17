from dataclasses import dataclass
from datetime import time
from typing import Tuple

'''config.py - Centralized configuration module'''

@dataclass
class Config:
    """Centralized configuration - no magic numbers"""
    # Data requirements
    MIN_BARS_FOR_ANALYSIS: int = 30
    MAX_BARS_STORED: int = 1000
    
    # Volume profile
    VOLUME_PROFILE_WINDOW_MINUTES: int = 120
    VALUE_AREA_THRESHOLD: float = 0.7
    
    # Entry conditions
    BREAKOUT_DISTANCE_THRESHOLD: float = 0.3
    LOW_VOLUME_THRESHOLD: float = 0.8
    HIGH_VOLUME_THRESHOLD: float = 1.2
    IMBALANCE_THRESHOLD: float = 0.3
    
    # Exit conditions
    MIN_STOP_POINTS: float = 8.0
    TIME_STOP_MINUTES: int = 45
    NO_PROGRESS_MINUTES: int = 15
    
    # Risk management
    RISK_PER_TRADE: float = 0.005  # 0.5%
    MAX_DAILY_LOSS: float = 0.02   # 2%
    MAX_CONSECUTIVE_LOSSES: int = 3
    MAX_DAILY_TRADES: int = 20
    MAX_POSITION_SIZE: int = 10
    
    # Cooldown
    COOLDOWN_MINUTES_AFTER_LOSS: int = 60
    
    # Session times (NY time)
    SESSION_START: time = time(9, 30)
    SESSION_END: time = time(16, 0)
    NO_TRADE_START: time = time(10, 0)    # Wait 30min after open
    NO_TRADE_END: time = time(15, 30)     # Stop 30min before close
    LUNCH_START: time = time(12, 0)
    LUNCH_END: time = time(13, 0)
    
    # Market hours (converted to UTC for comparison)
    # Assuming data is in UTC, NY is UTC-4/5
    # This should be set based on your data source
    TIMEZONE_OFFSET_HOURS: int = -4  # NY is UTC-4 during DST
    
    # Adaptive parameters
    VOLATILITY_LOOKBACK_RANGE: Tuple[int, int] = (5, 40)
    VOLATILITY_BASELINE: float = 0.15  # 15% annualized
