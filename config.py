# config.py
"""
Centralized configuration module
V0.2 — Edge Validation Phase

Strategy parameters are frozen.
Only execution, cost, and reporting parameters were added.
"""

from dataclasses import dataclass
from datetime import time
from typing import Tuple


@dataclass
class Config:
    # =========================
    # DATA REQUIREMENTS
    # =========================
    MIN_BARS_FOR_ANALYSIS: int = 30
    MAX_BARS_STORED: int = 1000

    # =========================
    # VOLUME PROFILE
    # =========================
    VOLUME_PROFILE_WINDOW_MINUTES: int = 120
    VALUE_AREA_THRESHOLD: float = 0.7

    # =========================
    # ENTRY CONDITIONS (FROZEN)
    # =========================
    BREAKOUT_DISTANCE_THRESHOLD: float = 0.3
    LOW_VOLUME_THRESHOLD: float = 0.8
    HIGH_VOLUME_THRESHOLD: float = 1.2
    IMBALANCE_THRESHOLD: float = 0.3

    # =========================
    # EXIT CONDITIONS (FROZEN)
    # =========================
    MIN_STOP_POINTS: float = 8.0
    TIME_STOP_MINUTES: int = 45
    NO_PROGRESS_MINUTES: int = 15

    # =========================
    # RISK MANAGEMENT (FROZEN)
    # =========================
    RISK_PER_TRADE: float = 0.005
    MAX_DAILY_LOSS: float = 0.02
    MAX_CONSECUTIVE_LOSSES: int = 3
    MAX_DAILY_TRADES: int = 20
    MAX_POSITION_SIZE: int = 10

    # =========================
    # COOLDOWN
    # =========================
    COOLDOWN_MINUTES_AFTER_LOSS: int = 60

    # =========================
    # SESSION TIMES (NY)
    # =========================
    SESSION_START: time = time(9, 30)
    SESSION_END: time = time(16, 0)
    NO_TRADE_START: time = time(10, 0)
    NO_TRADE_END: time = time(15, 30)
    LUNCH_START: time = time(12, 0)
    LUNCH_END: time = time(13, 0)

    # =========================
    # ADAPTIVE PARAMETERS
    # =========================
    VOLATILITY_LOOKBACK_RANGE: Tuple[int, int] = (5, 40)
    VOLATILITY_BASELINE: float = 0.15

    # =====================================================
    # V0.2 — PAPER EXECUTION & COST ASSUMPTIONS (NEW)
    # =====================================================

    # Execution model
    EXECUTION_MODEL: str = "BAR_CLOSE"

    # Commission (per contract per side)
    COMMISSION_PER_CONTRACT_PER_SIDE: float = 1.20

    # Slippage (deterministic)
    SLIPPAGE_MODEL: str = "fixed_ticks"
    TICK_SIZE: float = 0.25
    SLIPPAGE_TICKS: int = 1

    # Range-proxy slippage (if enabled)
    RANGE_SLIPPAGE_MIN_TICKS: float = 0.5
    RANGE_SLIPPAGE_MAX_TICKS: float = 2.0
    BASE_ACCEPTANCE_WINDOW_MINUTES: int = 10  # used by AdaptiveParameters
    # =====================================================
