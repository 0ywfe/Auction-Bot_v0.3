'''Market.py '''


from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional


@dataclass
class MarketState:
    """Market microstructure state"""
    timestamp: datetime  # Must be in consistent timezone (NY time preferred)
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    bid_volume: int = 0
    ask_volume: int = 0
    trades: int = 0
    has_bid_ask: bool = False  # Flag for bid/ask data availability
    
    @property
    def imbalance(self) -> Optional[float]:
        """Bid-ask imbalance (-1 to 1), None if not available"""
        if not self.has_bid_ask or (self.bid_volume + self.ask_volume) == 0:
            return None
        return (self.bid_volume - self.ask_volume) / (self.bid_volume + self.ask_volume)
    
    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3
    
@dataclass
class BarVolumeProfile:
    """
    Approximate volume profile based on bar data (not tick data)
    For prototype only - real implementation needs tick data
    """
    window_minutes: int
    price_levels: Dict[float, int] = field(default_factory=dict)
    timestamps: List[datetime] = field(default_factory=list)
    prices: List[float] = field(default_factory=list)
    volumes: List[int] = field(default_factory=list)
    
    def update(self, timestamp: datetime, price: float, volume: int):
        """Add new bar to profile"""
        self.timestamps.append(timestamp)
        self.prices.append(price)
        self.volumes.append(volume)
        
        # Round price to nearest tick (0.25 for MES)
        rounded_price = round(price * 4) / 4
        
        # Distribute volume across high-low range for better approximation
        # This is a simplification - real volume profile needs tick data
        self.price_levels[rounded_price] = self.price_levels.get(rounded_price, 0) + volume
        
        # Remove old data
        cutoff = timestamp - timedelta(minutes=self.window_minutes)
        while self.timestamps and self.timestamps[0] < cutoff:
            old_price = round(self.prices[0] * 4) / 4
            old_volume = self.volumes[0]
            
            self.price_levels[old_price] -= old_volume
            if self.price_levels[old_price] <= 0:
                del self.price_levels[old_price]
            
            self.timestamps.pop(0)
            self.prices.pop(0)
            self.volumes.pop(0)
    
    def get_value_area(self, threshold: float = 0.7) -> Tuple[float, float]:
        """Calculate value area (POC ± threshold% of volume)"""
        if not self.price_levels:
            return 0.0, 0.0
        
        # Sort price levels by price
        sorted_prices = sorted(self.price_levels.keys())
        
        # Find Point of Control (POC)
        poc_price = max(self.price_levels.items(), key=lambda x: x[1])[0]
        
        # Calculate total volume
        total_volume = sum(self.price_levels.values())
        target_volume = total_volume * threshold
        
        # Expand from POC until we have threshold% of volume
        current_volume = self.price_levels[poc_price]
        low_idx = sorted_prices.index(poc_price)
        high_idx = low_idx
        
        while current_volume < target_volume:
            can_expand_low = low_idx > 0
            can_expand_high = high_idx < len(sorted_prices) - 1
            
            if can_expand_low and can_expand_high:
                low_volume = self.price_levels[sorted_prices[low_idx - 1]]
                high_volume = self.price_levels[sorted_prices[high_idx + 1]]
                
                if low_volume >= high_volume:
                    low_idx -= 1
                    current_volume += low_volume
                else:
                    high_idx += 1
                    current_volume += high_volume
            elif can_expand_low:
                low_idx -= 1
                current_volume += self.price_levels[sorted_prices[low_idx]]
            elif can_expand_high:
                high_idx += 1
                current_volume += self.price_levels[sorted_prices[high_idx]]
            else:
                break
        
        return sorted_prices[low_idx], sorted_prices[high_idx]
    
    def get_poc(self) -> float:
        """Point of Control (price with highest volume)"""
        if not self.price_levels:
            return 0.0
        return max(self.price_levels.items(), key=lambda x: x[1])[0]
