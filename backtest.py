"""
backtest.py - Backtesting engine
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from tqdm import tqdm

from market import MarketState
from system import AuctionFailureSystem

@dataclass
class BacktestResult:
    """Results from backtest"""
    total_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    avg_daily_pnl: float = 0.0
    daily_std: float = 0.0
    
    # Trade list
    trades: List[Dict] = field(default_factory=list)
    
    # Equity curve
    timestamps: List[datetime] = field(default_factory=list)
    equity: List[float] = field(default_factory=list)
    drawdown: List[float] = field(default_factory=list)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        return pd.DataFrame({
            'Metric': [
                'Total PnL', 'Total Trades', 'Win Rate', 'Profit Factor',
                'Max Drawdown', 'Sharpe Ratio', 'Avg Win', 'Avg Loss',
                'Avg Trade', 'Best Day', 'Worst Day', 'Avg Daily PnL'
            ],
            'Value': [
                f"${self.total_pnl:.2f}",
                f"{self.total_trades}",
                f"{self.win_rate*100:.1f}%",
                f"{self.profit_factor:.2f}",
                f"{self.max_drawdown*100:.1f}%",
                f"{self.sharpe_ratio:.2f}",
                f"${self.avg_win:.2f}",
                f"${self.avg_loss:.2f}",
                f"${self.avg_trade:.2f}",
                f"${self.best_day:.2f}",
                f"${self.worst_day:.2f}",
                f"${self.avg_daily_pnl:.2f}"
            ]
        })

class Backtester:
    """Backtesting engine for AuctionFailureSystem"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        
    def load_data_from_csv(self, filepath: str) -> List[MarketState]:
        """
        Load 1-minute bar data from CSV
        Expected columns: timestamp, open, high, low, close, volume
        Optional: vwap, bid_volume, ask_volume, trades
        """
        df = pd.read_csv(filepath)
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        market_states = []
        for _, row in df.iterrows():
            market_states.append(MarketState(
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=int(row['volume']),
                vwap=row.get('vwap', (row['open'] + row['high'] + row['low'] + row['close']) / 4),
                bid_volume=int(row.get('bid_volume', 0)),
                ask_volume=int(row.get('ask_volume', 0)),
                trades=int(row.get('trades', 0)),
                has_bid_ask='bid_volume' in row.columns and 'ask_volume' in row.columns
            ))
        
        return market_states
    
    def generate_synthetic_data(self, 
                            days: int = 30,
                            start_price: float = 4500.0,
                            seed: int = 42) -> List[MarketState]:
        """
        Generate realistic synthetic data for testing
        """
        np.random.seed(seed)
        market_states = []
        
        # Generate base returns
        n_bars = days * 390  # 390 minutes per trading day
        returns = np.random.normal(0, 0.0005, n_bars)
        
        # Add volatility clustering
        vol = 0.0005
        for i in range(1, len(returns)):
            vol = 0.95 * vol + 0.05 * abs(returns[i-1])
            returns[i] = returns[i] * vol / 0.0005
        
        # Generate prices
        prices = start_price * np.cumprod(1 + returns)
        
        # Generate volume
        time_of_day = np.tile(np.linspace(0, 1, 390), days)
        volume_pattern = 1000 + 500 * np.sin(time_of_day * np.pi) ** 2
        volume = volume_pattern * np.random.lognormal(0, 0.2, len(prices))
        
        # Generate timestamps
        # Start with today's date at 9:30 AM
        base_date = datetime.now().date()
        market_states = []
        
        for day in range(days):
            current_date = base_date + timedelta(days=day)
            
            # Generate 390 minutes of data for this day (9:30 AM to 4:00 PM)
            for minute_offset in range(390):
                current_time = datetime.combine(
                    current_date, 
                    time(9, 30)
                ) + timedelta(minutes=minute_offset)
                
                bar_index = day * 390 + minute_offset
                if bar_index >= len(prices):
                    break
                    
                price = prices[bar_index]
                
                # Create OHLC
                open_price = price * (1 + np.random.normal(0, 0.0002))
                high_price = max(open_price, price) * (1 + abs(np.random.normal(0, 0.0003)))
                low_price = min(open_price, price) * (1 - abs(np.random.normal(0, 0.0003)))
                close_price = price
                
                # Create some fake breakouts
                if bar_index % 100 == 0 and bar_index > 100:
                    close_price = price * (1 + np.random.normal(0.002, 0.0005))
                
                market_states.append(MarketState(
                    timestamp=current_time,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=int(volume[bar_index]),
                    vwap=(open_price + high_price + low_price + close_price) / 4,
                    bid_volume=int(volume[bar_index] * 0.4),
                    ask_volume=int(volume[bar_index] * 0.6),
                    trades=int(volume[bar_index] / 10),
                    has_bid_ask=True
                ))
        
        return market_states
    
    def run_backtest(self, 
                     market_states: List[MarketState],
                     system_params: Dict = None) -> BacktestResult:
        """
        Run full backtest on historical data
        """
        # Initialize system with parameters
        params = system_params or {}
        system = AuctionFailureSystem(
            instrument="MES",
            initial_capital=self.initial_capital,
            **params
        )
        
        # Track results
        trades = []
        equity_curve = []
        daily_pnl = {}
        current_date = None
        daily_cumulative = 0.0
        
        # Add last_closed_trade attribute to system
        system.last_closed_trade = None
        
        # Run simulation
        print(f"Running backtest on {len(market_states)} bars...")
        for market_state in tqdm(market_states, desc="Processing bars"):
            # Track daily PnL
            date_key = market_state.timestamp.date()
            if date_key != current_date:
                if current_date is not None:
                    daily_pnl[current_date] = daily_cumulative
                current_date = date_key
                daily_cumulative = 0.0
            
            # Process market update
            system.process_market_update(market_state)
            
            # Record trade if one just closed
            if hasattr(system, 'last_closed_trade') and system.last_closed_trade:
                trades.append(system.last_closed_trade.copy())
                daily_cumulative += system.last_closed_trade['pnl']
                system.last_closed_trade = None
            
            # Record equity curve point
            unrealized_pnl = 0.0
            if system.position.is_active:
                if system.position.direction.name == "LONG":
                    unrealized_pnl = (market_state.close - system.position.entry_price) * system.position.size * 5
                else:
                    unrealized_pnl = (system.position.entry_price - market_state.close) * system.position.size * 5
            
            total_equity = system.risk_manager.current_capital + unrealized_pnl
            equity_curve.append((market_state.timestamp, total_equity))
        
        # Process final day
        if current_date:
            daily_pnl[current_date] = daily_cumulative
        
        # Calculate metrics
        result = self._calculate_metrics(trades, equity_curve, daily_pnl)
        result.trades = trades
        
        return result
    
    def _calculate_metrics(self, 
                          trades: List[Dict],
                          equity_curve: List[Tuple[datetime, float]],
                          daily_pnl: Dict) -> BacktestResult:
        """Calculate performance metrics from backtest results"""
        result = BacktestResult()
        
        if not trades:
            return result
        
        # Basic trade metrics
        result.total_trades = len(trades)
        result.winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        result.losing_trades = result.total_trades - result.winning_trades
        result.win_rate = result.winning_trades / result.total_trades
        result.total_pnl = sum(t['pnl'] for t in trades)
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade metrics
        winning_pnls = [t['pnl'] for t in trades if t['pnl'] > 0]
        losing_pnls = [t['pnl'] for t in trades if t['pnl'] < 0]
        
        result.avg_win = np.mean(winning_pnls) if winning_pnls else 0
        result.avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        result.avg_trade = result.total_pnl / result.total_trades
        
        # Equity curve metrics
        timestamps, equity = zip(*equity_curve)
        result.timestamps = list(timestamps)
        result.equity = list(equity)
        
        # Drawdown calculation
        result.drawdown = []
        peak = equity[0]
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            result.drawdown.append(dd)
        
        result.max_drawdown = max(result.drawdown) if result.drawdown else 0
        
        # Max drawdown duration
        in_dd = False
        dd_start = 0
        max_dd_duration = 0
        
        for i, dd in enumerate(result.drawdown):
            if dd > 0.05:  # In drawdown (5% threshold)
                if not in_dd:
                    in_dd = True
                    dd_start = i
            else:
                if in_dd:
                    in_dd = False
                    duration = i - dd_start
                    max_dd_duration = max(max_dd_duration, duration)
        
        result.max_drawdown_duration = max_dd_duration
        
        # Sharpe ratio (simplified - assumes 252 trading days)
        if daily_pnl:
            daily_returns = list(daily_pnl.values())
            result.avg_daily_pnl = np.mean(daily_returns)
            result.daily_std = np.std(daily_returns)
            
            if result.daily_std > 0:
                # Annualized Sharpe (assuming 0 risk-free rate for simplicity)
                result.sharpe_ratio = (result.avg_daily_pnl / result.daily_std) * np.sqrt(252)
            
            result.best_day = max(daily_returns) if daily_returns else 0
            result.worst_day = min(daily_returns) if daily_returns else 0
        
        # Sortino ratio (only downside deviation)
        if daily_pnl:
            downside_returns = [r for r in daily_returns if r < 0]
            downside_std = np.std(downside_returns) if downside_returns else 0
            if downside_std > 0:
                result.sortino_ratio = (result.avg_daily_pnl / downside_std) * np.sqrt(252)
        
        return result
    
    def plot_results(self, result: BacktestResult):
        """Create visualization of backtest results"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 1. Equity Curve
        axes[0, 0].plot(result.timestamps, result.equity, linewidth=1)
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_ylabel('Account Value ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Drawdown
        axes[0, 1].fill_between(result.timestamps, 0, result.drawdown, alpha=0.5, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Daily PnL distribution
        if result.trades:
            trade_pnls = [t['pnl'] for t in result.trades]
            axes[1, 0].hist(trade_pnls, bins=30, edgecolor='black', alpha=0.7)
            axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
            axes[1, 0].set_title('Trade PnL Distribution')
            axes[1, 0].set_xlabel('PnL ($)')
            axes[1, 0].set_ylabel('Frequency')
        
        # 4. Win/Loss ratio
        win_loss_data = [result.winning_trades, result.losing_trades]
        axes[1, 1].bar(['Wins', 'Losses'], win_loss_data, color=['green', 'red'], alpha=0.7)
        axes[1, 1].set_title(f'Win/Loss Ratio ({result.win_rate*100:.1f}% win rate)')
        axes[1, 1].set_ylabel('Number of Trades')
        
        # 5. Rolling Sharpe (if enough data)
        if len(result.equity) > 30:
            equity_series = pd.Series(result.equity, index=result.timestamps)
            returns = equity_series.pct_change().dropna()
            rolling_sharpe = returns.rolling(window=20).mean() / returns.rolling(window=20).std() * np.sqrt(252)
            axes[2, 0].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1)
            axes[2, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[2, 0].set_title('Rolling 20-day Sharpe Ratio')
            axes[2, 0].set_ylabel('Sharpe Ratio')
            axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Metrics table
        metrics_text = f"""
        Total PnL: ${result.total_pnl:,.2f}
        Total Trades: {result.total_trades}
        Win Rate: {result.win_rate*100:.1f}%
        Profit Factor: {result.profit_factor:.2f}
        Max Drawdown: {result.max_drawdown*100:.1f}%
        Sharpe Ratio: {result.sharpe_ratio:.2f}
        Avg Win: ${result.avg_win:.2f}
        Avg Loss: ${result.avg_loss:.2f}
        """
        axes[2, 1].axis('off')
        axes[2, 1].text(0.1, 0.5, metrics_text, fontsize=10, 
                       verticalalignment='center', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    
    def run_walkforward_analysis(self,
                                 market_states: List[MarketState],
                                 train_days: int = 60,
                                 test_days: int = 20,
                                 step_days: int = 10):
        """
        Walkforward analysis to test robustness
        """
        # Convert to DataFrame for easier slicing
        df = pd.DataFrame([{
            'timestamp': ms.timestamp,
            'close': ms.close,
            'volume': ms.volume
        } for ms in market_states])
        
        df.set_index('timestamp', inplace=True)
        
        results = []
        start_idx = 0
        
        while start_idx + (train_days + test_days) * 390 < len(df):
            # Split data
            train_end = start_idx + train_days * 390
            test_end = train_end + test_days * 390
            
            train_data = market_states[start_idx:train_end]
            test_data = market_states[train_end:test_end]
            
            # Run backtest on test set
            result = self.run_backtest(test_data)
            result.period = f"{df.index[start_idx].date()} to {df.index[test_end-1].date()}"
            results.append(result)
            
            print(f"Period {len(results)}: {result.period}")
            print(f"  PnL: ${result.total_pnl:.2f}, Trades: {result.total_trades}, Win Rate: {result.win_rate*100:.1f}%")
            print(f"  Sharpe: {result.sharpe_ratio:.2f}, Max DD: {result.max_drawdown*100:.1f}%")
            print("-" * 50)
            
            start_idx += step_days * 390
        
        return results