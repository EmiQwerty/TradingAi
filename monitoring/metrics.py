"""
Performance Metrics Calculator
Calculates trading performance metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculates comprehensive trading performance metrics
    """
    
    def __init__(self):
        self.trade_history = []
        self.equity_curve = []
        
        logger.info("MetricsCalculator initialized")
    
    def add_trade(self, trade: Dict):
        """
        Add completed trade to history
        
        Args:
            trade: Dict with trade details
        """
        self.trade_history.append(trade)
    
    def add_equity_point(self, timestamp: datetime, equity: float):
        """Add equity snapshot"""
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity
        })
    
    def calculate_metrics(
        self,
        trades: List[Dict] = None,
        initial_capital: float = 10000
    ) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Args:
            trades: Optional list of trades (uses internal history if None)
            initial_capital: Starting capital
        
        Returns:
            Dict with all performance metrics
        """
        if trades is None:
            trades = self.trade_history
        
        if not trades:
            return self._empty_metrics()
        
        # Extract P&L values
        pnls = [t.get('pnl', 0) for t in trades]
        pnl_pcts = [t.get('pnl_pct', 0) for t in trades]
        
        # Win/Loss analysis
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        total_trades = len(trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        
        win_rate = (winning_trades / total_trades) if total_trades > 0 else 0
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0
        
        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
        
        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Total P&L
        total_pnl = sum(pnls)
        total_return = (total_pnl / initial_capital) * 100
        
        # Consecutive wins/losses
        max_consecutive_wins = self._calculate_max_consecutive(pnls, positive=True)
        max_consecutive_losses = self._calculate_max_consecutive(pnls, positive=False)
        
        # Average trade duration
        durations = []
        for trade in trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600
                durations.append(duration)
        
        avg_duration = np.mean(durations) if durations else 0
        
        # Risk/Reward ratio
        avg_rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Sharpe ratio (simplified)
        if len(pnl_pcts) > 1:
            returns_std = np.std(pnl_pcts)
            avg_return = np.mean(pnl_pcts)
            sharpe = (avg_return / returns_std) * np.sqrt(252) if returns_std > 0 else 0
        else:
            sharpe = 0
        
        # Sortino ratio (downside deviation)
        downside_returns = [r for r in pnl_pcts if r < 0]
        if downside_returns and len(downside_returns) > 1:
            downside_std = np.std(downside_returns)
            sortino = (np.mean(pnl_pcts) / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        else:
            sortino = 0
        
        # Calculate drawdown
        if self.equity_curve:
            max_dd, max_dd_pct = self._calculate_max_drawdown()
        else:
            max_dd = 0
            max_dd_pct = 0
        
        return {
            'summary': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'total_return_pct': total_return
            },
            'wins_losses': {
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss
            },
            'ratios': {
                'profit_factor': profit_factor,
                'expectancy': expectancy,
                'avg_risk_reward': avg_rr,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino
            },
            'streaks': {
                'max_consecutive_wins': max_consecutive_wins,
                'max_consecutive_losses': max_consecutive_losses
            },
            'duration': {
                'avg_trade_duration_hours': avg_duration
            },
            'drawdown': {
                'max_drawdown': max_dd,
                'max_drawdown_pct': max_dd_pct
            }
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            'summary': {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return_pct': 0
            },
            'wins_losses': {
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'gross_profit': 0,
                'gross_loss': 0
            },
            'ratios': {
                'profit_factor': 0,
                'expectancy': 0,
                'avg_risk_reward': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0
            },
            'streaks': {
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0
            },
            'duration': {
                'avg_trade_duration_hours': 0
            },
            'drawdown': {
                'max_drawdown': 0,
                'max_drawdown_pct': 0
            }
        }
    
    def _calculate_max_consecutive(
        self,
        pnls: List[float],
        positive: bool = True
    ) -> int:
        """Calculate maximum consecutive wins or losses"""
        max_consecutive = 0
        current_consecutive = 0
        
        for pnl in pnls:
            if (positive and pnl > 0) or (not positive and pnl < 0):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_max_drawdown(self) -> tuple:
        """
        Calculate maximum drawdown from equity curve
        
        Returns:
            Tuple of (max_drawdown_amount, max_drawdown_percent)
        """
        if not self.equity_curve:
            return 0, 0
        
        df = pd.DataFrame(self.equity_curve)
        df['peak'] = df['equity'].cummax()
        df['drawdown'] = df['peak'] - df['equity']
        df['drawdown_pct'] = (df['drawdown'] / df['peak']) * 100
        
        max_dd = df['drawdown'].max()
        max_dd_pct = df['drawdown_pct'].max()
        
        return max_dd, max_dd_pct
    
    def get_daily_performance(self) -> Dict:
        """Get performance metrics for today"""
        today = datetime.utcnow().date()
        
        today_trades = [
            t for t in self.trade_history
            if t.get('exit_time', datetime.min).date() == today
        ]
        
        if not today_trades:
            return self._empty_metrics()
        
        return self.calculate_metrics(today_trades)
    
    def get_weekly_performance(self) -> Dict:
        """Get performance metrics for current week"""
        week_ago = datetime.utcnow() - timedelta(days=7)
        
        week_trades = [
            t for t in self.trade_history
            if t.get('exit_time', datetime.min) >= week_ago
        ]
        
        if not week_trades:
            return self._empty_metrics()
        
        return self.calculate_metrics(week_trades)
    
    def get_monthly_performance(self) -> Dict:
        """Get performance metrics for current month"""
        month_ago = datetime.utcnow() - timedelta(days=30)
        
        month_trades = [
            t for t in self.trade_history
            if t.get('exit_time', datetime.min) >= month_ago
        ]
        
        if not month_trades:
            return self._empty_metrics()
        
        return self.calculate_metrics(month_trades)
