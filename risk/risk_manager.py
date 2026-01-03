"""
Risk Manager
Global risk management: exposure limits, drawdown control, correlation management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Global risk management system
    Controls exposure, drawdown, correlations, and risk limits
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.risk_config = config.get('risk', {})
        self.global_risk = self.risk_config.get('global_risk', {})
        
        self.initial_capital = self.global_risk.get('initial_capital', 10000)
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        
        # Risk limits
        self.max_risk_per_trade = self.global_risk.get('max_risk_per_trade', 0.01)
        self.max_total_exposure = self.global_risk.get('max_total_exposure', 0.15)
        self.max_drawdown = self.global_risk.get('max_drawdown_percent', 0.20)
        
        # Tracking
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.trades_today = 0
        self.consecutive_losses = 0
        self.trading_locked = False
        
        # Trade history
        self.trade_history = []
        
        logger.info(f"RiskManager initialized with capital: ${self.initial_capital}")
    
    def calculate_current_exposure(self, positions: List[Dict]) -> Dict[str, float]:
        """
        Calculate current risk exposure across all positions
        
        Returns:
            Dict with exposure metrics
        """
        if not positions:
            return {
                'total_exposure': 0.0,
                'total_risk_amount': 0.0,
                'exposure_percent': 0.0,
                'position_count': 0
            }
        
        total_risk = 0.0
        
        for position in positions:
            # Calculate risk for each position (distance to stop loss)
            entry_price = position.get('entry_price', 0)
            stop_loss = position.get('stop_loss', 0)
            size = position.get('size', 0)
            
            if entry_price > 0 and stop_loss > 0:
                risk_per_position = abs(entry_price - stop_loss) * size
                # Normalize to account currency (simplified)
                total_risk += risk_per_position * entry_price
        
        exposure_percent = (total_risk / self.current_capital) if self.current_capital > 0 else 0
        
        return {
            'total_exposure': total_risk,
            'total_risk_amount': total_risk,
            'exposure_percent': exposure_percent,
            'position_count': len(positions)
        }
    
    def calculate_drawdown(self) -> Dict[str, float]:
        """
        Calculate current drawdown
        
        Returns:
            Dict with drawdown metrics
        """
        if self.peak_capital <= 0:
            return {'current_drawdown': 0.0, 'max_drawdown': 0.0}
        
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        # Update peak if new high
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        return {
            'current_drawdown': current_drawdown,
            'peak_capital': self.peak_capital,
            'current_capital': self.current_capital,
            'drawdown_amount': self.peak_capital - self.current_capital
        }
    
    def check_risk_limits(
        self,
        decision: Dict,
        positions: List[Dict]
    ) -> Dict[str, any]:
        """
        Check if decision violates any risk limits
        
        Returns:
            Dict with approval status and reasons
        """
        if self.trading_locked:
            return {
                'approved': False,
                'reason': 'trading_locked',
                'action': 'reject'
            }
        
        # Check if exit decision (always allow exits)
        if decision.get('action') == 'exit':
            return {'approved': True, 'reason': 'exit_allowed'}
        
        # Entry checks
        
        # 1. Check maximum positions
        max_positions = self.config.get('trading', {}).get('max_positions', 10)
        if len(positions) >= max_positions:
            return {
                'approved': False,
                'reason': f'max_positions_reached ({len(positions)}/{max_positions})',
                'action': 'reject'
            }
        
        # 2. Check total exposure
        current_exposure = self.calculate_current_exposure(positions)
        
        # Estimate new position risk
        entry_price = decision.get('price', 0)
        stop_loss = decision.get('stop_loss', 0)
        size = decision.get('size', 0)
        
        new_position_risk = abs(entry_price - stop_loss) * size * entry_price if entry_price > 0 else 0
        new_exposure_percent = (current_exposure['total_risk_amount'] + new_position_risk) / self.current_capital
        
        if new_exposure_percent > self.max_total_exposure:
            return {
                'approved': False,
                'reason': f'exposure_limit_exceeded ({new_exposure_percent:.1%} > {self.max_total_exposure:.1%})',
                'action': 'reject'
            }
        
        # 3. Check drawdown
        drawdown = self.calculate_drawdown()
        if drawdown['current_drawdown'] >= self.max_drawdown:
            return {
                'approved': False,
                'reason': f'max_drawdown_reached ({drawdown["current_drawdown"]:.1%})',
                'action': 'lock_trading'
            }
        
        # 4. Check daily loss limit
        daily_loss_limit = self.global_risk.get('daily_loss_limit', 0.10)
        daily_loss_pct = abs(self.daily_pnl / self.initial_capital) if self.daily_pnl < 0 else 0
        
        if daily_loss_pct >= daily_loss_limit:
            return {
                'approved': False,
                'reason': f'daily_loss_limit_reached ({daily_loss_pct:.1%})',
                'action': 'lock_trading_today'
            }
        
        # 5. Check consecutive losses
        max_consecutive = self.global_risk.get('max_consecutive_losses', 5)
        if self.consecutive_losses >= max_consecutive:
            return {
                'approved': False,
                'reason': f'max_consecutive_losses ({self.consecutive_losses})',
                'action': 'pause_trading'
            }
        
        # 6. Check daily trade limit
        max_trades_daily = self.risk_config.get('time_based_limits', {}).get('daily', {}).get('max_trades', 20)
        if self.trades_today >= max_trades_daily:
            return {
                'approved': False,
                'reason': f'daily_trade_limit ({self.trades_today}/{max_trades_daily})',
                'action': 'reject'
            }
        
        # All checks passed
        return {
            'approved': True,
            'reason': 'all_limits_passed',
            'new_exposure_percent': new_exposure_percent
        }
    
    def update_position_result(self, closed_position: Dict):
        """
        Update risk metrics after position is closed
        
        Args:
            closed_position: Dict with position details and P&L
        """
        pnl = closed_position.get('pnl', 0)
        pnl_pct = closed_position.get('pnl_pct', 0)
        
        # Update capital
        self.current_capital += pnl
        
        # Update daily/weekly P&L
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        
        # Update consecutive losses/wins
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Add to trade history
        self.trade_history.append({
            'timestamp': datetime.utcnow(),
            'symbol': closed_position.get('symbol'),
            'direction': closed_position.get('direction'),
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': closed_position.get('exit_reason')
        })
        
        logger.info(
            f"Position closed: {closed_position.get('symbol')} "
            f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%), "
            f"New capital: ${self.current_capital:.2f}"
        )
    
    def reset_daily_counters(self):
        """Reset daily counters (call at start of new trading day)"""
        self.daily_pnl = 0.0
        self.trades_today = 0
        logger.info("Daily risk counters reset")
    
    def reset_weekly_counters(self):
        """Reset weekly counters"""
        self.weekly_pnl = 0.0
        logger.info("Weekly risk counters reset")
    
    def get_risk_metrics(self) -> Dict:
        """
        Get comprehensive risk metrics
        
        Returns:
            Dict with all risk metrics
        """
        drawdown = self.calculate_drawdown()
        
        # Calculate win rate from recent trades
        recent_trades = self.trade_history[-50:] if len(self.trade_history) >= 50 else self.trade_history
        
        if recent_trades:
            wins = sum(1 for t in recent_trades if t['pnl'] > 0)
            win_rate = wins / len(recent_trades)
            
            winning_trades = [t['pnl'] for t in recent_trades if t['pnl'] > 0]
            losing_trades = [t['pnl'] for t in recent_trades if t['pnl'] < 0]
            
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0
            
            profit_factor = (sum(winning_trades) / abs(sum(losing_trades))) if losing_trades else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'capital': {
                'initial': self.initial_capital,
                'current': self.current_capital,
                'peak': self.peak_capital,
                'return_pct': ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
            },
            'drawdown': drawdown,
            'daily': {
                'pnl': self.daily_pnl,
                'trades': self.trades_today,
                'pnl_pct': (self.daily_pnl / self.initial_capital) * 100
            },
            'weekly': {
                'pnl': self.weekly_pnl,
                'pnl_pct': (self.weekly_pnl / self.initial_capital) * 100
            },
            'performance': {
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_trades': len(self.trade_history),
                'consecutive_losses': self.consecutive_losses
            },
            'status': {
                'trading_locked': self.trading_locked,
                'trading_allowed': not self.trading_locked and drawdown['current_drawdown'] < self.max_drawdown
            }
        }
    
    def lock_trading(self, reason: str = 'manual'):
        """Lock all trading"""
        self.trading_locked = True
        logger.warning(f"Trading LOCKED: {reason}")
    
    def unlock_trading(self):
        """Unlock trading"""
        self.trading_locked = False
        logger.info("Trading UNLOCKED")
    
    def adjust_risk_for_conditions(
        self,
        base_risk: float,
        drawdown_level: float,
        volatility_state: str
    ) -> float:
        """
        Adjust risk based on current conditions
        
        Returns:
            Adjusted risk percentage
        """
        adjusted_risk = base_risk
        
        # Reduce risk as drawdown increases
        if drawdown_level > 0.10:
            adjusted_risk *= 0.8
        if drawdown_level > 0.15:
            adjusted_risk *= 0.6
        
        # Adjust for volatility
        if volatility_state == 'HIGH':
            adjusted_risk *= 0.7
        elif volatility_state == 'EXTREME':
            adjusted_risk *= 0.5
        
        # Reduce risk after consecutive losses
        if self.consecutive_losses >= 3:
            adjusted_risk *= 0.75
        if self.consecutive_losses >= 5:
            adjusted_risk *= 0.5
        
        return max(adjusted_risk, 0.001)  # Minimum 0.1% risk
