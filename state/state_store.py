"""
State Store
Thread-safe central state management for the trading system
"""

import threading
from datetime import datetime
from typing import Dict, List, Optional
import json
import logging

from state.models import SystemState, SymbolState, TFState, Position

logger = logging.getLogger(__name__)


class StateStore:
    """
    Thread-safe central state store
    Maintains current state of all symbols, timeframes, and positions
    """
    
    def __init__(self):
        self._state = SystemState(timestamp=datetime.utcnow())
        self._lock = threading.RLock()
        
        logger.info("StateStore initialized")
    
    def update_symbol_timeframe(
        self,
        symbol: str,
        timeframe: str,
        tf_state: TFState
    ):
        """
        Update state for specific symbol and timeframe
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M1, M5, H1, etc.)
            tf_state: TFState object with updated data
        """
        with self._lock:
            # Get or create symbol state
            if symbol not in self._state.symbols:
                self._state.symbols[symbol] = SymbolState(
                    symbol=symbol,
                    last_update=datetime.utcnow()
                )
            
            symbol_state = self._state.symbols[symbol]
            symbol_state.update_timeframe(tf_state)
            
            logger.debug(f"Updated state: {symbol} {timeframe}")
    
    def get_symbol_state(self, symbol: str) -> Optional[SymbolState]:
        """Get complete state for symbol"""
        with self._lock:
            return self._state.symbols.get(symbol)
    
    def get_timeframe_state(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[TFState]:
        """Get state for specific symbol and timeframe"""
        with self._lock:
            symbol_state = self._state.symbols.get(symbol)
            if symbol_state:
                return symbol_state.timeframes.get(timeframe)
            return None
    
    def get_all_symbols(self) -> List[str]:
        """Get list of all symbols in state"""
        with self._lock:
            return list(self._state.symbols.keys())
    
    def update_positions(self, positions: List[Position]):
        """
        Update all positions
        
        Args:
            positions: List of Position objects
        """
        with self._lock:
            self._state.positions = positions
            self._state.timestamp = datetime.utcnow()
            
            logger.debug(f"Updated positions: {len(positions)} open")
    
    def add_position(self, position: Position):
        """Add new position to state"""
        with self._lock:
            self._state.add_position(position)
            logger.info(f"Position added: {position.symbol} {position.direction}")
    
    def remove_position(self, position_id: str):
        """Remove position from state"""
        with self._lock:
            self._state.remove_position(position_id)
            logger.info(f"Position removed: {position_id}")
    
    def get_positions(self) -> List[Position]:
        """Get all open positions"""
        with self._lock:
            return self._state.positions.copy()
    
    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol"""
        with self._lock:
            return self._state.get_position_by_symbol(symbol)
    
    def update_account_info(self, account_info: Dict):
        """
        Update account information
        
        Args:
            account_info: Dict with account data
        """
        with self._lock:
            self._state.account = account_info
            self._state.timestamp = datetime.utcnow()
            
            logger.debug(
                f"Account updated: Balance=${account_info.get('balance', 0):.2f}, "
                f"Equity=${account_info.get('equity', 0):.2f}"
            )
    
    def get_account_info(self) -> Dict:
        """Get current account information"""
        with self._lock:
            return self._state.account.copy()
    
    def update_risk_metrics(self, risk_metrics: Dict):
        """Update risk management metrics"""
        with self._lock:
            self._state.risk_metrics = risk_metrics
            self._state.timestamp = datetime.utcnow()
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        with self._lock:
            return self._state.risk_metrics.copy()
    
    def set_trading_enabled(self, enabled: bool, reason: str = ""):
        """Enable or disable trading"""
        with self._lock:
            self._state.trading_enabled = enabled
            
            if not enabled:
                self._state.errors.append(f"Trading disabled: {reason}")
                logger.warning(f"Trading disabled: {reason}")
            else:
                logger.info("Trading enabled")
    
    def is_trading_enabled(self) -> bool:
        """Check if trading is enabled"""
        with self._lock:
            return self._state.trading_enabled
    
    def add_error(self, error: str):
        """Add system error"""
        with self._lock:
            self._state.errors.append(f"{datetime.utcnow().isoformat()}: {error}")
            
            # Keep only last 100 errors
            if len(self._state.errors) > 100:
                self._state.errors = self._state.errors[-100:]
            
            logger.error(f"System error: {error}")
    
    def get_errors(self) -> List[str]:
        """Get recent system errors"""
        with self._lock:
            return self._state.errors.copy()
    
    def clear_errors(self):
        """Clear error list"""
        with self._lock:
            self._state.errors = []
    
    def set_system_health(self, health: str):
        """
        Set system health status
        
        Args:
            health: 'OK', 'WARNING', 'ERROR', 'CRITICAL'
        """
        with self._lock:
            self._state.system_health = health
    
    def get_system_health(self) -> str:
        """Get system health status"""
        with self._lock:
            return self._state.system_health
    
    def get_full_state(self) -> Dict:
        """
        Get complete system state as dictionary
        
        Returns:
            Dict with all state data
        """
        with self._lock:
            return self._state.to_dict()
    
    def get_state_summary(self) -> Dict:
        """
        Get summary of current state
        
        Returns:
            Dict with key metrics
        """
        with self._lock:
            account = self._state.account
            positions = self._state.positions
            risk = self._state.risk_metrics
            
            total_pnl = sum(p.pnl for p in positions)
            
            return {
                'timestamp': self._state.timestamp.isoformat(),
                'symbols_tracked': len(self._state.symbols),
                'open_positions': len(positions),
                'account_balance': account.get('balance', 0),
                'account_equity': account.get('equity', 0),
                'total_position_pnl': total_pnl,
                'trading_enabled': self._state.trading_enabled,
                'system_health': self._state.system_health,
                'error_count': len(self._state.errors),
                'risk_metrics': {
                    'drawdown': risk.get('drawdown', {}).get('current_drawdown', 0),
                    'daily_pnl': risk.get('daily', {}).get('pnl', 0),
                    'win_rate': risk.get('performance', {}).get('win_rate', 0)
                }
            }
    
    def export_state_to_file(self, filepath: str):
        """
        Export current state to JSON file
        
        Args:
            filepath: Path to output file
        """
        with self._lock:
            state_dict = self._state.to_dict()
            
            try:
                with open(filepath, 'w') as f:
                    json.dump(state_dict, f, indent=2, default=str)
                
                logger.info(f"State exported to {filepath}")
            except Exception as e:
                logger.error(f"Failed to export state: {e}")
    
    def get_market_overview(self) -> Dict:
        """
        Get overview of current market conditions
        
        Returns:
            Dict with market summary for all symbols
        """
        with self._lock:
            overview = {}
            
            for symbol, symbol_state in self._state.symbols.items():
                # Get primary timeframe (H1 or first available)
                primary_tf = None
                for tf in ['H1', 'M15', 'M5']:
                    if tf in symbol_state.timeframes:
                        primary_tf = symbol_state.timeframes[tf]
                        break
                
                if primary_tf:
                    overview[symbol] = {
                        'price': primary_tf.current_price,
                        'last_update': primary_tf.last_update.isoformat(),
                        'ml_regime': primary_tf.ml_insights.get('regime', {}).get('regime'),
                        'confidence': primary_tf.confidence.get('confidence', 0),
                        'direction': primary_tf.confidence.get('direction'),
                        'volatility': primary_tf.volatility.get('state')
                    }
            
            return overview
