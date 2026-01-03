"""
State Data Models
Defines data structures for system state
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List
import pandas as pd


@dataclass
class TFState:
    """State for a single timeframe"""
    timeframe: str
    last_update: datetime
    current_price: float
    
    # Market data
    candles: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Indicators
    indicators: Dict = field(default_factory=dict)
    
    # Structure analysis
    structure: Dict = field(default_factory=dict)
    
    # Volatility
    volatility: Dict = field(default_factory=dict)
    
    # ML insights
    ml_insights: Dict = field(default_factory=dict)
    
    # Macro adjustment
    macro_adjustment: Dict = field(default_factory=dict)
    
    # Confidence score
    confidence: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timeframe': self.timeframe,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'current_price': self.current_price,
            'indicators': self.indicators,
            'structure': self.structure,
            'volatility': self.volatility,
            'ml_insights': self.ml_insights,
            'macro_adjustment': self.macro_adjustment,
            'confidence': self.confidence
        }


@dataclass
class SymbolState:
    """State for a single trading symbol"""
    symbol: str
    last_update: datetime
    
    # Multi-timeframe data
    timeframes: Dict[str, TFState] = field(default_factory=dict)
    
    # Symbol-specific data
    spread: float = 0.0
    volume: float = 0.0
    session: str = ""
    
    # Correlation data
    correlations: Dict[str, float] = field(default_factory=dict)
    
    def get_timeframe(self, tf: str) -> Optional[TFState]:
        """Get timeframe state"""
        return self.timeframes.get(tf)
    
    def update_timeframe(self, tf_state: TFState):
        """Update timeframe state"""
        self.timeframes[tf_state.timeframe] = tf_state
        self.last_update = datetime.utcnow()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'spread': self.spread,
            'volume': self.volume,
            'session': self.session,
            'correlations': self.correlations,
            'timeframes': {
                tf: state.to_dict() 
                for tf, state in self.timeframes.items()
            }
        }


@dataclass
class Position:
    """Represents an open position"""
    position_id: str
    symbol: str
    direction: str  # 'long' or 'short'
    size: float
    entry_price: float
    entry_time: datetime
    
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    current_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    
    # Metadata
    entry_confidence: float = 0.0
    entry_reason: str = ""
    trailing_stop_enabled: bool = False
    trailing_stop_level: float = 0.0
    
    # Broker data
    broker_position_id: str = ""
    swap: float = 0.0
    commission: float = 0.0
    
    def update_current_price(self, price: float):
        """Update current price and recalculate P&L"""
        self.current_price = price
        
        if self.direction == 'long':
            self.pnl = (price - self.entry_price) * self.size
            self.pnl_pct = ((price - self.entry_price) / self.entry_price) * 100
        else:  # short
            self.pnl = (self.entry_price - price) * self.size
            self.pnl_pct = ((self.entry_price - price) / self.entry_price) * 100
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'size': self.size,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'current_price': self.current_price,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'entry_confidence': self.entry_confidence,
            'entry_reason': self.entry_reason,
            'trailing_stop_enabled': self.trailing_stop_enabled,
            'broker_position_id': self.broker_position_id
        }


@dataclass
class SystemState:
    """Global system state"""
    timestamp: datetime
    
    # Market states per symbol
    symbols: Dict[str, SymbolState] = field(default_factory=dict)
    
    # Open positions
    positions: List[Position] = field(default_factory=list)
    
    # Account info
    account: Dict = field(default_factory=dict)
    
    # Risk metrics
    risk_metrics: Dict = field(default_factory=dict)
    
    # System status
    trading_enabled: bool = True
    system_health: str = "OK"
    errors: List[str] = field(default_factory=list)
    
    def get_symbol_state(self, symbol: str) -> Optional[SymbolState]:
        """Get state for symbol"""
        return self.symbols.get(symbol)
    
    def update_symbol_state(self, symbol_state: SymbolState):
        """Update symbol state"""
        self.symbols[symbol_state.symbol] = symbol_state
        self.timestamp = datetime.utcnow()
    
    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """Get open position for symbol"""
        for pos in self.positions:
            if pos.symbol == symbol:
                return pos
        return None
    
    def add_position(self, position: Position):
        """Add new position"""
        self.positions.append(position)
    
    def remove_position(self, position_id: str):
        """Remove closed position"""
        self.positions = [p for p in self.positions if p.position_id != position_id]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'symbols': {
                symbol: state.to_dict() 
                for symbol, state in self.symbols.items()
            },
            'positions': [pos.to_dict() for pos in self.positions],
            'account': self.account,
            'risk_metrics': self.risk_metrics,
            'trading_enabled': self.trading_enabled,
            'system_health': self.system_health,
            'errors': self.errors
        }
