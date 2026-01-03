"""
Event Logger
Centralized logging for trading events and system activities
"""

import logging
from datetime import datetime
from typing import Dict, List
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class EventLogger:
    """
    Logs and tracks trading events, signals, and system activities
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.log_config = config.get('logging', {})
        
        self.events = []
        self.max_events = 1000
        
        # Event categories
        self.signals = []
        self.trades = []
        self.errors = []
        
        logger.info("EventLogger initialized")
    
    def log_signal(
        self,
        symbol: str,
        timeframe: str,
        signal_type: str,
        direction: str,
        confidence: float,
        metadata: Dict = None
    ):
        """
        Log trading signal
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            signal_type: 'entry' or 'exit'
            direction: 'long' or 'short'
            confidence: Signal confidence (0-1)
            metadata: Additional signal data
        """
        event = {
            'timestamp': datetime.utcnow(),
            'type': 'signal',
            'symbol': symbol,
            'timeframe': timeframe,
            'signal_type': signal_type,
            'direction': direction,
            'confidence': confidence,
            'metadata': metadata or {}
        }
        
        self.signals.append(event)
        self._add_event(event)
        
        logger.info(
            f"SIGNAL: {symbol} {timeframe} {signal_type.upper()} {direction.upper()} "
            f"(confidence: {confidence:.2f})"
        )
    
    def log_trade(
        self,
        action: str,
        symbol: str,
        direction: str = None,
        size: float = None,
        price: float = None,
        pnl: float = None,
        reason: str = None,
        metadata: Dict = None
    ):
        """
        Log trade execution
        
        Args:
            action: 'entry' or 'exit'
            symbol: Trading symbol
            direction: Trade direction
            size: Position size
            price: Execution price
            pnl: P&L for exits
            reason: Trade reason
            metadata: Additional data
        """
        event = {
            'timestamp': datetime.utcnow(),
            'type': 'trade',
            'action': action,
            'symbol': symbol,
            'direction': direction,
            'size': size,
            'price': price,
            'pnl': pnl,
            'reason': reason,
            'metadata': metadata or {}
        }
        
        self.trades.append(event)
        self._add_event(event)
        
        if action == 'entry':
            logger.info(
                f"TRADE ENTRY: {symbol} {direction.upper() if direction else ''} "
                f"size={size}, price={price}"
            )
        elif action == 'exit':
            logger.info(
                f"TRADE EXIT: {symbol} price={price}, P&L=${pnl:.2f}, reason={reason}"
            )
    
    def log_error(
        self,
        error_type: str,
        message: str,
        symbol: str = None,
        metadata: Dict = None
    ):
        """
        Log system error
        
        Args:
            error_type: Type of error
            message: Error message
            symbol: Optional related symbol
            metadata: Additional context
        """
        event = {
            'timestamp': datetime.utcnow(),
            'type': 'error',
            'error_type': error_type,
            'message': message,
            'symbol': symbol,
            'metadata': metadata or {}
        }
        
        self.errors.append(event)
        self._add_event(event)
        
        logger.error(f"ERROR ({error_type}): {message}")
    
    def log_system_event(
        self,
        event_type: str,
        message: str,
        level: str = 'info',
        metadata: Dict = None
    ):
        """
        Log general system event
        
        Args:
            event_type: Type of event
            message: Event message
            level: Log level ('info', 'warning', 'error')
            metadata: Additional data
        """
        event = {
            'timestamp': datetime.utcnow(),
            'type': 'system',
            'event_type': event_type,
            'message': message,
            'level': level,
            'metadata': metadata or {}
        }
        
        self._add_event(event)
        
        if level == 'warning':
            logger.warning(f"SYSTEM ({event_type}): {message}")
        elif level == 'error':
            logger.error(f"SYSTEM ({event_type}): {message}")
        else:
            logger.info(f"SYSTEM ({event_type}): {message}")
    
    def _add_event(self, event: Dict):
        """Add event to history"""
        self.events.append(event)
        
        # Keep only recent events
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
    
    def get_recent_events(self, count: int = 50, event_type: str = None) -> List[Dict]:
        """
        Get recent events
        
        Args:
            count: Number of events to return
            event_type: Optional filter by type
        
        Returns:
            List of event dicts
        """
        if event_type:
            filtered = [e for e in self.events if e['type'] == event_type]
            return filtered[-count:]
        
        return self.events[-count:]
    
    def get_recent_signals(self, count: int = 20) -> List[Dict]:
        """Get recent trading signals"""
        return self.signals[-count:]
    
    def get_recent_trades(self, count: int = 20) -> List[Dict]:
        """Get recent trades"""
        return self.trades[-count:]
    
    def get_recent_errors(self, count: int = 20) -> List[Dict]:
        """Get recent errors"""
        return self.errors[-count:]
    
    def export_events_to_file(self, filepath: str):
        """
        Export events to JSON file
        
        Args:
            filepath: Output file path
        """
        try:
            events_serializable = [
                {**event, 'timestamp': event['timestamp'].isoformat()}
                for event in self.events
            ]
            
            with open(filepath, 'w') as f:
                json.dump(events_serializable, f, indent=2)
            
            logger.info(f"Events exported to {filepath}")
        
        except Exception as e:
            logger.error(f"Failed to export events: {e}")
    
    def get_event_statistics(self) -> Dict:
        """
        Get statistics about logged events
        
        Returns:
            Dict with event stats
        """
        return {
            'total_events': len(self.events),
            'signals': len(self.signals),
            'trades': len(self.trades),
            'errors': len(self.errors),
            'recent_signal_count': len([e for e in self.events[-100:] if e['type'] == 'signal']),
            'recent_trade_count': len([e for e in self.events[-100:] if e['type'] == 'trade']),
            'recent_error_count': len([e for e in self.events[-100:] if e['type'] == 'error'])
        }
