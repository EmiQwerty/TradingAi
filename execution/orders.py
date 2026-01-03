"""
Order Execution Module
Manages order execution with throttling, retry logic, and logging
"""

import time
from typing import Dict, Optional
from datetime import datetime
import logging

from execution.broker_api import BrokerAPI

logger = logging.getLogger(__name__)


class OrderExecutor:
    """
    Handles order execution with proper controls and error handling
    """
    
    def __init__(self, config: dict, broker_api: BrokerAPI):
        self.config = config
        self.broker_api = broker_api
        self.execution_config = config.get('execution', {})
        
        self.throttle_ms = self.execution_config.get('throttle_ms', 100)
        self.max_retries = self.execution_config.get('max_order_retries', 3)
        self.order_timeout = self.execution_config.get('order_timeout', 10)
        self.slippage_tolerance = self.execution_config.get('slippage_tolerance', 0.0005)
        
        self.last_order_time = 0
        self.orders_executed = []
        
        logger.info("OrderExecutor initialized")
    
    def _throttle(self):
        """Enforce throttle between orders"""
        current_time = time.time() * 1000  # Convert to milliseconds
        time_since_last = current_time - self.last_order_time
        
        if time_since_last < self.throttle_ms:
            sleep_time = (self.throttle_ms - time_since_last) / 1000
            time.sleep(sleep_time)
        
        self.last_order_time = time.time() * 1000
    
    def execute_decision(self, decision: Dict) -> Dict:
        """
        Execute trading decision
        
        Args:
            decision: Decision dict from DecisionEngine
        
        Returns:
            Execution result dict
        """
        action = decision.get('action')
        
        if action == 'enter':
            return self.execute_entry(decision)
        elif action == 'exit':
            return self.execute_exit(decision)
        else:
            logger.warning(f"Unknown action: {action}")
            return {'success': False, 'error': 'unknown_action'}
    
    def execute_entry(self, decision: Dict) -> Dict:
        """
        Execute entry order
        
        Args:
            decision: Entry decision dict
        
        Returns:
            Execution result
        """
        symbol = decision['symbol']
        direction = decision['direction']
        size = decision['size']
        stop_loss = decision.get('stop_loss')
        take_profit = decision.get('take_profit')
        
        logger.info(
            f"Executing ENTRY: {symbol} {direction.upper()} "
            f"size={size:.2f}, SL={stop_loss:.5f}, TP={take_profit:.5f}"
        )
        
        # Apply throttle
        self._throttle()
        
        # Execute with retries
        for attempt in range(self.max_retries):
            try:
                result = self.broker_api.place_market_order(
                    symbol=symbol,
                    direction=direction,
                    size=size,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    comment=f"confidence_{decision.get('confidence', 0):.2f}"
                )
                
                if result['success']:
                    # Log execution
                    execution_record = {
                        'timestamp': datetime.utcnow(),
                        'action': 'entry',
                        'symbol': symbol,
                        'direction': direction,
                        'size': size,
                        'price': result.get('price'),
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'order_id': result.get('order_id'),
                        'position_id': result.get('position_id'),
                        'decision_confidence': decision.get('confidence')
                    }
                    
                    self.orders_executed.append(execution_record)
                    
                    logger.info(
                        f"✓ Entry executed successfully: {symbol} @ {result.get('price')} "
                        f"(Position ID: {result.get('position_id')})"
                    )
                    
                    return {
                        'success': True,
                        'execution': execution_record,
                        'broker_response': result
                    }
                else:
                    logger.warning(f"Entry attempt {attempt + 1} failed: {result.get('error')}")
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(1)  # Wait before retry
                    else:
                        return {
                            'success': False,
                            'error': result.get('error'),
                            'attempts': attempt + 1
                        }
            
            except Exception as e:
                logger.error(f"Entry execution error (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                else:
                    return {
                        'success': False,
                        'error': str(e),
                        'attempts': attempt + 1
                    }
        
        return {'success': False, 'error': 'max_retries_exceeded'}
    
    def execute_exit(self, decision: Dict) -> Dict:
        """
        Execute exit order (close position)
        
        Args:
            decision: Exit decision dict
        
        Returns:
            Execution result
        """
        position_id = decision.get('position_id')
        symbol = decision['symbol']
        size = decision.get('size')
        reason = decision.get('reason', 'strategy_exit')
        
        if not position_id:
            logger.error(f"No position_id provided for exit: {symbol}")
            return {'success': False, 'error': 'missing_position_id'}
        
        logger.info(f"Executing EXIT: {symbol} position {position_id} (reason: {reason})")
        
        # Apply throttle
        self._throttle()
        
        # Execute with retries
        for attempt in range(self.max_retries):
            try:
                result = self.broker_api.close_position(
                    position_id=position_id,
                    size=size
                )
                
                if result['success']:
                    # Log execution
                    execution_record = {
                        'timestamp': datetime.utcnow(),
                        'action': 'exit',
                        'symbol': symbol,
                        'position_id': position_id,
                        'price': result.get('close_price'),
                        'pnl': result.get('pnl'),
                        'reason': reason
                    }
                    
                    self.orders_executed.append(execution_record)
                    
                    logger.info(
                        f"✓ Exit executed successfully: {symbol} @ {result.get('close_price')} "
                        f"P&L: ${result.get('pnl', 0):.2f}"
                    )
                    
                    return {
                        'success': True,
                        'execution': execution_record,
                        'broker_response': result
                    }
                else:
                    logger.warning(f"Exit attempt {attempt + 1} failed: {result.get('error')}")
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(1)
                    else:
                        return {
                            'success': False,
                            'error': result.get('error'),
                            'attempts': attempt + 1
                        }
            
            except Exception as e:
                logger.error(f"Exit execution error (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                else:
                    return {
                        'success': False,
                        'error': str(e),
                        'attempts': attempt + 1
                    }
        
        return {'success': False, 'error': 'max_retries_exceeded'}
    
    def modify_stop_loss(
        self,
        position_id: str,
        new_stop_loss: float
    ) -> Dict:
        """
        Modify stop loss for existing position
        """
        self._throttle()
        
        result = self.broker_api.modify_position(
            position_id=position_id,
            stop_loss=new_stop_loss
        )
        
        if result['success']:
            logger.info(f"✓ Stop loss modified for position {position_id}: {new_stop_loss}")
        else:
            logger.error(f"Failed to modify stop loss: {result.get('error')}")
        
        return result
    
    def modify_take_profit(
        self,
        position_id: str,
        new_take_profit: float
    ) -> Dict:
        """
        Modify take profit for existing position
        """
        self._throttle()
        
        result = self.broker_api.modify_position(
            position_id=position_id,
            take_profit=new_take_profit
        )
        
        if result['success']:
            logger.info(f"✓ Take profit modified for position {position_id}: {new_take_profit}")
        else:
            logger.error(f"Failed to modify take profit: {result.get('error')}")
        
        return result
    
    def cancel_all_orders(self, symbol: str = None) -> Dict:
        """
        Cancel all pending orders (optionally filtered by symbol)
        
        Returns:
            Dict with cancellation results
        """
        pending_orders = self.broker_api.get_pending_orders()
        
        if symbol:
            pending_orders = [o for o in pending_orders if o['symbol'] == symbol]
        
        results = {
            'cancelled': [],
            'failed': []
        }
        
        for order in pending_orders:
            result = self.broker_api.cancel_order(order['order_id'])
            
            if result['success']:
                results['cancelled'].append(order['order_id'])
            else:
                results['failed'].append({
                    'order_id': order['order_id'],
                    'error': result.get('error')
                })
        
        logger.info(
            f"Cancelled {len(results['cancelled'])} orders, "
            f"{len(results['failed'])} failed"
        )
        
        return results
    
    def close_all_positions(self, symbol: str = None) -> Dict:
        """
        Close all open positions (optionally filtered by symbol)
        Emergency function
        
        Returns:
            Dict with close results
        """
        positions = self.broker_api.get_open_positions()
        
        if symbol:
            positions = [p for p in positions if p['symbol'] == symbol]
        
        results = {
            'closed': [],
            'failed': []
        }
        
        for position in positions:
            result = self.broker_api.close_position(position['position_id'])
            
            if result['success']:
                results['closed'].append(position['position_id'])
            else:
                results['failed'].append({
                    'position_id': position['position_id'],
                    'error': result.get('error')
                })
        
        logger.warning(
            f"EMERGENCY CLOSE: Closed {len(results['closed'])} positions, "
            f"{len(results['failed'])} failed"
        )
        
        return results
    
    def get_execution_summary(self) -> Dict:
        """
        Get summary of executed orders
        
        Returns:
            Summary statistics
        """
        if not self.orders_executed:
            return {
                'total_orders': 0,
                'entries': 0,
                'exits': 0
            }
        
        entries = [o for o in self.orders_executed if o['action'] == 'entry']
        exits = [o for o in self.orders_executed if o['action'] == 'exit']
        
        return {
            'total_orders': len(self.orders_executed),
            'entries': len(entries),
            'exits': len(exits),
            'last_order_time': self.orders_executed[-1]['timestamp'] if self.orders_executed else None
        }
