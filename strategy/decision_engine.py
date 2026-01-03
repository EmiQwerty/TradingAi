"""
Decision Engine
Main trading logic: generates concrete trading decisions based on all inputs
"""

import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Core trading decision engine
    Analyzes all inputs and generates actionable trading decisions
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.symbols_config = config.get('symbols', [])
        self.execution_config = config.get('execution', {})
        
        # Create symbol-specific configs
        self.symbol_settings = {
            s['symbol']: s for s in self.symbols_config if s.get('enabled', True)
        }
        
        logger.info(f"DecisionEngine initialized for {len(self.symbol_settings)} symbols")
    
    def analyze_entry(
        self,
        symbol: str,
        timeframe: str,
        current_price: float,
        confidence: Dict,
        indicators: Dict,
        structure: Dict,
        ml_insights: Dict,
        volatility: Dict,
        macro_adjustment: Dict
    ) -> Optional[Dict]:
        """
        Analyze if entry conditions are met
        
        Returns:
            Entry decision dict or None
        """
        if symbol not in self.symbol_settings:
            logger.warning(f"No configuration for symbol: {symbol}")
            return None
        
        symbol_config = self.symbol_settings[symbol]
        strategy_config = symbol_config.get('strategy', {})
        
        # Get entry criteria from config
        min_confidence = strategy_config.get('entry_conditions', {}).get('min_confidence', 0.65)
        min_trend_strength = strategy_config.get('entry_conditions', {}).get('min_trend_strength', 0.6)
        
        # Check confidence threshold
        if confidence['confidence'] < min_confidence:
            logger.debug(f"{symbol}: Confidence too low ({confidence['confidence']:.2f})")
            return None
        
        # Check trend strength
        trend_strength = ml_insights.get('trend', {}).get('confidence', 0)
        if trend_strength < min_trend_strength:
            logger.debug(f"{symbol}: Trend strength too low ({trend_strength:.2f})")
            return None
        
        # Check direction consistency
        direction = confidence['direction']
        if direction == 'neutral':
            logger.debug(f"{symbol}: Direction is neutral")
            return None
        
        # Check structure confirmation if required
        if strategy_config.get('entry_conditions', {}).get('require_structure_confirmation'):
            structure_score = structure.get('structure_score', {}).get('trend_structure', 0)
            
            if direction == 'long' and structure_score < 0.3:
                logger.debug(f"{symbol}: No bullish structure confirmation")
                return None
            elif direction == 'short' and structure_score > -0.3:
                logger.debug(f"{symbol}: No bearish structure confirmation")
                return None
        
        # Check MA alignment if required
        if strategy_config.get('entry_conditions', {}).get('require_ma_alignment'):
            if 'ma_fast' in indicators and 'ma_medium' in indicators:
                ma_fast = indicators['ma_fast']
                ma_medium = indicators['ma_medium']
                
                if direction == 'long' and ma_fast <= ma_medium:
                    logger.debug(f"{symbol}: MA not aligned for long")
                    return None
                elif direction == 'short' and ma_fast >= ma_medium:
                    logger.debug(f"{symbol}: MA not aligned for short")
                    return None
        
        # Check macro filter
        if macro_adjustment['adjustment_factor'] < 0.5:
            logger.info(f"{symbol}: Macro filter blocks entry ({macro_adjustment['reason']})")
            return None
        
        # Calculate stop loss and take profit
        atr = indicators.get('atr', 0)
        stop_loss, take_profit = self._calculate_stop_take_profit(
            current_price, 
            direction, 
            atr, 
            symbol_config
        )
        
        # Calculate position size
        position_size = self._calculate_position_size(
            symbol,
            current_price,
            stop_loss,
            confidence['confidence'],
            volatility,
            macro_adjustment
        )
        
        # Generate entry decision
        decision = {
            'action': 'enter',
            'symbol': symbol,
            'direction': direction,
            'order_type': 'market',
            'price': current_price,
            'size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': confidence['confidence'],
            'timeframe': timeframe,
            'timestamp': datetime.utcnow(),
            'reason': self._generate_entry_reason(confidence, ml_insights, structure),
            'metadata': {
                'ml_score': confidence['breakdown']['ml_score'],
                'indicator_score': confidence['breakdown']['indicator_score'],
                'structure_score': confidence['breakdown']['structure_score'],
                'macro_factor': macro_adjustment['adjustment_factor'],
                'volatility_state': volatility.get('state', 'UNKNOWN')
            }
        }
        
        logger.info(
            f"ENTRY SIGNAL: {symbol} {direction.upper()} @ {current_price:.5f}, "
            f"SL={stop_loss:.5f}, TP={take_profit:.5f}, Size={position_size:.2f}, "
            f"Confidence={confidence['confidence']:.2f}"
        )
        
        return decision
    
    def analyze_exit(
        self,
        position: Dict,
        current_price: float,
        indicators: Dict,
        structure: Dict,
        ml_insights: Dict
    ) -> Optional[Dict]:
        """
        Analyze if exit conditions are met for existing position
        
        Returns:
            Exit decision dict or None
        """
        symbol = position['symbol']
        entry_price = position['entry_price']
        direction = position['direction']
        
        # Calculate current P&L
        if direction == 'long':
            pnl = current_price - entry_price
            pnl_pct = (pnl / entry_price) * 100
        else:  # short
            pnl = entry_price - current_price
            pnl_pct = (pnl / entry_price) * 100
        
        # Check stop loss hit
        if 'stop_loss' in position:
            if direction == 'long' and current_price <= position['stop_loss']:
                return self._create_exit_decision(
                    position, current_price, 'stop_loss_hit', pnl_pct
                )
            elif direction == 'short' and current_price >= position['stop_loss']:
                return self._create_exit_decision(
                    position, current_price, 'stop_loss_hit', pnl_pct
                )
        
        # Check take profit hit
        if 'take_profit' in position:
            if direction == 'long' and current_price >= position['take_profit']:
                return self._create_exit_decision(
                    position, current_price, 'take_profit_hit', pnl_pct
                )
            elif direction == 'short' and current_price <= position['take_profit']:
                return self._create_exit_decision(
                    position, current_price, 'take_profit_hit', pnl_pct
                )
        
        # Check for reversal signals
        structure_score = structure.get('structure_score', {}).get('trend_structure', 0)
        
        # CHOCH detected (reversal)
        if structure.get('choch', {}).get('detected'):
            choch_direction = structure['choch'].get('direction')
            
            if direction == 'long' and choch_direction == 'bearish':
                return self._create_exit_decision(
                    position, current_price, 'reversal_signal_choch', pnl_pct
                )
            elif direction == 'short' and choch_direction == 'bullish':
                return self._create_exit_decision(
                    position, current_price, 'reversal_signal_choch', pnl_pct
                )
        
        # ML regime change to CHAOS
        regime = ml_insights.get('regime', {}).get('regime')
        if regime == 'CHAOS':
            return self._create_exit_decision(
                position, current_price, 'regime_change_chaos', pnl_pct
            )
        
        # Trailing stop if enabled
        if position.get('trailing_stop_enabled'):
            trailing_stop = self._calculate_trailing_stop(position, current_price, indicators)
            if trailing_stop:
                if direction == 'long' and current_price <= trailing_stop:
                    return self._create_exit_decision(
                        position, current_price, 'trailing_stop', pnl_pct
                    )
                elif direction == 'short' and current_price >= trailing_stop:
                    return self._create_exit_decision(
                        position, current_price, 'trailing_stop', pnl_pct
                    )
        
        return None
    
    def _calculate_stop_take_profit(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        symbol_config: Dict
    ) -> tuple:
        """
        Calculate stop loss and take profit levels
        
        Returns:
            Tuple of (stop_loss, take_profit)
        """
        strategy_config = symbol_config.get('strategy', {})
        exit_config = strategy_config.get('exit_conditions', {})
        
        # Get pip value for symbol
        pip_value = symbol_config.get('pip_value', 0.0001)
        
        # Calculate stop loss
        stop_pips = exit_config.get('stop_loss_pips', 25)
        
        # Use ATR-based stop if ATR available and reasonable
        if atr > 0:
            atr_multiplier = 2.0
            atr_stop_distance = atr * atr_multiplier
            
            # Convert to pips and compare
            atr_pips = atr_stop_distance / pip_value
            stop_pips = max(min(atr_pips, 100), 10)  # Clamp between 10-100 pips
        
        stop_distance = stop_pips * pip_value
        
        if direction == 'long':
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        
        # Calculate take profit
        tp_ratio = exit_config.get('take_profit_ratio', 2.0)
        tp_distance = stop_distance * tp_ratio
        
        if direction == 'long':
            take_profit = entry_price + tp_distance
        else:
            take_profit = entry_price - tp_distance
        
        return stop_loss, take_profit
    
    def _calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        confidence: float,
        volatility: Dict,
        macro_adjustment: Dict
    ) -> float:
        """
        Calculate position size based on risk parameters
        
        Returns:
            Position size in lots
        """
        symbol_config = self.symbol_settings[symbol]
        risk_config = symbol_config.get('risk', {})
        
        # Base risk per trade
        base_risk = risk_config.get('max_risk_per_trade', 0.01)
        
        # Adjust based on confidence
        confidence_adjustment = 0.5 + (confidence * 0.5)  # 0.5 to 1.0
        
        # Adjust based on volatility
        vol_adjustment = volatility.get('implications', {}).get('position_size_adjustment', 1.0)
        
        # Macro adjustment
        macro_factor = macro_adjustment.get('adjustment_factor', 1.0)
        
        # Final risk
        final_risk = base_risk * confidence_adjustment * vol_adjustment * macro_factor
        
        # Calculate position size (simplified - assumes fixed account size)
        account_size = 10000  # This should come from actual account state
        risk_amount = account_size * final_risk
        
        # Distance to stop loss
        stop_distance = abs(entry_price - stop_loss)
        
        # Position size calculation (for forex)
        contract_size = symbol_config.get('contract_size', 100000)
        pip_value_per_lot = contract_size * symbol_config.get('pip_value', 0.0001)
        
        if stop_distance > 0:
            position_size = risk_amount / (stop_distance / symbol_config.get('pip_value', 0.0001))
            position_size = position_size / contract_size  # Convert to lots
        else:
            position_size = symbol_config.get('min_lot', 0.01)
        
        # Clamp to symbol limits
        min_lot = symbol_config.get('min_lot', 0.01)
        max_lot = symbol_config.get('max_lot', 10.0)
        position_size = max(min(position_size, max_lot), min_lot)
        
        # Round to lot step
        lot_step = symbol_config.get('lot_step', 0.01)
        position_size = round(position_size / lot_step) * lot_step
        
        return position_size
    
    def _generate_entry_reason(
        self,
        confidence: Dict,
        ml_insights: Dict,
        structure: Dict
    ) -> str:
        """Generate human-readable entry reason"""
        reasons = []
        
        regime = ml_insights.get('regime', {}).get('regime')
        if regime:
            reasons.append(f"regime:{regime}")
        
        if structure.get('bos', {}).get('detected'):
            bos_dir = structure['bos'].get('direction')
            reasons.append(f"BOS_{bos_dir}")
        
        if confidence['confidence'] > 0.8:
            reasons.append("high_confidence")
        
        return " | ".join(reasons) if reasons else "confluence"
    
    def _create_exit_decision(
        self,
        position: Dict,
        exit_price: float,
        reason: str,
        pnl_pct: float
    ) -> Dict:
        """Create exit decision dict"""
        return {
            'action': 'exit',
            'symbol': position['symbol'],
            'position_id': position.get('position_id'),
            'price': exit_price,
            'size': position.get('size', 0),
            'reason': reason,
            'pnl_pct': pnl_pct,
            'timestamp': datetime.utcnow()
        }
    
    def _calculate_trailing_stop(
        self,
        position: Dict,
        current_price: float,
        indicators: Dict
    ) -> Optional[float]:
        """Calculate trailing stop level"""
        # Simplified trailing stop logic
        entry_price = position['entry_price']
        direction = position['direction']
        
        # Check if in profit enough to activate trailing
        profit_threshold = 1.5 * abs(entry_price - position.get('stop_loss', entry_price))
        
        if direction == 'long':
            profit = current_price - entry_price
            if profit >= profit_threshold:
                # Trail at 50% of current profit
                trailing_stop = entry_price + (profit * 0.5)
                return max(trailing_stop, position.get('trailing_stop_level', 0))
        else:
            profit = entry_price - current_price
            if profit >= profit_threshold:
                trailing_stop = entry_price - (profit * 0.5)
                return min(trailing_stop, position.get('trailing_stop_level', float('inf')))
        
        return None
    
    def generate_decisions(
        self,
        market_state: Dict,
        positions: List[Dict]
    ) -> List[Dict]:
        """
        Generate all trading decisions for current market state
        
        Args:
            market_state: Complete market analysis for all symbols/timeframes
            positions: List of current open positions
        
        Returns:
            List of decision dicts (entries and exits)
        """
        decisions = []
        
        # Check exits for existing positions
        for position in positions:
            symbol = position['symbol']
            
            if symbol not in market_state:
                continue
            
            symbol_state = market_state[symbol]
            primary_tf = self.symbol_settings[symbol].get('strategy', {}).get('primary_tf', 'H1')
            
            if primary_tf not in symbol_state:
                continue
            
            tf_state = symbol_state[primary_tf]
            
            exit_decision = self.analyze_exit(
                position,
                tf_state['current_price'],
                tf_state.get('indicators', {}),
                tf_state.get('structure', {}),
                tf_state.get('ml_insights', {})
            )
            
            if exit_decision:
                decisions.append(exit_decision)
        
        # Check entries for each symbol
        for symbol in self.symbol_settings.keys():
            if symbol not in market_state:
                continue
            
            symbol_state = market_state[symbol]
            strategy_config = self.symbol_settings[symbol].get('strategy', {})
            primary_tf = strategy_config.get('primary_tf', 'H1')
            
            if primary_tf not in symbol_state:
                continue
            
            tf_state = symbol_state[primary_tf]
            
            # Check if already have position in this symbol
            has_position = any(p['symbol'] == symbol for p in positions)
            if has_position:
                continue  # Skip entry analysis if already in position
            
            entry_decision = self.analyze_entry(
                symbol,
                primary_tf,
                tf_state['current_price'],
                tf_state.get('confidence', {}),
                tf_state.get('indicators', {}),
                tf_state.get('structure', {}),
                tf_state.get('ml_insights', {}),
                tf_state.get('volatility', {}),
                tf_state.get('macro_adjustment', {})
            )
            
            if entry_decision:
                decisions.append(entry_decision)
        
        return decisions
