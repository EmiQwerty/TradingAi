"""
Market Structure Detection Module
Identifies Break of Structure (BOS), Change of Character (CHOCH), and Order Blocks
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class StructureDetector:
    """
    Detects market structure patterns for smart money concepts
    - Break of Structure (BOS): Continuation pattern
    - Change of Character (CHOCH): Reversal pattern
    - Order Blocks: Institutional supply/demand zones
    """
    
    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        
        self.structure_config = config.get('structure', {})
        self.bos_lookback = self.structure_config.get('bos_lookback', 50)
        self.choch_lookback = self.structure_config.get('choch_lookback', 30)
        self.ob_threshold = self.structure_config.get('order_block_threshold', 0.7)
        self.swing_strength = self.structure_config.get('swing_strength', 5)
        
        logger.info("StructureDetector initialized")
    
    def identify_swing_points(
        self, 
        df: pd.DataFrame,
        strength: int = None
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Identify swing highs and swing lows
        
        Args:
            df: DataFrame with OHLC data
            strength: Number of bars before/after for swing validation
        
        Returns:
            Tuple of (swing_highs, swing_lows) as boolean Series
        """
        if strength is None:
            strength = self.swing_strength
        
        high = df['high'].values
        low = df['low'].values
        
        swing_highs = pd.Series(False, index=df.index)
        swing_lows = pd.Series(False, index=df.index)
        
        for i in range(strength, len(df) - strength):
            # Check if it's a swing high
            is_swing_high = True
            for j in range(1, strength + 1):
                if high[i] <= high[i - j] or high[i] <= high[i + j]:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.iloc[i] = True
            
            # Check if it's a swing low
            is_swing_low = True
            for j in range(1, strength + 1):
                if low[i] >= low[i - j] or low[i] >= low[i + j]:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.iloc[i] = True
        
        return swing_highs, swing_lows
    
    def detect_bos(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Detect Break of Structure (BOS)
        BOS indicates trend continuation when price breaks previous swing high/low
        
        Returns:
            Dict with BOS information: {
                'detected': bool,
                'direction': 'bullish' or 'bearish',
                'level': float (price level broken),
                'strength': float (0-1)
            }
        """
        if len(df) < self.bos_lookback:
            return {'detected': False}
        
        recent_df = df.tail(self.bos_lookback).copy()
        swing_highs, swing_lows = self.identify_swing_points(recent_df)
        
        current_close = recent_df['close'].iloc[-1]
        current_high = recent_df['high'].iloc[-1]
        current_low = recent_df['low'].iloc[-1]
        
        # Find last swing high and low
        last_swing_high_idx = recent_df[swing_highs].index[-1] if swing_highs.any() else None
        last_swing_low_idx = recent_df[swing_lows].index[-1] if swing_lows.any() else None
        
        # Bullish BOS: Price breaks above previous swing high
        if last_swing_high_idx is not None:
            swing_high_level = recent_df.loc[last_swing_high_idx, 'high']
            
            if current_high > swing_high_level:
                strength = min((current_high - swing_high_level) / swing_high_level * 100, 1.0)
                
                return {
                    'detected': True,
                    'direction': 'bullish',
                    'level': swing_high_level,
                    'strength': strength,
                    'bars_ago': len(recent_df) - recent_df.index.get_loc(last_swing_high_idx)
                }
        
        # Bearish BOS: Price breaks below previous swing low
        if last_swing_low_idx is not None:
            swing_low_level = recent_df.loc[last_swing_low_idx, 'low']
            
            if current_low < swing_low_level:
                strength = min((swing_low_level - current_low) / swing_low_level * 100, 1.0)
                
                return {
                    'detected': True,
                    'direction': 'bearish',
                    'level': swing_low_level,
                    'strength': strength,
                    'bars_ago': len(recent_df) - recent_df.index.get_loc(last_swing_low_idx)
                }
        
        return {'detected': False}
    
    def detect_choch(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Detect Change of Character (CHOCH)
        CHOCH indicates potential trend reversal
        
        Returns:
            Dict with CHOCH information
        """
        if len(df) < self.choch_lookback:
            return {'detected': False}
        
        recent_df = df.tail(self.choch_lookback).copy()
        swing_highs, swing_lows = self.identify_swing_points(recent_df)
        
        # Get sequence of swing points
        swing_high_indices = recent_df[swing_highs].index.tolist()
        swing_low_indices = recent_df[swing_lows].index.tolist()
        
        if len(swing_high_indices) < 2 or len(swing_low_indices) < 2:
            return {'detected': False}
        
        # Bullish CHOCH: Lower lows followed by higher high
        # Check if we had downtrend (lower lows) then broke structure upward
        last_two_lows = [recent_df.loc[idx, 'low'] for idx in swing_low_indices[-2:]]
        
        if last_two_lows[1] < last_two_lows[0]:  # Downtrend
            # Check if recent price broke above recent swing high
            recent_high_level = recent_df.loc[swing_high_indices[-1], 'high'] if swing_high_indices else 0
            current_high = recent_df['high'].iloc[-1]
            
            if current_high > recent_high_level:
                return {
                    'detected': True,
                    'direction': 'bullish',
                    'level': recent_high_level,
                    'strength': 0.8,
                    'type': 'choch'
                }
        
        # Bearish CHOCH: Higher highs followed by lower low
        last_two_highs = [recent_df.loc[idx, 'high'] for idx in swing_high_indices[-2:]]
        
        if last_two_highs[1] > last_two_highs[0]:  # Uptrend
            # Check if recent price broke below recent swing low
            recent_low_level = recent_df.loc[swing_low_indices[-1], 'low'] if swing_low_indices else 0
            current_low = recent_df['low'].iloc[-1]
            
            if current_low < recent_low_level:
                return {
                    'detected': True,
                    'direction': 'bearish',
                    'level': recent_low_level,
                    'strength': 0.8,
                    'type': 'choch'
                }
        
        return {'detected': False}
    
    def identify_order_blocks(
        self, 
        df: pd.DataFrame,
        direction: str = None
    ) -> List[Dict]:
        """
        Identify Order Blocks (institutional buying/selling zones)
        
        Args:
            df: DataFrame with OHLC data
            direction: 'bullish' or 'bearish' or None (both)
        
        Returns:
            List of order block dicts with zone information
        """
        order_blocks = []
        
        if len(df) < 20:
            return order_blocks
        
        swing_highs, swing_lows = self.identify_swing_points(df)
        
        # Bullish Order Blocks (demand zones)
        if direction in [None, 'bullish']:
            for i in range(len(df) - 10):
                if swing_lows.iloc[i]:
                    # Check for strong move up after this low
                    low_price = df['low'].iloc[i]
                    subsequent_high = df['high'].iloc[i:i+10].max()
                    
                    move_strength = (subsequent_high - low_price) / low_price
                    
                    if move_strength > self.ob_threshold * 0.01:  # Significant move
                        order_blocks.append({
                            'type': 'bullish',
                            'zone_low': df['low'].iloc[i],
                            'zone_high': df['high'].iloc[i],
                            'timestamp': df['timestamp'].iloc[i] if 'timestamp' in df.columns else i,
                            'strength': min(move_strength * 10, 1.0),
                            'tested': False  # Will be updated if price returns
                        })
        
        # Bearish Order Blocks (supply zones)
        if direction in [None, 'bearish']:
            for i in range(len(df) - 10):
                if swing_highs.iloc[i]:
                    # Check for strong move down after this high
                    high_price = df['high'].iloc[i]
                    subsequent_low = df['low'].iloc[i:i+10].min()
                    
                    move_strength = (high_price - subsequent_low) / high_price
                    
                    if move_strength > self.ob_threshold * 0.01:
                        order_blocks.append({
                            'type': 'bearish',
                            'zone_low': df['low'].iloc[i],
                            'zone_high': df['high'].iloc[i],
                            'timestamp': df['timestamp'].iloc[i] if 'timestamp' in df.columns else i,
                            'strength': min(move_strength * 10, 1.0),
                            'tested': False
                        })
        
        # Sort by timestamp (most recent first)
        order_blocks = sorted(order_blocks, key=lambda x: x['timestamp'], reverse=True)
        
        return order_blocks[:5]  # Return top 5 most recent
    
    def get_structure_score(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate overall structure score for current market
        
        Returns:
            Dict with structure metrics: {
                'trend_structure': float (-1 to 1, bearish to bullish),
                'structure_strength': float (0 to 1),
                'bos_score': float,
                'choch_score': float
            }
        """
        bos = self.detect_bos(df)
        choch = self.detect_choch(df)
        
        # Calculate trend structure score
        trend_score = 0.0
        structure_strength = 0.0
        
        if bos['detected']:
            if bos['direction'] == 'bullish':
                trend_score += bos['strength']
            else:
                trend_score -= bos['strength']
            structure_strength = bos['strength']
        
        if choch['detected']:
            # CHOCH indicates potential reversal, reduces structure strength
            structure_strength *= 0.5
            if choch['direction'] == 'bullish':
                trend_score += choch['strength'] * 0.7
            else:
                trend_score -= choch['strength'] * 0.7
        
        # Normalize trend score to -1 to 1
        trend_score = max(min(trend_score, 1.0), -1.0)
        
        return {
            'trend_structure': trend_score,
            'structure_strength': structure_strength,
            'bos_score': bos['strength'] if bos['detected'] else 0.0,
            'choch_score': choch['strength'] if choch['detected'] else 0.0,
            'bos_direction': bos.get('direction'),
            'choch_direction': choch.get('direction')
        }
    
    def is_price_in_order_block(
        self, 
        current_price: float, 
        order_blocks: List[Dict]
    ) -> Optional[Dict]:
        """
        Check if current price is within an order block zone
        
        Returns:
            Order block dict if price is in zone, None otherwise
        """
        for ob in order_blocks:
            if ob['zone_low'] <= current_price <= ob['zone_high']:
                return ob
        return None
    
    def analyze_structure(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive structure analysis
        
        Returns:
            Complete structure analysis dict
        """
        bos = self.detect_bos(df)
        choch = self.detect_choch(df)
        order_blocks = self.identify_order_blocks(df)
        structure_score = self.get_structure_score(df)
        
        current_price = df['close'].iloc[-1]
        active_ob = self.is_price_in_order_block(current_price, order_blocks)
        
        return {
            'bos': bos,
            'choch': choch,
            'order_blocks': order_blocks,
            'structure_score': structure_score,
            'active_order_block': active_ob,
            'timestamp': df['timestamp'].iloc[-1] if 'timestamp' in df.columns else None
        }
