"""
Chart Pattern Recognition - Riconoscimento pattern tecnici professionali
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class PatternType(Enum):
    """Tipi di pattern riconoscibili"""
    # Reversal patterns
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_SHOULDERS = "head_shoulders"
    INVERSE_HEAD_SHOULDERS = "inverse_head_shoulders"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    
    # Continuation patterns
    FLAG_BULLISH = "flag_bullish"
    FLAG_BEARISH = "flag_bearish"
    PENNANT = "pennant"
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    TRIANGLE_SYMMETRICAL = "triangle_symmetrical"
    
    # Candlestick patterns
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    DOJI = "doji"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"


@dataclass
class ChartPattern:
    """Pattern riconosciuto"""
    pattern_type: PatternType
    start_index: int
    end_index: int
    confidence: float  # 0-1
    direction: str  # 'bullish', 'bearish', 'neutral'
    target_price: Optional[float]  # Prezzo target del pattern
    metadata: Dict  # Info aggiuntive


class PatternRecognizer:
    """
    Riconoscitore di pattern tecnici professionali.
    Usa algoritmi geometrici e statistici per identificare pattern validi.
    """
    
    def __init__(self, min_pattern_bars: int = 10, tolerance: float = 0.02):
        """
        Args:
            min_pattern_bars: Minimo numero di barre per un pattern
            tolerance: Tolleranza percentuale per match (2% default)
        """
        self.min_pattern_bars = min_pattern_bars
        self.tolerance = tolerance
        self.patterns_found: List[ChartPattern] = []
    
    def scan_all_patterns(self, data: pd.DataFrame) -> List[ChartPattern]:
        """Scansiona tutti i pattern su dati storici"""
        self.patterns_found = []
        
        # Candlestick patterns (single/multi candela)
        self.patterns_found.extend(self._find_hammer(data))
        self.patterns_found.extend(self._find_engulfing(data))
        self.patterns_found.extend(self._find_doji(data))
        self.patterns_found.extend(self._find_morning_evening_star(data))
        
        # Chart patterns (swing-based)
        self.patterns_found.extend(self._find_double_top_bottom(data))
        self.patterns_found.extend(self._find_head_shoulders(data))
        self.patterns_found.extend(self._find_triangles(data))
        self.patterns_found.extend(self._find_flags(data))
        
        return self.patterns_found
    
    # --- CANDLESTICK PATTERNS ---
    
    def _find_hammer(self, data: pd.DataFrame) -> List[ChartPattern]:
        """Hammer / Shooting Star"""
        patterns = []
        
        for i in range(2, len(data)):
            current = data.iloc[i]
            
            body = abs(current['close'] - current['open'])
            total_range = current['high'] - current['low']
            
            if total_range == 0:
                continue
            
            lower_wick = min(current['open'], current['close']) - current['low']
            upper_wick = current['high'] - max(current['open'], current['close'])
            
            # Hammer: long lower wick, small body, small upper wick
            if (lower_wick > 2 * body and
                upper_wick < 0.3 * total_range and
                body < 0.3 * total_range):
                
                patterns.append(ChartPattern(
                    pattern_type=PatternType.HAMMER,
                    start_index=i,
                    end_index=i,
                    confidence=0.7,
                    direction='bullish',
                    target_price=current['close'] + total_range,
                    metadata={
                        'body_size': body,
                        'lower_wick': lower_wick,
                        'upper_wick': upper_wick
                    }
                ))
            
            # Shooting Star: long upper wick, small body, small lower wick
            elif (upper_wick > 2 * body and
                  lower_wick < 0.3 * total_range and
                  body < 0.3 * total_range):
                
                patterns.append(ChartPattern(
                    pattern_type=PatternType.SHOOTING_STAR,
                    start_index=i,
                    end_index=i,
                    confidence=0.7,
                    direction='bearish',
                    target_price=current['close'] - total_range,
                    metadata={
                        'body_size': body,
                        'lower_wick': lower_wick,
                        'upper_wick': upper_wick
                    }
                ))
        
        return patterns
    
    def _find_engulfing(self, data: pd.DataFrame) -> List[ChartPattern]:
        """Bullish/Bearish Engulfing"""
        patterns = []
        
        for i in range(1, len(data)):
            prev = data.iloc[i-1]
            current = data.iloc[i]
            
            prev_body = abs(prev['close'] - prev['open'])
            curr_body = abs(current['close'] - current['open'])
            
            # Bullish Engulfing
            if (prev['close'] < prev['open'] and  # prev bearish
                current['close'] > current['open'] and  # current bullish
                current['open'] < prev['close'] and
                current['close'] > prev['open'] and
                curr_body > prev_body):
                
                patterns.append(ChartPattern(
                    pattern_type=PatternType.ENGULFING_BULLISH,
                    start_index=i-1,
                    end_index=i,
                    confidence=0.75,
                    direction='bullish',
                    target_price=current['close'] + (current['close'] - prev['close']),
                    metadata={'body_ratio': curr_body / prev_body}
                ))
            
            # Bearish Engulfing
            elif (prev['close'] > prev['open'] and  # prev bullish
                  current['close'] < current['open'] and  # current bearish
                  current['open'] > prev['close'] and
                  current['close'] < prev['open'] and
                  curr_body > prev_body):
                
                patterns.append(ChartPattern(
                    pattern_type=PatternType.ENGULFING_BEARISH,
                    start_index=i-1,
                    end_index=i,
                    confidence=0.75,
                    direction='bearish',
                    target_price=current['close'] - (prev['close'] - current['close']),
                    metadata={'body_ratio': curr_body / prev_body}
                ))
        
        return patterns
    
    def _find_doji(self, data: pd.DataFrame) -> List[ChartPattern]:
        """Doji - Indecision candle"""
        patterns = []
        
        for i in range(len(data)):
            current = data.iloc[i]
            
            body = abs(current['close'] - current['open'])
            total_range = current['high'] - current['low']
            
            if total_range == 0:
                continue
            
            # Doji: body < 10% of total range
            if body < 0.1 * total_range:
                patterns.append(ChartPattern(
                    pattern_type=PatternType.DOJI,
                    start_index=i,
                    end_index=i,
                    confidence=0.6,
                    direction='neutral',
                    target_price=None,
                    metadata={'body_ratio': body / total_range}
                ))
        
        return patterns
    
    def _find_morning_evening_star(self, data: pd.DataFrame) -> List[ChartPattern]:
        """Morning Star / Evening Star (3-candle reversal)"""
        patterns = []
        
        for i in range(2, len(data)):
            c1 = data.iloc[i-2]
            c2 = data.iloc[i-1]
            c3 = data.iloc[i]
            
            body1 = abs(c1['close'] - c1['open'])
            body2 = abs(c2['close'] - c2['open'])
            body3 = abs(c3['close'] - c3['open'])
            
            # Morning Star: bearish + small + bullish
            if (c1['close'] < c1['open'] and  # c1 bearish
                body2 < 0.3 * body1 and  # c2 small
                c3['close'] > c3['open'] and  # c3 bullish
                c3['close'] > (c1['open'] + c1['close']) / 2):  # c3 closes above midpoint
                
                patterns.append(ChartPattern(
                    pattern_type=PatternType.MORNING_STAR,
                    start_index=i-2,
                    end_index=i,
                    confidence=0.8,
                    direction='bullish',
                    target_price=c3['close'] + body1,
                    metadata={'reversal_strength': body3 / body1}
                ))
            
            # Evening Star: bullish + small + bearish
            elif (c1['close'] > c1['open'] and  # c1 bullish
                  body2 < 0.3 * body1 and  # c2 small
                  c3['close'] < c3['open'] and  # c3 bearish
                  c3['close'] < (c1['open'] + c1['close']) / 2):
                
                patterns.append(ChartPattern(
                    pattern_type=PatternType.EVENING_STAR,
                    start_index=i-2,
                    end_index=i,
                    confidence=0.8,
                    direction='bearish',
                    target_price=c3['close'] - body1,
                    metadata={'reversal_strength': body3 / body1}
                ))
        
        return patterns
    
    # --- SWING PATTERNS ---
    
    def _find_double_top_bottom(self, data: pd.DataFrame) -> List[ChartPattern]:
        """Double Top / Double Bottom"""
        patterns = []
        
        # Find swing highs and lows
        swing_highs = self._find_swing_points(data, 'high')
        swing_lows = self._find_swing_points(data, 'low')
        
        # Double Top
        for i in range(len(swing_highs) - 1):
            for j in range(i+1, len(swing_highs)):
                idx1, price1 = swing_highs[i]
                idx2, price2 = swing_highs[j]
                
                if abs(price1 - price2) / price1 < self.tolerance and idx2 - idx1 > self.min_pattern_bars:
                    neckline = data.iloc[idx1:idx2+1]['low'].min()
                    target = neckline - (price1 - neckline)
                    
                    patterns.append(ChartPattern(
                        pattern_type=PatternType.DOUBLE_TOP,
                        start_index=idx1,
                        end_index=idx2,
                        confidence=0.75,
                        direction='bearish',
                        target_price=target,
                        metadata={'peak1': price1, 'peak2': price2, 'neckline': neckline}
                    ))
                    break
        
        # Double Bottom
        for i in range(len(swing_lows) - 1):
            for j in range(i+1, len(swing_lows)):
                idx1, price1 = swing_lows[i]
                idx2, price2 = swing_lows[j]
                
                if abs(price1 - price2) / price1 < self.tolerance and idx2 - idx1 > self.min_pattern_bars:
                    neckline = data.iloc[idx1:idx2+1]['high'].max()
                    target = neckline + (neckline - price1)
                    
                    patterns.append(ChartPattern(
                        pattern_type=PatternType.DOUBLE_BOTTOM,
                        start_index=idx1,
                        end_index=idx2,
                        confidence=0.75,
                        direction='bullish',
                        target_price=target,
                        metadata={'trough1': price1, 'trough2': price2, 'neckline': neckline}
                    ))
                    break
        
        return patterns
    
    def _find_head_shoulders(self, data: pd.DataFrame) -> List[ChartPattern]:
        """Head & Shoulders / Inverse H&S"""
        patterns = []
        swing_highs = self._find_swing_points(data, 'high')
        swing_lows = self._find_swing_points(data, 'low')
        
        # Head & Shoulders (3 peaks)
        if len(swing_highs) >= 3:
            for i in range(len(swing_highs) - 2):
                idx1, left_shoulder = swing_highs[i]
                idx2, head = swing_highs[i+1]
                idx3, right_shoulder = swing_highs[i+2]
                
                if (head > left_shoulder * 1.05 and head > right_shoulder * 1.05 and
                    abs(left_shoulder - right_shoulder) / left_shoulder < self.tolerance):
                    
                    neckline = data.iloc[idx1:idx3+1]['low'].min()
                    target = neckline - (head - neckline)
                    
                    patterns.append(ChartPattern(
                        pattern_type=PatternType.HEAD_SHOULDERS,
                        start_index=idx1,
                        end_index=idx3,
                        confidence=0.85,
                        direction='bearish',
                        target_price=target,
                        metadata={'left_shoulder': left_shoulder, 'head': head, 'right_shoulder': right_shoulder}
                    ))
        
        return patterns
    
    def _find_triangles(self, data: pd.DataFrame) -> List[ChartPattern]:
        """Ascending/Descending/Symmetrical Triangles"""
        # Implementazione semplificata - da espandere
        return []
    
    def _find_flags(self, data: pd.DataFrame) -> List[ChartPattern]:
        """Bull/Bear Flags"""
        # Implementazione semplificata - da espandere
        return []
    
    def _find_swing_points(self, data: pd.DataFrame, column: str, window: int = 5) -> List[Tuple[int, float]]:
        """Trova swing highs/lows"""
        swings = []
        
        for i in range(window, len(data) - window):
            value = data.iloc[i][column]
            
            if column == 'high':
                # Swing high
                if all(value >= data.iloc[j]['high'] for j in range(i-window, i+window+1) if j != i):
                    swings.append((i, value))
            else:  # low
                # Swing low
                if all(value <= data.iloc[j]['low'] for j in range(i-window, i+window+1) if j != i):
                    swings.append((i, value))
        
        return swings
