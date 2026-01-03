"""
Breakout Strategy - Strategia basata su rotture di livelli chiave
"""

import pandas as pd
import numpy as np
from typing import List
from strategies.strategy_base import StrategyBase, TradingSignal, SignalType
from strategies.pattern_recognition import PatternRecognizer, PatternType


class BreakoutStrategy(StrategyBase):
    """
    Strategia Breakout professionale:
    
    LONG Entry:
    - Price breaks above resistance (swing high)
    - Volume > 1.5x average (conferma breakout)
    - Pullback to broken level (optional, higher confidence)
    - Bollinger Bands expansion
    - Supporto pattern: Triangle, Flag
    
    SHORT Entry:
    - Price breaks below support (swing low)
    - Volume spike
    - Retest from below
    - BB expansion
    
    Stop Loss: Below/above breakout level + buffer (ATR)
    Take Profit: Measured move (range proiettato)
    """
    
    def __init__(self,
                 lookback_period: int = 20,
                 volume_threshold: float = 1.5,
                 atr_period: int = 14,
                 atr_multiplier: float = 1.5,
                 bb_period: int = 20,
                 bb_std: float = 2.0):
        super().__init__(name="Breakout")
        
        self.lookback_period = lookback_period
        self.volume_threshold = volume_threshold
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.pattern_recognizer = PatternRecognizer()
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcola indicatori per breakout"""
        df = data.copy()
        
        # Swing highs/lows (resistance/support)
        df['resistance'] = df['high'].rolling(self.lookback_period).max()
        df['support'] = df['low'].rolling(self.lookback_period).min()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(self.atr_period).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(self.bb_period).mean()
        df['bb_std'] = df['close'].rolling(self.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (self.bb_std * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (self.bb_std * df['bb_std'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Volume
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Range expansion
        df['range'] = df['high'] - df['low']
        df['range_ma'] = df['range'].rolling(10).mean()
        df['range_expansion'] = df['range'] / df['range_ma']
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Genera segnali breakout"""
        signals = []
        
        # Riconosci pattern per filtraggio
        patterns = self.pattern_recognizer.scan_all_patterns(data)
        pattern_indices = {p.end_index: p for p in patterns}
        
        for i in range(self.lookback_period + 10, len(data)):
            current = data.iloc[i]
            prev = data.iloc[i-1]
            
            if pd.isna(current['atr']) or pd.isna(current['resistance']):
                continue
            
            # --- BULLISH BREAKOUT ---
            # Prezzo rompe resistenza + volume spike + BB expansion
            if (current['close'] > prev['resistance'] and
                prev['close'] <= prev['resistance'] and
                current['volume_ratio'] > self.volume_threshold and
                current['bb_width'] > data.iloc[i-10:i]['bb_width'].mean()):
                
                entry_price = current['close']
                
                # Stop loss sotto breakout level
                stop_loss = prev['resistance'] - (self.atr_multiplier * current['atr'])
                
                # Measured move: proietta range precedente
                prev_range = prev['resistance'] - data.iloc[i-self.lookback_period:i]['low'].min()
                take_profit = entry_price + prev_range
                
                # Confidence aumentata se c'è pattern supportivo
                confidence = 0.7
                pattern_boost = ""
                if i in pattern_indices:
                    pattern = pattern_indices[i]
                    if pattern.direction == 'bullish' and pattern.pattern_type in [
                        PatternType.TRIANGLE_ASCENDING, PatternType.FLAG_BULLISH, PatternType.ENGULFING_BULLISH
                    ]:
                        confidence = 0.85
                        pattern_boost = f" + {pattern.pattern_type.value}"
                
                signal = TradingSignal(
                    timestamp=current.name if hasattr(current, 'name') else i,
                    signal_type=SignalType.BUY,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    reason=f"Bullish breakout @ {prev['resistance']:.5f} + volume spike ({current['volume_ratio']:.1f}x){pattern_boost}",
                    metadata={
                        'breakout_level': prev['resistance'],
                        'volume_ratio': current['volume_ratio'],
                        'bb_width': current['bb_width'],
                        'range_expansion': current['range_expansion'],
                        'atr': current['atr'],
                        'pattern': pattern_indices[i].pattern_type.value if i in pattern_indices else None
                    }
                )
                signals.append(signal)
            
            # --- BEARISH BREAKOUT ---
            elif (current['close'] < prev['support'] and
                  prev['close'] >= prev['support'] and
                  current['volume_ratio'] > self.volume_threshold and
                  current['bb_width'] > data.iloc[i-10:i]['bb_width'].mean()):
                
                entry_price = current['close']
                stop_loss = prev['support'] + (self.atr_multiplier * current['atr'])
                
                prev_range = data.iloc[i-self.lookback_period:i]['high'].max() - prev['support']
                take_profit = entry_price - prev_range
                
                confidence = 0.7
                pattern_boost = ""
                if i in pattern_indices:
                    pattern = pattern_indices[i]
                    if pattern.direction == 'bearish' and pattern.pattern_type in [
                        PatternType.TRIANGLE_DESCENDING, PatternType.FLAG_BEARISH, PatternType.ENGULFING_BEARISH
                    ]:
                        confidence = 0.85
                        pattern_boost = f" + {pattern.pattern_type.value}"
                
                signal = TradingSignal(
                    timestamp=current.name if hasattr(current, 'name') else i,
                    signal_type=SignalType.SELL,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    reason=f"Bearish breakdown @ {prev['support']:.5f} + volume spike ({current['volume_ratio']:.1f}x){pattern_boost}",
                    metadata={
                        'breakout_level': prev['support'],
                        'volume_ratio': current['volume_ratio'],
                        'bb_width': current['bb_width'],
                        'range_expansion': current['range_expansion'],
                        'atr': current['atr'],
                        'pattern': pattern_indices[i].pattern_type.value if i in pattern_indices else None
                    }
                )
                signals.append(signal)
        
        return signals
    
    def calculate_stop_loss(self, entry_price: float, signal_type: SignalType,
                           atr: float, data: pd.DataFrame, index: int) -> float:
        """Stop loss sotto/sopra breakout level"""
        if signal_type == SignalType.BUY:
            breakout_level = data.iloc[index-1]['resistance']
            return breakout_level - (self.atr_multiplier * atr)
        else:
            breakout_level = data.iloc[index-1]['support']
            return breakout_level + (self.atr_multiplier * atr)
    
    def calculate_take_profit(self, entry_price: float, signal_type: SignalType,
                             stop_loss: float, risk_reward_ratio: float = 2.0) -> float:
        """TP basato su measured move (già calcolato in generate_signals)"""
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward_ratio
        
        if signal_type == SignalType.BUY:
            return entry_price + reward
        else:
            return entry_price - reward
