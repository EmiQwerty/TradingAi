"""
Trend Following Strategy - Cavalca i trend seguendo pullback
"""

import pandas as pd
import numpy as np
from typing import List
from strategies.strategy_base import StrategyBase, TradingSignal, SignalType
from strategies.pattern_recognition import PatternRecognizer, PatternType


class TrendFollowingStrategy(StrategyBase):
    """
    Strategia Trend Following professionale:
    
    LONG Entry (Uptrend):
    - Price > EMA50 > EMA200 (trend confermato)
    - Pullback verso EMA20 (non rompe)
    - Bullish candle di rimbalzo
    - MACD > 0 (momentum positivo)
    - ADX > 25 (trend forte)
    - Supporto pattern: Bullish engulfing, Hammer sul pullback
    
    SHORT Entry (Downtrend):
    - Price < EMA50 < EMA200
    - Pullback verso EMA20 (retest dall'alto)
    - Bearish rejection candle
    - MACD < 0
    - ADX > 25
    
    Stop Loss: Sotto/sopra EMA50 + ATR buffer
    Take Profit: Trailing stop o fixed 1:3 R/R
    """
    
    def __init__(self,
                 ema_fast: int = 20,
                 ema_medium: int = 50,
                 ema_slow: int = 200,
                 adx_period: int = 14,
                 adx_threshold: float = 25.0,
                 atr_period: int = 14,
                 atr_multiplier: float = 2.0):
        super().__init__(name="TrendFollowing")
        
        self.ema_fast = ema_fast
        self.ema_medium = ema_medium
        self.ema_slow = ema_slow
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.pattern_recognizer = PatternRecognizer()
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcola indicatori per trend following"""
        df = data.copy()
        
        # EMAs
        df['ema20'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=self.ema_medium, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ADX (Average Directional Index)
        df['atr'] = self._calculate_atr(df)
        df['adx'] = self._calculate_adx(df)
        
        # Distance from EMAs (per riconoscere pullback)
        df['dist_from_ema20'] = ((df['close'] - df['ema20']) / df['ema20']) * 100
        df['dist_from_ema50'] = ((df['close'] - df['ema50']) / df['ema50']) * 100
        
        # Candle body strength
        df['candle_body'] = df['close'] - df['open']
        df['candle_range'] = df['high'] - df['low']
        df['body_ratio'] = np.abs(df['candle_body']) / df['candle_range']
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """ATR calculation"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(self.atr_period).mean()
    
    def _calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """ADX calculation - misura forza del trend"""
        # +DM e -DM
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Smoothed
        atr = self._calculate_atr(df)
        plus_di = 100 * (plus_dm.rolling(self.adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(self.adx_period).mean() / atr)
        
        # DX e ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(self.adx_period).mean()
        
        return adx
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Genera segnali trend following su pullback"""
        signals = []
        
        # Pattern recognition
        patterns = self.pattern_recognizer.scan_all_patterns(data)
        pattern_indices = {p.end_index: p for p in patterns}
        
        for i in range(self.ema_slow + 10, len(data)):
            current = data.iloc[i]
            prev = data.iloc[i-1]
            prev2 = data.iloc[i-2]
            
            if pd.isna(current['adx']) or pd.isna(current['ema200']):
                continue
            
            # ADX check (trend forte richiesto)
            if current['adx'] < self.adx_threshold:
                continue
            
            # --- BULLISH TREND PULLBACK ---
            # Trend: price > ema50 > ema200
            # Pullback: price touched ema20, now bouncing
            if (current['close'] > current['ema50'] > current['ema200'] and  # Trend UP
                current['ema50'] > prev['ema50'] and  # EMA50 rising
                prev['close'] <= prev['ema20'] and  # Pullback touched EMA20
                current['close'] > current['ema20'] and  # Now above EMA20 again
                current['candle_body'] > 0 and  # Bullish candle
                current['body_ratio'] > 0.5 and  # Strong body
                current['macd'] > 0):  # Momentum positive
                
                entry_price = current['close']
                
                # Stop loss sotto EMA50 + ATR buffer
                stop_loss = current['ema50'] - (self.atr_multiplier * current['atr'])
                
                # Take profit 1:3 R/R
                risk = entry_price - stop_loss
                take_profit = entry_price + (3 * risk)
                
                # Confidence boost con pattern
                confidence = 0.75
                pattern_boost = ""
                if i in pattern_indices:
                    pattern = pattern_indices[i]
                    if pattern.direction == 'bullish' and pattern.pattern_type in [
                        PatternType.ENGULFING_BULLISH, PatternType.HAMMER, PatternType.MORNING_STAR
                    ]:
                        confidence = 0.88
                        pattern_boost = f" + {pattern.pattern_type.value}"
                
                signal = TradingSignal(
                    timestamp=current.name if hasattr(current, 'name') else i,
                    signal_type=SignalType.BUY,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    reason=f"Uptrend pullback @ EMA20 + bounce (ADX={current['adx']:.1f}){pattern_boost}",
                    metadata={
                        'adx': current['adx'],
                        'macd': current['macd'],
                        'dist_from_ema20': current['dist_from_ema20'],
                        'dist_from_ema50': current['dist_from_ema50'],
                        'body_ratio': current['body_ratio'],
                        'atr': current['atr'],
                        'pattern': pattern_indices[i].pattern_type.value if i in pattern_indices else None
                    }
                )
                signals.append(signal)
            
            # --- BEARISH TREND PULLBACK ---
            elif (current['close'] < current['ema50'] < current['ema200'] and  # Trend DOWN
                  current['ema50'] < prev['ema50'] and  # EMA50 falling
                  prev['close'] >= prev['ema20'] and  # Pullback reached EMA20
                  current['close'] < current['ema20'] and  # Now rejected below
                  current['candle_body'] < 0 and  # Bearish candle
                  current['body_ratio'] > 0.5 and
                  current['macd'] < 0):
                
                entry_price = current['close']
                stop_loss = current['ema50'] + (self.atr_multiplier * current['atr'])
                
                risk = stop_loss - entry_price
                take_profit = entry_price - (3 * risk)
                
                confidence = 0.75
                pattern_boost = ""
                if i in pattern_indices:
                    pattern = pattern_indices[i]
                    if pattern.direction == 'bearish' and pattern.pattern_type in [
                        PatternType.ENGULFING_BEARISH, PatternType.SHOOTING_STAR, PatternType.EVENING_STAR
                    ]:
                        confidence = 0.88
                        pattern_boost = f" + {pattern.pattern_type.value}"
                
                signal = TradingSignal(
                    timestamp=current.name if hasattr(current, 'name') else i,
                    signal_type=SignalType.SELL,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    reason=f"Downtrend pullback @ EMA20 + rejection (ADX={current['adx']:.1f}){pattern_boost}",
                    metadata={
                        'adx': current['adx'],
                        'macd': current['macd'],
                        'dist_from_ema20': current['dist_from_ema20'],
                        'dist_from_ema50': current['dist_from_ema50'],
                        'body_ratio': current['body_ratio'],
                        'atr': current['atr'],
                        'pattern': pattern_indices[i].pattern_type.value if i in pattern_indices else None
                    }
                )
                signals.append(signal)
        
        return signals
    
    def calculate_stop_loss(self, entry_price: float, signal_type: SignalType,
                           atr: float, data: pd.DataFrame, index: int) -> float:
        """Stop loss sotto/sopra EMA50"""
        ema50 = data.iloc[index]['ema50']
        
        if signal_type == SignalType.BUY:
            return ema50 - (self.atr_multiplier * atr)
        else:
            return ema50 + (self.atr_multiplier * atr)
    
    def calculate_take_profit(self, entry_price: float, signal_type: SignalType,
                             stop_loss: float, risk_reward_ratio: float = 3.0) -> float:
        """TP con 1:3 R/R"""
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward_ratio
        
        if signal_type == SignalType.BUY:
            return entry_price + reward
        else:
            return entry_price - reward
