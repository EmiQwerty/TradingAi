"""
RSI Mean Reversion Strategy - Strategia professionale basata su RSI + conferme
"""

import pandas as pd
import numpy as np
from typing import List
from strategies.strategy_base import StrategyBase, TradingSignal, SignalType


class RSIMeanReversionStrategy(StrategyBase):
    """
    Strategia Mean Reversion con RSI:
    
    LONG Entry:
    - RSI < 30 (oversold)
    - MACD histogram turning positive
    - Price above 200 EMA (trend filter)
    - Volume spike (conferma)
    
    SHORT Entry:
    - RSI > 70 (overbought)
    - MACD histogram turning negative
    - Price below 200 EMA
    - Volume spike
    
    Stop Loss: ATR-based (2x ATR)
    Take Profit: Risk/Reward 1:2
    """
    
    def __init__(self, 
                 rsi_period: int = 14,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70,
                 ema_period: int = 200,
                 atr_period: int = 14,
                 atr_multiplier: float = 2.0,
                 volume_threshold: float = 1.5):
        super().__init__(name="RSI_MeanReversion")
        
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.volume_threshold = volume_threshold
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcola RSI, MACD, EMA, ATR, Volume"""
        df = data.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # EMA 200
        df['ema200'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(self.atr_period).mean()
        
        # Volume moving average
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Genera segnali BUY/SELL basati su logica RSI mean reversion"""
        signals = []
        
        for i in range(self.ema_period + 20, len(data)):
            current = data.iloc[i]
            prev = data.iloc[i-1]
            
            # Skip se dati mancanti
            if pd.isna(current['rsi']) or pd.isna(current['atr']):
                continue
            
            # --- LONG SIGNAL ---
            if (current['rsi'] < self.rsi_oversold and
                current['macd_histogram'] > 0 and prev['macd_histogram'] <= 0 and
                current['close'] > current['ema200'] and
                current['volume_ratio'] > self.volume_threshold):
                
                entry_price = current['close']
                stop_loss = self.calculate_stop_loss(
                    entry_price, SignalType.BUY, current['atr'], data, i
                )
                take_profit = self.calculate_take_profit(
                    entry_price, SignalType.BUY, stop_loss, risk_reward_ratio=2.0
                )
                
                # Confidence basato su quanto RSI è oversold
                confidence = min(1.0, (self.rsi_oversold - current['rsi']) / 20 + 0.5)
                
                signal = TradingSignal(
                    timestamp=current.name if hasattr(current, 'name') else i,
                    signal_type=SignalType.BUY,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    reason=f"RSI oversold ({current['rsi']:.1f}) + MACD crossover + above EMA200 + volume spike",
                    metadata={
                        'rsi': current['rsi'],
                        'macd_histogram': current['macd_histogram'],
                        'ema200': current['ema200'],
                        'volume_ratio': current['volume_ratio'],
                        'atr': current['atr']
                    }
                )
                signals.append(signal)
            
            # --- SHORT SIGNAL ---
            elif (current['rsi'] > self.rsi_overbought and
                  current['macd_histogram'] < 0 and prev['macd_histogram'] >= 0 and
                  current['close'] < current['ema200'] and
                  current['volume_ratio'] > self.volume_threshold):
                
                entry_price = current['close']
                stop_loss = self.calculate_stop_loss(
                    entry_price, SignalType.SELL, current['atr'], data, i
                )
                take_profit = self.calculate_take_profit(
                    entry_price, SignalType.SELL, stop_loss, risk_reward_ratio=2.0
                )
                
                confidence = min(1.0, (current['rsi'] - self.rsi_overbought) / 20 + 0.5)
                
                signal = TradingSignal(
                    timestamp=current.name if hasattr(current, 'name') else i,
                    signal_type=SignalType.SELL,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    reason=f"RSI overbought ({current['rsi']:.1f}) + MACD crossover + below EMA200 + volume spike",
                    metadata={
                        'rsi': current['rsi'],
                        'macd_histogram': current['macd_histogram'],
                        'ema200': current['ema200'],
                        'volume_ratio': current['volume_ratio'],
                        'atr': current['atr']
                    }
                )
                signals.append(signal)
        
        return signals
    
    def calculate_stop_loss(self, entry_price: float, signal_type: SignalType,
                           atr: float, data: pd.DataFrame, index: int) -> float:
        """Stop Loss basato su ATR"""
        if signal_type == SignalType.BUY:
            return entry_price - (self.atr_multiplier * atr)
        else:  # SELL
            return entry_price + (self.atr_multiplier * atr)
    
    def calculate_take_profit(self, entry_price: float, signal_type: SignalType,
                             stop_loss: float, risk_reward_ratio: float = 2.0) -> float:
        """Take Profit basato su Risk/Reward ratio"""
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward_ratio
        
        if signal_type == SignalType.BUY:
            return entry_price + reward
        else:  # SELL
            return entry_price - reward
