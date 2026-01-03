"""
Feature Engineering Module - Estrai feature significative da OHLC + Context
Per training modelli ML che predicono successo/fallimento dei trades
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class TrendRegime(Enum):
    """Regime di mercato"""
    STRONG_UPTREND = 1.0
    UPTREND = 0.75
    NEUTRAL_UP = 0.5
    SIDEWAYS = 0.25
    NEUTRAL_DOWN = -0.5
    DOWNTREND = -0.75
    STRONG_DOWNTREND = -1.0


class FeatureEngineer:
    """
    Estrae feature significative per training modelli.
    Combina:
    - Technical Indicators (RSI, MACD, BB, ATR)
    - Price Action (support/resistance, breakouts)
    - Momentum & Volatility
    - Market Structure (HH/HL, LL/LH)
    - Macro Context (trend strength, regime)
    """
    
    def __init__(self, lookback: int = 50):
        """
        Args:
            lookback: Numero di candele storiche per calcoli
        """
        self.lookback = lookback
        logger.info(f"FeatureEngineer initialized: lookback={lookback}")
    
    def extract_all_features(
        self,
        data: pd.DataFrame,
        symbol: str,
        entry_price: float,
        entry_side: str = 'BUY'
    ) -> Dict[str, float]:
        """
        Estrae TUTTE le feature per un potenziale trade.
        
        Args:
            data: DataFrame OHLCV con indicatori calcolati
            symbol: Simbolo (per logging)
            entry_price: Prezzo di entry
            entry_side: 'BUY' o 'SELL'
        
        Returns:
            Dict con 100+ feature
        """
        features = {}
        
        # Assicura almeno lookback candles
        if len(data) < self.lookback:
            logger.warning(f"Not enough data for {symbol}: {len(data)} < {self.lookback}")
            return {}
        
        # Prendi ultimi lookback candles
        data = data.tail(self.lookback).copy()
        
        # 1. PRICE ACTION FEATURES
        features.update(self._extract_price_action(data, entry_price, entry_side))
        
        # 2. TECHNICAL INDICATORS
        features.update(self._extract_technical_indicators(data))
        
        # 3. MOMENTUM FEATURES
        features.update(self._extract_momentum(data))
        
        # 4. VOLATILITY FEATURES
        features.update(self._extract_volatility(data))
        
        # 5. MARKET STRUCTURE
        features.update(self._extract_market_structure(data, entry_side))
        
        # 6. TREND & REGIME
        features.update(self._extract_trend_regime(data))
        
        # 7. PATTERN FEATURES
        features.update(self._extract_patterns(data, entry_side))
        
        # 8. EDGE FEATURES (confluenza di segnali)
        features.update(self._extract_edge(features))
        
        return features
    
    def _extract_price_action(
        self,
        data: pd.DataFrame,
        entry_price: float,
        entry_side: str
    ) -> Dict[str, float]:
        """Price action: distanza da support/resistance, breakout strength"""
        features = {}
        
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # Ultimi 20 candles per calcoli SR
        lookback_sr = min(20, len(data))
        
        # Support & Resistance
        recent_high = high[-lookback_sr:].max()
        recent_low = low[-lookback_sr:].min()
        current_price = close[-1]
        
        # Distance da SR (percentuale)
        dist_to_resistance = ((recent_high - current_price) / current_price) * 100 if current_price > 0 else 0
        dist_to_support = ((current_price - recent_low) / current_price) * 100 if current_price > 0 else 0
        
        features['price_dist_to_resistance'] = dist_to_resistance
        features['price_dist_to_support'] = dist_to_support
        
        # Distance da entry
        entry_distance_pips = abs(current_price - entry_price) * 10000 if entry_side == 'BUY' else abs(entry_price - current_price) * 10000
        features['entry_distance_pips'] = entry_distance_pips
        
        # Candle body size (% of ATR)
        atr = data['atr'].iloc[-1] if 'atr' in data.columns else (high[-1] - low[-1])
        candle_body = abs(close[-1] - close[-2]) if len(close) > 1 else 0
        features['candle_body_atr_ratio'] = (candle_body / atr * 100) if atr > 0 else 0
        
        # Wicks strength
        upper_wick = high[-1] - max(close[-1], close[-2]) if len(close) > 1 else 0
        lower_wick = min(close[-1], close[-2]) - low[-1] if len(close) > 1 else 0
        features['upper_wick_ratio'] = (upper_wick / atr * 100) if atr > 0 else 0
        features['lower_wick_ratio'] = (lower_wick / atr * 100) if atr > 0 else 0
        
        # Recent breakout strength
        if len(data) >= 2:
            breakout_high = high[-20:].max() if len(high) >= 20 else high.max()
            breakout_low = low[-20:].min() if len(low) >= 20 else low.min()
            
            if entry_side == 'BUY':
                breakout_strength = ((current_price - breakout_low) / (breakout_high - breakout_low) * 100) if breakout_high > breakout_low else 50
            else:
                breakout_strength = ((breakout_high - current_price) / (breakout_high - breakout_low) * 100) if breakout_high > breakout_low else 50
            
            features['breakout_strength'] = breakout_strength
        
        return features
    
    def _extract_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """RSI, MACD, Bollinger Bands, Stochastic"""
        features = {}
        
        # RSI
        if 'rsi' in data.columns:
            rsi = data['rsi'].iloc[-1]
            features['rsi_current'] = rsi
            features['rsi_overbought'] = 1.0 if rsi > 70 else 0.0
            features['rsi_oversold'] = 1.0 if rsi < 30 else 0.0
            features['rsi_neutral'] = 1.0 if 40 < rsi < 60 else 0.0
        
        # MACD
        if 'macd' in data.columns and 'macd_signal' in data.columns:
            macd = data['macd'].iloc[-1]
            macd_signal = data['macd_signal'].iloc[-1]
            macd_histogram = macd - macd_signal
            
            features['macd_value'] = macd
            features['macd_signal_value'] = macd_signal
            features['macd_histogram'] = macd_histogram
            features['macd_positive'] = 1.0 if macd_histogram > 0 else 0.0
            
            if len(data) > 1:
                prev_histogram = data['macd'].iloc[-2] - data['macd_signal'].iloc[-2]
                features['macd_histogram_increasing'] = 1.0 if macd_histogram > prev_histogram else 0.0
        
        # Bollinger Bands
        if 'bb_upper' in data.columns and 'bb_middle' in data.columns and 'bb_lower' in data.columns:
            close = data['close'].iloc[-1]
            bb_upper = data['bb_upper'].iloc[-1]
            bb_middle = data['bb_middle'].iloc[-1]
            bb_lower = data['bb_lower'].iloc[-1]
            bb_width = bb_upper - bb_lower
            
            features['bb_upper'] = bb_upper
            features['bb_middle'] = bb_middle
            features['bb_lower'] = bb_lower
            features['bb_width'] = bb_width
            
            # Position relative to BB
            if bb_width > 0:
                bb_position = (close - bb_lower) / bb_width
                features['bb_position'] = np.clip(bb_position, 0, 1)
            
            features['bb_touch_upper'] = 1.0 if close >= bb_upper else 0.0
            features['bb_touch_lower'] = 1.0 if close <= bb_lower else 0.0
        
        # ATR
        if 'atr' in data.columns:
            atr = data['atr'].iloc[-1]
            features['atr_current'] = atr
            features['atr_ma_atr'] = atr / data['atr'].rolling(20).mean().iloc[-1] if len(data) > 20 else 1.0
        
        return features
    
    def _extract_momentum(self, data: pd.DataFrame) -> Dict[str, float]:
        """Rate of change, momentum, acceleration"""
        features = {}
        
        close = data['close'].values
        
        # ROC (Rate of Change)
        if len(close) >= 10:
            roc_5 = ((close[-1] - close[-5]) / close[-5] * 100) if close[-5] != 0 else 0
            roc_10 = ((close[-1] - close[-10]) / close[-10] * 100) if close[-10] != 0 else 0
            
            features['roc_5'] = roc_5
            features['roc_10'] = roc_10
            features['momentum_direction'] = 1.0 if roc_10 > 0 else -1.0
        
        # Acceleration (ROC of ROC)
        if len(close) >= 15:
            prev_roc_10 = ((close[-11] - close[-16]) / close[-16] * 100) if close[-16] != 0 else 0
            roc_10_now = ((close[-1] - close[-10]) / close[-10] * 100) if close[-10] != 0 else 0
            
            acceleration = roc_10_now - prev_roc_10
            features['momentum_acceleration'] = acceleration
        
        # Consecutive closes direction
        if len(close) >= 5:
            consecutive_up = sum(1 for i in range(-4, 0) if close[i] > close[i-1])
            consecutive_down = 5 - consecutive_up
            features['consecutive_up_closes'] = consecutive_up
            features['consecutive_down_closes'] = consecutive_down
        
        return features
    
    def _extract_volatility(self, data: pd.DataFrame) -> Dict[str, float]:
        """Volatility metrics"""
        features = {}
        
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        # Standard deviation
        if len(close) >= 20:
            std_20 = np.std(close[-20:])
            std_50 = np.std(close[-50:]) if len(close) >= 50 else np.std(close)
            
            features['volatility_std_20'] = std_20
            features['volatility_std_50'] = std_50
            features['volatility_ratio'] = (std_20 / std_50) if std_50 > 0 else 1.0
        
        # Range expansion
        if len(data) >= 10:
            range_current = high[-1] - low[-1]
            range_avg_10 = np.mean([h - l for h, l in zip(high[-10:], low[-10:])])
            
            features['range_expansion'] = (range_current / range_avg_10) if range_avg_10 > 0 else 1.0
        
        # True range (per ATR-like metric)
        if len(data) >= 2:
            tr_current = max(
                high[-1] - low[-1],
                abs(high[-1] - close[-2]),
                abs(low[-1] - close[-2])
            )
            features['true_range_current'] = tr_current
        
        return features
    
    def _extract_market_structure(self, data: pd.DataFrame, entry_side: str) -> Dict[str, float]:
        """Higher High/Low, support/resistance patterns"""
        features = {}
        
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        if len(data) < 5:
            return features
        
        # Identify swing highs/lows (10 period lookback)
        period = 5
        
        swing_highs = []
        swing_lows = []
        
        for i in range(period, len(high) - period):
            is_swing_high = high[i] == max(high[i-period:i+period+1])
            is_swing_low = low[i] == min(low[i-period:i+period+1])
            
            if is_swing_high:
                swing_highs.append((i, high[i]))
            if is_swing_low:
                swing_lows.append((i, low[i]))
        
        if len(swing_highs) >= 2:
            # Higher highs or lower highs?
            recent_hh = swing_highs[-1][1]
            prev_hh = swing_highs[-2][1] if len(swing_highs) >= 2 else recent_hh
            
            features['higher_highs'] = 1.0 if recent_hh > prev_hh else 0.0
            features['hh_strength'] = ((recent_hh - prev_hh) / prev_hh * 100) if prev_hh > 0 else 0
        
        if len(swing_lows) >= 2:
            # Higher lows or lower lows?
            recent_hl = swing_lows[-1][1]
            prev_hl = swing_lows[-2][1] if len(swing_lows) >= 2 else recent_hl
            
            features['higher_lows'] = 1.0 if recent_hl > prev_hl else 0.0
            features['hl_strength'] = ((recent_hl - prev_hl) / prev_hl * 100) if prev_hl > 0 else 0
        
        # Uptrend: HH and HL, Downtrend: LH and LL
        if entry_side == 'BUY':
            features['structure_alignment'] = (
                features.get('higher_highs', 0) + features.get('higher_lows', 0)
            ) / 2.0
        else:
            features['structure_alignment'] = (
                (1.0 - features.get('higher_highs', 1)) + (1.0 - features.get('higher_lows', 1))
            ) / 2.0
        
        return features
    
    def _extract_trend_regime(self, data: pd.DataFrame) -> Dict[str, float]:
        """Trend strength, regime classification"""
        features = {}
        
        close = data['close'].values
        
        # Simple MA-based trend
        if len(close) >= 50:
            ma_9 = np.mean(close[-9:])
            ma_21 = np.mean(close[-21:])
            ma_50 = np.mean(close[-50:])
            
            features['ma_9'] = ma_9
            features['ma_21'] = ma_21
            features['ma_50'] = ma_50
            
            # Trend alignment
            if ma_9 > ma_21 > ma_50:
                features['trend_type'] = 1.0  # Strong uptrend
                features['trend_strength'] = 0.95
            elif ma_9 > ma_21:
                features['trend_type'] = 0.5  # Uptrend
                features['trend_strength'] = 0.65
            elif ma_9 < ma_21 < ma_50:
                features['trend_type'] = -1.0  # Strong downtrend
                features['trend_strength'] = 0.95
            elif ma_9 < ma_21:
                features['trend_type'] = -0.5  # Downtrend
                features['trend_strength'] = 0.65
            else:
                features['trend_type'] = 0.0  # Sideways
                features['trend_strength'] = 0.3
            
            # Linear regression slope for trend
            x = np.arange(len(close[-20:]))
            y = close[-20:]
            slope = np.polyfit(x, y, 1)[0]
            features['trend_slope'] = slope
        
        return features
    
    def _extract_patterns(self, data: pd.DataFrame, entry_side: str) -> Dict[str, float]:
        """Pattern recognition: pin bars, engulfing, etc"""
        features = {}
        
        open_prices = data['open'].values
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        if len(data) < 2:
            return features
        
        # Pin Bar (long wick, small body)
        atr = high[-1] - low[-1] if len(data) == 1 else np.mean([h - l for h, l in zip(high[-5:], low[-5:])])
        
        candle_body = abs(close[-1] - open_prices[-1])
        upper_wick = high[-1] - max(close[-1], open_prices[-1])
        lower_wick = min(close[-1], open_prices[-1]) - low[-1]
        
        is_pin_bar = (candle_body < atr * 0.3) and (max(upper_wick, lower_wick) > atr * 0.5)
        features['is_pin_bar'] = 1.0 if is_pin_bar else 0.0
        
        # Engulfing
        if len(data) >= 2:
            prev_body = abs(close[-2] - open_prices[-2])
            curr_body = abs(close[-1] - open_prices[-1])
            
            is_bullish_engulfing = (
                (close[-2] < open_prices[-2]) and  # prev bearish
                (close[-1] > open_prices[-1]) and  # curr bullish
                (open_prices[-1] < close[-2]) and
                (close[-1] > open_prices[-2])
            )
            
            is_bearish_engulfing = (
                (close[-2] > open_prices[-2]) and  # prev bullish
                (close[-1] < open_prices[-1]) and  # curr bearish
                (open_prices[-1] > close[-2]) and
                (close[-1] < open_prices[-2])
            )
            
            features['bullish_engulfing'] = 1.0 if is_bullish_engulfing else 0.0
            features['bearish_engulfing'] = 1.0 if is_bearish_engulfing else 0.0
        
        # Hammer/Shooting Star
        body_ratio = candle_body / atr if atr > 0 else 0
        wick_ratio = max(upper_wick, lower_wick) / atr if atr > 0 else 0
        
        is_hammer = (lower_wick > upper_wick * 2) and (body_ratio < 0.3)
        is_shooting_star = (upper_wick > lower_wick * 2) and (body_ratio < 0.3)
        
        features['is_hammer'] = 1.0 if is_hammer else 0.0
        features['is_shooting_star'] = 1.0 if is_shooting_star else 0.0
        
        return features
    
    def _extract_edge(self, features: Dict[str, float]) -> Dict[str, float]:
        """Edge features: confluenza di segnali"""
        edge_features = {}
        
        # Count quanti segnali bullish/bearish abbiamo
        bullish_signals = 0
        bearish_signals = 0
        
        if features.get('rsi_oversold', 0) == 1.0:
            bullish_signals += 1
        if features.get('rsi_overbought', 0) == 1.0:
            bearish_signals += 1
        
        if features.get('macd_positive', 0) == 1.0:
            bullish_signals += 1
        if features.get('macd_positive', 0) == 0.0:
            bearish_signals += 1
        
        if features.get('higher_highs', 0) == 1.0:
            bullish_signals += 1
        if features.get('higher_lows', 0) == 1.0:
            bullish_signals += 0.5
        
        if features.get('trend_type', 0) > 0:
            bullish_signals += abs(features.get('trend_type', 0))
        else:
            bearish_signals += abs(features.get('trend_type', 0))
        
        if features.get('is_pin_bar', 0) == 1.0:
            bullish_signals += 0.5
        
        if features.get('bullish_engulfing', 0) == 1.0:
            bullish_signals += 1.0
        
        edge_features['bullish_signal_count'] = bullish_signals
        edge_features['bearish_signal_count'] = bearish_signals
        edge_features['signal_confluence'] = (bullish_signals - bearish_signals) / max(bullish_signals + bearish_signals, 1)
        
        # Entry quality score (0-1)
        total_signals = bullish_signals + bearish_signals
        if total_signals > 0:
            edge_features['entry_quality'] = max(bullish_signals, bearish_signals) / total_signals
        else:
            edge_features['entry_quality'] = 0.5
        
        return edge_features
    
    def get_feature_names(self) -> List[str]:
        """Restituisce nomi di tutte le feature estratte"""
        return [
            # Price action
            'price_dist_to_resistance', 'price_dist_to_support',
            'entry_distance_pips', 'candle_body_atr_ratio',
            'upper_wick_ratio', 'lower_wick_ratio', 'breakout_strength',
            
            # Technical
            'rsi_current', 'rsi_overbought', 'rsi_oversold', 'rsi_neutral',
            'macd_value', 'macd_signal_value', 'macd_histogram',
            'macd_positive', 'macd_histogram_increasing',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'bb_touch_upper', 'bb_touch_lower',
            'atr_current', 'atr_ma_atr',
            
            # Momentum
            'roc_5', 'roc_10', 'momentum_direction',
            'momentum_acceleration', 'consecutive_up_closes', 'consecutive_down_closes',
            
            # Volatility
            'volatility_std_20', 'volatility_std_50', 'volatility_ratio',
            'range_expansion', 'true_range_current',
            
            # Structure
            'higher_highs', 'hh_strength',
            'higher_lows', 'hl_strength',
            'structure_alignment',
            
            # Trend
            'ma_9', 'ma_21', 'ma_50',
            'trend_type', 'trend_strength', 'trend_slope',
            
            # Patterns
            'is_pin_bar', 'bullish_engulfing', 'bearish_engulfing',
            'is_hammer', 'is_shooting_star',
            
            # Edge
            'bullish_signal_count', 'bearish_signal_count',
            'signal_confluence', 'entry_quality'
        ]
