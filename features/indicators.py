"""
Technical Indicators Module
Calculates RSI, Stochastic, Moving Averages, PMax, ATR, and other indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Comprehensive technical indicators calculator
    All methods return calculated values ready for strategy use
    """
    
    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index
        
        Args:
            close: Close prices
            period: RSI period (default 14)
        
        Returns:
            RSI values (0-100)
        """
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_stochastic(
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator (%K and %D)
        
        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    @staticmethod
    def calculate_sma(close: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return close.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(close: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return close.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_moving_averages(
        close: pd.Series, 
        fast: int = 10, 
        medium: int = 50, 
        slow: int = 200,
        ma_type: str = 'EMA'
    ) -> Dict[str, pd.Series]:
        """
        Calculate multiple moving averages
        
        Returns:
            Dict with 'fast', 'medium', 'slow' MAs
        """
        if ma_type.upper() == 'EMA':
            return {
                'fast': TechnicalIndicators.calculate_ema(close, fast),
                'medium': TechnicalIndicators.calculate_ema(close, medium),
                'slow': TechnicalIndicators.calculate_ema(close, slow)
            }
        else:
            return {
                'fast': TechnicalIndicators.calculate_sma(close, fast),
                'medium': TechnicalIndicators.calculate_sma(close, medium),
                'slow': TechnicalIndicators.calculate_sma(close, slow)
            }
    
    @staticmethod
    def calculate_atr(
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range
        """
        high_low = high - low
        high_close = abs(high - close.shift())
        low_close = abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_pmax(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 10,
        multiplier: float = 3.0,
        atr_length: int = 10
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate PMax indicator (trend-following indicator)
        
        Returns:
            Tuple of (pmax_values, trend_direction)
            trend_direction: 1 for uptrend, -1 for downtrend
        """
        # Calculate ATR
        atr = TechnicalIndicators.calculate_atr(high, low, close, atr_length)
        
        # Calculate moving average
        ma = close.rolling(window=period).mean()
        
        # Calculate upper and lower bands
        upper_band = ma + (multiplier * atr)
        lower_band = ma - (multiplier * atr)
        
        # Initialize PMax and trend
        pmax = pd.Series(index=close.index, dtype=float)
        trend = pd.Series(index=close.index, dtype=float)
        
        for i in range(1, len(close)):
            if pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]):
                pmax.iloc[i] = np.nan
                trend.iloc[i] = np.nan
                continue
            
            # Determine trend
            if close.iloc[i] > upper_band.iloc[i-1]:
                trend.iloc[i] = 1
                pmax.iloc[i] = lower_band.iloc[i]
            elif close.iloc[i] < lower_band.iloc[i-1]:
                trend.iloc[i] = -1
                pmax.iloc[i] = upper_band.iloc[i]
            else:
                trend.iloc[i] = trend.iloc[i-1] if not pd.isna(trend.iloc[i-1]) else 0
                if trend.iloc[i] == 1:
                    pmax.iloc[i] = lower_band.iloc[i]
                else:
                    pmax.iloc[i] = upper_band.iloc[i]
        
        return pmax, trend
    
    @staticmethod
    def calculate_bollinger_bands(
        close: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return upper, middle, lower
    
    @staticmethod
    def calculate_macd(
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD
        
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_momentum(close: pd.Series, period: int = 10) -> pd.Series:
        """Calculate price momentum"""
        return close.diff(period)
    
    @staticmethod
    def calculate_rate_of_change(close: pd.Series, period: int = 10) -> pd.Series:
        """Calculate Rate of Change (ROC)"""
        return ((close - close.shift(period)) / close.shift(period)) * 100
    
    @staticmethod
    def calculate_adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average Directional Index (ADX)
        Measures trend strength
        """
        # Calculate +DM and -DM
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = pd.Series(0.0, index=close.index)
        minus_dm = pd.Series(0.0, index=close.index)
        
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
        
        # Calculate ATR
        atr = TechnicalIndicators.calculate_atr(high, low, close, period)
        
        # Calculate smoothed DM
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def identify_pivot_points(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate pivot points for support/resistance
        
        Returns:
            Dict with pivot, support (s1, s2, s3), resistance (r1, r2, r3)
        """
        pivot = (high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 3
        
        r1 = 2 * pivot - low.iloc[-1]
        s1 = 2 * pivot - high.iloc[-1]
        
        r2 = pivot + (high.iloc[-1] - low.iloc[-1])
        s2 = pivot - (high.iloc[-1] - low.iloc[-1])
        
        r3 = high.iloc[-1] + 2 * (pivot - low.iloc[-1])
        s3 = low.iloc[-1] - 2 * (high.iloc[-1] - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    
    @staticmethod
    def calculate_all_indicators(
        df: pd.DataFrame,
        config: dict = None
    ) -> pd.DataFrame:
        """
        Calculate all indicators for a DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            config: Optional config dict with indicator parameters
        
        Returns:
            DataFrame with all indicators added as columns
        """
        if df.empty or len(df) < 50:
            logger.warning("Insufficient data for indicator calculation")
            return df
        
        result = df.copy()
        
        # Get config parameters or use defaults
        if config is None:
            config = {}
        
        ind_config = config.get('indicators', {})
        
        # RSI
        rsi_period = ind_config.get('rsi', {}).get('period', 14)
        result['rsi'] = TechnicalIndicators.calculate_rsi(df['close'], rsi_period)
        
        # Stochastic
        stoch_config = ind_config.get('stochastic', {})
        stoch_k, stoch_d = TechnicalIndicators.calculate_stochastic(
            df['high'], df['low'], df['close'],
            stoch_config.get('k_period', 14),
            stoch_config.get('d_period', 3)
        )
        result['stoch_k'] = stoch_k
        result['stoch_d'] = stoch_d
        
        # Moving Averages
        ma_config = ind_config.get('moving_averages', {})
        mas = TechnicalIndicators.calculate_moving_averages(
            df['close'],
            ma_config.get('fast_period', 10),
            ma_config.get('medium_period', 50),
            ma_config.get('slow_period', 200),
            ma_config.get('type', 'EMA')
        )
        result['ma_fast'] = mas['fast']
        result['ma_medium'] = mas['medium']
        result['ma_slow'] = mas['slow']
        
        # ATR
        atr_period = ind_config.get('atr', {}).get('period', 14)
        result['atr'] = TechnicalIndicators.calculate_atr(
            df['high'], df['low'], df['close'], atr_period
        )
        
        # PMax
        pmax_config = ind_config.get('pmax', {})
        pmax_values, pmax_trend = TechnicalIndicators.calculate_pmax(
            df['high'], df['low'], df['close'],
            pmax_config.get('period', 10),
            pmax_config.get('multiplier', 3.0),
            pmax_config.get('atr_length', 10)
        )
        result['pmax'] = pmax_values
        result['pmax_trend'] = pmax_trend
        
        # MACD
        macd, signal, hist = TechnicalIndicators.calculate_macd(df['close'])
        result['macd'] = macd
        result['macd_signal'] = signal
        result['macd_hist'] = hist
        
        # ADX
        result['adx'] = TechnicalIndicators.calculate_adx(
            df['high'], df['low'], df['close']
        )
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(df['close'])
        result['bb_upper'] = bb_upper
        result['bb_middle'] = bb_middle
        result['bb_lower'] = bb_lower
        
        # Additional metrics
        result['momentum'] = TechnicalIndicators.calculate_momentum(df['close'])
        result['roc'] = TechnicalIndicators.calculate_rate_of_change(df['close'])
        
        logger.debug(f"Calculated all indicators for {len(df)} bars")
        
        return result
