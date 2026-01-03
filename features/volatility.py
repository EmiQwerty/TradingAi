"""
Volatility Analysis Module
Measures and analyzes market volatility using ATR, standard deviation, and other metrics
"""

import numpy as np
import pandas as pd
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class VolatilityAnalyzer:
    """
    Analyzes market volatility for risk management and strategy adaptation
    """
    
    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        
        self.vol_config = config.get('indicators', {}).get('volatility', {})
        self.std_period = self.vol_config.get('std_period', 20)
        self.percentile_period = self.vol_config.get('percentile_period', 100)
        
        logger.info("VolatilityAnalyzer initialized")
    
    def calculate_historical_volatility(
        self, 
        close: pd.Series, 
        period: int = 20
    ) -> pd.Series:
        """
        Calculate historical volatility (annualized standard deviation of returns)
        
        Args:
            close: Close prices
            period: Lookback period
        
        Returns:
            Historical volatility as percentage
        """
        returns = close.pct_change()
        volatility = returns.rolling(window=period).std() * np.sqrt(252) * 100
        
        return volatility
    
    def calculate_volatility_percentile(
        self,
        current_vol: float,
        historical_vol: pd.Series,
        lookback: int = None
    ) -> float:
        """
        Calculate percentile rank of current volatility
        
        Returns:
            Percentile (0-100)
        """
        if lookback is None:
            lookback = self.percentile_period
        
        recent_vol = historical_vol.tail(lookback).dropna()
        
        if len(recent_vol) == 0:
            return 50.0
        
        percentile = (recent_vol < current_vol).sum() / len(recent_vol) * 100
        
        return percentile
    
    def get_volatility_state(
        self,
        df: pd.DataFrame,
        atr_period: int = 14
    ) -> Dict[str, any]:
        """
        Determine current volatility state
        
        Returns:
            Dict with volatility metrics: {
                'state': 'LOW' | 'MEDIUM' | 'HIGH' | 'EXTREME',
                'atr': float,
                'atr_percent': float (ATR as % of price),
                'std': float,
                'percentile': float,
                'expanding': bool (True if volatility increasing)
            }
        """
        if len(df) < max(atr_period, self.std_period, self.percentile_period):
            return {'state': 'UNKNOWN'}
        
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=atr_period).mean().iloc[-1]
        
        # ATR as percentage of price
        current_price = df['close'].iloc[-1]
        atr_percent = (atr / current_price) * 100
        
        # Calculate standard deviation
        returns = df['close'].pct_change()
        std = returns.rolling(window=self.std_period).std().iloc[-1] * 100
        
        # Historical volatility
        hist_vol = self.calculate_historical_volatility(df['close'], self.std_period)
        current_hist_vol = hist_vol.iloc[-1]
        
        # Volatility percentile
        percentile = self.calculate_volatility_percentile(
            current_hist_vol, 
            hist_vol, 
            self.percentile_period
        )
        
        # Check if volatility is expanding (compare last 5 bars to previous 15)
        recent_atr = true_range.tail(5).mean()
        previous_atr = true_range.tail(20).head(15).mean()
        expanding = recent_atr > previous_atr * 1.2
        
        # Determine state based on percentile
        if percentile < 30:
            state = 'LOW'
        elif percentile < 60:
            state = 'MEDIUM'
        elif percentile < 85:
            state = 'HIGH'
        else:
            state = 'EXTREME'
        
        return {
            'state': state,
            'atr': atr,
            'atr_percent': atr_percent,
            'std': std,
            'hist_vol': current_hist_vol,
            'percentile': percentile,
            'expanding': expanding,
            'contracting': not expanding and (recent_atr < previous_atr * 0.8)
        }
    
    def calculate_parkinson_volatility(
        self,
        df: pd.DataFrame,
        period: int = 20
    ) -> float:
        """
        Calculate Parkinson volatility estimator (uses high-low range)
        More efficient than close-to-close volatility
        """
        if len(df) < period:
            return 0.0
        
        hl_ratio = np.log(df['high'] / df['low'])
        parkinson = np.sqrt(
            (1 / (4 * len(df) * np.log(2))) * 
            (hl_ratio ** 2).tail(period).sum()
        ) * np.sqrt(252) * 100
        
        return parkinson
    
    def calculate_garman_klass_volatility(
        self,
        df: pd.DataFrame,
        period: int = 20
    ) -> float:
        """
        Calculate Garman-Klass volatility estimator
        Uses open, high, low, close for better estimate
        """
        if len(df) < period or 'open' not in df.columns:
            return 0.0
        
        recent = df.tail(period)
        
        hl = np.log(recent['high'] / recent['low']) ** 2
        co = np.log(recent['close'] / recent['open']) ** 2
        
        gk = np.sqrt(
            (0.5 * hl.mean() - (2 * np.log(2) - 1) * co.mean())
        ) * np.sqrt(252) * 100
        
        return gk
    
    def get_volatility_regime(
        self,
        df: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Identify current volatility regime with multiple estimators
        
        Returns:
            Comprehensive volatility regime analysis
        """
        vol_state = self.get_volatility_state(df)
        
        # Additional volatility estimators
        parkinson = self.calculate_parkinson_volatility(df)
        garman_klass = self.calculate_garman_klass_volatility(df)
        
        # Calculate volatility trend (5-day vs 20-day)
        if len(df) >= 20:
            short_vol = self.calculate_historical_volatility(df['close'], 5).iloc[-1]
            long_vol = self.calculate_historical_volatility(df['close'], 20).iloc[-1]
            vol_trend = (short_vol - long_vol) / long_vol if long_vol > 0 else 0
        else:
            vol_trend = 0
        
        # Determine trading implications
        if vol_state['state'] == 'LOW':
            implications = {
                'position_size_adjustment': 1.2,  # Increase size by 20%
                'stop_distance_adjustment': 0.8,  # Tighter stops
                'breakout_probability': 'high',
                'mean_reversion_favorable': True
            }
        elif vol_state['state'] == 'MEDIUM':
            implications = {
                'position_size_adjustment': 1.0,
                'stop_distance_adjustment': 1.0,
                'breakout_probability': 'medium',
                'mean_reversion_favorable': True
            }
        elif vol_state['state'] == 'HIGH':
            implications = {
                'position_size_adjustment': 0.7,  # Reduce size by 30%
                'stop_distance_adjustment': 1.3,  # Wider stops
                'breakout_probability': 'low',
                'mean_reversion_favorable': False
            }
        else:  # EXTREME
            implications = {
                'position_size_adjustment': 0.5,  # Reduce size by 50%
                'stop_distance_adjustment': 1.5,
                'breakout_probability': 'very_low',
                'mean_reversion_favorable': False
            }
        
        return {
            'state': vol_state['state'],
            'metrics': {
                'atr': vol_state['atr'],
                'atr_percent': vol_state['atr_percent'],
                'std': vol_state['std'],
                'hist_vol': vol_state['hist_vol'],
                'parkinson': parkinson,
                'garman_klass': garman_klass,
                'percentile': vol_state['percentile']
            },
            'trend': {
                'expanding': vol_state['expanding'],
                'contracting': vol_state['contracting'],
                'vol_trend': vol_trend,
                'direction': 'rising' if vol_trend > 0.1 else 'falling' if vol_trend < -0.1 else 'stable'
            },
            'implications': implications
        }
    
    def calculate_realized_volatility(
        self,
        df: pd.DataFrame,
        periods: List[int] = [5, 10, 20]
    ) -> Dict[int, float]:
        """
        Calculate realized volatility for multiple periods
        
        Returns:
            Dict mapping period to realized vol
        """
        realized = {}
        
        for period in periods:
            if len(df) >= period:
                vol = self.calculate_historical_volatility(df['close'], period).iloc[-1]
                realized[period] = vol
        
        return realized
    
    def get_volatility_breakout_signal(
        self,
        df: pd.DataFrame,
        threshold: float = 70
    ) -> Dict[str, any]:
        """
        Detect volatility breakout (expansion from low to high)
        
        Returns:
            Signal dict with breakout information
        """
        vol_state = self.get_volatility_state(df)
        
        # Check for volatility expansion
        if vol_state['expanding'] and vol_state['percentile'] > threshold:
            return {
                'signal': True,
                'type': 'volatility_breakout',
                'strength': min((vol_state['percentile'] - threshold) / (100 - threshold), 1.0),
                'direction': 'expanding',
                'recommendation': 'reduce_size_or_wait'
            }
        
        # Check for volatility contraction (potential breakout setup)
        if vol_state.get('contracting') and vol_state['percentile'] < 30:
            return {
                'signal': True,
                'type': 'volatility_compression',
                'strength': (30 - vol_state['percentile']) / 30,
                'direction': 'contracting',
                'recommendation': 'prepare_for_breakout'
            }
        
        return {'signal': False}
