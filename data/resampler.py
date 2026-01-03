"""
Multi-Timeframe Resampler
Converts M1 (1-minute) data to higher timeframes (M5, M15, H1, H4)
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class TimeframeResampler:
    """
    Resamples tick/minute data to multiple timeframes
    Maintains synchronized candles across all timeframes
    """
    
    # Timeframe mapping to pandas resample rules
    TF_RULES = {
        'M1': '1T',   # 1 minute
        'M5': '5T',   # 5 minutes
        'M15': '15T', # 15 minutes
        'M30': '30T', # 30 minutes
        'H1': '1H',   # 1 hour
        'H4': '4H',   # 4 hours
        'D1': '1D'    # 1 day
    }
    
    def __init__(self, timeframes: List[str] = None):
        """
        Initialize resampler with target timeframes
        
        Args:
            timeframes: List of timeframes to resample to (e.g., ['M5', 'M15', 'H1'])
        """
        self.timeframes = timeframes or ['M5', 'M15', 'H1', 'H4']
        logger.info(f"TimeframeResampler initialized for: {self.timeframes}")
    
    def resample_ohlcv(self, df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """
        Resample OHLCV data to target timeframe
        
        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]
            target_tf: Target timeframe (M5, M15, H1, H4, etc.)
        
        Returns:
            Resampled DataFrame
        """
        if df.empty:
            return pd.DataFrame()
        
        if target_tf not in self.TF_RULES:
            logger.error(f"Unknown timeframe: {target_tf}")
            return pd.DataFrame()
        
        try:
            # Ensure timestamp is datetime and set as index
            df_copy = df.copy()
            if not isinstance(df_copy['timestamp'].iloc[0], pd.Timestamp):
                df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            
            df_copy.set_index('timestamp', inplace=True)
            
            # Resample OHLCV
            resampled = df_copy.resample(self.TF_RULES[target_tf], label='left').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            resampled.reset_index(inplace=True)
            
            logger.debug(f"Resampled {len(df)} bars to {len(resampled)} {target_tf} bars")
            return resampled
        
        except Exception as e:
            logger.error(f"Error resampling to {target_tf}: {e}")
            return pd.DataFrame()
    
    def resample_all_timeframes(
        self, 
        df: pd.DataFrame, 
        symbol: str = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Resample data to all configured timeframes
        
        Args:
            df: Source DataFrame (typically M1 data)
            symbol: Optional symbol name for logging
        
        Returns:
            Dict mapping timeframe to resampled DataFrame
        """
        result = {}
        
        for tf in self.timeframes:
            resampled = self.resample_ohlcv(df, tf)
            if not resampled.empty:
                result[tf] = resampled
                if symbol:
                    logger.debug(f"{symbol} {tf}: {len(resampled)} candles")
        
        return result
    
    def update_incremental(
        self, 
        existing_data: Dict[str, pd.DataFrame],
        new_m1_data: pd.DataFrame,
        lookback_bars: int = 200
    ) -> Dict[str, pd.DataFrame]:
        """
        Incrementally update higher timeframe data with new M1 bars
        More efficient than full resampling
        
        Args:
            existing_data: Dict of existing timeframe data
            new_m1_data: New M1 bars to incorporate
            lookback_bars: Number of bars to keep for each timeframe
        
        Returns:
            Updated dict of timeframe data
        """
        if new_m1_data.empty:
            return existing_data
        
        result = {}
        
        for tf in self.timeframes:
            # Combine existing data with new M1 data
            if tf in existing_data and not existing_data[tf].empty:
                # Get last timestamp from existing data
                last_ts = existing_data[tf]['timestamp'].iloc[-1]
                
                # Get M1 data from before last bar (to properly recalculate incomplete bar)
                tf_minutes = self._get_tf_minutes(tf)
                lookback_time = last_ts - timedelta(minutes=tf_minutes * 2)
                
                m1_for_update = new_m1_data[
                    pd.to_datetime(new_m1_data['timestamp']) >= lookback_time
                ]
                
                if not m1_for_update.empty:
                    # Resample the update period
                    updated_tf = self.resample_ohlcv(m1_for_update, tf)
                    
                    # Remove old incomplete bars and append new
                    existing_trimmed = existing_data[tf][
                        existing_data[tf]['timestamp'] < lookback_time
                    ]
                    
                    result[tf] = pd.concat(
                        [existing_trimmed, updated_tf], 
                        ignore_index=True
                    ).tail(lookback_bars)
                else:
                    result[tf] = existing_data[tf].tail(lookback_bars)
            else:
                # No existing data, do full resample
                result[tf] = self.resample_ohlcv(new_m1_data, tf).tail(lookback_bars)
        
        return result
    
    def _get_tf_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        tf_map = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240, 'D1': 1440
        }
        return tf_map.get(timeframe, 60)
    
    def align_timeframes(
        self, 
        data_dict: Dict[str, pd.DataFrame],
        reference_time: datetime = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Align all timeframes to same reference time
        Ensures all TF bars end at the same moment
        
        Args:
            data_dict: Dict of timeframe DataFrames
            reference_time: Reference time to align to (default: latest time)
        
        Returns:
            Aligned dict of DataFrames
        """
        if not reference_time:
            # Use latest timestamp across all timeframes
            latest_times = [
                df['timestamp'].max() 
                for df in data_dict.values() 
                if not df.empty
            ]
            if not latest_times:
                return data_dict
            reference_time = max(latest_times)
        
        aligned = {}
        
        for tf, df in data_dict.items():
            if df.empty:
                aligned[tf] = df
                continue
            
            # Keep only bars that end before or at reference time
            aligned[tf] = df[
                pd.to_datetime(df['timestamp']) <= reference_time
            ].copy()
        
        return aligned
    
    def get_current_bar_completion(
        self, 
        current_time: datetime, 
        timeframe: str
    ) -> float:
        """
        Calculate how complete the current bar is (0.0 to 1.0)
        
        Args:
            current_time: Current datetime
            timeframe: Timeframe to check
        
        Returns:
            Completion percentage (0.0 = just opened, 1.0 = about to close)
        """
        tf_minutes = self._get_tf_minutes(timeframe)
        
        # Get time since last bar open
        minutes_into_hour = current_time.minute
        hours_into_day = current_time.hour
        
        if timeframe.startswith('M'):
            # Minute-based timeframes
            elapsed = minutes_into_hour % tf_minutes
        elif timeframe == 'H1':
            elapsed = minutes_into_hour
        elif timeframe == 'H4':
            elapsed = (hours_into_day % 4) * 60 + minutes_into_hour
        elif timeframe == 'D1':
            elapsed = hours_into_day * 60 + minutes_into_hour
        else:
            elapsed = 0
        
        completion = elapsed / tf_minutes if tf_minutes > 0 else 0
        return min(completion, 1.0)
    
    def is_bar_closed(
        self, 
        current_time: datetime, 
        timeframe: str,
        threshold: float = 0.95
    ) -> bool:
        """
        Check if current bar is closed (or nearly closed)
        
        Args:
            current_time: Current datetime
            timeframe: Timeframe to check
            threshold: Completion threshold (default 0.95 = 95% complete)
        
        Returns:
            True if bar is closed/complete
        """
        completion = self.get_current_bar_completion(current_time, timeframe)
        return completion >= threshold
    
    def get_synchronized_data(
        self,
        symbol: str,
        m1_data: pd.DataFrame,
        required_bars: Dict[str, int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get synchronized multi-timeframe data
        
        Args:
            symbol: Trading symbol
            m1_data: M1 source data
            required_bars: Dict of {timeframe: num_bars} required
        
        Returns:
            Dict of synchronized timeframe DataFrames
        """
        if required_bars is None:
            required_bars = {tf: 200 for tf in self.timeframes}
        
        # Resample to all timeframes
        tf_data = self.resample_all_timeframes(m1_data, symbol)
        
        # Align timeframes
        tf_data = self.align_timeframes(tf_data)
        
        # Trim to required bars
        for tf, num_bars in required_bars.items():
            if tf in tf_data and not tf_data[tf].empty:
                tf_data[tf] = tf_data[tf].tail(num_bars)
        
        return tf_data
