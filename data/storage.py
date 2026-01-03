"""
Historical Data Storage Module
Handles persistent storage of market data in Parquet/CSV format
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataStorage:
    """
    Manages persistent storage of historical market data
    Supports efficient incremental updates and data retrieval
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.storage_config = config.get('storage', {})
        
        self.base_path = Path(self.storage_config.get('path', 'data/historical'))
        self.format = self.storage_config.get('format', 'parquet')
        self.retention_days = self.storage_config.get('retention_days', 90)
        self.compression = self.storage_config.get('compression', 'snappy')
        
        # Create directory structure
        self._create_directories()
        
        logger.info(f"DataStorage initialized at {self.base_path}, format: {self.format}")
    
    def _create_directories(self):
        """Create necessary directory structure"""
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each data type
        for subdir in ['candles', 'ticks', 'indicators', 'signals']:
            (self.base_path / subdir).mkdir(exist_ok=True)
    
    def _get_file_path(
        self, 
        data_type: str, 
        symbol: str, 
        timeframe: str = None,
        date: datetime = None
    ) -> Path:
        """
        Generate file path for data storage
        
        Args:
            data_type: 'candles', 'ticks', 'indicators', 'signals'
            symbol: Trading symbol
            timeframe: Optional timeframe
            date: Optional date for daily partitioning
        
        Returns:
            Path object
        """
        path = self.base_path / data_type / symbol
        path.mkdir(parents=True, exist_ok=True)
        
        # Build filename
        filename_parts = [symbol]
        if timeframe:
            filename_parts.append(timeframe)
        if date:
            filename_parts.append(date.strftime('%Y%m%d'))
        
        filename = '_'.join(filename_parts)
        
        if self.format == 'parquet':
            filename += '.parquet'
        else:
            filename += '.csv'
        
        return path / filename
    
    def save_candles(
        self, 
        symbol: str, 
        timeframe: str, 
        df: pd.DataFrame,
        append: bool = False
    ) -> bool:
        """
        Save OHLCV candle data
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M1, M5, H1, etc.)
            df: DataFrame with candle data
            append: If True, append to existing data
        
        Returns:
            True if successful
        """
        if df.empty:
            logger.warning(f"Empty DataFrame, nothing to save for {symbol} {timeframe}")
            return False
        
        try:
            file_path = self._get_file_path('candles', symbol, timeframe)
            
            if append and file_path.exists():
                # Load existing data
                existing_df = self.load_candles(symbol, timeframe)
                if not existing_df.empty:
                    # Combine and remove duplicates
                    df = pd.concat([existing_df, df], ignore_index=True)
                    df = df.drop_duplicates(subset=['timestamp'], keep='last')
                    df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Save data
            if self.format == 'parquet':
                df.to_parquet(
                    file_path, 
                    compression=self.compression,
                    index=False
                )
            else:
                df.to_csv(file_path, index=False)
            
            logger.info(f"Saved {len(df)} candles to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving candles for {symbol} {timeframe}: {e}")
            return False
    
    def load_candles(
        self, 
        symbol: str, 
        timeframe: str,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """
        Load OHLCV candle data
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Optional start date filter
            end_date: Optional end date filter
        
        Returns:
            DataFrame with candle data
        """
        file_path = self._get_file_path('candles', symbol, timeframe)
        
        if not file_path.exists():
            logger.debug(f"No stored data found at {file_path}")
            return pd.DataFrame()
        
        try:
            if self.format == 'parquet':
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path, parse_dates=['timestamp'])
            
            # Apply date filters
            if start_date:
                df = df[pd.to_datetime(df['timestamp']) >= start_date]
            if end_date:
                df = df[pd.to_datetime(df['timestamp']) <= end_date]
            
            logger.debug(f"Loaded {len(df)} candles from {file_path}")
            return df
        
        except Exception as e:
            logger.error(f"Error loading candles from {file_path}: {e}")
            return pd.DataFrame()
    
    def save_ticks(
        self, 
        symbol: str, 
        ticks: List[dict],
        date: datetime = None
    ) -> bool:
        """
        Save tick data (bid/ask)
        
        Args:
            symbol: Trading symbol
            ticks: List of tick dicts
            date: Date for partitioning (default: today)
        
        Returns:
            True if successful
        """
        if not ticks:
            return False
        
        try:
            df = pd.DataFrame(ticks)
            
            if date is None:
                date = datetime.utcnow()
            
            file_path = self._get_file_path('ticks', symbol, date=date)
            
            # Append to existing ticks for the day
            if file_path.exists():
                existing_df = pd.read_parquet(file_path) if self.format == 'parquet' else pd.read_csv(file_path)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            if self.format == 'parquet':
                df.to_parquet(file_path, compression=self.compression, index=False)
            else:
                df.to_csv(file_path, index=False)
            
            logger.debug(f"Saved {len(ticks)} ticks to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving ticks for {symbol}: {e}")
            return False
    
    def save_indicators(
        self, 
        symbol: str, 
        timeframe: str, 
        indicators_df: pd.DataFrame
    ) -> bool:
        """
        Save calculated indicators
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            indicators_df: DataFrame with indicator values
        
        Returns:
            True if successful
        """
        try:
            file_path = self._get_file_path('indicators', symbol, timeframe)
            
            if self.format == 'parquet':
                indicators_df.to_parquet(
                    file_path, 
                    compression=self.compression,
                    index=False
                )
            else:
                indicators_df.to_csv(file_path, index=False)
            
            logger.debug(f"Saved indicators to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving indicators for {symbol} {timeframe}: {e}")
            return False
    
    def load_indicators(
        self, 
        symbol: str, 
        timeframe: str
    ) -> pd.DataFrame:
        """Load saved indicators"""
        file_path = self._get_file_path('indicators', symbol, timeframe)
        
        if not file_path.exists():
            return pd.DataFrame()
        
        try:
            if self.format == 'parquet':
                return pd.read_parquet(file_path)
            else:
                return pd.read_csv(file_path, parse_dates=['timestamp'])
        except Exception as e:
            logger.error(f"Error loading indicators: {e}")
            return pd.DataFrame()
    
    def cleanup_old_data(self):
        """
        Remove data older than retention_days
        """
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        deleted_count = 0
        
        logger.info(f"Cleaning up data older than {cutoff_date.strftime('%Y-%m-%d')}")
        
        try:
            for data_type in ['candles', 'ticks', 'indicators', 'signals']:
                data_path = self.base_path / data_type
                
                if not data_path.exists():
                    continue
                
                # Iterate through all files
                for file_path in data_path.rglob(f"*.{self.format if self.format == 'parquet' else 'csv'}"):
                    try:
                        # Check file modification time
                        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        
                        if file_mtime < cutoff_date:
                            file_path.unlink()
                            deleted_count += 1
                            logger.debug(f"Deleted old file: {file_path}")
                    
                    except Exception as e:
                        logger.error(f"Error deleting {file_path}: {e}")
            
            logger.info(f"Cleanup complete: {deleted_count} files deleted")
        
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_data_summary(self) -> Dict[str, dict]:
        """
        Get summary of stored data
        
        Returns:
            Dict with statistics about stored data
        """
        summary = {
            'candles': {},
            'ticks': {},
            'indicators': {},
            'total_size_mb': 0
        }
        
        try:
            for data_type in ['candles', 'ticks', 'indicators']:
                data_path = self.base_path / data_type
                
                if not data_path.exists():
                    continue
                
                file_count = 0
                total_size = 0
                
                for file_path in data_path.rglob('*'):
                    if file_path.is_file():
                        file_count += 1
                        total_size += file_path.stat().st_size
                
                summary[data_type] = {
                    'file_count': file_count,
                    'size_mb': total_size / (1024 * 1024)
                }
                summary['total_size_mb'] += summary[data_type]['size_mb']
        
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
        
        return summary
    
    def export_to_csv(self, symbol: str, timeframe: str, output_path: str) -> bool:
        """
        Export data to CSV format (regardless of storage format)
        """
        try:
            df = self.load_candles(symbol, timeframe)
            if df.empty:
                logger.warning(f"No data to export for {symbol} {timeframe}")
                return False
            
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(df)} rows to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False
