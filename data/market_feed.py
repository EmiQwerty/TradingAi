"""
Market Data Feed Module
Handles real-time and historical market data via broker API and WebSocket
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import requests
import json
import logging
from websocket import create_connection, WebSocketConnectionClosedException

logger = logging.getLogger(__name__)


class MarketFeed:
    """
    Manages market data feed from broker API
    Supports both REST API for historical data and WebSocket for real-time ticks
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.broker_config = config['broker']
        self.symbols = [s['symbol'] for s in config['symbols'] if s.get('enabled', True)]
        
        self.api_url = self.broker_config['api_url']
        self.ws_url = self.broker_config['websocket_url']
        self.api_key = self.broker_config.get('api_key', '')
        self.api_secret = self.broker_config.get('api_secret', '')
        
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        
        self.ws_connection = None
        self.is_connected = False
        self.last_prices: Dict[str, float] = {}
        self.tick_buffer: Dict[str, List[dict]] = {symbol: [] for symbol in self.symbols}
        
        logger.info(f"MarketFeed initialized for symbols: {self.symbols}")
    
    def connect_websocket(self) -> bool:
        """
        Establish WebSocket connection for real-time data
        """
        try:
            logger.info(f"Connecting to WebSocket: {self.ws_url}")
            self.ws_connection = create_connection(
                self.ws_url,
                timeout=self.broker_config.get('timeout', 30)
            )
            
            # Subscribe to symbols
            subscribe_msg = {
                'action': 'subscribe',
                'symbols': self.symbols,
                'api_key': self.api_key
            }
            self.ws_connection.send(json.dumps(subscribe_msg))
            
            self.is_connected = True
            logger.info("WebSocket connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.is_connected = False
            return False
    
    def disconnect_websocket(self):
        """Close WebSocket connection"""
        if self.ws_connection:
            try:
                self.ws_connection.close()
                logger.info("WebSocket disconnected")
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
        self.is_connected = False
    
    def get_latest_tick(self, symbol: str) -> Optional[dict]:
        """
        Get latest tick from WebSocket stream
        Returns: {'symbol': str, 'bid': float, 'ask': float, 'timestamp': datetime}
        """
        if not self.is_connected:
            logger.warning("WebSocket not connected, attempting reconnection...")
            self.connect_websocket()
        
        try:
            if self.ws_connection:
                data = self.ws_connection.recv()
                tick_data = json.loads(data)
                
                if tick_data.get('symbol') == symbol:
                    tick = {
                        'symbol': tick_data['symbol'],
                        'bid': float(tick_data['bid']),
                        'ask': float(tick_data['ask']),
                        'timestamp': datetime.fromtimestamp(tick_data['timestamp'])
                    }
                    self.last_prices[symbol] = (tick['bid'] + tick['ask']) / 2
                    self.tick_buffer[symbol].append(tick)
                    return tick
        
        except WebSocketConnectionClosedException:
            logger.error("WebSocket connection closed unexpectedly")
            self.is_connected = False
            return None
        except Exception as e:
            logger.error(f"Error receiving tick data: {e}")
            return None
        
        return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current mid price for symbol
        Uses last WebSocket tick or falls back to REST API
        """
        if symbol in self.last_prices:
            return self.last_prices[symbol]
        
        # Fallback to REST API
        try:
            url = f"{self.api_url}/quotes/{symbol}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                bid = float(data['bid'])
                ask = float(data['ask'])
                price = (bid + ask) / 2
                self.last_prices[symbol] = price
                return price
            else:
                logger.error(f"Failed to get price for {symbol}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from broker API
        
        Args:
            symbol: Trading symbol
            timeframe: M1, M5, M15, H1, H4, D1
            start_date: Start datetime
            end_date: End datetime
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            logger.info(f"Fetching historical data: {symbol} {timeframe} from {start_date} to {end_date}")
            
            url = f"{self.api_url}/candles/{symbol}"
            params = {
                'timeframe': timeframe,
                'start': int(start_date.timestamp()),
                'end': int(end_date.timestamp()),
                'limit': 10000
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if not data.get('candles'):
                    logger.warning(f"No data returned for {symbol} {timeframe}")
                    return pd.DataFrame()
                
                df = pd.DataFrame(data['candles'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                logger.info(f"Loaded {len(df)} candles for {symbol} {timeframe}")
                return df
            
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def get_latest_candles(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        """
        Get most recent N candles for symbol and timeframe
        """
        end_date = datetime.utcnow()
        
        # Calculate start date based on timeframe
        tf_minutes = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240, 'D1': 1440
        }
        minutes = tf_minutes.get(timeframe, 60)
        start_date = end_date - timedelta(minutes=minutes * count * 2)  # Buffer for gaps
        
        df = self.get_historical_data(symbol, timeframe, start_date, end_date)
        
        if not df.empty:
            return df.tail(count).reset_index(drop=True)
        return df
    
    def get_multiple_symbols_data(
        self, 
        symbols: List[str], 
        timeframe: str, 
        count: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """
        Get latest candles for multiple symbols
        Returns dict mapping symbol to DataFrame
        """
        result = {}
        
        for symbol in symbols:
            df = self.get_latest_candles(symbol, timeframe, count)
            if not df.empty:
                result[symbol] = df
            else:
                logger.warning(f"No data available for {symbol} {timeframe}")
        
        return result
    
    def stream_ticks(self, symbol: str, callback, duration_seconds: int = None):
        """
        Stream real-time ticks and call callback function for each tick
        
        Args:
            symbol: Trading symbol
            callback: Function to call with each tick dict
            duration_seconds: Optional duration to stream (None = infinite)
        """
        if not self.is_connected:
            self.connect_websocket()
        
        start_time = time.time()
        
        try:
            while True:
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    break
                
                tick = self.get_latest_tick(symbol)
                if tick:
                    callback(tick)
                
                time.sleep(0.01)  # Small delay to prevent CPU overload
        
        except KeyboardInterrupt:
            logger.info("Tick streaming stopped by user")
        finally:
            self.disconnect_websocket()
    
    def get_server_time(self) -> Optional[datetime]:
        """Get broker server time"""
        try:
            url = f"{self.api_url}/time"
            response = self.session.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return datetime.fromtimestamp(data['timestamp'])
        except Exception as e:
            logger.error(f"Error getting server time: {e}")
        
        return None
    
    def is_market_open(self, symbol: str) -> bool:
        """
        Check if market is currently open for trading
        """
        try:
            url = f"{self.api_url}/market/status/{symbol}"
            response = self.session.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('is_open', False)
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
        
        # Default to True for forex (24/5 market)
        return True
