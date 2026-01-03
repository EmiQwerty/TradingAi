"""
Binance Data Fetcher - Dati reali OHLCV da Binance API pubblica
Scarica candele storiche per crypto spot trading
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Optional
import logging
import time

logger = logging.getLogger(__name__)


class BinanceDataFetcher:
    """
    Fetcher per scaricare dati OHLCV da Binance API pubblica (no autenticazione).
    Supporta tutte le coppie spot disponibili su Binance.
    """
    
    BASE_URL = "https://api.binance.com/api/v3"
    
    # Timeframe mapping
    TIMEFRAME_MAP = {
        'M1': '1m',
        'M5': '5m',
        'M15': '15m',
        'M30': '30m',
        'H1': '1h',
        'H4': '4h',
        'D1': '1d',
        'W1': '1w',
        'MN': '1M'
    }
    
    def __init__(self, cache_timeout: int = 300):
        """
        Args:
            cache_timeout: Secondi prima di ri-scaricare dati (default 5 min)
        """
        self.cache = {}
        self.cache_timeout = cache_timeout
        logger.info("BinanceDataFetcher initialized")
    
    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalizza simbolo per Binance.
        
        Args:
            symbol: Simbolo (es. BTC_USD, BTCUSDT, BTC/USDT)
        
        Returns:
            Simbolo Binance (es. BTCUSDT)
        """
        # Rimuovi separatori
        symbol = symbol.replace('_', '').replace('/', '').upper()
        
        # Converti USD in USDT se necessario
        if symbol.endswith('USD') and not symbol.endswith('USDT'):
            symbol = symbol[:-3] + 'USDT'
        
        return symbol
    
    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str = 'H1',
        lookback_bars: int = 500,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Scarica dati storici da Binance.
        
        Args:
            symbol: Simbolo (es. BTC_USD, BTCUSDT, ETH_USD)
            timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN)
            lookback_bars: Numero di candele da scaricare (max 1000)
            use_cache: Usa cache se disponibile
        
        Returns:
            DataFrame con colonne [open, high, low, close, volume]
            o None se errore
        """
        cache_key = f"{symbol}_{timeframe}_{lookback_bars}"
        
        # Check cache
        if use_cache and cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                logger.debug(f"Using cached data for {symbol}")
                return data.copy()
        
        # Normalizza simbolo
        binance_symbol = self._normalize_symbol(symbol)
        binance_interval = self.TIMEFRAME_MAP.get(timeframe, '1h')
        
        # Limita a max 1000 bars (limite API Binance)
        limit = min(lookback_bars, 1000)
        
        try:
            logger.info(f"Downloading {binance_symbol} {timeframe} from Binance")
            
            # Chiamata API
            url = f"{self.BASE_URL}/klines"
            params = {
                'symbol': binance_symbol,
                'interval': binance_interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            klines = response.json()
            
            if not klines:
                logger.error(f"No data received for {binance_symbol}")
                return None
            
            # Converti in DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Converti timestamp in datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Seleziona solo colonne OHLCV
            df = df[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # Converti a float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Salva in cache
            self.cache[cache_key] = (df.copy(), time.time())
            
            logger.info(f"Downloaded {len(df)} bars for {binance_symbol}")
            
            return df
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error downloading {binance_symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error downloading {binance_symbol}: {e}")
            return None
    
    def get_available_symbols(self) -> Optional[list]:
        """
        Ottiene lista simboli disponibili su Binance.
        
        Returns:
            Lista di simboli o None se errore
        """
        try:
            url = f"{self.BASE_URL}/exchangeInfo"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            symbols = [s['symbol'] for s in data['symbols'] if s['status'] == 'TRADING']
            
            logger.info(f"Found {len(symbols)} trading symbols on Binance")
            return symbols
        
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return None


# Test standalone
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    fetcher = BinanceDataFetcher()
    
    # Test download BTCUSDT
    data = fetcher.fetch_historical_data('BTC_USD', 'H1', 100)
    
    if data is not None:
        print(f"\n✅ Downloaded {len(data)} bars")
        print(f"\nFirst 5 rows:")
        print(data.head())
        print(f"\nLast 5 rows:")
        print(data.tail())
        print(f"\nData range: {data.index[0]} to {data.index[-1]}")
    else:
        print("❌ Download failed")
