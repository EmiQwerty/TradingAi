"""
Yahoo Finance Data Fetcher - Dati reali OHLCV da Yahoo Finance
Scarica candele storiche e recenti per simboli forex, indici, crypto
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import time

logger = logging.getLogger(__name__)


class YahooFinanceDataFetcher:
    """
    Fetcher per scaricare dati OHLCV reali da Yahoo Finance.
    Gestisce mapping simboli trading → Yahoo Finance ticker.
    """
    
    # Mapping simboli OANDA → Yahoo Finance tickers
    SYMBOL_MAP = {
        'EUR_USD': 'EURUSD=X',
        'GBP_USD': 'GBPUSD=X',
        'USD_JPY': 'USDJPY=X',
        'AUD_USD': 'AUDUSD=X',
        'USD_CAD': 'USDCAD=X',
        'USD_CHF': 'USDCHF=X',
        'NZD_USD': 'NZDUSD=X',
        'EUR_GBP': 'EURGBP=X',
        'EUR_JPY': 'EURJPY=X',
        'GBP_JPY': 'GBPJPY=X',
        'XAU_USD': 'GC=F',        # Gold futures
        'XAG_USD': 'SI=F',        # Silver futures
        'BCO_USD': 'BZ=F',        # Brent Crude Oil
        'WTI_USD': 'CL=F',        # WTI Crude Oil
        'BTC_USD': 'BTC-USD',     # Bitcoin
        'ETH_USD': 'ETH-USD',     # Ethereum
        'SPX': '^GSPC',           # S&P 500
        'NDX': '^NDX',            # NASDAQ 100
        'DJI': '^DJI',            # Dow Jones
    }
    
    # Timeframe mapping: nostro formato → Yahoo Finance
    TIMEFRAME_MAP = {
        'M1': '1m',
        'M5': '5m',
        'M15': '15m',
        'M30': '30m',
        'H1': '1h',
        'H4': '1h',   # Download 1h e poi resample
        'D1': '1d',
        'W1': '1wk',
        'MN': '1mo'
    }
    
    def __init__(self, cache_timeout: int = 300):
        """
        Args:
            cache_timeout: Secondi prima di ri-scaricare dati (default 5 min)
        """
        self.cache: Dict[str, Tuple[pd.DataFrame, float]] = {}  # (data, timestamp)
        self.cache_timeout = cache_timeout
        
        logger.info("YahooFinanceDataFetcher initialized")
    
    def _get_yahoo_ticker(self, symbol: str) -> str:
        """
        Converte simbolo trading in Yahoo Finance ticker.
        
        Args:
            symbol: Simbolo formato OANDA (es. EUR_USD)
        
        Returns:
            Yahoo ticker (es. EURUSD=X)
        """
        ticker = self.SYMBOL_MAP.get(symbol)
        
        if ticker is None:
            # Fallback: prova conversione automatica
            if '_' in symbol:
                # Forex: EUR_USD → EURUSD=X
                ticker = symbol.replace('_', '') + '=X'
            else:
                # Assume sia già ticker valido
                ticker = symbol
            
            logger.warning(f"Symbol {symbol} not in map, using fallback: {ticker}")
        
        return ticker
    
    def _get_cache_key(self, symbol: str, timeframe: str, lookback_bars: int) -> str:
        """Genera chiave cache."""
        return f"{symbol}_{timeframe}_{lookback_bars}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Controlla se cache è ancora valida."""
        if cache_key not in self.cache:
            return False
        
        _, timestamp = self.cache[cache_key]
        elapsed = time.time() - timestamp
        
        return elapsed < self.cache_timeout
    
    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str = 'H1',
        lookback_bars: int = 500,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Scarica dati storici da Yahoo Finance.
        
        Args:
            symbol: Simbolo (es. EUR_USD)
            timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN)
            lookback_bars: Numero di candele da scaricare
            use_cache: Usa cache se disponibile
        
        Returns:
            DataFrame con colonne [Open, High, Low, Close, Volume]
            o None se errore
        """
        cache_key = self._get_cache_key(symbol, timeframe, lookback_bars)
        
        # Check cache
        if use_cache and self._is_cache_valid(cache_key):
            logger.debug(f"Using cached data for {symbol} {timeframe}")
            data, _ = self.cache[cache_key]
            return data.copy()
        
        # Download da Yahoo Finance
        yahoo_ticker = self._get_yahoo_ticker(symbol)
        yahoo_interval = self.TIMEFRAME_MAP.get(timeframe, '1h')
        
        try:
            # Calcola periodo da scaricare
            period = self._calculate_period(timeframe, lookback_bars)
            
            logger.info(f"Downloading {symbol} ({yahoo_ticker}) {timeframe} last {period}")
            
            # Download con yfinance
            ticker = yf.Ticker(yahoo_ticker)
            data = ticker.history(
                period=period,
                interval=yahoo_interval,
                auto_adjust=True,  # Adjust per splits/dividends
                actions=False      # Non servono dividends/splits info
            )
            
            if data.empty:
                logger.error(f"No data received for {symbol} from Yahoo Finance")
                return None
            
            # Resample se necessario (es. H4 da H1)
            if timeframe == 'H4' and yahoo_interval == '1h':
                data = self._resample_to_h4(data)
            
            # Limita al numero di barre richieste
            data = data.tail(lookback_bars)
            
            # Standardizza nomi colonne
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Assicura che abbiamo le colonne necessarie
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_cols):
                logger.error(f"Missing required columns in data for {symbol}")
                return None
            
            # Salva in cache
            self.cache[cache_key] = (data.copy(), time.time())
            
            logger.info(f"Downloaded {len(data)} bars for {symbol} {timeframe}")
            
            return data
        
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {e}")
            return None
    
    def _calculate_period(self, timeframe: str, lookback_bars: int) -> str:
        """
        Calcola periodo Yahoo Finance da timeframe e bars.
        
        Args:
            timeframe: Nostro timeframe (M1, H1, D1, etc.)
            lookback_bars: Numero di barre
        
        Returns:
            Periodo Yahoo (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        """
        # Stima giorni necessari
        timeframe_minutes = {
            'M1': 1,
            'M5': 5,
            'M15': 15,
            'M30': 30,
            'H1': 60,
            'H4': 240,
            'D1': 1440,
            'W1': 10080,
            'MN': 43200
        }
        
        minutes = timeframe_minutes.get(timeframe, 60)
        total_minutes = lookback_bars * minutes
        days = total_minutes / (60 * 24)
        
        # Aggiungi buffer (mercato non aperto 24/7)
        days = days * 1.5
        
        # Mappa a periodi Yahoo
        if days <= 7:
            return '7d'
        elif days <= 30:
            return '1mo'
        elif days <= 90:
            return '3mo'
        elif days <= 180:
            return '6mo'
        elif days <= 365:
            return '1y'
        elif days <= 730:
            return '2y'
        elif days <= 1825:
            return '5y'
        else:
            return 'max'
    
    def _resample_to_h4(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Resample dati H1 in H4.
        
        Args:
            data: DataFrame con dati H1
        
        Returns:
            DataFrame resampled a H4
        """
        try:
            resampled = data.resample('4H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            return resampled
        
        except Exception as e:
            logger.error(f"Error resampling to H4: {e}")
            return data
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Ottiene prezzo corrente (last close) da Yahoo Finance.
        
        Args:
            symbol: Simbolo (es. EUR_USD)
        
        Returns:
            Prezzo corrente o None
        """
        try:
            yahoo_ticker = self._get_yahoo_ticker(symbol)
            ticker = yf.Ticker(yahoo_ticker)
            
            # Info contiene current price
            info = ticker.info
            
            # Prova vari campi possibili
            price = (
                info.get('regularMarketPrice') or
                info.get('currentPrice') or
                info.get('previousClose')
            )
            
            if price is None:
                # Fallback: scarica ultima candela
                data = ticker.history(period='1d', interval='1m')
                if not data.empty:
                    price = data['Close'].iloc[-1]
            
            if price is not None:
                logger.debug(f"Current price for {symbol}: {price:.5f}")
                return float(price)
            
            logger.warning(f"Could not get current price for {symbol}")
            return None
        
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def get_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = 'H1',
        lookback_bars: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """
        Scarica dati per multipli simboli in parallelo.
        
        Args:
            symbols: Lista simboli
            timeframe: Timeframe
            lookback_bars: Numero barre
        
        Returns:
            Dict {symbol: DataFrame}
        """
        results = {}
        
        for symbol in symbols:
            data = self.fetch_historical_data(symbol, timeframe, lookback_bars)
            if data is not None:
                results[symbol] = data
            else:
                logger.warning(f"Failed to fetch data for {symbol}")
        
        return results
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Ottiene prezzi correnti per multipli simboli.
        
        Args:
            symbols: Lista simboli
        
        Returns:
            Dict {symbol: price}
        """
        prices = {}
        
        for symbol in symbols:
            price = self.get_current_price(symbol)
            if price is not None:
                prices[symbol] = price
        
        return prices
    
    def clear_cache(self) -> None:
        """Pulisce cache."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def add_symbol_mapping(self, symbol: str, yahoo_ticker: str) -> None:
        """
        Aggiunge mapping custom simbolo → Yahoo ticker.
        
        Args:
            symbol: Nostro simbolo (es. CUSTOM_PAIR)
            yahoo_ticker: Yahoo Finance ticker
        """
        self.SYMBOL_MAP[symbol] = yahoo_ticker
        logger.info(f"Added symbol mapping: {symbol} → {yahoo_ticker}")


# Utility function per download veloce
def download_symbol_data(
    symbol: str,
    timeframe: str = 'H1',
    lookback_bars: int = 500
) -> Optional[pd.DataFrame]:
    """
    Utility function per download veloce di un simbolo.
    
    Args:
        symbol: Simbolo (es. EUR_USD)
        timeframe: Timeframe
        lookback_bars: Numero barre
    
    Returns:
        DataFrame o None
    """
    fetcher = YahooFinanceDataFetcher()
    return fetcher.fetch_historical_data(symbol, timeframe, lookback_bars)
