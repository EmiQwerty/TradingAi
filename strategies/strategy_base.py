"""
Base Strategy Class - Framework per strategie di trading
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Tipo di segnale"""
    BUY = "BUY"
    SELL = "SELL"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


@dataclass
class TradingSignal:
    """Segnale di trading generato da una strategia"""
    timestamp: pd.Timestamp
    signal_type: SignalType
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float  # 0-1
    reason: str  # Motivazione del segnale
    metadata: Dict  # Dati aggiuntivi (RSI, MACD, pattern, etc.)


class StrategyBase(ABC):
    """
    Classe base per tutte le strategie.
    Ogni strategia implementa logica specifica per generare segnali.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.signals: List[TradingSignal] = []
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcola indicatori tecnici necessari per la strategia.
        
        Args:
            data: DataFrame OHLCV
        
        Returns:
            DataFrame con indicatori aggiunti
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        Genera segnali di trading basati su logica strategica.
        
        Args:
            data: DataFrame con OHLCV + indicatori
        
        Returns:
            Lista di TradingSignal
        """
        pass
    
    @abstractmethod
    def calculate_stop_loss(self, entry_price: float, signal_type: SignalType, 
                           atr: float, data: pd.DataFrame, index: int) -> float:
        """Calcola stop loss ottimale per il trade"""
        pass
    
    @abstractmethod
    def calculate_take_profit(self, entry_price: float, signal_type: SignalType,
                             stop_loss: float, risk_reward_ratio: float = 2.0) -> float:
        """Calcola take profit ottimale (tipicamente RR ratio)"""
        pass
    
    def backtest(self, data: pd.DataFrame) -> Dict:
        """
        Esegue backtest completo della strategia.
        
        Returns:
            Dict con metriche performance
        """
        # Calcola indicatori
        data = self.calculate_indicators(data)
        
        # Genera segnali
        self.signals = self.generate_signals(data)
        
        return {
            'total_signals': len(self.signals),
            'strategy_name': self.name
        }
