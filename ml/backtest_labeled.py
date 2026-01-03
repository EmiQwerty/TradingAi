"""
Backtester con Label Generation per ML Training
Simula operazioni chiuse, estrae feature e genera label (win/loss)
Produce training dataset per modelli predittivi
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Rappresentazione di un trade chiuso"""
    trade_id: str
    symbol: str
    side: str  # BUY/SELL
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    stop_loss: float
    take_profit: float
    pnl: float  # Realized PnL in USD
    pnl_pips: float  # PnL in pips
    pnl_percent: float  # PnL % of entry
    duration_bars: int  # Quante candele da entry a exit
    
    # ML labels
    label_win_loss: int  # 1 win, 0 loss
    label_magnitude: float  # 0-1, quanto è stata grande la win/loss
    label_exit_reason: str  # 'tp', 'sl', 'timeout'
    
    # Context al moment o di entry
    entry_rsi: float = None
    entry_macd_histogram: float = None
    entry_trend_type: float = None
    entry_volatility: float = None
    entry_confluence: float = None


class BacktestEngine:
    """
    Esegue backtest deterministico e genera training labels.
    """
    
    def __init__(
        self,
        initial_balance: float = 100000.0,
        risk_per_trade: float = 0.01,  # 1% per trade
        leverage: int = 30,
        commission_per_lot: float = 7.0
    ):
        """
        Args:
            initial_balance: Capitale iniziale
            risk_per_trade: Rischio per trade (% di balance)
            leverage: Leva finanziaria
            commission_per_lot: Commissione per lotto standard
        """
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        self.commission_per_lot = commission_per_lot
        
        self.trades: List[BacktestTrade] = []
        self.open_positions: Dict[str, Dict] = {}
        
        logger.info(f"BacktestEngine initialized: Balance=${initial_balance}, Risk={risk_per_trade*100}%")
    
    def simulate_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        entry_time: datetime,
        stop_loss: float,
        take_profit: float,
        historical_data: pd.DataFrame,
        starting_row_index: int,
        entry_features: Dict[str, float] = None
    ) -> Optional[BacktestTrade]:
        """
        Simula esecuzione di un trade partendo da entry_time.
        Itera through candele storiche fino al hit di SL/TP.
        
        Args:
            symbol: Simbolo
            side: 'BUY' o 'SELL'
            entry_price: Prezzo di entry
            entry_time: Timestamp entry
            stop_loss: Prezzo SL
            take_profit: Prezzo TP
            historical_data: DataFrame con OHLCV storico
            starting_row_index: Index nella data dove cominciare la simulazione
            entry_features: Dict con feature al momento dell'entry
        
        Returns:
            BacktestTrade con outcome, o None se non chiuso
        """
        if starting_row_index >= len(historical_data) - 1:
            logger.warning(f"Not enough data to simulate trade for {symbol}")
            return None
        
        trade_id = f"{symbol}_{entry_time.timestamp()}"
        
        # Itera through candele future fino a hit SL/TP
        for i in range(starting_row_index + 1, len(historical_data)):
            row = historical_data.iloc[i]
            
            candle_high = row['high']
            candle_low = row['low']
            candle_close = row['close']
            candle_time = row.name if hasattr(row.name, 'to_pydatetime') else datetime.now()
            
            # Controlla hit SL
            if side == 'BUY':
                if candle_low <= stop_loss:
                    exit_price = stop_loss
                    exit_time = candle_time
                    exit_reason = 'sl'
                    break
                elif candle_high >= take_profit:
                    exit_price = take_profit
                    exit_time = candle_time
                    exit_reason = 'tp'
                    break
            else:  # SELL
                if candle_high >= stop_loss:
                    exit_price = stop_loss
                    exit_time = candle_time
                    exit_reason = 'sl'
                    break
                elif candle_low <= take_profit:
                    exit_price = take_profit
                    exit_time = candle_time
                    exit_reason = 'tp'
                    break
            
            # Timeout: se abbiamo aspettato 100 candele, close a market
            duration = i - starting_row_index
            if duration >= 100:
                exit_price = candle_close
                exit_time = candle_time
                exit_reason = 'timeout'
                break
        else:
            # Nessun hit SL/TP/timeout in range disponibile
            logger.debug(f"Trade {trade_id} not closed in available data")
            return None
        
        # Calcola PnL
        if side == 'BUY':
            pnl_pips = (exit_price - entry_price) * 10000
            pnl_percent = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl_pips = (entry_price - exit_price) * 10000
            pnl_percent = ((entry_price - exit_price) / entry_price) * 100
        
        # PnL in USD (approssimativo, assume 1 lotto standard = 100k units)
        lot_size = 1.0  # 1 lotto
        pnl_usd = pnl_pips * 0.0001 * 100000 * lot_size - self.commission_per_lot
        
        # Labels
        label_win = 1 if pnl_usd > 0 else 0
        label_magnitude = min(abs(pnl_percent) / 2.0, 1.0)  # Normalizza 0-2% a 0-1
        
        duration_bars = i - starting_row_index
        
        trade = BacktestTrade(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            entry_time=entry_time,
            exit_price=exit_price,
            exit_time=exit_time,
            stop_loss=stop_loss,
            take_profit=take_profit,
            pnl=pnl_usd,
            pnl_pips=pnl_pips,
            pnl_percent=pnl_percent,
            duration_bars=duration_bars,
            label_win_loss=label_win,
            label_magnitude=label_magnitude,
            label_exit_reason=exit_reason
        )
        
        # Aggiungi feature entry se fornite
        if entry_features:
            if 'rsi_current' in entry_features:
                trade.entry_rsi = entry_features['rsi_current']
            if 'macd_histogram' in entry_features:
                trade.entry_macd_histogram = entry_features['macd_histogram']
            if 'trend_type' in entry_features:
                trade.entry_trend_type = entry_features['trend_type']
            if 'volatility_std_20' in entry_features:
                trade.entry_volatility = entry_features['volatility_std_20']
            if 'signal_confluence' in entry_features:
                trade.entry_confluence = entry_features['signal_confluence']
        
        self.trades.append(trade)
        
        logger.debug(f"Trade closed: {trade_id} {side} PnL={pnl_usd:+.2f} ({exit_reason})")
        
        return trade
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Restituisce DataFrame con tutti i trades"""
        if not self.trades:
            return pd.DataFrame()
        
        trades_dicts = [asdict(t) for t in self.trades]
        return pd.DataFrame(trades_dicts)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calcola metriche di performance dal backtest"""
        if not self.trades:
            return {}
        
        trades_df = self.get_trades_dataframe()
        
        total_trades = len(trades_df)
        winning_trades = (trades_df['label_win_loss'] == 1).sum()
        losing_trades = (trades_df['label_win_loss'] == 0).sum()
        
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['label_win_loss'] == 1]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['label_win_loss'] == 0]['pnl'].mean() if losing_trades > 0 else 0
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 1.0
        
        # Expectancy
        expectancy = (win_rate / 100.0 * avg_win) + ((1 - win_rate/100.0) * avg_loss) if avg_win + abs(avg_loss) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'payoff_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else 0
        }
    
    def analyze_win_conditions(self) -> Dict[str, float]:
        """Analizza feature comuni nei trades vincenti vs perdenti"""
        if not self.trades:
            return {}
        
        trades_df = self.get_trades_dataframe()
        
        wins = trades_df[trades_df['label_win_loss'] == 1]
        losses = trades_df[trades_df['label_win_loss'] == 0]
        
        analysis = {}
        
        # RSI medio
        if wins['entry_rsi'].notna().any():
            analysis['avg_rsi_wins'] = wins['entry_rsi'].mean()
            analysis['avg_rsi_losses'] = losses['entry_rsi'].mean()
        
        # Trend type
        if wins['entry_trend_type'].notna().any():
            analysis['avg_trend_wins'] = wins['entry_trend_type'].mean()
            analysis['avg_trend_losses'] = losses['entry_trend_type'].mean()
        
        # Confluence
        if wins['entry_confluence'].notna().any():
            analysis['avg_confluence_wins'] = wins['entry_confluence'].mean()
            analysis['avg_confluence_losses'] = losses['entry_confluence'].mean()
        
        # Exit reason distribution
        analysis['tp_hit_rate'] = (trades_df['label_exit_reason'] == 'tp').sum() / len(trades_df) * 100
        analysis['sl_hit_rate'] = (trades_df['label_exit_reason'] == 'sl').sum() / len(trades_df) * 100
        
        return analysis
    
    def get_training_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara dataset per training ML.
        
        Returns:
            (X_features, y_labels) dove y = label_win_loss
        """
        trades_df = self.get_trades_dataframe()
        
        # Feature columns (escludiamo le label columns)
        feature_cols = [col for col in trades_df.columns 
                       if not col.startswith('label_') and col not in 
                       ['trade_id', 'symbol', 'side', 'entry_time', 'exit_time']]
        
        X = trades_df[feature_cols].fillna(0)
        y = trades_df['label_win_loss']
        
        return X, y
    
    def reset(self):
        """Reset engine per nuovo backtest"""
        self.trades = []
        self.open_positions = {}
        logger.info("BacktestEngine reset")
