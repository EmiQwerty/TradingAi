"""
Backtest Engine
Simula il trading system su dati storici con pipeline completa.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from data.storage import DataStorage
from data.resampler import TimeframeResampler
from features.indicators import TechnicalIndicators
from features.structure import StructureDetector
from features.volatility import VolatilityAnalyzer
from ml.inference import MLInference
from macro.macro_filter import MacroFilter
from strategy.confidence import ConfidenceAggregator
from strategy.decision_engine import DecisionEngine
from risk.risk_manager import RiskManager
from risk.correlation import CorrelationAnalyzer
from monitoring.metrics import MetricsCalculator
from state.models import Position, TFState, SymbolState
from state.state_store import StateStore


logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Trade simulato nel backtest."""
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime] = None
    side: str = "buy"  # buy/sell
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    volume: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    commission: float = 0.0
    exit_reason: Optional[str] = None  # sl, tp, signal, manual
    max_favorable: float = 0.0  # MAE
    max_adverse: float = 0.0  # MFE
    bars_held: int = 0


@dataclass
class BacktestConfig:
    """Configurazione backtest."""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)
    commission: float = 0.0001  # 0.01% per side
    slippage: float = 0.0001  # 0.01%
    primary_timeframe: str = "H1"
    # Opzioni simulazione
    realistic_fills: bool = True
    check_liquidity: bool = False
    use_bid_ask: bool = False


class BacktestEngine:
    """
    Engine per backtest completo con simulazione realistica.
    Replay dati storici ed esegue pipeline completa strategy.
    """
    
    def __init__(
        self,
        config: BacktestConfig,
        settings: Dict,
        symbol_configs: Dict,
        risk_config: Dict
    ):
        self.config = config
        self.settings = settings
        self.symbol_configs = symbol_configs
        self.risk_config = risk_config
        
        # Componenti
        self.storage = DataStorage(self.settings.get('storage', {}))
        self.resampler = TimeframeResampler(self.config.timeframes)
        self.indicators = TechnicalIndicators(self.settings.get('indicators', {}))
        self.structure = StructureDetector(self.settings.get('structure', {}))
        self.volatility = VolatilityAnalyzer()
        self.ml_inference = MLInference(self.settings.get('ml', {}))
        self.macro_filter = MacroFilter(self.settings.get('macro', {}))
        self.confidence = ConfidenceAggregator(self.settings)
        self.decision_engine = DecisionEngine(
            self.settings,
            self.symbol_configs,
            self.confidence,
            self.indicators,
            self.structure,
            self.ml_inference,
            self.volatility,
            self.macro_filter
        )
        self.risk_manager = RiskManager(self.risk_config, self.config.initial_capital)
        self.correlation = CorrelationAnalyzer()
        self.metrics_calc = MetricsCalculator()
        self.state_store = StateStore()
        
        # Stato backtest
        self.capital = self.config.initial_capital
        self.equity = self.config.initial_capital
        self.open_trades: Dict[str, BacktestTrade] = {}
        self.closed_trades: List[BacktestTrade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.current_time: Optional[datetime] = None
        
        # Cache dati
        self.historical_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        
        logger.info(f"BacktestEngine inizializzato: {config.start_date} -> {config.end_date}")
    
    def run(self) -> Dict:
        """
        Esegue backtest completo.
        
        Returns:
            Risultati backtest con metriche e trades
        """
        logger.info("Avvio backtest...")
        
        # 1. Carica dati storici
        if not self._load_historical_data():
            logger.error("Errore caricamento dati storici")
            return {}
        
        # 2. Preprocessing: resample tutti i TF
        if not self._preprocess_data():
            logger.error("Errore preprocessing dati")
            return {}
        
        # 3. Main loop: itera su barre primary TF
        self._run_simulation()
        
        # 4. Chiudi trade aperti
        self._close_all_open_trades()
        
        # 5. Calcola metriche finali
        results = self._calculate_results()
        
        logger.info(f"Backtest completato: {len(self.closed_trades)} trades")
        return results
    
    def _load_historical_data(self) -> bool:
        """Carica dati storici per tutti i simboli."""
        try:
            for symbol in self.config.symbols:
                self.historical_data[symbol] = {}
                
                # Carica M1 (base per resample)
                df_m1 = self.storage.load_candles(
                    symbol,
                    "M1",
                    self.config.start_date,
                    self.config.end_date
                )
                
                if df_m1 is None or df_m1.empty:
                    logger.warning(f"Nessun dato M1 per {symbol}, genero dati dummy")
                    df_m1 = self._generate_dummy_data(symbol, "M1")
                
                self.historical_data[symbol]["M1"] = df_m1
                logger.info(f"Caricati {len(df_m1)} candles M1 per {symbol}")
            
            return True
        
        except Exception as e:
            logger.error(f"Errore caricamento dati: {e}")
            return False
    
    def _generate_dummy_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Genera dati dummy per testing (random walk)."""
        logger.warning(f"Generazione dati dummy per {symbol} {timeframe}")
        
        # Base price
        base_prices = {"EURUSD": 1.1000, "GBPUSD": 1.3000, "XAUUSD": 1900.0}
        base = base_prices.get(symbol, 1.0)
        
        # Calcola numero di barre
        minutes = {"M1": 1, "M5": 5, "M15": 15, "H1": 60, "H4": 240}
        delta_minutes = (self.config.end_date - self.config.start_date).total_seconds() / 60
        n_bars = int(delta_minutes / minutes.get(timeframe, 1))
        
        # Genera timestamp
        freq = f"{minutes.get(timeframe, 1)}T"
        timestamps = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq=freq
        )[:n_bars]
        
        # Random walk
        np.random.seed(42)
        returns = np.random.normal(0, 0.001, n_bars)
        close_prices = base * np.exp(np.cumsum(returns))
        
        # OHLC
        high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.0005, n_bars)))
        low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.0005, n_bars)))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = base
        
        volume = np.random.randint(100, 1000, n_bars)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        })
        
        return df
    
    def _preprocess_data(self) -> bool:
        """Resample tutti i timeframes."""
        try:
            for symbol in self.config.symbols:
                df_m1 = self.historical_data[symbol]["M1"]
                
                # Resample a tutti i TF
                resampled = self.resampler.resample_all_timeframes(df_m1)
                
                for tf, df in resampled.items():
                    if df is not None and not df.empty:
                        self.historical_data[symbol][tf] = df
                        logger.info(f"Resampled {symbol} {tf}: {len(df)} bars")
            
            return True
        
        except Exception as e:
            logger.error(f"Errore preprocessing: {e}")
            return False
    
    def _run_simulation(self):
        """Loop principale simulazione."""
        # Usa primary TF come driver
        primary_tf = self.config.primary_timeframe
        
        # Trova simbolo con più dati per primary TF
        max_bars = 0
        driver_symbol = self.config.symbols[0]
        for symbol in self.config.symbols:
            if symbol in self.historical_data and primary_tf in self.historical_data[symbol]:
                n_bars = len(self.historical_data[symbol][primary_tf])
                if n_bars > max_bars:
                    max_bars = n_bars
                    driver_symbol = symbol
        
        df_driver = self.historical_data[driver_symbol][primary_tf]
        logger.info(f"Simulazione: {len(df_driver)} barre {primary_tf} su {driver_symbol}")
        
        # Itera su ogni barra
        for idx in range(100, len(df_driver)):  # Skip primi 100 per warmup indicatori
            current_bar = df_driver.iloc[idx]
            self.current_time = current_bar['timestamp']
            
            # Update equity curve
            self._update_equity()
            self.equity_curve.append((self.current_time, self.equity))
            
            # Process ogni simbolo
            for symbol in self.config.symbols:
                self._process_symbol(symbol, idx)
            
            # Log periodico
            if idx % 1000 == 0:
                logger.debug(f"Bar {idx}/{len(df_driver)}: Equity={self.equity:.2f}, Open={len(self.open_trades)}")
    
    def _process_symbol(self, symbol: str, bar_idx: int):
        """Processa un simbolo alla barra corrente."""
        try:
            # 1. Ottieni dati per tutti i TF fino a bar corrente
            tf_data = {}
            for tf in self.config.timeframes:
                if tf not in self.historical_data[symbol]:
                    continue
                
                df = self.historical_data[symbol][tf]
                # Trova indice corrispondente (timestamp <= current_time)
                mask = df['timestamp'] <= self.current_time
                df_subset = df[mask].copy()
                
                if not df_subset.empty:
                    tf_data[tf] = df_subset
            
            if not tf_data:
                return
            
            # 2. Calcola features per ogni TF
            tf_states = {}
            for tf, df in tf_data.items():
                # Indicators
                df_ind = self.indicators.calculate_all_indicators(df, tf)
                
                # Structure
                structure = self.structure.analyze_structure(df_ind)
                
                # Volatility
                vol_state = self.volatility.get_volatility_state(df_ind)
                
                # ML inference (su latest bar)
                regime = self.ml_inference.predict_regime(df_ind)
                trend = self.ml_inference.predict_trend(df_ind)
                vol_pred = self.ml_inference.predict_volatility(df_ind)
                
                # Create TFState
                tf_state = TFState(
                    timeframe=tf,
                    last_update=self.current_time,
                    ohlcv=df_ind.iloc[-1].to_dict() if not df_ind.empty else {},
                    indicators=df_ind.iloc[-1].to_dict() if not df_ind.empty else {},
                    structure=structure,
                    ml_regime=regime,
                    ml_trend_strength=trend,
                    ml_volatility=vol_pred,
                    volatility_state=vol_state
                )
                tf_states[tf] = tf_state
            
            # 3. Update state store
            symbol_state = SymbolState(
                symbol=symbol,
                timeframes=tf_states,
                last_update=self.current_time
            )
            self.state_store.update_symbol_state(symbol, symbol_state)
            
            # 4. Gestione posizioni aperte (check SL/TP)
            if symbol in self.open_trades:
                self._check_open_trade(symbol, tf_data)
            
            # 5. Genera decisioni se non abbiamo posizione aperta
            if symbol not in self.open_trades:
                self._check_entry_signals(symbol, tf_data, tf_states)
        
        except Exception as e:
            logger.error(f"Errore processing {symbol}: {e}")
    
    def _check_open_trade(self, symbol: str, tf_data: Dict[str, pd.DataFrame]):
        """Verifica SL/TP/exit signal per trade aperto."""
        trade = self.open_trades[symbol]
        
        # Ottieni prezzo corrente (close primary TF)
        primary_tf = self.config.primary_timeframe
        if primary_tf not in tf_data:
            return
        
        df = tf_data[primary_tf]
        if df.empty:
            return
        
        current_bar = df.iloc[-1]
        current_price = current_bar['close']
        high = current_bar['high']
        low = current_bar['low']
        
        # Update MAE/MFE
        if trade.side == "buy":
            pnl = current_price - trade.entry_price
            mae = low - trade.entry_price
            mfe = high - trade.entry_price
        else:  # sell
            pnl = trade.entry_price - current_price
            mae = trade.entry_price - high
            mfe = trade.entry_price - low
        
        trade.max_adverse = min(trade.max_adverse, mae)
        trade.max_favorable = max(trade.max_favorable, mfe)
        
        # Check SL
        if trade.stop_loss is not None:
            if trade.side == "buy" and low <= trade.stop_loss:
                self._close_trade(symbol, trade.stop_loss, "sl")
                return
            elif trade.side == "sell" and high >= trade.stop_loss:
                self._close_trade(symbol, trade.stop_loss, "sl")
                return
        
        # Check TP
        if trade.take_profit is not None:
            if trade.side == "buy" and high >= trade.take_profit:
                self._close_trade(symbol, trade.take_profit, "tp")
                return
            elif trade.side == "sell" and low <= trade.take_profit:
                self._close_trade(symbol, trade.take_profit, "tp")
                return
        
        # Check exit signal (run decision engine)
        symbol_state = self.state_store.get_symbol_state(symbol)
        if symbol_state:
            decisions = self.decision_engine.generate_decisions(symbol, symbol_state)
            
            for decision in decisions:
                if decision['action'] == 'exit' and decision['symbol'] == symbol:
                    self._close_trade(symbol, current_price, "signal")
                    return
    
    def _check_entry_signals(
        self,
        symbol: str,
        tf_data: Dict[str, pd.DataFrame],
        tf_states: Dict[str, TFState]
    ):
        """Verifica segnali di entrata."""
        # Ottieni state
        symbol_state = self.state_store.get_symbol_state(symbol)
        if not symbol_state:
            return
        
        # Genera decisioni
        decisions = self.decision_engine.generate_decisions(symbol, symbol_state)
        
        for decision in decisions:
            if decision['action'] != 'entry':
                continue
            
            # Check risk limits
            if not self.risk_manager.check_risk_limits([decision]):
                continue
            
            # Execute entry
            primary_tf = self.config.primary_timeframe
            if primary_tf not in tf_data:
                continue
            
            df = tf_data[primary_tf]
            current_bar = df.iloc[-1]
            entry_price = current_bar['close']
            
            # Apply slippage
            if self.config.slippage > 0:
                if decision['side'] == 'buy':
                    entry_price *= (1 + self.config.slippage)
                else:
                    entry_price *= (1 - self.config.slippage)
            
            # Create trade
            trade = BacktestTrade(
                symbol=symbol,
                entry_time=self.current_time,
                side=decision['side'],
                entry_price=entry_price,
                volume=decision['volume'],
                stop_loss=decision.get('stop_loss'),
                take_profit=decision.get('take_profit'),
                commission=decision['volume'] * entry_price * self.config.commission * 2  # entry + exit
            )
            
            self.open_trades[symbol] = trade
            
            # Update risk manager
            position = Position(
                symbol=symbol,
                side=decision['side'],
                volume=decision['volume'],
                entry_price=entry_price,
                current_price=entry_price,
                stop_loss=decision.get('stop_loss'),
                take_profit=decision.get('take_profit'),
                open_time=self.current_time,
                unrealized_pnl=0.0
            )
            self.risk_manager.open_positions[symbol] = position
            
            logger.debug(f"Aperto {symbol} {decision['side']} @ {entry_price:.5f}")
    
    def _close_trade(self, symbol: str, exit_price: float, reason: str):
        """Chiude trade."""
        if symbol not in self.open_trades:
            return
        
        trade = self.open_trades[symbol]
        trade.exit_time = self.current_time
        trade.exit_price = exit_price
        trade.exit_reason = reason
        
        # Calcola P&L
        if trade.side == "buy":
            trade.pnl = (exit_price - trade.entry_price) * trade.volume
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.volume
        
        trade.pnl -= trade.commission
        trade.pnl_pct = (trade.pnl / (trade.entry_price * trade.volume)) * 100
        
        # Bars held
        trade.bars_held = int((trade.exit_time - trade.entry_time).total_seconds() / 3600)  # ore
        
        # Update capital
        self.capital += trade.pnl
        
        # Update risk manager
        self.risk_manager.update_position_result(symbol, trade.pnl)
        if symbol in self.risk_manager.open_positions:
            del self.risk_manager.open_positions[symbol]
        
        # Move to closed
        self.closed_trades.append(trade)
        del self.open_trades[symbol]
        
        logger.debug(f"Chiuso {symbol} {reason} @ {exit_price:.5f}: PnL={trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
    
    def _update_equity(self):
        """Aggiorna equity includendo unrealized PnL."""
        unrealized_pnl = 0.0
        
        for symbol, trade in self.open_trades.items():
            # Get current price
            primary_tf = self.config.primary_timeframe
            if symbol not in self.historical_data or primary_tf not in self.historical_data[symbol]:
                continue
            
            df = self.historical_data[symbol][primary_tf]
            mask = df['timestamp'] <= self.current_time
            df_subset = df[mask]
            
            if df_subset.empty:
                continue
            
            current_price = df_subset.iloc[-1]['close']
            
            # Calcola unrealized
            if trade.side == "buy":
                pnl = (current_price - trade.entry_price) * trade.volume
            else:
                pnl = (trade.entry_price - current_price) * trade.volume
            
            unrealized_pnl += pnl
        
        self.equity = self.capital + unrealized_pnl
    
    def _close_all_open_trades(self):
        """Chiude tutti i trade aperti a fine backtest."""
        symbols_to_close = list(self.open_trades.keys())
        
        for symbol in symbols_to_close:
            # Usa ultimo prezzo disponibile
            primary_tf = self.config.primary_timeframe
            if symbol in self.historical_data and primary_tf in self.historical_data[symbol]:
                df = self.historical_data[symbol][primary_tf]
                if not df.empty:
                    exit_price = df.iloc[-1]['close']
                    self._close_trade(symbol, exit_price, "end_of_backtest")
    
    def _calculate_results(self) -> Dict:
        """Calcola risultati finali."""
        # Converti trades in formato per MetricsCalculator
        trade_history = []
        for trade in self.closed_trades:
            if trade.exit_time and trade.pnl is not None:
                trade_history.append({
                    'symbol': trade.symbol,
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'side': trade.side,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'volume': trade.volume,
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl_pct,
                    'commission': trade.commission,
                    'exit_reason': trade.exit_reason
                })
        
        # Calcola metriche
        metrics = self.metrics_calc.calculate_metrics(trade_history)
        
        # Equity curve
        equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        
        # Summary
        results = {
            'config': {
                'start_date': self.config.start_date.isoformat(),
                'end_date': self.config.end_date.isoformat(),
                'initial_capital': self.config.initial_capital,
                'symbols': self.config.symbols,
                'timeframes': self.config.timeframes,
                'commission': self.config.commission,
                'slippage': self.config.slippage
            },
            'summary': {
                'final_capital': self.capital,
                'final_equity': self.equity,
                'total_return': ((self.equity - self.config.initial_capital) / self.config.initial_capital) * 100,
                'total_trades': len(self.closed_trades),
                'open_trades': len(self.open_trades)
            },
            'metrics': metrics,
            'trades': trade_history,
            'equity_curve': equity_df.to_dict('records')
        }
        
        logger.info(f"Backtest Results: Return={results['summary']['total_return']:.2f}%, Trades={results['summary']['total_trades']}")
        
        return results
    
    def export_results(self, results: Dict, output_dir: str = "backtest_results"):
        """Esporta risultati in file."""
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON summary
        summary_path = os.path.join(output_dir, f"backtest_summary_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump({
                'config': results['config'],
                'summary': results['summary'],
                'metrics': results['metrics']
            }, f, indent=2)
        
        # CSV trades
        trades_path = os.path.join(output_dir, f"backtest_trades_{timestamp}.csv")
        df_trades = pd.DataFrame(results['trades'])
        df_trades.to_csv(trades_path, index=False)
        
        # CSV equity curve
        equity_path = os.path.join(output_dir, f"backtest_equity_{timestamp}.csv")
        df_equity = pd.DataFrame(results['equity_curve'])
        df_equity.to_csv(equity_path, index=False)
        
        logger.info(f"Risultati esportati in {output_dir}")
        return {
            'summary': summary_path,
            'trades': trades_path,
            'equity': equity_path
        }
