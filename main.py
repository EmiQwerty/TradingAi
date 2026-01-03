"""
Trading System Main Orchestrator
Punto di ingresso centrale per il sistema di trading multi-simbolo/multi-TF.
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List
import yaml
from pathlib import Path

# Imports moduli
from data.market_feed import MarketFeed
from data.resampler import TimeframeResampler
from data.storage import DataStorage
from features.indicators import TechnicalIndicators
from features.structure import StructureDetector
from features.volatility import VolatilityAnalyzer
from ml.inference import MLInference
from macro.macro_filter import MacroFilter
from strategy.confidence import ConfidenceAggregator
from strategy.decision_engine import DecisionEngine
from risk.risk_manager import RiskManager
from risk.correlation import CorrelationAnalyzer
from execution.broker_api import BrokerAPI
from execution.orders import OrderExecutor
from state.state_store import StateStore
from state.models import TFState, SymbolState, Position
from monitoring.metrics import MetricsCalculator
from monitoring.events import EventLogger
from backtest.engine import BacktestEngine, BacktestConfig
from backtest.walk_forward import WalkForwardValidator


# Setup logging
def setup_logging(log_config: Dict):
    """Configura logging system."""
    import colorlog
    
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        log_format,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    ))
    
    logging.basicConfig(
        level=log_level,
        handlers=[handler]
    )
    
    # File logging
    if log_config.get('file_enabled', True):
        log_dir = Path(log_config.get('directory', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)


logger = logging.getLogger(__name__)


class TradingSystem:
    """
    Sistema di trading completo.
    Orchestrator principale che coordina tutti i moduli.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Inizializza sistema di trading.
        
        Args:
            config_dir: Directory configurazioni
        """
        self.config_dir = Path(config_dir)
        self.running = False
        self.shutdown_requested = False
        
        # Load configs
        self.settings = self._load_yaml(self.config_dir / "settings.yaml")
        self.symbol_configs = self._load_yaml(self.config_dir / "symbols.yaml")
        self.risk_config = self._load_yaml(self.config_dir / "risk.yaml")
        
        # Setup logging
        setup_logging(self.settings.get('logging', {}))
        
        # Extract trading config
        trading_config = self.settings.get('trading', {})
        self.symbols = trading_config.get('symbols', ['EURUSD', 'GBPUSD', 'XAUUSD'])
        self.timeframes = trading_config.get('timeframes', ['M1', 'M5', 'M15', 'H1', 'H4'])
        self.primary_timeframe = trading_config.get('primary_timeframe', 'H1')
        self.update_interval = trading_config.get('update_interval_seconds', 60)
        
        # Inizializza componenti
        self._init_components()
        
        logger.info(f"TradingSystem inizializzato: {len(self.symbols)} simboli, {len(self.timeframes)} timeframes")
    
    def _load_yaml(self, path: Path) -> Dict:
        """Carica file YAML."""
        with open(path) as f:
            return yaml.safe_load(f)
    
    def _init_components(self):
        """Inizializza tutti i componenti del sistema."""
        logger.info("Inizializzazione componenti...")
        
        # Data layer
        self.market_feed = MarketFeed(self.settings.get('broker', {}))
        self.resampler = TimeframeResampler(self.timeframes)
        self.storage = DataStorage(self.settings.get('storage', {}))
        
        # Features
        self.indicators = TechnicalIndicators(self.settings.get('indicators', {}))
        self.structure = StructureDetector(self.settings.get('structure', {}))
        self.volatility = VolatilityAnalyzer()
        
        # ML & Macro
        self.ml_inference = MLInference(self.settings.get('ml', {}))
        self.macro_filter = MacroFilter(self.settings.get('macro', {}))
        
        # Strategy
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
        
        # Risk management
        initial_capital = self.settings.get('trading', {}).get('initial_capital', 100000.0)
        self.risk_manager = RiskManager(self.risk_config, initial_capital)
        self.correlation = CorrelationAnalyzer()
        
        # Execution
        self.broker_api = BrokerAPI(self.settings.get('broker', {}))
        self.order_executor = OrderExecutor(self.broker_api, self.settings.get('execution', {}))
        
        # State & Monitoring
        self.state_store = StateStore()
        self.metrics_calc = MetricsCalculator()
        self.event_logger = EventLogger()
        
        logger.info("Componenti inizializzati")
    
    def start(self, mode: str = "demo"):
        """
        Avvia sistema di trading.
        
        Args:
            mode: Modalità operativa (demo/live)
        """
        if self.running:
            logger.warning("Sistema già in esecuzione")
            return
        
        logger.info(f"Avvio sistema in modalità {mode.upper()}")
        self.running = True
        self.shutdown_requested = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Connetti broker
        if not self.broker_api.connect():
            logger.error("Impossibile connettersi al broker")
            self.running = False
            return
        
        # Update account info
        account_info = self.broker_api.get_account_info()
        self.state_store.update_account_info(account_info)
        self.event_logger.log_system_event("system_started", {"mode": mode, "account": account_info})
        
        # Update risk manager con capitale reale
        if account_info and 'balance' in account_info:
            self.risk_manager.initial_capital = account_info['balance']
            self.risk_manager.current_capital = account_info['balance']
        
        # Carica posizioni aperte dal broker
        self._sync_positions()
        
        # Main loop
        try:
            self._main_loop()
        except Exception as e:
            logger.error(f"Errore main loop: {e}", exc_info=True)
            self.event_logger.log_error("main_loop_error", str(e))
        finally:
            self.shutdown()
    
    def _main_loop(self):
        """Loop principale del sistema."""
        logger.info("Main loop avviato")
        iteration = 0
        
        while self.running and not self.shutdown_requested:
            iteration += 1
            cycle_start = time.time()
            
            try:
                # 1. Update market data per tutti i simboli
                for symbol in self.symbols:
                    self._update_symbol(symbol)
                
                # 2. Update correlation matrix
                self._update_correlations()
                
                # 3. Update risk metrics
                self._update_risk_metrics()
                
                # 4. Log status periodico
                if iteration % 10 == 0:
                    self._log_status()
                
            except Exception as e:
                logger.error(f"Errore ciclo {iteration}: {e}", exc_info=True)
                self.event_logger.log_error("cycle_error", str(e))
            
            # Sleep fino a prossimo update
            cycle_time = time.time() - cycle_start
            sleep_time = max(0, self.update_interval - cycle_time)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        logger.info("Main loop terminato")
    
    def _update_symbol(self, symbol: str):
        """
        Update completo per un simbolo.
        
        Pipeline:
        1. Fetch latest data (M1)
        2. Resample a tutti i TF
        3. Calculate indicators
        4. Detect structure
        5. Analyze volatility
        6. ML inference
        7. Aggregate confidence
        8. Generate decisions
        9. Check risk limits
        10. Execute orders
        """
        try:
            # 1. Fetch M1 data
            candles_m1 = self.market_feed.get_latest_candles(symbol, "M1", limit=500)
            
            if candles_m1 is None or candles_m1.empty:
                logger.warning(f"Nessun dato per {symbol}")
                return
            
            # Save to storage
            self.storage.save_candles(symbol, "M1", candles_m1)
            
            # 2. Resample a tutti i TF
            resampled = self.resampler.resample_all_timeframes(candles_m1)
            
            # 3. Process ogni timeframe
            tf_states = {}
            
            for tf, df in resampled.items():
                if df is None or df.empty:
                    continue
                
                # Save resampled data
                self.storage.save_candles(symbol, tf, df)
                
                # Calculate indicators
                df_indicators = self.indicators.calculate_all_indicators(df, tf)
                
                # Save indicators
                self.storage.save_indicators(symbol, tf, df_indicators)
                
                # Detect structure
                structure = self.structure.analyze_structure(df_indicators)
                
                # Analyze volatility
                vol_state = self.volatility.get_volatility_state(df_indicators)
                
                # ML inference
                regime = self.ml_inference.predict_regime(df_indicators)
                trend = self.ml_inference.predict_trend(df_indicators)
                vol_pred = self.ml_inference.predict_volatility(df_indicators)
                
                # Create TFState
                latest_bar = df_indicators.iloc[-1].to_dict() if not df_indicators.empty else {}
                
                tf_state = TFState(
                    timeframe=tf,
                    last_update=datetime.now(),
                    ohlcv=latest_bar,
                    indicators=latest_bar,
                    structure=structure,
                    ml_regime=regime,
                    ml_trend_strength=trend,
                    ml_volatility=vol_pred,
                    volatility_state=vol_state
                )
                
                tf_states[tf] = tf_state
            
            # 4. Update state store
            symbol_state = SymbolState(
                symbol=symbol,
                timeframes=tf_states,
                last_update=datetime.now()
            )
            self.state_store.update_symbol_state(symbol, symbol_state)
            
            # 5. Generate decisions
            decisions = self.decision_engine.generate_decisions(symbol, symbol_state)
            
            if not decisions:
                return
            
            # Log signals
            for decision in decisions:
                self.event_logger.log_signal(
                    decision['symbol'],
                    decision['action'],
                    decision.get('side', 'N/A'),
                    decision.get('confidence', 0),
                    decision.get('reasoning', {})
                )
            
            # 6. Check risk limits
            valid_decisions = self.risk_manager.check_risk_limits(decisions)
            
            if not valid_decisions:
                logger.debug(f"Nessuna decisione valida per {symbol} dopo risk check")
                return
            
            # 7. Execute orders
            for decision in valid_decisions:
                self._execute_decision(decision)
        
        except Exception as e:
            logger.error(f"Errore update {symbol}: {e}", exc_info=True)
            self.event_logger.log_error(f"update_error_{symbol}", str(e))
    
    def _execute_decision(self, decision: Dict):
        """Esegue una decisione di trading."""
        try:
            # Execute via order executor
            result = self.order_executor.execute_decision(decision)
            
            if result and result.get('success'):
                # Log trade
                self.event_logger.log_trade(
                    decision['symbol'],
                    decision['action'],
                    decision.get('side', 'N/A'),
                    decision.get('entry_price', 0),
                    decision.get('volume', 0),
                    decision.get('stop_loss'),
                    decision.get('take_profit')
                )
                
                # Update state store (posizioni già aggiornate da OrderExecutor)
                
                logger.info(f"Eseguito: {decision['action']} {decision['symbol']} @ {decision.get('entry_price', 'N/A')}")
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result'
                logger.warning(f"Esecuzione fallita: {error_msg}")
                self.event_logger.log_error("execution_failed", error_msg)
        
        except Exception as e:
            logger.error(f"Errore esecuzione decisione: {e}", exc_info=True)
            self.event_logger.log_error("execution_error", str(e))
    
    def _sync_positions(self):
        """Sincronizza posizioni aperte dal broker."""
        try:
            positions = self.broker_api.get_open_positions()
            
            if not positions:
                logger.info("Nessuna posizione aperta")
                return
            
            logger.info(f"Sincronizzate {len(positions)} posizioni aperte")
            
            for pos in positions:
                position = Position(
                    symbol=pos['symbol'],
                    side=pos['side'],
                    volume=pos['volume'],
                    entry_price=pos['entry_price'],
                    current_price=pos['current_price'],
                    stop_loss=pos.get('stop_loss'),
                    take_profit=pos.get('take_profit'),
                    open_time=pos.get('open_time', datetime.now()),
                    unrealized_pnl=pos.get('unrealized_pnl', 0)
                )
                
                self.state_store.add_position(position)
                self.risk_manager.open_positions[pos['symbol']] = position
        
        except Exception as e:
            logger.error(f"Errore sync posizioni: {e}")
    
    def _update_correlations(self):
        """Update correlation matrix."""
        try:
            # Carica dati storici recenti per tutti i simboli
            historical_data = {}
            
            for symbol in self.symbols:
                df = self.storage.load_candles(
                    symbol,
                    self.primary_timeframe,
                    datetime.now() - timedelta(days=30),
                    datetime.now()
                )
                if df is not None and not df.empty:
                    historical_data[symbol] = df
            
            if len(historical_data) > 1:
                corr_matrix = self.correlation.calculate_correlation_matrix(historical_data)
                # Potresti salvare corr_matrix nello state_store se necessario
        
        except Exception as e:
            logger.debug(f"Errore update correlations: {e}")
    
    def _update_risk_metrics(self):
        """Update risk metrics."""
        try:
            risk_metrics = self.risk_manager.get_risk_metrics()
            self.state_store.update_risk_metrics(risk_metrics)
        except Exception as e:
            logger.debug(f"Errore update risk metrics: {e}")
    
    def _log_status(self):
        """Log status periodico."""
        state_summary = self.state_store.get_state_summary()
        
        logger.info(f"=== Status ===")
        logger.info(f"Simboli attivi: {len(state_summary['symbols'])}")
        logger.info(f"Posizioni aperte: {state_summary['positions']['total']}")
        logger.info(f"Capitale: {state_summary['account']['balance']:.2f}")
        logger.info(f"Equity: {state_summary['account']['equity']:.2f}")
        logger.info(f"Exposure: {state_summary['risk']['current_exposure']:.2f}%")
        logger.info(f"Drawdown: {state_summary['risk']['current_drawdown']:.2f}%")
    
    def _signal_handler(self, signum, frame):
        """Gestione segnali shutdown."""
        logger.info(f"Ricevuto segnale {signum}, shutdown...")
        self.shutdown_requested = True
    
    def shutdown(self):
        """Shutdown graceful del sistema."""
        if not self.running:
            return
        
        logger.info("Shutdown sistema...")
        self.running = False
        
        # Chiudi tutte le posizioni (opzionale, configurabile)
        close_on_shutdown = self.settings.get('trading', {}).get('close_positions_on_shutdown', False)
        
        if close_on_shutdown:
            logger.info("Chiusura posizioni aperte...")
            self.order_executor.close_all_positions()
        
        # Export state
        self.state_store.export_state_to_file()
        
        # Export events
        self.event_logger.export_events_to_file()
        
        # Log final stats
        self.event_logger.log_system_event("system_shutdown", self.state_store.get_state_summary())
        
        logger.info("Sistema arrestato")
    
    def run_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0
    ):
        """
        Esegue backtest.
        
        Args:
            start_date: Data inizio
            end_date: Data fine
            initial_capital: Capitale iniziale
        """
        logger.info(f"Avvio backtest: {start_date.date()} -> {end_date.date()}")
        
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            symbols=self.symbols,
            timeframes=self.timeframes,
            primary_timeframe=self.primary_timeframe
        )
        
        engine = BacktestEngine(
            config,
            self.settings,
            self.symbol_configs,
            self.risk_config
        )
        
        results = engine.run()
        
        # Export results
        paths = engine.export_results(results)
        
        logger.info(f"Backtest completato: {results['summary']['total_trades']} trades")
        logger.info(f"Return: {results['summary']['total_return']:.2f}%")
        logger.info(f"Risultati salvati: {paths}")
        
        return results
    
    def run_walk_forward(
        self,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0
    ):
        """
        Esegue walk-forward validation.
        
        Args:
            start_date: Data inizio
            end_date: Data fine
            initial_capital: Capitale iniziale
        """
        logger.info(f"Avvio walk-forward: {start_date.date()} -> {end_date.date()}")
        
        validator = WalkForwardValidator(
            self.settings,
            self.symbol_configs,
            self.risk_config
        )
        
        results = validator.run(
            start_date=start_date,
            end_date=end_date,
            symbols=self.symbols,
            timeframes=self.timeframes,
            initial_capital=initial_capital
        )
        
        # Export
        paths = validator.export_results()
        validator.plot_results()
        
        logger.info(f"Walk-forward completato: {results['summary']['num_windows']} windows")
        logger.info(f"Combined return: {results['summary']['total_return']:.2f}%")
        logger.info(f"Stability score: {results['summary']['stability_score']:.2f}")
        logger.info(f"Risultati salvati: {paths}")
        
        return results


def main():
    """Entry point principale."""
    parser = argparse.ArgumentParser(description="Trading System Multi-Symbol/Multi-TF")
    parser.add_argument(
        '--mode',
        choices=['demo', 'live', 'backtest', 'walk-forward'],
        default='demo',
        help="Modalità operativa"
    )
    parser.add_argument(
        '--config',
        default='config',
        help="Directory configurazioni"
    )
    parser.add_argument(
        '--start-date',
        type=lambda s: datetime.fromisoformat(s),
        help="Data inizio (per backtest/walk-forward, formato: YYYY-MM-DD)"
    )
    parser.add_argument(
        '--end-date',
        type=lambda s: datetime.fromisoformat(s),
        help="Data fine (per backtest/walk-forward, formato: YYYY-MM-DD)"
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=100000.0,
        help="Capitale iniziale"
    )
    
    args = parser.parse_args()
    
    # Crea sistema
    system = TradingSystem(config_dir=args.config)
    
    # Esegui modalità richiesta
    if args.mode in ['demo', 'live']:
        system.start(mode=args.mode)
    
    elif args.mode == 'backtest':
        if not args.start_date or not args.end_date:
            print("Errore: --start-date e --end-date richiesti per backtest")
            sys.exit(1)
        
        system.run_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.capital
        )
    
    elif args.mode == 'walk-forward':
        if not args.start_date or not args.end_date:
            print("Errore: --start-date e --end-date richiesti per walk-forward")
            sys.exit(1)
        
        system.run_walk_forward(
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.capital
        )


if __name__ == "__main__":
    main()
