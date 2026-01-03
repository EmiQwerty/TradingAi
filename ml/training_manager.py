"""
Training Manager - Manage ML pipeline training with progress tracking and monitoring
Supports real-time callbacks for dashboard monitoring
NOW WITH PROFESSIONAL STRATEGIES!
"""

import pandas as pd
import numpy as np
from typing import Dict, Callable, Optional, Tuple, List
import logging
from pathlib import Path
from datetime import datetime
import json
import threading
import time
from dataclasses import dataclass, asdict
from enum import Enum
import sys
import os

# Import strategies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from strategies.strategy_ensemble import StrategyEnsemble
from strategies.rsi_strategy import RSIMeanReversionStrategy
from strategies.breakout_strategy import BreakoutStrategy
from strategies.trend_following_strategy import TrendFollowingStrategy

logger = logging.getLogger(__name__)


class TrainingStatus(Enum):
    """Training status states"""
    IDLE = "idle"
    LOADING_DATA = "loading_data"
    FEATURE_EXTRACTION = "feature_extraction"
    BACKTESTING = "backtesting"
    MODEL_TRAINING = "model_training"
    CONTEXT_ANALYSIS = "context_analysis"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class TrainingProgress:
    """Training progress data"""
    status: TrainingStatus
    current_step: int
    total_steps: int
    percentage: float
    message: str
    timestamp: datetime
    elapsed_seconds: float
    estimated_remaining_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'status': self.status.value,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'percentage': self.percentage,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'elapsed_seconds': self.elapsed_seconds,
            'estimated_remaining_seconds': self.estimated_remaining_seconds,
        }


@dataclass
class TrainingMetrics:
    """Training results metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win_pnl: float = 0.0
    avg_loss_pnl: float = 0.0
    
    # Model metrics
    best_model: str = ""
    model_accuracy: float = 0.0
    model_f1: float = 0.0
    model_roc_auc: float = 0.0
    
    # Feature metrics
    top_features: List[Tuple[str, float]] = None
    feature_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        if self.top_features:
            data['top_features'] = [(f, float(s)) for f, s in self.top_features]
        return data


class TrainingManager:
    """
    Manages ML pipeline training with progress monitoring.
    Supports callbacks for real-time dashboard updates.
    NOW WITH STRATEGY-DRIVEN BACKTESTING!
    """
    
    def __init__(self, 
                 pipeline=None,
                 data_dir: str = 'data',
                 results_dir: str = 'ml/training_results',
                 strategy_mode: str = "WEIGHTED",
                 selected_strategies: List[str] = None):
        """
        Args:
            pipeline: MLPipeline instance
            data_dir: Directory with training data
            results_dir: Directory to save training results
            strategy_mode: VOTING | WEIGHTED | ALL | SINGLE
            selected_strategies: Lista strategie ['RSI', 'Breakout', 'TrendFollowing']
        """
        self.pipeline = pipeline
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Strategy configuration
        self.strategy_mode = strategy_mode
        self.selected_strategies = selected_strategies or ['RSI', 'Breakout', 'TrendFollowing']
        self.strategy_ensemble = None
        
        # State
        self.is_training = False
        self.progress = TrainingProgress(
            status=TrainingStatus.IDLE,
            current_step=0,
            total_steps=100,
            percentage=0,
            message="Ready to train",
            timestamp=datetime.now(),
            elapsed_seconds=0
        )
        self.metrics = TrainingMetrics()
        self.training_history = []
        
        # Callbacks
        self.progress_callbacks: List[Callable] = []
        self.metrics_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        # Timing
        self.start_time = None
        self.step_start_time = None
        
        logger.info("TrainingManager initialized")
    
    def add_progress_callback(self, callback: Callable):
        """Register callback for progress updates"""
        self.progress_callbacks.append(callback)
    
    def add_metrics_callback(self, callback: Callable):
        """Register callback for metrics updates"""
        self.metrics_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """Register callback for error notifications"""
        self.error_callbacks.append(callback)
    
    def _update_progress(self, status: TrainingStatus, 
                        current_step: int, total_steps: int,
                        message: str):
        """Update training progress"""
        
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        percentage = (current_step / total_steps * 100) if total_steps > 0 else 0
        
        # Estimate remaining time
        if current_step > 0 and elapsed > 0:
            rate = elapsed / current_step
            remaining = rate * (total_steps - current_step)
        else:
            remaining = None
        
        self.progress = TrainingProgress(
            status=status,
            current_step=current_step,
            total_steps=total_steps,
            percentage=percentage,
            message=message,
            timestamp=datetime.now(),
            elapsed_seconds=elapsed,
            estimated_remaining_seconds=remaining
        )
        
        # Notify callbacks
        for callback in self.progress_callbacks:
            try:
                callback(self.progress)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
    
    def _update_metrics(self, metrics: TrainingMetrics):
        """Update training metrics"""
        self.metrics = metrics
        
        # Notify callbacks
        for callback in self.metrics_callbacks:
            try:
                callback(self.metrics)
            except Exception as e:
                logger.error(f"Metrics callback error: {e}")
    
    def _notify_error(self, error: str, details: str = ""):
        """Notify error callbacks"""
        for callback in self.error_callbacks:
            try:
                callback(error, details)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")
    
    def train(self, symbol: str = 'EUR_USD',
              historical_data: Optional[pd.DataFrame] = None,
              data_file: Optional[str] = None,
              test_size: float = 0.2,
              cv_folds: int = 5,
              strategy_mode: str = "WEIGHTED",
              selected_strategies: List[str] = None) -> bool:
        """
        Execute training with STRATEGY-DRIVEN backtesting.
        
        Args:
            symbol: Trading symbol
            historical_data: DataFrame with OHLCV (if not loading from file)
            data_file: CSV file with historical data
            test_size: Test set proportion
            cv_folds: Cross-validation folds
            strategy_mode: VOTING | WEIGHTED | ALL | SINGLE
            selected_strategies: List of strategies ['RSI', 'Breakout', 'TrendFollowing']
        
        Returns:
            True if training succeeded, False otherwise
        """
        
        # Update strategy config
        self.strategy_mode = strategy_mode
        self.selected_strategies = selected_strategies or ['RSI', 'Breakout', 'TrendFollowing']
        
        if self.is_training:
            self._notify_error("Training already in progress")
            return False
        
        self.is_training = True
        self.start_time = datetime.now()
        self.training_history = []
        
        try:
            # Step 1: Load data
            self._update_progress(
                TrainingStatus.LOADING_DATA, 1, 6,
                f"Loading historical data for {symbol}..."
            )
            
            if historical_data is None:
                if data_file:
                    historical_data = pd.read_csv(data_file)
                else:
                    # Try to find data file
                    data_files = list(self.data_dir.glob(f"*{symbol}*.csv"))
                    if data_files:
                        historical_data = pd.read_csv(data_files[0])
                    else:
                        raise FileNotFoundError(f"No data found for {symbol}")
            
            self.training_history.append({
                'step': 'data_loaded',
                'timestamp': datetime.now().isoformat(),
                'samples': len(historical_data)
            })
            
            # Step 2: Feature Extraction
            self._update_progress(
                TrainingStatus.FEATURE_EXTRACTION, 2, 6,
                "Extracting 40+ technical features..."
            )
            
            features_list = self._extract_features(historical_data, symbol)
            features_df = pd.DataFrame(features_list)
            
            self.training_history.append({
                'step': 'features_extracted',
                'timestamp': datetime.now().isoformat(),
                'feature_count': len(features_df.columns)
            })
            
            # Step 3: Backtesting WITH STRATEGIES
            self._update_progress(
                TrainingStatus.BACKTESTING, 3, 6,
                f"Running STRATEGY-DRIVEN backtest on {symbol}..."
            )
            
            # Initialize strategy ensemble
            self._initialize_strategies()
            
            trades_df = self._run_strategy_backtest(historical_data, features_df, symbol)

            if trades_df is None or trades_df.empty:
                logger.error("No trades generated - strategies didn't produce signals")
                self._notify_error("Training failed", 
                                 f"Strategies generated ZERO trades. Possible causes:\n"
                                 f"- Data range too short (try 1-2 years)\n"
                                 f"- Timeframe too large (try H1 or H4)\n"
                                 f"- Market conditions don't match strategy criteria")
                return False
            
            self.training_history.append({
                'step': 'backtest_complete',
                'timestamp': datetime.now().isoformat(),
                'trades_count': len(trades_df),
                'strategy_mode': self.strategy_mode,
                'strategies_used': self.selected_strategies,
                'win_rate': float((trades_df['label_win_loss'] == 1).sum() / len(trades_df)) if (len(trades_df) > 0 and 'label_win_loss' in trades_df.columns) else 0.0
            })
            
            # Step 4: Model Training
            self._update_progress(
                TrainingStatus.MODEL_TRAINING, 4, 6,
                "Training 4 ML models (RandomForest, GB, XGBoost, LightGBM)..."
            )
            
            if 'label_win_loss' not in trades_df.columns:
                logger.error("Trades dataframe missing label_win_loss column")
                self._notify_error("Training failed", "Backtest did not produce labels. Try increasing data range/timeframe.")
                return False

            training_results = self._train_models(
                trades_df, test_size=test_size, cv_folds=cv_folds
            )
            
            self.training_history.append({
                'step': 'models_trained',
                'timestamp': datetime.now().isoformat(),
                'models_count': len(training_results)
            })
            
            # Step 5: Context Analysis
            self._update_progress(
                TrainingStatus.CONTEXT_ANALYSIS, 5, 6,
                "Analyzing trading context - why trades win/lose..."
            )
            
            analysis = self._analyze_context(trades_df)
            
            self.training_history.append({
                'step': 'analysis_complete',
                'timestamp': datetime.now().isoformat()
            })
            
            # Step 6: Results
            self._update_progress(
                TrainingStatus.COMPLETED, 6, 6,
                "Training completed successfully!"
            )
            
            # Update metrics
            self._update_metrics_from_results(
                trades_df, training_results, analysis
            )
            
            # Save results
            self._save_training_results(
                symbol, trades_df, training_results, analysis
            )
            
            self.training_history.append({
                'step': 'completed',
                'timestamp': datetime.now().isoformat(),
                'total_seconds': (datetime.now() - self.start_time).total_seconds()
            })
            
            logger.info("Training completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self._notify_error(
                "Training failed",
                str(e)
            )
            self._update_progress(
                TrainingStatus.FAILED, -1, -1,
                f"Training failed: {str(e)}"
            )
            return False
        
        finally:
            self.is_training = False
    
    def _extract_features(self, data: pd.DataFrame, symbol: str) -> List[Dict]:
        """Extract features from historical data"""
        from ml.feature_engineer import FeatureEngineer
        
        fe = FeatureEngineer()
        features_list = []
        
        min_lookback = 50
        for i in range(min_lookback, len(data)):
            data_slice = data.iloc[:i+1].copy()
            entry_price = data.iloc[i]['close'] if 'close' in data.columns else data.iloc[i][3]
            
            try:
                features = fe.extract_all_features(
                    data_slice, symbol, entry_price, 'BUY'
                )
                features['timestamp'] = data.iloc[i].get('time', i)
                features['entry_price'] = entry_price
                features_list.append(features)
            except Exception as e:
                logger.warning(f"Feature extraction error at row {i}: {e}")
                continue
        
        return features_list
    
    def _run_backtest(self, data: pd.DataFrame, 
                     features_df: pd.DataFrame,
                     symbol: str) -> pd.DataFrame:
        """Run backtest"""
        from ml.backtest_labeled import BacktestEngine
        
        be = BacktestEngine()
        
        pip_value = 0.0001
        stop_loss_pips = 50
        take_profit_pips = 100
        
        for i in range(len(features_df)):
            if i >= len(data) - 100:
                continue
            
            entry_price = features_df.iloc[i].get('entry_price', data.iloc[i][3])
            
            try:
                trade = be.simulate_trade(
                    symbol=symbol,
                    side='BUY',
                    entry_price=entry_price,
                    entry_time=data.iloc[i].get('time', i),
                    stop_loss=entry_price - (stop_loss_pips * pip_value),
                    take_profit=entry_price + (take_profit_pips * pip_value),
                    historical_data=data,
                    starting_row_index=i,
                    entry_features={
                        'rsi_current': features_df.iloc[i].get('rsi_current', 50),
                        'macd_histogram': features_df.iloc[i].get('macd_histogram', 0),
                        'trend_type': features_df.iloc[i].get('trend_type', 'NEUTRAL'),
                        'volatility_atr': features_df.iloc[i].get('volatility_atr', 0),
                        'signal_confluence': features_df.iloc[i].get('signal_confluence', 0),
                    }
                )
            except Exception as e:
                logger.debug(f"Backtest error at row {i}: {e}")
                continue
        
        # Get trades dataframe
        trades_df = be.get_trades_dataframe()

        if trades_df.empty:
            logger.warning("Backtest produced zero trades")
            return trades_df
        
        # Merge features with trades (matching by index)
        if len(trades_df) > 0 and len(features_df) > 0:
            # Add features to trades (use first N trades to match features)
            for col in features_df.columns:
                if col not in ['timestamp', 'entry_price', 'symbol']:
                    # Match trades with features by order (simplified)
                    feature_values = features_df[col].iloc[:len(trades_df)].values
                    if len(feature_values) == len(trades_df):
                        trades_df[col] = feature_values
        
        return trades_df
    
    def _initialize_strategies(self):
        """Initialize strategy ensemble based on configuration"""
        if self.strategy_mode == "SINGLE":
            # Single strategy mode
            if 'RSI' in self.selected_strategies:
                self.strategy_ensemble = RSIMeanReversionStrategy()
            elif 'Breakout' in self.selected_strategies:
                self.strategy_ensemble = BreakoutStrategy()
            elif 'TrendFollowing' in self.selected_strategies:
                self.strategy_ensemble = TrendFollowingStrategy()
            else:
                # Default to RSI
                self.strategy_ensemble = RSIMeanReversionStrategy()
        else:
            # Ensemble mode (VOTING, WEIGHTED, ALL)
            self.strategy_ensemble = StrategyEnsemble(
                mode=self.strategy_mode,
                min_votes=2 if self.strategy_mode == "VOTING" else 1
            )
    
    def _run_strategy_backtest(self, data: pd.DataFrame, features_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        NEW: Strategy-driven backtest
        Only trades when strategies generate signals!
        """
        logger.info(f"Running strategy-driven backtest with mode: {self.strategy_mode}")
        
        # Generate signals from strategies
        signals = self.strategy_ensemble.generate_signals(data)
        
        if not signals:
            logger.warning("⚠️ Strategies generated ZERO signals")
            return pd.DataFrame()
        
        logger.info(f"✅ Strategies generated {len(signals)} signals")
        
        # Simulate each signal
        from ml.backtest_engine import BacktestEngine
        backtest_engine = BacktestEngine(initial_capital=10000.0, risk_per_trade=0.02)
        
        trades = []
        for signal in signals:
            # Find signal in data
            signal_idx = None
            if isinstance(signal.timestamp, pd.Timestamp):
                try:
                    signal_idx = data.index.get_loc(signal.timestamp)
                except KeyError:
                    # Try finding nearest timestamp
                    time_diffs = abs(data.index - signal.timestamp)
                    signal_idx = time_diffs.argmin()
            else:
                signal_idx = signal.timestamp
            
            if signal_idx is None or signal_idx >= len(data):
                continue
            
            # Simulate trade from this signal
            trade_result = self._simulate_signal_trade(
                signal, data, signal_idx, symbol, backtest_engine
            )
            
            if trade_result:
                trades.append(trade_result)
        
        if not trades:
            logger.warning("⚠️ No valid trades after simulation")
            return pd.DataFrame()
        
        trades_df = pd.DataFrame(trades)
        
        # Add labels
        trades_df['label_win_loss'] = (trades_df['pnl'] > 0).astype(int)
        trades_df['label_magnitude'] = pd.cut(
            trades_df['pnl_percent'],
            bins=[-float('inf'), -2, -0.5, 0.5, 2, float('inf')],
            labels=['big_loss', 'small_loss', 'breakeven', 'small_win', 'big_win']
        )
        
        # Merge features
        if len(trades_df) > 0 and len(features_df) > 0:
            # Match by entry_time
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            features_df_copy = features_df.copy()
            
            # Merge on closest timestamp
            for idx, trade in trades_df.iterrows():
                # Find closest feature row
                time_diff = abs(features_df_copy.index - trade['entry_time'])
                closest_idx = time_diff.argmin()
                
                # Copy feature values
                for col in features_df_copy.columns:
                    if col not in ['timestamp', 'entry_price', 'symbol']:
                        trades_df.at[idx, col] = features_df_copy.iloc[closest_idx][col]
        
        logger.info(f"✅ Backtest complete: {len(trades_df)} trades with labels")
        return trades_df
    
    def _simulate_signal_trade(self, signal, data: pd.DataFrame, signal_idx: int, 
                              symbol: str, backtest_engine) -> Optional[Dict]:
        """Simulate a single trade from a strategy signal"""
        
        entry_bar = data.iloc[signal_idx]
        entry_time = entry_bar.name if hasattr(entry_bar, 'name') else signal_idx
        
        # Determine side
        from strategies.strategy_base import SignalType
        side = 'long' if signal.signal_type == SignalType.BUY else 'short'
        
        # Find exit
        exit_idx = None
        exit_price = None
        exit_reason = 'timeout'
        
        for i in range(signal_idx + 1, len(data)):
            bar = data.iloc[i]
            
            if side == 'long':
                # Check SL
                if bar['low'] <= signal.stop_loss:
                    exit_idx = i
                    exit_price = signal.stop_loss
                    exit_reason = 'stop_loss'
                    break
                # Check TP
                if bar['high'] >= signal.take_profit:
                    exit_idx = i
                    exit_price = signal.take_profit
                    exit_reason = 'take_profit'
                    break
            else:  # short
                # Check SL
                if bar['high'] >= signal.stop_loss:
                    exit_idx = i
                    exit_price = signal.stop_loss
                    exit_reason = 'stop_loss'
                    break
                # Check TP
                if bar['low'] <= signal.take_profit:
                    exit_idx = i
                    exit_price = signal.take_profit
                    exit_reason = 'take_profit'
                    break
            
            # Timeout after 50 bars
            if i - signal_idx > 50:
                exit_idx = i
                exit_price = bar['close']
                exit_reason = 'timeout'
                break
        
        if exit_idx is None:
            # Never exited
            return None
        
        # Calculate PnL
        if side == 'long':
            pnl_pips = (exit_price - signal.entry_price) * 10000  # assuming 4 decimal prices
            pnl_percent = ((exit_price - signal.entry_price) / signal.entry_price) * 100
        else:
            pnl_pips = (signal.entry_price - exit_price) * 10000
            pnl_percent = ((signal.entry_price - exit_price) / signal.entry_price) * 100
        
        exit_bar = data.iloc[exit_idx]
        exit_time = exit_bar.name if hasattr(exit_bar, 'name') else exit_idx
        
        return {
            'symbol': symbol,
            'side': side,
            'entry_price': signal.entry_price,
            'entry_time': entry_time,
            'exit_price': exit_price,
            'exit_time': exit_time,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'pnl': pnl_pips * 10,  # rough conversion
            'pnl_pips': pnl_pips,
            'pnl_percent': pnl_percent,
            'duration_bars': exit_idx - signal_idx,
            'label_exit_reason': exit_reason,
            'confidence': signal.confidence,
            'strategy_reason': signal.reason,
            **({'strategy_metadata': str(signal.metadata)} if signal.metadata else {})
        }
    
    def _train_models(self, trades_df: pd.DataFrame,
                     test_size: float = 0.2,
                     cv_folds: int = 5) -> Dict:
        """Train models"""
        from ml.model_trainer import ModelTrainer
        
        if trades_df.empty:
            raise ValueError("No trades to train on (empty trades_df)")

        if 'label_win_loss' not in trades_df.columns:
            raise ValueError("Missing label_win_loss column in trades_df")

        feature_cols = [
            col for col in trades_df.columns
            if col not in {
                'trade_id', 'symbol', 'side', 'entry_price', 'entry_time',
                'exit_price', 'exit_time', 'stop_loss', 'take_profit',
                'pnl', 'pnl_pips', 'pnl_percent', 'duration_bars',
                'label_win_loss', 'label_magnitude', 'label_exit_reason'
            }
        ]
        
        X = trades_df[feature_cols].fillna(0)
        y = trades_df['label_win_loss']
        
        trainer = ModelTrainer()
        results = trainer.train_all_models(
            X, y,
            test_size=test_size,
            cv_folds=cv_folds
        )
        
        return results
    
    def _analyze_context(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze trading context"""
        from ml.context_analyzer import ContextAnalyzer
        
        ca = ContextAnalyzer()
        analysis = ca.analyze_trades(trades_df)
        
        return analysis
    
    def _update_metrics_from_results(self, 
                                     trades_df: pd.DataFrame,
                                     training_results: Dict,
                                     analysis: Dict):
        """Update metrics from training results"""
        
        overview = analysis.get('overview', {})
        
        # Get best model
        best_model = max(training_results.items(), 
                        key=lambda x: x[1].get('f1', 0))
        
        self.metrics = TrainingMetrics(
            total_trades=overview.get('total_trades', 0),
            winning_trades=overview.get('winning_trades', 0),
            win_rate=overview.get('win_rate', 0),
            profit_factor=overview.get('profit_factor', 0),
            avg_win_pnl=overview.get('avg_win_pnl', 0),
            avg_loss_pnl=overview.get('avg_loss_pnl', 0),
            best_model=best_model[0],
            model_accuracy=best_model[1].get('accuracy', 0),
            model_f1=best_model[1].get('f1', 0),
            model_roc_auc=best_model[1].get('roc_auc', 0),
            top_features=list(analysis.get('feature_comparison', {}).items())[:10],
            feature_count=len(analysis.get('feature_comparison', {}))
        )
        
        self._update_metrics(self.metrics)
    
    def _save_training_results(self, symbol: str,
                               trades_df: pd.DataFrame,
                               training_results: Dict,
                               analysis: Dict):
        """Save training results to disk"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.results_dir / f"training_{symbol}_{timestamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trades
        trades_df.to_csv(session_dir / "trades.csv", index=False)
        
        # Save metrics
        metrics_file = session_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics.to_dict(), f, indent=2, default=str)
        
        # Save history
        history_file = session_dir / "history.json"
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        logger.info(f"Training results saved to {session_dir}")
    
    def get_progress(self) -> Dict:
        """Get current progress"""
        return self.progress.to_dict()
    
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        return self.metrics.to_dict()
    
    def get_history(self) -> List[Dict]:
        """Get training history"""
        return self.training_history
    
    def pause(self):
        """Pause training (not currently implemented)"""
        self._update_progress(
            TrainingStatus.PAUSED, -1, -1,
            "Training paused"
        )
    
    def resume(self):
        """Resume training (not currently implemented)"""
        pass
    
    def cancel(self):
        """Cancel training (not currently implemented)"""
        self.is_training = False
        self._update_progress(
            TrainingStatus.IDLE, 0, 0,
            "Training cancelled"
        )
