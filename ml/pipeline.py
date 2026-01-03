"""
ML Pipeline Integration - Mette insieme tutto il flusso ML
1. Estrae feature dai dati storici
2. Fa backtest e genera labels
3. Addestra modelli
4. Analizza contesto
5. Predice trade futuri
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

from ml.feature_engineer import FeatureEngineer
from ml.backtest_labeled import BacktestEngine, BacktestTrade
from ml.model_trainer import ModelTrainer
from ml.prediction_engine import PredictionEngine
from ml.context_analyzer import ContextAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLPipeline:
    """
    Pipeline ML end-to-end:
    Historical Data → Features → Backtest+Labels → Model Training → Predictions
    """
    
    def __init__(self, data_dir: str = 'data', model_dir: str = 'ml/models'):
        """
        Args:
            data_dir: Directory con dati storici
            model_dir: Directory per salvare modelli
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        
        # Componenti
        self.feature_engineer = FeatureEngineer()
        self.backtest_engine = BacktestEngine()
        self.model_trainer = ModelTrainer(str(model_dir))
        self.prediction_engine = PredictionEngine(self.model_trainer, self.feature_engineer)
        self.context_analyzer = ContextAnalyzer()
        
        logger.info("MLPipeline initialized")
    
    def run_full_pipeline(
        self,
        historical_data: pd.DataFrame,
        symbol: str = 'EUR_USD',
        entry_price_column: str = 'close',
        test_size: float = 0.2,
        cv_folds: int = 5
    ):
        """
        Esegue pipeline completo.
        
        Args:
            historical_data: DataFrame con OHLCV (columns: open, high, low, close, volume, time)
            symbol: Symbol da tradare
            entry_price_column: Colonna per prezzo entry (default: close)
            test_size: Proporzione test set
            cv_folds: Folds per cross-validation
        """
        
        logger.info(f"Starting ML Pipeline for {symbol}")
        
        # Step 1: Feature extraction
        logger.info("STEP 1: Feature Extraction")
        features_list = self._extract_features(historical_data, symbol)
        features_df = pd.DataFrame(features_list)
        logger.info(f"Extracted {len(features_df.columns)} features from {len(features_df)} samples")
        
        # Step 2: Backtesting with labels
        logger.info("\nSTEP 2: Backtesting & Label Generation")
        trades_df = self._run_backtest(
            historical_data, features_df, symbol
        )
        logger.info(f"Generated {len(trades_df)} backtested trades with labels")
        
        # Step 3: Model training
        logger.info("\nSTEP 3: Model Training")
        if len(trades_df) < 50:
            logger.error(f"Not enough trades ({len(trades_df)}) to train models. Need at least 50.")
            return None
        
        training_results = self._train_models(
            trades_df, test_size=test_size, cv_folds=cv_folds
        )
        logger.info("Models trained successfully")
        
        # Step 4: Context analysis
        logger.info("\nSTEP 4: Context Analysis")
        analysis = self._analyze_context(trades_df)
        
        # Stampa report
        self.context_analyzer.print_analysis_report(analysis)
        self.model_trainer.print_summary()
        
        return {
            'features_df': features_df,
            'trades_df': trades_df,
            'training_results': training_results,
            'context_analysis': analysis,
        }
    
    def _extract_features(
        self,
        historical_data: pd.DataFrame,
        symbol: str
    ) -> list:
        """
        Estrae feature da dati storici.
        Simula entry a ogni candela e estrae feature per quella candela.
        """
        
        features_list = []
        
        # Prendi ultime 50 candele per features
        min_lookback = 50
        
        for i in range(min_lookback, len(historical_data)):
            # Slice dati fino a current
            data_slice = historical_data.iloc[:i+1].copy()
            
            # Prezzo entry = close della candela
            entry_price = historical_data.iloc[i]['close']
            
            # Estrai feature
            try:
                features = self.feature_engineer.extract_all_features(
                    data_slice,
                    symbol,
                    entry_price,
                    side='BUY'  # Semplicemente BUY per analisi
                )
                
                # Aggiungi metadata
                features['timestamp'] = historical_data.iloc[i].get('time', i)
                features['entry_price'] = entry_price
                features['symbol'] = symbol
                
                features_list.append(features)
            
            except Exception as e:
                logger.warning(f"Error extracting features at row {i}: {e}")
                continue
        
        return features_list
    
    def _run_backtest(
        self,
        historical_data: pd.DataFrame,
        features_df: pd.DataFrame,
        symbol: str,
        stop_loss_pips: int = 50,
        take_profit_pips: int = 100
    ) -> pd.DataFrame:
        """
        Esegue backtest su dati storici.
        Per ogni punto di entry, simula un trade fino a SL/TP/timeout.
        """
        
        trades_list = []
        
        # Conversione: 1 pip = 0.0001 per coppie come EUR_USD
        pip_value = 0.0001
        
        for i in range(len(features_df)):
            # Salta se non ci sono abbastanza dati futuri per il backtest
            if i >= len(historical_data) - 100:
                continue
            
            entry_price = features_df.iloc[i]['entry_price']
            side = 'BUY'
            
            # Calcola SL/TP
            stop_loss = entry_price - (stop_loss_pips * pip_value)
            take_profit = entry_price + (take_profit_pips * pip_value)
            
            # Simula trade
            try:
                trade = self.backtest_engine.simulate_trade(
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    entry_time=historical_data.iloc[i].get('time', i),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    historical_data=historical_data,
                    starting_row_index=i,
                    entry_features={
                        'rsi_current': features_df.iloc[i].get('rsi_current', 50),
                        'macd_histogram': features_df.iloc[i].get('macd_histogram', 0),
                        'trend_type': features_df.iloc[i].get('trend_type', 'NEUTRAL'),
                        'volatility_atr': features_df.iloc[i].get('volatility_atr', 0),
                        'signal_confluence': features_df.iloc[i].get('signal_confluence', 0),
                    }
                )
                
                trades_list.append(trade)
            
            except Exception as e:
                logger.debug(f"Error simulating trade at row {i}: {e}")
                continue
        
        # Converti a DataFrame
        trades_df = self.backtest_engine.get_trades_dataframe()
        
        # Merge con features
        if len(trades_df) > 0 and len(features_df) > 0:
            # Aggiungi feature columns dal features_df
            for feat_col in self.feature_engineer.get_feature_names():
                if feat_col in features_df.columns:
                    # Questo è semplificato - in pratica serve matching corretto
                    pass
        
        return trades_df
    
    def _train_models(
        self,
        trades_df: pd.DataFrame,
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> dict:
        """
        Addestra modelli su trades.
        """
        
        # Prepara X, y
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
        
        logger.info(f"Training on {len(X)} samples with {len(feature_cols)} features")
        
        # Addestra modelli
        results = self.model_trainer.train_all_models(
            X, y,
            test_size=test_size,
            cv_folds=cv_folds
        )
        
        return results
    
    def _analyze_context(self, trades_df: pd.DataFrame) -> dict:
        """Analizza contesto"""
        
        analysis = self.context_analyzer.analyze_trades(trades_df)
        
        return analysis
    
    def predict_new_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        current_data: pd.DataFrame,
        model_name: str = 'xgboost',
        confidence_threshold: float = 0.55
    ) -> dict:
        """
        Predice outcome di un nuovo trade.
        
        Args:
            symbol: Symbol
            side: BUY/SELL
            entry_price: Prezzo entry
            current_data: OHLCV data corrente (ultimi ~50 bar)
            model_name: Nome modello
            confidence_threshold: Soglia confidenza
        
        Returns:
            Dict con predizione e analisi
        """
        
        # Carica modelli se non già in memoria
        if model_name not in self.model_trainer.models:
            self.model_trainer.load_model(model_name)
        
        # Predici
        prediction = self.prediction_engine.predict_trade_outcome(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            current_data=current_data,
            model_name=model_name,
            confidence_threshold=confidence_threshold
        )
        
        return prediction
    
    def print_winning_trade_profile(self):
        """Stampa profilo del trade vincente ideale"""
        
        profile = self.context_analyzer.get_winning_trade_profile()
        
        if not profile:
            logger.warning("No winning trade profile available")
            return
        
        print("\n" + "="*70)
        print("WINNING TRADE PROFILE (Ideal Setup)")
        print("="*70 + "\n")
        
        for feature, stats in list(profile.items())[:20]:
            print(f"{feature:<30} | Mean: {stats['mean']:>8.4f} | Median: {stats['median']:>8.4f}")
        
        print("\n" + "="*70 + "\n")


# Example usage
if __name__ == "__main__":
    """
    Esempio di utilizzo della pipeline ML.
    
    Flusso:
    1. Carica dati storici (es. da Yahoo Finance)
    2. Esegui pipeline completo
    3. Addestra modelli
    4. Predici nuovi trade
    """
    
    logger.info("ML Pipeline Example")
    
    # TODO: Carica dati storici (es. con YahooFinanceDataFetcher)
    # historical_data = load_historical_data('EUR_USD', '2023-01-01', '2024-01-01')
    
    # Inizializza pipeline
    # pipeline = MLPipeline()
    
    # Esegui pipeline
    # results = pipeline.run_full_pipeline(
    #     historical_data,
    #     symbol='EUR_USD'
    # )
    
    # Predici nuovo trade
    # current_data = load_current_data('EUR_USD', last_50_bars=True)
    # prediction = pipeline.predict_new_trade(
    #     symbol='EUR_USD',
    #     side='BUY',
    #     entry_price=1.0950,
    #     current_data=current_data
    # )
    # print(pipeline.prediction_engine.get_prediction_report())
