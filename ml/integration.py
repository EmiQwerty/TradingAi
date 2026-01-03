"""
Trading System Integration with ML Pipeline

Mostra come integrare il ML pipeline con:
1. BrokerAPI (OANDA/Mock)
2. DecisionEngine
3. DataFetcher (Yahoo Finance)
4. OrderExecutor
"""

import logging
from typing import Dict, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MLEnhancedDecisionEngine:
    """
    DecisionEngine potenziato con ML predictions.
    
    Flusso:
    1. Carica dati storici
    2. Genera features
    3. Usa modello ML per predicare win_probability
    4. Filtra trade con alta confidenza
    5. Decide se eseguire basato su probabilità + regole tecniche
    """
    
    def __init__(self, 
                 ml_pipeline=None,
                 decision_engine=None,
                 data_fetcher=None,
                 min_win_prob: float = 0.55,
                 min_ml_confidence: float = 0.30):
        """
        Args:
            ml_pipeline: MLPipeline instance (trained models)
            decision_engine: Original DecisionEngine (technical rules)
            data_fetcher: DataFetcher instance (Yahoo/OANDA)
            min_win_prob: Minimum win probability to trade
            min_ml_confidence: Minimum model confidence
        """
        self.ml_pipeline = ml_pipeline
        self.decision_engine = decision_engine
        self.data_fetcher = data_fetcher
        self.min_win_prob = min_win_prob
        self.min_ml_confidence = min_ml_confidence
        
        self.last_decision = None
        self.decision_history = []
        
        logger.info(
            f"MLEnhancedDecisionEngine initialized "
            f"(min_win_prob={min_win_prob}, min_confidence={min_ml_confidence})"
        )
    
    def decide_trade(self, symbol: str, entry_price: float, timeframe: str = '1H') -> Tuple[bool, str]:
        """
        Decide whether to trade basato su ML + technical signals.
        
        Args:
            symbol: Es. 'EUR_USD'
            entry_price: Prezzo entry proposto
            timeframe: Es. '1H', '4H', 'D'
        
        Returns:
            (should_trade: bool, reason: str)
        """
        
        reasons = []
        
        # Step 1: Valutazione Tecnica (Original DecisionEngine)
        logger.info(f"Step 1: Technical Analysis for {symbol}...")
        
        if self.decision_engine:
            try:
                # Fetch data for technical analysis
                data = self._get_historical_data(symbol, lookback_bars=100)
                
                # Original decision engine
                tech_signal, tech_reason = self.decision_engine.decide_trade(symbol, data)
                reasons.append(f"Technical: {tech_reason}")
                
                if not tech_signal:
                    logger.info(f"Technical signal negative: {tech_reason}")
                    return False, " → ".join(reasons)
            
            except Exception as e:
                logger.warning(f"Technical analysis failed: {e}")
                return False, f"Technical analysis error: {e}"
        
        else:
            tech_signal = True
            reasons.append("Technical: Skipped (no engine)")
        
        # Step 2: ML Prediction
        logger.info(f"Step 2: ML Prediction for {symbol}...")
        
        if not self.ml_pipeline:
            logger.warning("ML Pipeline not available")
            return tech_signal, " → ".join(reasons)
        
        try:
            # Fetch recent data (last 50 bars for feature extraction)
            current_data = self._get_historical_data(symbol, lookback_bars=50)
            
            if len(current_data) < 30:
                reasons.append("ML: Insufficient data")
                logger.warning(f"Not enough data for ML: {len(current_data)} bars")
                return tech_signal, " → ".join(reasons)
            
            # ML Prediction
            ml_prediction = self.ml_pipeline.predict_new_trade(
                symbol=symbol,
                side='BUY',  # In realtà decidi BUY/SELL da logica tecnica
                entry_price=entry_price,
                current_data=current_data,
                model_name='xgboost',
                confidence_threshold=self.min_win_prob
            )
            
            win_prob = ml_prediction['win_probability']
            confidence = ml_prediction['confidence']
            
            logger.info(
                f"ML Prediction: {symbol} | Win: {win_prob:.1%} | Conf: {confidence:.1%}"
            )
            
            reasons.append(
                f"ML: Win prob {win_prob:.1%}, Confidence {confidence:.1%}"
            )
            
            # Step 3: Combined Decision
            logger.info(f"Step 3: Combined Decision...")
            
            should_trade = (
                tech_signal and
                win_prob >= self.min_win_prob and
                confidence >= self.min_ml_confidence
            )
            
            # Detailed reasoning
            if should_trade:
                reasons.append("✓ EXECUTE TRADE")
                decision_reason = " → ".join(reasons)
            else:
                if win_prob < self.min_win_prob:
                    reasons.append(f"✗ LOW WIN PROB (< {self.min_win_prob:.1%})")
                if confidence < self.min_ml_confidence:
                    reasons.append(f"✗ LOW CONFIDENCE (< {self.min_ml_confidence:.1%})")
                decision_reason = " → ".join(reasons)
            
            # Store decision for analysis
            self.last_decision = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'entry_price': entry_price,
                'decision': should_trade,
                'win_probability': win_prob,
                'ml_confidence': confidence,
                'tech_signal': tech_signal,
                'reasons': decision_reason,
                'top_features': ml_prediction.get('top_influential_features', {})
            }
            
            self.decision_history.append(self.last_decision)
            
            return should_trade, decision_reason
        
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to technical signal
            reasons.append(f"ML Error (fallback to technical): {e}")
            return tech_signal, " → ".join(reasons)
    
    def _get_historical_data(self, symbol: str, lookback_bars: int = 50) -> pd.DataFrame:
        """
        Carica dati storici per il symbol.
        
        Usa data_fetcher se disponibile, altrimenti dummy data.
        """
        
        if self.data_fetcher:
            try:
                # Carica ultimi N bar
                data = self.data_fetcher.fetch_last_n_bars(symbol, lookback_bars)
                return data
            except Exception as e:
                logger.warning(f"DataFetcher failed: {e}")
        
        # Dummy data for testing
        logger.warning(f"Using dummy data for {symbol}")
        
        import numpy as np
        dates = pd.date_range(datetime.now() - timedelta(hours=lookback_bars), 
                             datetime.now(), freq='1H')
        
        close_prices = 1.0900 + np.cumsum(np.random.randn(lookback_bars) * 0.001)
        
        data = pd.DataFrame({
            'time': dates[-lookback_bars:],
            'open': close_prices - 0.0002,
            'high': close_prices + 0.0003,
            'low': close_prices - 0.0003,
            'close': close_prices,
            'volume': np.random.uniform(100000, 500000, lookback_bars)
        })
        
        return data
    
    def analyze_decisions(self, n_last: Optional[int] = 10) -> Dict:
        """
        Analizza decisions prese.
        
        Returns: Stats su decisions
        """
        
        if not self.decision_history:
            return {'total_decisions': 0}
        
        history = self.decision_history[-n_last:] if n_last else self.decision_history
        
        executed = sum(1 for d in history if d['decision'])
        skipped = sum(1 for d in history if not d['decision'])
        
        avg_win_prob = sum(d['win_probability'] for d in history) / len(history)
        avg_confidence = sum(d['ml_confidence'] for d in history) / len(history)
        
        return {
            'total_decisions': len(history),
            'executed_trades': executed,
            'skipped_trades': skipped,
            'execution_rate': executed / len(history) if history else 0,
            'avg_win_probability': avg_win_prob,
            'avg_ml_confidence': avg_confidence,
            'symbols': list(set(d['symbol'] for d in history)),
        }
    
    def print_decision_report(self):
        """Stampa report delle decisions prese"""
        
        if not self.last_decision:
            print("No decisions made yet")
            return
        
        d = self.last_decision
        
        print("\n" + "="*70)
        print("TRADING DECISION REPORT")
        print("="*70)
        
        print(f"\nDecision: {'✓ EXECUTE' if d['decision'] else '✗ SKIP'}")
        print(f"Timestamp: {d['timestamp']}")
        print(f"Symbol: {d['symbol']} @ {d['entry_price']:.5f}")
        
        print(f"\nAnalysis:")
        print(f"  Win Probability:  {d['win_probability']:.1%}")
        print(f"  ML Confidence:    {d['ml_confidence']:.1%}")
        print(f"  Technical Signal: {'✓ POSITIVE' if d['tech_signal'] else '✗ NEGATIVE'}")
        
        print(f"\nReasoning:")
        for part in d['reasons'].split(' → '):
            prefix = '  ' if not part.startswith('✓') and not part.startswith('✗') else '  '
            print(f"{prefix}{part}")
        
        print(f"\nTop Influential Features:")
        for feat, score in d['top_features'].items():
            print(f"  {feat:<30} {score:.4f}")
        
        print("\n" + "="*70 + "\n")


class MLIntegrationExample:
    """Esempio di integrazione completa"""
    
    @staticmethod
    def example_usage():
        """
        Mostra come usare MLEnhancedDecisionEngine
        """
        
        # 1. Setup Components
        from ml.pipeline import MLPipeline
        from decision_engine import DecisionEngine  # Original
        from apis.data_fetcher_yahoo import YahooFinanceDataFetcher
        
        # Initialize
        ml_pipeline = MLPipeline()
        ml_pipeline.model_trainer.load_all_models()  # Load trained models
        
        original_engine = DecisionEngine()
        data_fetcher = YahooFinanceDataFetcher()
        
        # 2. Create ML-Enhanced Engine
        enhanced_engine = MLEnhancedDecisionEngine(
            ml_pipeline=ml_pipeline,
            decision_engine=original_engine,
            data_fetcher=data_fetcher,
            min_win_prob=0.55,  # Trade solo con 55%+ win probability
            min_ml_confidence=0.30
        )
        
        # 3. Make Decision
        should_trade, reason = enhanced_engine.decide_trade(
            symbol='EUR_USD',
            entry_price=1.0950,
            timeframe='1H'
        )
        
        print(f"Decision: {should_trade}")
        print(f"Reason: {reason}")
        
        # 4. Analyze Decisions
        stats = enhanced_engine.analyze_decisions(n_last=20)
        print(f"\nDecision Statistics:")
        print(f"  Total Decisions: {stats['total_decisions']}")
        print(f"  Executed: {stats['executed_trades']}")
        print(f"  Execution Rate: {stats['execution_rate']:.1%}")
        
        # 5. Print Report
        enhanced_engine.print_decision_report()


if __name__ == '__main__':
    # Example usage
    # MLIntegrationExample.example_usage()
    
    print("""
    ML Integration Example
    
    To use MLEnhancedDecisionEngine in your trading system:
    
    1. Train ML models:
       python -c "from ml.pipeline import MLPipeline; p = MLPipeline(); p.run_full_pipeline(data)"
    
    2. Create enhanced engine:
       engine = MLEnhancedDecisionEngine(ml_pipeline, original_engine, data_fetcher)
    
    3. Make decisions:
       should_trade, reason = engine.decide_trade('EUR_USD', 1.0950)
    
    4. Execute if True:
       if should_trade:
           broker.place_order(...)
    
    See code above for complete example.
    """)
