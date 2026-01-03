"""
Prediction Engine - Usa modelli addestrati per predicare outcome di nuovi trade
Genera confidence scores e analizza quali feature hanno influenzato la predizione
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PredictionEngine:
    """
    Motore di predizione che:
    1. Estrae feature dal contesto di trading corrente
    2. Usa modelli addestrati per predicare win/loss
    3. Assegna confidence score
    4. Fornisce interpretability (quali feature hanno portato alla decisione)
    """
    
    def __init__(self, model_trainer, feature_engineer):
        """
        Args:
            model_trainer: ModelTrainer instance con modelli addestrati
            feature_engineer: FeatureEngineer instance
        """
        self.model_trainer = model_trainer
        self.feature_engineer = feature_engineer
        
        self.last_prediction = None
        self.last_features = None
        
        logger.info("PredictionEngine initialized")
    
    def predict_trade_outcome(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        current_data: pd.DataFrame,
        model_name: str = 'xgboost',
        confidence_threshold: float = 0.5
    ) -> Dict:
        """
        Predice outcome di un potenziale trade.
        
        Args:
            symbol: Es. 'EUR_USD'
            side: 'BUY' o 'SELL'
            entry_price: Prezzo di entrata
            current_data: DataFrame con OHLCV (deve avere almeno 50 bar)
            model_name: Nome del modello
            confidence_threshold: Soglia di confidenza (0.5-1.0)
        
        Returns:
            Dict con:
            - prediction: 0 (loss) o 1 (win)
            - win_probability: Probabilità di win (0-1)
            - confidence: Quanto sicuro è il modello
            - should_trade: Bool, True se probabilità > threshold
            - features: Feature vector usato
            - top_influential_features: Top feature che hanno influenzato la predizione
        """
        
        # Extract features
        features = self.feature_engineer.extract_all_features(
            current_data, symbol, entry_price, side
        )
        
        self.last_features = features
        
        # Crea DataFrame (una sola riga)
        X = pd.DataFrame([features])
        
        # Predizione
        prediction = self.model_trainer.predict(X, model_name)[0]
        proba = self.model_trainer.predict_proba(X, model_name)[0]
        
        win_probability = float(proba[1])  # Probabilità di classe 1 (win)
        confidence = abs(win_probability - 0.5) * 2  # 0-1, dove 1 = massima certezza
        
        # Decisione
        should_trade = win_probability > confidence_threshold
        
        # Top influential features
        top_features = self._get_influential_features(X, model_name, top_n=5)
        
        result = {
            'prediction': int(prediction),
            'win_probability': win_probability,
            'loss_probability': float(proba[0]),
            'confidence': confidence,
            'should_trade': should_trade,
            'confidence_threshold': confidence_threshold,
            'model_name': model_name,
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'num_features': len(features),
            'top_influential_features': top_features,
        }
        
        self.last_prediction = result
        
        logger.info(
            f"Prediction: {symbol} {side} @ {entry_price:.5f} | "
            f"Win: {win_probability:.2%} | Conf: {confidence:.2%} | "
            f"Trade: {should_trade}"
        )
        
        return result
    
    def _get_influential_features(
        self,
        X: pd.DataFrame,
        model_name: str,
        top_n: int = 5
    ) -> Dict[str, float]:
        """
        Identifica quali feature hanno influenzato la predizione più di altri.
        Usa feature importance del modello.
        
        Args:
            X: Feature vector (una riga)
            model_name: Nome modello
            top_n: Numero di feature da restituire
        
        Returns:
            Dict {feature_name: importance_score}
        """
        
        # Ottieni feature importance globale
        importance = self.model_trainer.get_feature_importance(model_name, top_n=top_n)
        
        # Filtra solo le feature presenti in X
        X_cols = X.columns.tolist()
        influential = {
            feat: score
            for feat, score in importance.items()
            if feat in X_cols
        }
        
        return influential
    
    def analyze_prediction_context(self) -> Dict:
        """
        Analizza il contesto della last_prediction.
        Cosa ha portato il modello a fare quella predizione?
        
        Returns:
            Dict con analisi dettagliata
        """
        
        if self.last_prediction is None or self.last_features is None:
            logger.warning("No last prediction to analyze")
            return {}
        
        pred = self.last_prediction
        features = self.last_features
        
        analysis = {
            'prediction': pred['prediction'],
            'win_probability': pred['win_probability'],
            'confidence': pred['confidence'],
            'symbol': pred['symbol'],
            'side': pred['side'],
            'feature_context': {}
        }
        
        # Aggiungi context su key feature
        for feat_name in pred['top_influential_features'].keys():
            if feat_name in features:
                value = features[feat_name]
                analysis['feature_context'][feat_name] = {
                    'value': float(value),
                    'normalized': float(np.clip(value, 0, 1))
                }
        
        # Analisi aggiuntiva
        if 'rsi_current' in features:
            rsi = features['rsi_current']
            if rsi < 30:
                analysis['market_condition'] = 'OVERSOLD'
            elif rsi > 70:
                analysis['market_condition'] = 'OVERBOUGHT'
            else:
                analysis['market_condition'] = 'NEUTRAL'
        
        if 'trend_type' in features:
            analysis['trend'] = features['trend_type']
        
        return analysis
    
    def batch_predict(
        self,
        trades: pd.DataFrame,
        model_name: str = 'xgboost',
        confidence_threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Predice outcome per batch di trade potenziali.
        
        Args:
            trades: DataFrame con colonne:
                - symbol, side, entry_price, (deve avere context per extract_all_features)
                - current_data: Serie con OHLCV data
            model_name: Nome modello
            confidence_threshold: Soglia
        
        Returns:
            DataFrame con predizioni aggiunte
        """
        
        results = []
        
        for idx, trade in trades.iterrows():
            
            # Assumendo che ogni riga abbia accesso ai dati OHLCV
            # Questa è una semplificazione, in pratica serve accesso a dati storici
            pred = self.predict_trade_outcome(
                symbol=trade['symbol'],
                side=trade['side'],
                entry_price=trade['entry_price'],
                current_data=trade.get('current_data'),
                model_name=model_name,
                confidence_threshold=confidence_threshold
            )
            
            results.append({
                **trade.to_dict(),
                'predicted_win_prob': pred['win_probability'],
                'predicted_outcome': pred['prediction'],
                'prediction_confidence': pred['confidence'],
                'should_execute': pred['should_trade']
            })
        
        return pd.DataFrame(results)
    
    def filter_high_confidence_trades(
        self,
        trades: pd.DataFrame,
        min_win_probability: float = 0.60,
        min_confidence: float = 0.30,
        model_name: str = 'xgboost'
    ) -> pd.DataFrame:
        """
        Filtra trade basato su win probability e model confidence.
        Utile per trading selettivo (trade solo i migliori).
        
        Args:
            trades: DataFrame di potenziali trade
            min_win_probability: Minima probabilità di win (0-1)
            min_confidence: Minima confidenza del modello (0-1)
            model_name: Nome modello
        
        Returns:
            DataFrame filtrato (solo trade ad alta probabilità)
        """
        
        # Batch predict
        predictions = self.batch_predict(trades, model_name)
        
        # Filtra
        filtered = predictions[
            (predictions['predicted_win_prob'] >= min_win_probability) &
            (predictions['prediction_confidence'] >= min_confidence)
        ]
        
        logger.info(
            f"Filtered {len(trades)} trades → {len(filtered)} high-confidence trades "
            f"(min_win_prob={min_win_probability:.2%}, min_conf={min_confidence:.2%})"
        )
        
        return filtered
    
    def get_prediction_report(self) -> str:
        """Genera report leggibile della last prediction"""
        
        if self.last_prediction is None:
            return "No prediction available"
        
        pred = self.last_prediction
        
        report = f"""
╔══════════════════════════════════════════════════════╗
║           TRADE PREDICTION ANALYSIS                  ║
╚══════════════════════════════════════════════════════╝

Trade Setup:
  Symbol:        {pred['symbol']}
  Side:          {pred['side']}
  Entry Price:   {pred['entry_price']:.5f}

Prediction Results:
  Model:         {pred['model_name'].upper()}
  Prediction:    {'WIN' if pred['prediction'] == 1 else 'LOSS'}
  Win Prob:      {pred['win_probability']:.1%}
  Loss Prob:     {pred['loss_probability']:.1%}
  Confidence:    {pred['confidence']:.1%}
  
Decision:
  Execute Trade: {'✓ YES' if pred['should_trade'] else '✗ NO'}
  Threshold:     {pred['confidence_threshold']:.1%}

Top Influential Features:
"""
        
        for feat_name, score in pred['top_influential_features'].items():
            report += f"  • {feat_name:<30} {score:.4f}\n"
        
        # Aggiungi context
        context = self.analyze_prediction_context()
        if 'market_condition' in context:
            report += f"\nMarket Context:\n"
            report += f"  Condition:   {context.get('market_condition', 'N/A')}\n"
            report += f"  Trend:       {context.get('trend', 'N/A')}\n"
        
        report += "\n" + "="*50 + "\n"
        
        return report
    
    def print_prediction_report(self):
        """Stampa prediction report"""
        print(self.get_prediction_report())
