"""
Machine Learning Inference Module
Loads pre-trained models and performs inference for regime, trend, and volatility classification
"""

import os
import pickle
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MLInference:
    """
    Manages ML model inference for trading decisions
    Models are pre-trained and frozen - only inference is performed
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.ml_config = config.get('ml', {})
        
        self.models_path = Path(self.ml_config.get('models_path', 'ml/models'))
        self.enabled = self.ml_config.get('enabled', True)
        
        # Model containers
        self.regime_model = None
        self.trend_model = None
        self.volatility_model = None
        self.scalers = None
        
        # Feature configurations
        self.regime_features = self.ml_config.get('regime', {}).get('features', [])
        self.trend_features = self.ml_config.get('trend', {}).get('features', [])
        self.volatility_features = self.ml_config.get('volatility', {}).get('features', [])
        
        if self.enabled:
            self._load_models()
        
        logger.info(f"MLInference initialized (enabled: {self.enabled})")
    
    def _load_models(self):
        """Load pre-trained models from disk"""
        try:
            # Load regime classification model
            regime_path = self.models_path / 'regime.pkl'
            if regime_path.exists():
                with open(regime_path, 'rb') as f:
                    self.regime_model = pickle.load(f)
                logger.info("Loaded regime model")
            else:
                logger.warning(f"Regime model not found at {regime_path}")
                self._create_dummy_regime_model()
            
            # Load trend prediction model
            trend_path = self.models_path / 'trend.pkl'
            if trend_path.exists():
                with open(trend_path, 'rb') as f:
                    self.trend_model = pickle.load(f)
                logger.info("Loaded trend model")
            else:
                logger.warning(f"Trend model not found at {trend_path}")
                self._create_dummy_trend_model()
            
            # Load volatility classification model
            vol_path = self.models_path / 'volatility.pkl'
            if vol_path.exists():
                with open(vol_path, 'rb') as f:
                    self.volatility_model = pickle.load(f)
                logger.info("Loaded volatility model")
            else:
                logger.warning(f"Volatility model not found at {vol_path}")
                self._create_dummy_volatility_model()
            
            # Load feature scalers
            scaler_path = self.models_path / 'scalers.pkl'
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scalers = pickle.load(f)
                logger.info("Loaded feature scalers")
            else:
                logger.warning("Feature scalers not found, creating dummy scalers")
                self._create_dummy_scalers()
        
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self._create_dummy_models()
    
    def _create_dummy_regime_model(self):
        """Create a simple dummy regime classifier"""
        class DummyRegimeModel:
            def predict(self, X):
                # Simple rule-based regime detection
                regime_scores = []
                for row in X:
                    rsi = row[0] if len(row) > 0 else 50
                    atr = row[1] if len(row) > 1 else 0.5
                    
                    if rsi > 60 or rsi < 40:
                        regime_scores.append('TREND')
                    elif atr < 0.3:
                        regime_scores.append('RANGE')
                    else:
                        regime_scores.append('CHAOS')
                
                return np.array(regime_scores)
            
            def predict_proba(self, X):
                predictions = self.predict(X)
                # Return dummy probabilities
                proba = []
                for pred in predictions:
                    if pred == 'TREND':
                        proba.append([0.7, 0.2, 0.1])
                    elif pred == 'RANGE':
                        proba.append([0.2, 0.7, 0.1])
                    else:
                        proba.append([0.1, 0.2, 0.7])
                return np.array(proba)
        
        self.regime_model = DummyRegimeModel()
        logger.info("Created dummy regime model")
    
    def _create_dummy_trend_model(self):
        """Create a simple dummy trend predictor"""
        class DummyTrendModel:
            def predict(self, X):
                # Simple trend calculation
                trends = []
                for row in X:
                    price_change = row[0] if len(row) > 0 else 0
                    ma_cross = row[1] if len(row) > 1 else 0
                    
                    trend_strength = (price_change + ma_cross) / 2
                    trends.append(np.clip(trend_strength, -1, 1))
                
                return np.array(trends)
        
        self.trend_model = DummyTrendModel()
        logger.info("Created dummy trend model")
    
    def _create_dummy_volatility_model(self):
        """Create a simple dummy volatility classifier"""
        class DummyVolatilityModel:
            def predict(self, X):
                # Simple volatility classification
                vol_states = []
                for row in X:
                    atr = row[0] if len(row) > 0 else 0.5
                    
                    if atr < 0.3:
                        vol_states.append('LOW')
                    elif atr < 0.7:
                        vol_states.append('MEDIUM')
                    else:
                        vol_states.append('HIGH')
                
                return np.array(vol_states)
            
            def predict_proba(self, X):
                predictions = self.predict(X)
                proba = []
                for pred in predictions:
                    if pred == 'LOW':
                        proba.append([0.7, 0.2, 0.1])
                    elif pred == 'MEDIUM':
                        proba.append([0.2, 0.6, 0.2])
                    else:
                        proba.append([0.1, 0.2, 0.7])
                return np.array(proba)
        
        self.volatility_model = DummyVolatilityModel()
        logger.info("Created dummy volatility model")
    
    def _create_dummy_scalers(self):
        """Create dummy scalers"""
        class DummyScaler:
            def transform(self, X):
                return X
        
        self.scalers = {
            'regime': DummyScaler(),
            'trend': DummyScaler(),
            'volatility': DummyScaler()
        }
    
    def _create_dummy_models(self):
        """Create all dummy models"""
        self._create_dummy_regime_model()
        self._create_dummy_trend_model()
        self._create_dummy_volatility_model()
        self._create_dummy_scalers()
    
    def _prepare_features(
        self, 
        df: pd.DataFrame, 
        indicators: pd.DataFrame,
        feature_list: List[str]
    ) -> np.ndarray:
        """
        Prepare feature vector from dataframe and indicators
        
        Args:
            df: Raw OHLCV data
            indicators: Calculated indicators
            feature_list: List of feature names to extract
        
        Returns:
            Feature array ready for model input
        """
        features = []
        
        for feature_name in feature_list:
            if feature_name in indicators.columns:
                value = indicators[feature_name].iloc[-1]
            elif feature_name == 'price_change':
                value = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
            elif feature_name == 'ma_distance':
                if 'ma_medium' in indicators.columns:
                    value = (df['close'].iloc[-1] - indicators['ma_medium'].iloc[-1]) / indicators['ma_medium'].iloc[-1]
                else:
                    value = 0
            elif feature_name == 'volume_ratio':
                if 'volume' in df.columns:
                    avg_vol = df['volume'].tail(20).mean()
                    value = df['volume'].iloc[-1] / avg_vol if avg_vol > 0 else 1
                else:
                    value = 1
            elif feature_name == 'ma_cross':
                if 'ma_fast' in indicators.columns and 'ma_medium' in indicators.columns:
                    value = 1 if indicators['ma_fast'].iloc[-1] > indicators['ma_medium'].iloc[-1] else -1
                else:
                    value = 0
            elif feature_name == 'momentum':
                if 'momentum' in indicators.columns:
                    value = indicators['momentum'].iloc[-1]
                else:
                    value = 0
            elif feature_name == 'structure_score':
                value = 0  # Will be populated from structure analysis
            elif feature_name == 'price_range':
                value = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1]
            elif feature_name == 'volume':
                value = df['volume'].iloc[-1] if 'volume' in df.columns else 1000
            else:
                value = 0
            
            features.append(value if not pd.isna(value) else 0)
        
        return np.array(features).reshape(1, -1)
    
    def predict_regime(
        self, 
        df: pd.DataFrame, 
        indicators: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Predict market regime: TREND, RANGE, or CHAOS
        
        Returns:
            Dict with regime prediction and confidence
        """
        if not self.enabled or self.regime_model is None:
            return {'regime': 'UNKNOWN', 'confidence': 0.0}
        
        try:
            features = self._prepare_features(df, indicators, self.regime_features)
            
            if self.scalers and 'regime' in self.scalers:
                features = self.scalers['regime'].transform(features)
            
            prediction = self.regime_model.predict(features)[0]
            
            # Get probabilities if available
            if hasattr(self.regime_model, 'predict_proba'):
                probabilities = self.regime_model.predict_proba(features)[0]
                confidence = float(np.max(probabilities))
                
                return {
                    'regime': prediction,
                    'confidence': confidence,
                    'probabilities': {
                        'TREND': float(probabilities[0]),
                        'RANGE': float(probabilities[1]),
                        'CHAOS': float(probabilities[2])
                    }
                }
            else:
                return {
                    'regime': prediction,
                    'confidence': 0.7  # Default confidence
                }
        
        except Exception as e:
            logger.error(f"Error in regime prediction: {e}")
            return {'regime': 'UNKNOWN', 'confidence': 0.0}
    
    def predict_trend(
        self, 
        df: pd.DataFrame, 
        indicators: pd.DataFrame,
        structure_score: float = 0.0
    ) -> Dict[str, any]:
        """
        Predict trend direction and strength
        
        Returns:
            Dict with trend strength (-1 to 1) and confidence
        """
        if not self.enabled or self.trend_model is None:
            return {'trend_strength': 0.0, 'confidence': 0.0}
        
        try:
            features = self._prepare_features(df, indicators, self.trend_features)
            
            # Add structure score to features
            if len(features[0]) > 0:
                features[0][-1] = structure_score
            
            if self.scalers and 'trend' in self.scalers:
                features = self.scalers['trend'].transform(features)
            
            trend_strength = float(self.trend_model.predict(features)[0])
            trend_strength = np.clip(trend_strength, -1, 1)
            
            return {
                'trend_strength': trend_strength,
                'direction': 'bullish' if trend_strength > 0.2 else 'bearish' if trend_strength < -0.2 else 'neutral',
                'confidence': abs(trend_strength)
            }
        
        except Exception as e:
            logger.error(f"Error in trend prediction: {e}")
            return {'trend_strength': 0.0, 'confidence': 0.0}
    
    def predict_volatility(
        self, 
        df: pd.DataFrame, 
        indicators: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Predict volatility state
        
        Returns:
            Dict with volatility classification and confidence
        """
        if not self.enabled or self.volatility_model is None:
            return {'state': 'MEDIUM', 'confidence': 0.0}
        
        try:
            features = self._prepare_features(df, indicators, self.volatility_features)
            
            if self.scalers and 'volatility' in self.scalers:
                features = self.scalers['volatility'].transform(features)
            
            prediction = self.volatility_model.predict(features)[0]
            
            if hasattr(self.volatility_model, 'predict_proba'):
                probabilities = self.volatility_model.predict_proba(features)[0]
                confidence = float(np.max(probabilities))
            else:
                confidence = 0.7
            
            return {
                'state': prediction,
                'confidence': confidence
            }
        
        except Exception as e:
            logger.error(f"Error in volatility prediction: {e}")
            return {'state': 'MEDIUM', 'confidence': 0.0}
    
    def get_ml_insights(
        self, 
        df: pd.DataFrame, 
        indicators: pd.DataFrame,
        structure_score: float = 0.0
    ) -> Dict[str, any]:
        """
        Get comprehensive ML insights for trading decision
        
        Returns:
            Complete ML analysis dict
        """
        regime = self.predict_regime(df, indicators)
        trend = self.predict_trend(df, indicators, structure_score)
        volatility = self.predict_volatility(df, indicators)
        
        # Calculate overall ML score
        ml_score = 0.0
        
        if regime['regime'] == 'TREND':
            ml_score += trend['trend_strength'] * regime['confidence']
        elif regime['regime'] == 'RANGE':
            ml_score += trend['trend_strength'] * 0.5 * regime['confidence']
        # CHAOS regime reduces confidence
        
        return {
            'regime': regime,
            'trend': trend,
            'volatility': volatility,
            'ml_score': ml_score,
            'timestamp': pd.Timestamp.now()
        }
