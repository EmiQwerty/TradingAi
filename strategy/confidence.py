"""
Confidence Score Aggregator
Combines ML predictions, indicators, structure, and macro filter into final trading confidence
"""

import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class ConfidenceAggregator:
    """
    Aggregates multiple signal sources into unified confidence score
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
        # Weights for different components
        self.weights = {
            'ml': 0.35,           # 35% ML models
            'indicators': 0.30,   # 30% technical indicators
            'structure': 0.25,    # 25% market structure
            'macro': 0.10         # 10% macro filter
        }
        
        logger.info("ConfidenceAggregator initialized")
    
    def calculate_indicator_score(self, indicators: Dict) -> float:
        """
        Calculate score from technical indicators
        
        Returns:
            Score from -1 (bearish) to 1 (bullish)
        """
        score = 0.0
        count = 0
        
        # RSI score
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if not np.isnan(rsi):
                if rsi < 30:
                    score += 0.8  # Oversold - bullish
                elif rsi < 40:
                    score += 0.4
                elif rsi > 70:
                    score -= 0.8  # Overbought - bearish
                elif rsi > 60:
                    score -= 0.4
                count += 1
        
        # Stochastic score
        if 'stoch_k' in indicators and 'stoch_d' in indicators:
            stoch_k = indicators['stoch_k']
            stoch_d = indicators['stoch_d']
            
            if not (np.isnan(stoch_k) or np.isnan(stoch_d)):
                if stoch_k < 20:
                    score += 0.7
                elif stoch_k < 30:
                    score += 0.4
                elif stoch_k > 80:
                    score -= 0.7
                elif stoch_k > 70:
                    score -= 0.4
                
                # K/D crossover
                if stoch_k > stoch_d:
                    score += 0.3
                else:
                    score -= 0.3
                
                count += 1
        
        # Moving average alignment
        if all(k in indicators for k in ['ma_fast', 'ma_medium', 'ma_slow']):
            ma_fast = indicators['ma_fast']
            ma_medium = indicators['ma_medium']
            ma_slow = indicators['ma_slow']
            
            if not (np.isnan(ma_fast) or np.isnan(ma_medium) or np.isnan(ma_slow)):
                # Bullish alignment: fast > medium > slow
                if ma_fast > ma_medium > ma_slow:
                    score += 1.0
                # Bearish alignment: fast < medium < slow
                elif ma_fast < ma_medium < ma_slow:
                    score -= 1.0
                # Partial alignment
                elif ma_fast > ma_medium:
                    score += 0.5
                elif ma_fast < ma_medium:
                    score -= 0.5
                
                count += 1
        
        # MACD score
        if 'macd' in indicators and 'macd_signal' in indicators:
            macd = indicators['macd']
            signal = indicators['macd_signal']
            
            if not (np.isnan(macd) or np.isnan(signal)):
                if macd > signal:
                    score += 0.5
                else:
                    score -= 0.5
                count += 1
        
        # PMax trend
        if 'pmax_trend' in indicators:
            pmax_trend = indicators['pmax_trend']
            if not np.isnan(pmax_trend):
                score += pmax_trend * 0.8  # 1 or -1
                count += 1
        
        # ADX for trend strength (doesn't add direction, just weight)
        strength_multiplier = 1.0
        if 'adx' in indicators:
            adx = indicators['adx']
            if not np.isnan(adx):
                if adx > 25:
                    strength_multiplier = 1.2  # Strong trend
                elif adx < 20:
                    strength_multiplier = 0.7  # Weak trend
        
        # Normalize and apply strength
        if count > 0:
            score = (score / count) * strength_multiplier
        
        return np.clip(score, -1, 1)
    
    def calculate_structure_score(self, structure: Dict) -> float:
        """
        Calculate score from market structure
        
        Returns:
            Score from -1 to 1
        """
        score = 0.0
        
        if 'structure_score' in structure:
            trend_structure = structure['structure_score'].get('trend_structure', 0)
            strength = structure['structure_score'].get('structure_strength', 0)
            
            score = trend_structure * strength
        
        # Bonus for BOS (continuation)
        if structure.get('bos', {}).get('detected'):
            bos_direction = structure['bos'].get('direction')
            bos_strength = structure['bos'].get('strength', 0)
            
            if bos_direction == 'bullish':
                score += 0.3 * bos_strength
            elif bos_direction == 'bearish':
                score -= 0.3 * bos_strength
        
        # CHOCH indicates reversal (moderate weight)
        if structure.get('choch', {}).get('detected'):
            choch_direction = structure['choch'].get('direction')
            choch_strength = structure['choch'].get('strength', 0)
            
            if choch_direction == 'bullish':
                score += 0.2 * choch_strength
            elif choch_direction == 'bearish':
                score -= 0.2 * choch_strength
        
        return np.clip(score, -1, 1)
    
    def calculate_ml_score(self, ml_insights: Dict) -> float:
        """
        Calculate score from ML predictions
        
        Returns:
            Score from -1 to 1
        """
        score = ml_insights.get('ml_score', 0.0)
        
        # Adjust based on regime
        regime = ml_insights.get('regime', {}).get('regime')
        regime_confidence = ml_insights.get('regime', {}).get('confidence', 0)
        
        if regime == 'TREND':
            # Trend regime: follow trend prediction strongly
            score *= 1.2 * regime_confidence
        elif regime == 'RANGE':
            # Range regime: mean reversion, reduce trend following
            score *= 0.6 * regime_confidence
        elif regime == 'CHAOS':
            # Chaos regime: reduce all signals
            score *= 0.3 * regime_confidence
        
        return np.clip(score, -1, 1)
    
    def aggregate_confidence(
        self,
        ml_insights: Dict,
        indicators: Dict,
        structure: Dict,
        macro_adjustment: Dict,
        volatility_state: Dict = None
    ) -> Dict[str, any]:
        """
        Aggregate all components into final confidence score
        
        Returns:
            Dict with final confidence and breakdown
        """
        # Calculate individual scores
        ml_score = self.calculate_ml_score(ml_insights)
        indicator_score = self.calculate_indicator_score(indicators)
        structure_score = self.calculate_structure_score(structure)
        
        # Macro filter (multiplicative adjustment, not additive)
        macro_factor = macro_adjustment.get('adjustment_factor', 1.0)
        
        # Weighted combination
        base_score = (
            ml_score * self.weights['ml'] +
            indicator_score * self.weights['indicators'] +
            structure_score * self.weights['structure']
        )
        
        # Apply macro filter
        final_score = base_score * macro_factor
        
        # Volatility adjustment
        if volatility_state:
            vol_state = volatility_state.get('state', 'MEDIUM')
            if vol_state == 'EXTREME':
                final_score *= 0.5  # Reduce confidence in extreme volatility
            elif vol_state == 'HIGH':
                final_score *= 0.8
        
        # Convert to confidence (0 to 1) and direction
        confidence = abs(final_score)
        direction = 'long' if final_score > 0 else 'short' if final_score < 0 else 'neutral'
        
        return {
            'confidence': confidence,
            'direction': direction,
            'raw_score': final_score,
            'breakdown': {
                'ml_score': ml_score,
                'indicator_score': indicator_score,
                'structure_score': structure_score,
                'macro_factor': macro_factor
            },
            'components': {
                'ml': ml_score * self.weights['ml'],
                'indicators': indicator_score * self.weights['indicators'],
                'structure': structure_score * self.weights['structure']
            }
        }
    
    def meets_entry_criteria(
        self,
        confidence_result: Dict,
        min_confidence: float = 0.60,
        require_alignment: bool = True
    ) -> Dict[str, bool]:
        """
        Check if confidence meets entry criteria
        
        Returns:
            Dict with entry decision and reasons
        """
        confidence = confidence_result['confidence']
        breakdown = confidence_result['breakdown']
        
        meets_min = confidence >= min_confidence
        
        # Check alignment between components
        alignment = True
        if require_alignment:
            scores = [
                breakdown['ml_score'],
                breakdown['indicator_score'],
                breakdown['structure_score']
            ]
            
            # All should have same sign (or be near zero)
            positive_count = sum(1 for s in scores if s > 0.1)
            negative_count = sum(1 for s in scores if s < -0.1)
            
            # Good alignment: most signals agree
            alignment = (positive_count >= 2 or negative_count >= 2)
        
        can_enter = meets_min and alignment
        
        reasons = []
        if not meets_min:
            reasons.append(f"confidence_below_threshold ({confidence:.2f} < {min_confidence})")
        if not alignment:
            reasons.append("components_not_aligned")
        if can_enter:
            reasons.append("entry_criteria_met")
        
        return {
            'can_enter': can_enter,
            'meets_min_confidence': meets_min,
            'has_alignment': alignment,
            'reasons': reasons
        }
