"""
Correlation Analysis Module
Monitors correlations between trading pairs to avoid over-exposure
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Analyzes price correlations between trading pairs
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.correlation_config = config.get('correlation', {})
        
        self.enabled = self.correlation_config.get('enabled', True)
        self.calculation_period = self.correlation_config.get('calculation_period', 30)
        self.max_correlation = self.correlation_config.get('limits', {}).get('max_correlation', 0.7)
        
        # Cache for correlation data
        self.price_history: Dict[str, pd.DataFrame] = {}
        self.correlation_matrix = None
        self.last_update = None
        
        logger.info(f"CorrelationAnalyzer initialized (enabled: {self.enabled})")
    
    def update_price_history(self, symbol: str, prices: pd.Series):
        """
        Update price history for correlation calculation
        
        Args:
            symbol: Trading symbol
            prices: Series of prices (typically close prices)
        """
        if symbol not in self.price_history:
            self.price_history[symbol] = pd.DataFrame()
        
        # Keep only recent data
        if len(prices) > self.calculation_period:
            prices = prices.tail(self.calculation_period)
        
        self.price_history[symbol] = prices
    
    def calculate_correlation_matrix(
        self,
        symbols: List[str] = None
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between all symbols
        
        Returns:
            Correlation matrix as DataFrame
        """
        if not self.enabled:
            return pd.DataFrame()
        
        if symbols is None:
            symbols = list(self.price_history.keys())
        
        # Build price matrix
        price_data = {}
        
        for symbol in symbols:
            if symbol in self.price_history and not self.price_history[symbol].empty:
                price_data[symbol] = self.price_history[symbol]
        
        if len(price_data) < 2:
            return pd.DataFrame()
        
        # Align all series to same index
        df = pd.DataFrame(price_data)
        
        # Calculate returns
        returns = df.pct_change().dropna()
        
        # Calculate correlation
        if len(returns) >= 10:  # Need minimum data points
            self.correlation_matrix = returns.corr()
            return self.correlation_matrix
        
        return pd.DataFrame()
    
    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """
        Get correlation between two symbols
        
        Returns:
            Correlation coefficient (-1 to 1)
        """
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            self.calculate_correlation_matrix()
        
        if self.correlation_matrix is not None and not self.correlation_matrix.empty:
            if symbol1 in self.correlation_matrix.index and symbol2 in self.correlation_matrix.columns:
                return self.correlation_matrix.loc[symbol1, symbol2]
        
        return 0.0
    
    def check_correlation_risk(
        self,
        new_symbol: str,
        new_direction: str,
        existing_positions: List[Dict]
    ) -> Dict[str, any]:
        """
        Check if new position would create correlation risk
        
        Args:
            new_symbol: Symbol for potential new position
            new_direction: 'long' or 'short'
            existing_positions: List of current positions
        
        Returns:
            Dict with risk assessment
        """
        if not self.enabled or not existing_positions:
            return {
                'risk_detected': False,
                'max_correlation': 0.0,
                'correlated_positions': []
            }
        
        # Update correlation matrix
        self.calculate_correlation_matrix()
        
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            return {
                'risk_detected': False,
                'max_correlation': 0.0,
                'correlated_positions': []
            }
        
        correlated_positions = []
        max_correlation = 0.0
        
        for position in existing_positions:
            pos_symbol = position.get('symbol')
            pos_direction = position.get('direction')
            
            if pos_symbol == new_symbol:
                continue  # Skip same symbol
            
            correlation = self.get_correlation(new_symbol, pos_symbol)
            
            # Adjust correlation based on direction
            # If both long or both short, use positive correlation
            # If one long and one short, invert correlation
            if new_direction != pos_direction:
                effective_correlation = -correlation
            else:
                effective_correlation = correlation
            
            # Check if correlation exceeds threshold
            if abs(effective_correlation) > self.max_correlation:
                correlated_positions.append({
                    'symbol': pos_symbol,
                    'direction': pos_direction,
                    'correlation': effective_correlation,
                    'position_size': position.get('size', 0)
                })
                
                if abs(effective_correlation) > abs(max_correlation):
                    max_correlation = effective_correlation
        
        risk_detected = len(correlated_positions) > 0
        
        if risk_detected:
            logger.warning(
                f"Correlation risk: {new_symbol} correlated with "
                f"{len(correlated_positions)} existing positions "
                f"(max correlation: {max_correlation:.2f})"
            )
        
        return {
            'risk_detected': risk_detected,
            'max_correlation': max_correlation,
            'correlated_positions': correlated_positions,
            'recommendation': 'reduce_size' if risk_detected else 'proceed'
        }
    
    def get_portfolio_correlation_exposure(
        self,
        positions: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate overall portfolio correlation exposure
        
        Returns:
            Dict with exposure metrics
        """
        if not positions or len(positions) < 2:
            return {
                'avg_correlation': 0.0,
                'max_correlation': 0.0,
                'diversification_score': 1.0
            }
        
        self.calculate_correlation_matrix()
        
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            return {
                'avg_correlation': 0.0,
                'max_correlation': 0.0,
                'diversification_score': 1.0
            }
        
        # Calculate average correlation between all positions
        correlations = []
        
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i+1:]:
                symbol1 = pos1.get('symbol')
                symbol2 = pos2.get('symbol')
                
                if symbol1 != symbol2:
                    corr = self.get_correlation(symbol1, symbol2)
                    
                    # Adjust for direction
                    if pos1.get('direction') != pos2.get('direction'):
                        corr = -corr
                    
                    correlations.append(abs(corr))
        
        if correlations:
            avg_corr = np.mean(correlations)
            max_corr = np.max(correlations)
            
            # Diversification score: 1.0 = fully diversified, 0.0 = highly correlated
            diversification_score = 1.0 - avg_corr
        else:
            avg_corr = 0.0
            max_corr = 0.0
            diversification_score = 1.0
        
        return {
            'avg_correlation': avg_corr,
            'max_correlation': max_corr,
            'diversification_score': diversification_score,
            'position_pairs': len(correlations)
        }
    
    def suggest_hedge(
        self,
        positions: List[Dict]
    ) -> Optional[Dict]:
        """
        Suggest hedging position based on current portfolio
        
        Returns:
            Hedge suggestion or None
        """
        if not positions:
            return None
        
        exposure = self.get_portfolio_correlation_exposure(positions)
        
        # If portfolio is well diversified, no hedge needed
        if exposure['diversification_score'] > 0.7:
            return None
        
        # Find most exposed direction
        long_exposure = sum(p.get('size', 0) for p in positions if p.get('direction') == 'long')
        short_exposure = sum(p.get('size', 0) for p in positions if p.get('direction') == 'short')
        
        net_exposure = long_exposure - short_exposure
        
        if abs(net_exposure) > 0.5:  # Significant net exposure
            hedge_direction = 'short' if net_exposure > 0 else 'long'
            hedge_size = abs(net_exposure) * 0.5  # Hedge 50% of exposure
            
            return {
                'action': 'hedge',
                'direction': hedge_direction,
                'size': hedge_size,
                'reason': f'net_exposure_{net_exposure:.2f}',
                'diversification_score': exposure['diversification_score']
            }
        
        return None
