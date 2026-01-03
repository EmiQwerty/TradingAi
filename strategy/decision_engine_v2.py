"""
Decision Engine v2 - Deterministic Trading Logic
Implementa logica deterministica con bias multi-timeframe e switching automatico trend/range.
Output JSON compatibile OANDA v20 API.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import json


logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Decision engine con logica deterministica.
    - Bias H4 (confirmation)
    - Bias H1 (entry signal)
    - Entry M5 (precise entry)
    - Switching automatico TREND/RANGE
    """
    
    def __init__(self, config: Dict, symbol_configs: Dict, confidence_agg, 
                 indicators_obj, structure_obj, ml_inference, volatility_obj, macro_filter):
        """
        Inizializza decision engine.
        
        Args:
            config: Global settings
            symbol_configs: Symbol-specific configs
            confidence_agg: ConfidenceAggregator istanza
            indicators_obj: TechnicalIndicators istanza
            structure_obj: StructureDetector istanza
            ml_inference: MLInference istanza
            volatility_obj: VolatilityAnalyzer istanza
            macro_filter: MacroFilter istanza
        """
        self.config = config
        self.symbol_configs = symbol_configs
        self.confidence_agg = confidence_agg
        self.indicators = indicators_obj
        self.structure = structure_obj
        self.ml = ml_inference
        self.volatility = volatility_obj
        self.macro = macro_filter
        
        logger.info("DecisionEngine v2 initialized (deterministic logic)")
    
    def generate_decisions(self, symbol: str, symbol_state: Dict) -> List[Dict]:
        """
        Genera decisioni di trading per un simbolo.
        
        Args:
            symbol: Simbolo (es. EUR_USD)
            symbol_state: SymbolState con dati per tutti i TF
        
        Returns:
            Lista decisioni [{action, order_type, entry, stop, take_profit, size, confidence}]
        """
        decisions = []
        
        # 1. Ottieni dati per i timeframe critici
        h4_data = symbol_state.timeframes.get('H4')
        h1_data = symbol_state.timeframes.get('H1')
        m5_data = symbol_state.timeframes.get('M5')
        
        if not h1_data or not h4_data:
            logger.debug(f"{symbol}: Missing H1 or H4 data")
            return decisions
        
        # 2. Analizza regime (da ML su H4)
        regime = h4_data.ml_regime
        logger.info(f"{symbol}: Regime = {regime}")
        
        # 3. Determina strategia basata su regime
        if regime == 'TREND':
            decision = self._trend_following_logic(symbol, h4_data, h1_data, m5_data)
        elif regime == 'RANGE':
            decision = self._mean_reversion_logic(symbol, h4_data, h1_data, m5_data)
        else:  # CHAOS
            logger.info(f"{symbol}: CHAOS regime, no trading")
            decision = None
        
        if decision:
            decisions.append(decision)
        
        return decisions
    
    def _trend_following_logic(self, symbol: str, h4_data, h1_data, m5_data) -> Optional[Dict]:
        """
        Logica trend-following.
        
        Bias H4: Confirm trend direction
        Bias H1: Refine entry signal
        Entry M5: Precise entry
        """
        logger.info(f"{symbol}: TREND-FOLLOWING strategy")
        
        # Step 1: H4 Confirmation (macro trend)
        h4_trend = h4_data.ml_trend_strength
        h4_regime = h4_data.ml_regime
        
        if h4_trend < 0.5:
            logger.debug(f"{symbol}: H4 trend weak ({h4_trend:.2f}), skip")
            return None
        
        # Determinare direzione da H4
        h4_rsi = h4_data.indicators.get('rsi', 50)
        if h4_rsi > 50:
            h4_direction = 'BUY'
        else:
            h4_direction = 'SELL'
        
        logger.info(f"{symbol}: H4 direction = {h4_direction}, RSI={h4_rsi:.2f}")
        
        # Step 2: H1 Refinement (entry signal)
        h1_confidence = h1_data.indicators.get('confidence_score', 0.5)
        h1_rsi = h1_data.indicators.get('rsi', 50)
        h1_structure = h1_data.structure
        
        # Controlla H1 sia aligned con H4
        h1_direction = 'BUY' if h1_rsi > 50 else 'SELL'
        
        if h1_direction != h4_direction:
            logger.debug(f"{symbol}: H1 direction {h1_direction} != H4 {h4_direction}, skip")
            return None
        
        # Controlla BOS/CHOCH su H1
        if h1_structure.get('bos', {}).get('detected'):
            bos_direction = h1_structure['bos'].get('direction')
            if bos_direction == h1_direction.lower():
                logger.info(f"{symbol}: H1 BOS confirmed for {h1_direction}")
            else:
                logger.debug(f"{symbol}: H1 BOS against direction, skip")
                return None
        
        # Step 3: M5 Precise Entry
        m5_data = m5_data or h1_data  # Fallback a H1
        m5_rsi = m5_data.indicators.get('rsi', 50)
        m5_price = m5_data.ohlcv.get('close', 1.0)
        
        # M5 Entry conditions per direction
        if h1_direction == 'BUY':
            # Vuoi BUY su M5 quando RSI fra 30-50 (non overbought)
            if not (30 < m5_rsi < 55):
                logger.debug(f"{symbol}: M5 RSI {m5_rsi} not suitable for BUY")
                return None
            
            # Controlla che sia su uptrend M5
            m5_ema9 = m5_data.indicators.get('ema_9', m5_price)
            if m5_price < m5_ema9:
                logger.debug(f"{symbol}: M5 price below EMA9, not suitable")
                return None
            
            units = 100000  # 1 lot EUR_USD
        
        else:  # SELL
            # Vuoi SELL su M5 quando RSI fra 50-70 (not oversold)
            if not (45 < m5_rsi < 70):
                logger.debug(f"{symbol}: M5 RSI {m5_rsi} not suitable for SELL")
                return None
            
            m5_ema9 = m5_data.indicators.get('ema_9', m5_price)
            if m5_price > m5_ema9:
                logger.debug(f"{symbol}: M5 price above EMA9, not suitable")
                return None
            
            units = -100000  # 1 lot short
        
        # Step 4: Calculate SL/TP with ATR
        atr = h1_data.indicators.get('atr', 0.001)
        if atr == 0:
            atr = 0.001
        
        current_price = m5_price
        
        # SL at 2x ATR below/above entry
        if h1_direction == 'BUY':
            stop_loss = current_price - (2 * atr)
            take_profit = current_price + (3 * atr)  # 1.5:1 risk/reward
        else:
            stop_loss = current_price + (2 * atr)
            take_profit = current_price - (3 * atr)
        
        # Step 5: Calculate size based on risk
        account_balance = 100000  # Should come from state store
        risk_pct = 0.01  # 1% risk per trade
        risk_amount = account_balance * risk_pct
        
        distance_to_sl = abs(current_price - stop_loss)
        pip_value = 0.0001  # For EUR_USD
        risk_pips = distance_to_sl / pip_value
        
        # Size = risk_amount / (risk_pips * pip_value * contract_size)
        contract_size = 100000
        size = risk_amount / (risk_pips * pip_value * contract_size * 10)
        size = max(0.01, min(size, 10))  # Clamp to 0.01-10 lots
        
        # Converti units per OANDA format
        if h1_direction == 'BUY':
            final_units = int(size * contract_size)
        else:
            final_units = -int(size * contract_size)
        
        # Step 6: Build decision JSON
        decision = {
            'action': 'BUY' if units > 0 else 'SELL',
            'order_type': 'MARKET',
            'entry': float(round(current_price, 5)),
            'stop': float(round(stop_loss, 5)),
            'take_profit': float(round(take_profit, 5)),
            'size': float(round(size, 2)),
            'units': final_units,
            'confidence': float(round(h1_confidence, 2)),
            'symbol': symbol,
            'timeframe': 'M5',
            'regime': 'TREND',
            'reasoning': f"H4:{h4_direction}(RSI={h4_rsi:.1f}) H1:{h1_direction} M5:Entry(RSI={m5_rsi:.1f})",
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"DECISION: {symbol} {decision['action']} {size:.2f} @ {current_price:.5f} "
                   f"SL={stop_loss:.5f} TP={take_profit:.5f} Conf={h1_confidence:.2f}")
        
        return decision
    
    def _mean_reversion_logic(self, symbol: str, h4_data, h1_data, m5_data) -> Optional[Dict]:
        """
        Logica mean-reversion (RANGE regime).
        
        Compra su estremi bassi, vende su estremi alti.
        """
        logger.info(f"{symbol}: MEAN-REVERSION strategy")
        
        # Prendi dati da H1 per mean reversion
        h1_rsi = h1_data.indicators.get('rsi', 50)
        h1_price = h1_data.ohlcv.get('close', 1.0)
        h1_bb = h1_data.indicators.get('bollinger_bands', {})
        
        # Controlla se sei ai limiti delle Bollinger Bands
        bb_upper = h1_bb.get('upper', h1_price)
        bb_lower = h1_bb.get('lower', h1_price)
        bb_middle = h1_bb.get('middle', h1_price)
        
        # Mean reversion su RSI extremes
        if h1_rsi < 30:
            # Oversold, BUY (expect bounce to middle)
            direction = 'BUY'
            target_price = bb_middle
            take_profit = target_price
            units = 100000
        
        elif h1_rsi > 70:
            # Overbought, SELL (expect bounce down)
            direction = 'SELL'
            target_price = bb_middle
            take_profit = target_price
            units = -100000
        
        else:
            # RSI not at extremes
            logger.debug(f"{symbol}: RSI={h1_rsi:.1f}, not at extremes for mean reversion")
            return None
        
        # SL al di là del BB band opposto
        atr = h1_data.indicators.get('atr', 0.001)
        if direction == 'BUY':
            stop_loss = bb_lower - atr
        else:
            stop_loss = bb_upper + atr
        
        # Size calculation
        account_balance = 100000
        risk_pct = 0.01
        risk_amount = account_balance * risk_pct
        
        distance_to_sl = abs(h1_price - stop_loss)
        pip_value = 0.0001
        risk_pips = distance_to_sl / pip_value
        contract_size = 100000
        
        size = risk_amount / (risk_pips * pip_value * contract_size * 10)
        size = max(0.01, min(size, 10))
        
        # Confidence basato su RSI estremi
        if h1_rsi < 20 or h1_rsi > 80:
            confidence = 0.75
        elif h1_rsi < 30 or h1_rsi > 70:
            confidence = 0.65
        else:
            confidence = 0.55
        
        decision = {
            'action': direction,
            'order_type': 'MARKET',
            'entry': float(round(h1_price, 5)),
            'stop': float(round(stop_loss, 5)),
            'take_profit': float(round(take_profit, 5)),
            'size': float(round(size, 2)),
            'units': int(size * contract_size) if units > 0 else -int(size * contract_size),
            'confidence': float(round(confidence, 2)),
            'symbol': symbol,
            'timeframe': 'H1',
            'regime': 'RANGE',
            'reasoning': f"Mean-reversion on RSI={h1_rsi:.1f} (target={target_price:.5f})",
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"DECISION: {symbol} {decision['action']} {size:.2f} @ {h1_price:.5f} "
                   f"SL={stop_loss:.5f} TP={take_profit:.5f} Conf={confidence:.2f}")
        
        return decision
