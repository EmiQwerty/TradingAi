"""
Strategy Ensemble - Sistema che combina più strategie
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Type
from strategies.strategy_base import StrategyBase, TradingSignal, SignalType
from strategies.rsi_strategy import RSIMeanReversionStrategy
from strategies.breakout_strategy import BreakoutStrategy
from strategies.trend_following_strategy import TrendFollowingStrategy


class StrategyEnsemble:
    """
    Ensemble di strategie che:
    1. Esegue tutte le strategie in parallelo
    2. Combina segnali multipli sulla stessa barra
    3. Filtra segnali conflittuali
    4. Aumenta confidence se strategie concordano
    5. Assegna pesi diversi per strategia
    
    Modalità:
    - VOTING: Segnale solo se N strategie concordano
    - WEIGHTED: Somma pesata di confidence
    - ALL: Restituisce tutti i segnali separatamente
    """
    
    def __init__(self, 
                 mode: str = "WEIGHTED",
                 min_votes: int = 2,
                 strategy_weights: Dict[str, float] = None):
        """
        Args:
            mode: VOTING | WEIGHTED | ALL
            min_votes: Minimo numero strategie che concordano (per VOTING)
            strategy_weights: Pesi per ogni strategia {nome: peso}
        """
        self.mode = mode.upper()
        self.min_votes = min_votes
        
        # Inizializza strategie
        self.strategies = {
            'RSI': RSIMeanReversionStrategy(),
            'Breakout': BreakoutStrategy(),
            'TrendFollowing': TrendFollowingStrategy()
        }
        
        # Pesi default (modificabili)
        self.weights = strategy_weights or {
            'RSI': 1.0,
            'Breakout': 1.2,  # Breakout leggermente più affidabile
            'TrendFollowing': 1.3  # Trend following più affidabile
        }
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Genera segnali combinati da tutte le strategie"""
        
        # Raccogli segnali da ogni strategia
        all_signals = {}
        for name, strategy in self.strategies.items():
            try:
                signals = strategy.generate_signals(data)
                all_signals[name] = signals
                print(f"{name}: {len(signals)} segnali generati")
            except Exception as e:
                print(f"Errore in {name}: {e}")
                all_signals[name] = []
        
        # Modalità ALL: restituisci tutti i segnali
        if self.mode == "ALL":
            return self._merge_all_signals(all_signals)
        
        # Modalità VOTING o WEIGHTED: combina segnali sulla stessa barra
        elif self.mode in ["VOTING", "WEIGHTED"]:
            return self._combine_signals(all_signals, data)
        
        else:
            raise ValueError(f"Modalità non supportata: {self.mode}")
    
    def _merge_all_signals(self, all_signals: Dict[str, List[TradingSignal]]) -> List[TradingSignal]:
        """Restituisce tutti i segnali con tag strategia"""
        merged = []
        
        for strategy_name, signals in all_signals.items():
            for signal in signals:
                # Aggiungi info strategia al metadata
                signal.metadata['strategy'] = strategy_name
                signal.metadata['strategy_weight'] = self.weights.get(strategy_name, 1.0)
                merged.append(signal)
        
        # Ordina per timestamp
        merged.sort(key=lambda s: s.timestamp)
        return merged
    
    def _combine_signals(self, all_signals: Dict[str, List[TradingSignal]], 
                        data: pd.DataFrame) -> List[TradingSignal]:
        """Combina segnali sulla stessa barra con voting/weighted"""
        
        # Organizza segnali per timestamp
        signals_by_time = {}
        for strategy_name, signals in all_signals.items():
            for signal in signals:
                ts = signal.timestamp
                if ts not in signals_by_time:
                    signals_by_time[ts] = []
                signals_by_time[ts].append((strategy_name, signal))
        
        combined_signals = []
        
        for timestamp, strategy_signals in signals_by_time.items():
            # Separa BUY e SELL
            buy_signals = [(name, sig) for name, sig in strategy_signals if sig.signal_type == SignalType.BUY]
            sell_signals = [(name, sig) for name, sig in strategy_signals if sig.signal_type == SignalType.SELL]
            
            # Processa BUY
            if len(buy_signals) > 0:
                combined = self._create_combined_signal(buy_signals, timestamp, SignalType.BUY)
                if combined:
                    combined_signals.append(combined)
            
            # Processa SELL
            if len(sell_signals) > 0:
                combined = self._create_combined_signal(sell_signals, timestamp, SignalType.SELL)
                if combined:
                    combined_signals.append(combined)
        
        return combined_signals
    
    def _create_combined_signal(self, strategy_signals: List[tuple], 
                                timestamp, signal_type: SignalType) -> TradingSignal:
        """Crea segnale combinato da multiple strategie"""
        
        # VOTING mode: serve min_votes
        if self.mode == "VOTING":
            if len(strategy_signals) < self.min_votes:
                return None
        
        # Calcola medie pesate
        total_weight = sum(self.weights.get(name, 1.0) for name, _ in strategy_signals)
        
        weighted_entry = sum(
            sig.entry_price * self.weights.get(name, 1.0) 
            for name, sig in strategy_signals
        ) / total_weight
        
        weighted_sl = sum(
            sig.stop_loss * self.weights.get(name, 1.0) 
            for name, sig in strategy_signals
        ) / total_weight
        
        weighted_tp = sum(
            sig.take_profit * self.weights.get(name, 1.0) 
            for name, sig in strategy_signals
        ) / total_weight
        
        # Confidence: media pesata + bonus per concordanza
        base_confidence = sum(
            sig.confidence * self.weights.get(name, 1.0) 
            for name, sig in strategy_signals
        ) / total_weight
        
        # Bonus: +5% per ogni strategia addizionale oltre la prima
        agreement_bonus = (len(strategy_signals) - 1) * 0.05
        final_confidence = min(0.95, base_confidence + agreement_bonus)
        
        # Combina reasons
        strategies_list = ", ".join(name for name, _ in strategy_signals)
        combined_reason = f"{len(strategy_signals)} strategie concordano ({strategies_list})"
        
        # Combina metadata
        combined_metadata = {
            'strategies': [name for name, _ in strategy_signals],
            'votes': len(strategy_signals),
            'individual_confidences': {name: sig.confidence for name, sig in strategy_signals},
            'individual_reasons': {name: sig.reason for name, sig in strategy_signals}
        }
        
        # Merge metadata individuali
        for name, sig in strategy_signals:
            if sig.metadata:
                combined_metadata[f'{name}_metadata'] = sig.metadata
        
        return TradingSignal(
            timestamp=timestamp,
            signal_type=signal_type,
            entry_price=weighted_entry,
            stop_loss=weighted_sl,
            take_profit=weighted_tp,
            confidence=final_confidence,
            reason=combined_reason,
            metadata=combined_metadata
        )
    
    def backtest(self, data: pd.DataFrame) -> pd.DataFrame:
        """Backtest dell'ensemble completo"""
        signals = self.generate_signals(data)
        
        if len(signals) == 0:
            print("⚠️ Nessun segnale generato dall'ensemble")
            return pd.DataFrame()
        
        print(f"\n{'='*60}")
        print(f"ENSEMBLE BACKTEST - Mode: {self.mode}")
        print(f"{'='*60}")
        print(f"Totale segnali: {len(signals)}")
        print(f"  - BUY: {sum(1 for s in signals if s.signal_type == SignalType.BUY)}")
        print(f"  - SELL: {sum(1 for s in signals if s.signal_type == SignalType.SELL)}")
        
        if self.mode != "ALL":
            # Mostra distribuzione per numero di voti
            votes = [s.metadata.get('votes', 1) for s in signals]
            for v in range(1, 4):
                count = sum(1 for vote in votes if vote == v)
                if count > 0:
                    print(f"  - {v} strategie: {count} segnali")
        
        print(f"{'='*60}\n")
        
        # Crea DataFrame segnali
        results = []
        for signal in signals:
            results.append({
                'timestamp': signal.timestamp,
                'signal_type': signal.signal_type.value,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'confidence': signal.confidence,
                'reason': signal.reason,
                **signal.metadata
            })
        
        return pd.DataFrame(results)
    
    def get_strategy_stats(self, backtest_results: pd.DataFrame) -> Dict:
        """Analizza performance per strategia (se mode=ALL)"""
        if self.mode != "ALL" or 'strategy' not in backtest_results.columns:
            return {}
        
        stats = {}
        for strategy_name in self.strategies.keys():
            strategy_trades = backtest_results[backtest_results['strategy'] == strategy_name]
            
            if len(strategy_trades) == 0:
                continue
            
            stats[strategy_name] = {
                'total_signals': len(strategy_trades),
                'avg_confidence': strategy_trades['confidence'].mean(),
                'buy_signals': len(strategy_trades[strategy_trades['signal_type'] == 'BUY']),
                'sell_signals': len(strategy_trades[strategy_trades['signal_type'] == 'SELL'])
            }
        
        return stats
