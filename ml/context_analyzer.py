"""
Context Analyzer - Analizza il contesto di trading per capire PERCHÉ funzionano i trade
Confronta vincenti vs perdenti, identifica pattern ricorrenti
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class ContextAnalyzer:
    """
    Analizza il contesto di trading:
    - Che feature hanno i trade vincenti vs perdenti?
    - Quali condizioni di mercato favoriscono il successo?
    - Quali pattern ricorrono nei vincenti?
    - Come interpretare i risultati?
    """
    
    def __init__(self):
        """Inizializza analyzer"""
        self.trades_df = None
        self.feature_columns = None
        self.label_column = 'label_win_loss'
        
        logger.info("ContextAnalyzer initialized")
    
    def analyze_trades(self, trades_df: pd.DataFrame, label_col: str = 'label_win_loss') -> Dict:
        """
        Analizza trades per capire differenze tra vincenti e perdenti.
        
        Args:
            trades_df: DataFrame con trades (da BacktestEngine)
            label_col: Nome colonna label (0=loss, 1=win)
        
        Returns:
            Dict con analisi dettagliata
        """
        
        self.trades_df = trades_df.copy()
        self.label_column = label_col
        
        # Identifica feature columns (tutti tranne metadata)
        metadata_cols = {
            'trade_id', 'symbol', 'side', 'entry_price', 'entry_time',
            'exit_price', 'exit_time', 'stop_loss', 'take_profit',
            'pnl', 'pnl_pips', 'pnl_percent', 'duration_bars',
            'label_win_loss', 'label_magnitude', 'label_exit_reason'
        }
        self.feature_columns = [
            col for col in trades_df.columns
            if col not in metadata_cols
        ]
        
        # Split vincenti/perdenti
        wins = trades_df[trades_df[label_col] == 1]
        losses = trades_df[trades_df[label_col] == 0]
        
        logger.info(f"Analyzing {len(trades_df)} trades: {len(wins)} wins, {len(losses)} losses")
        
        analysis = {
            'overview': self._analyze_overview(trades_df, wins, losses),
            'feature_comparison': self._analyze_feature_comparison(wins, losses),
            'statistical_significance': self._analyze_statistical_significance(wins, losses),
            'market_conditions': self._analyze_market_conditions(wins, losses),
            'trading_rules': self._extract_trading_rules(wins, losses),
            'exit_analysis': self._analyze_exit_reasons(wins, losses),
        }
        
        return analysis
    
    def _analyze_overview(self, all_trades, wins, losses) -> Dict:
        """Overview statistiche base"""
        
        total = len(all_trades)
        win_count = len(wins)
        loss_count = len(losses)
        
        return {
            'total_trades': total,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': win_count / total if total > 0 else 0,
            'avg_win_pnl': float(wins['pnl_percent'].mean()) if len(wins) > 0 else 0,
            'avg_loss_pnl': float(losses['pnl_percent'].mean()) if len(losses) > 0 else 0,
            'profit_factor': self._calculate_profit_factor(wins, losses),
            'expectancy': self._calculate_expectancy(wins, losses),
        }
    
    def _analyze_feature_comparison(self, wins: pd.DataFrame, losses: pd.DataFrame) -> Dict:
        """
        Confronta valori feature tra vincenti e perdenti.
        
        Returns:
            Dict {feature_name: {wins_avg, losses_avg, diff, wins_better}}
        """
        
        comparison = {}
        
        for feat in self.feature_columns:
            if feat in wins.columns and feat in losses.columns:
                
                wins_vals = wins[feat].dropna()
                losses_vals = losses[feat].dropna()
                
                if len(wins_vals) == 0 or len(losses_vals) == 0:
                    continue
                
                wins_avg = float(wins_vals.mean())
                losses_avg = float(losses_vals.mean())
                wins_std = float(wins_vals.std())
                losses_std = float(losses_vals.std())
                
                # Determinazione se wins hanno valori "migliori"
                # (dipende dal feature, ma per semplificazione usiamo il valore assoluto)
                diff = wins_avg - losses_avg
                wins_better = diff > 0  # Semplificazione
                
                comparison[feat] = {
                    'wins_avg': wins_avg,
                    'losses_avg': losses_avg,
                    'wins_std': wins_std,
                    'losses_std': losses_std,
                    'difference': diff,
                    'wins_better': wins_better,
                }
        
        return comparison
    
    def _analyze_statistical_significance(self, wins: pd.DataFrame, losses: pd.DataFrame) -> Dict:
        """
        T-test per determinare se differenze sono statisticamente significative.
        p-value < 0.05 = significativo
        """
        
        significance = {}
        
        for feat in self.feature_columns:
            if feat not in wins.columns or feat not in losses.columns:
                continue
            
            wins_vals = wins[feat].dropna()
            losses_vals = losses[feat].dropna()
            
            if len(wins_vals) < 2 or len(losses_vals) < 2:
                continue
            
            try:
                t_stat, p_value = stats.ttest_ind(wins_vals, losses_vals)
                
                significance[feat] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'is_significant': p_value < 0.05,
                }
            except Exception as e:
                logger.warning(f"T-test failed for {feat}: {e}")
        
        return significance
    
    def _analyze_market_conditions(self, wins: pd.DataFrame, losses: pd.DataFrame) -> Dict:
        """
        Analizza condizioni di mercato quando vincono vs perdono i trade.
        """
        
        conditions = {}
        
        # Trend analysis
        if 'entry_trend' in self.feature_columns:
            trend_wins = wins['entry_trend'].value_counts() if 'entry_trend' in wins.columns else {}
            trend_losses = losses['entry_trend'].value_counts() if 'entry_trend' in losses.columns else {}
            
            conditions['trend_distribution'] = {
                'wins': trend_wins.to_dict(),
                'losses': trend_losses.to_dict(),
            }
        
        # RSI analysis
        if 'entry_rsi' in self.feature_columns:
            wins_rsi = wins['entry_rsi'].dropna()
            losses_rsi = losses['entry_rsi'].dropna()
            
            conditions['rsi'] = {
                'wins_avg': float(wins_rsi.mean()),
                'losses_avg': float(losses_rsi.mean()),
                'wins_oversold': (wins_rsi < 30).sum(),
                'wins_overbought': (wins_rsi > 70).sum(),
                'losses_oversold': (losses_rsi < 30).sum(),
                'losses_overbought': (losses_rsi > 70).sum(),
            }
        
        # Volatility analysis
        if 'entry_volatility' in self.feature_columns:
            wins_vol = wins['entry_volatility'].dropna()
            losses_vol = losses['entry_volatility'].dropna()
            
            conditions['volatility'] = {
                'wins_avg': float(wins_vol.mean()),
                'losses_avg': float(losses_vol.mean()),
            }
        
        return conditions
    
    def _extract_trading_rules(self, wins: pd.DataFrame, losses: pd.DataFrame) -> List[Dict]:
        """
        Estrae semplici regole di trading dai dati.
        Es: "Se RSI > 50 e trend=UP allora 65% win rate"
        """
        
        rules = []
        
        # Regola 1: Confluence threshold
        if 'entry_confluence' in wins.columns:
            wins_conf = wins['entry_confluence'].dropna()
            losses_conf = losses['entry_confluence'].dropna()
            
            wins_mean = wins_conf.mean()
            losses_mean = losses_conf.mean()
            
            rules.append({
                'rule': 'Confluence score',
                'wins_avg': float(wins_mean),
                'losses_avg': float(losses_mean),
                'recommendation': f"Enter only if confluence > {wins_mean:.2f}",
                'expected_win_rate': len(wins[wins_conf > wins_mean]) / len(wins) if len(wins) > 0 else 0,
            })
        
        # Regola 2: Entry quality
        if 'entry_quality_score' in wins.columns:
            wins_qs = wins['entry_quality_score'].dropna()
            losses_qs = losses['entry_quality_score'].dropna()
            
            rules.append({
                'rule': 'Entry quality score',
                'wins_avg': float(wins_qs.mean()),
                'losses_avg': float(losses_qs.mean()),
                'recommendation': f"High quality: score > {wins_qs.quantile(0.75):.2f}",
            })
        
        # Regola 3: Trend-based
        if 'entry_trend' in wins.columns and 'entry_trend' in losses.columns:
            trend_wins = wins['entry_trend'].value_counts()
            trend_losses = losses['entry_trend'].value_counts()
            
            for trend_type in trend_wins.index:
                if trend_type in trend_losses.index:
                    win_count = trend_wins[trend_type]
                    loss_count = trend_losses[trend_type]
                    win_rate = win_count / (win_count + loss_count)
                    
                    if win_rate > 0.55:  # Better than 50%
                        rules.append({
                            'rule': f'Trend type: {trend_type}',
                            'win_count': int(win_count),
                            'loss_count': int(loss_count),
                            'win_rate': float(win_rate),
                            'recommendation': f"Trade preferentially in {trend_type} trend",
                        })
        
        return rules
    
    def _analyze_exit_reasons(self, wins: pd.DataFrame, losses: pd.DataFrame) -> Dict:
        """Analizza come i trade sono usciti (TP, SL, timeout)"""
        
        result = {}
        
        if 'label_exit_reason' in wins.columns:
            exit_wins = wins['label_exit_reason'].value_counts()
            exit_losses = losses['label_exit_reason'].value_counts()
            
            result['exit_distribution_wins'] = exit_wins.to_dict()
            result['exit_distribution_losses'] = exit_losses.to_dict()
            
            # Analysis
            result['analysis'] = {}
            for exit_type in set(list(exit_wins.index) + list(exit_losses.index)):
                win_count = exit_wins.get(exit_type, 0)
                loss_count = exit_losses.get(exit_type, 0)
                total = win_count + loss_count
                
                if total > 0:
                    result['analysis'][exit_type] = {
                        'total': total,
                        'wins': int(win_count),
                        'losses': int(loss_count),
                        'win_rate': float(win_count / total),
                    }
        
        return result
    
    def _calculate_profit_factor(self, wins: pd.DataFrame, losses: pd.DataFrame) -> float:
        """Calcola profit factor = gross_win / abs(gross_loss)"""
        
        gross_win = wins['pnl'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 1
        
        if gross_loss == 0:
            return float(gross_win) if gross_win > 0 else 0
        
        return float(gross_win / gross_loss)
    
    def _calculate_expectancy(self, wins: pd.DataFrame, losses: pd.DataFrame) -> float:
        """Calcola expectancy = (% win * avg win) - (% loss * avg loss)"""
        
        total = len(wins) + len(losses)
        if total == 0:
            return 0
        
        win_pct = len(wins) / total
        loss_pct = len(losses) / total
        
        avg_win = wins['pnl_percent'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_percent'].mean() if len(losses) > 0 else 0
        
        expectancy = (win_pct * avg_win) - (loss_pct * abs(avg_loss))
        
        return float(expectancy)
    
    def print_analysis_report(self, analysis: Dict):
        """Stampa report analisi leggibile"""
        
        print("\n" + "="*70)
        print("TRADING CONTEXT ANALYSIS - WHY DO TRADES WIN OR LOSE?")
        print("="*70 + "\n")
        
        # Overview
        overview = analysis['overview']
        print("OVERVIEW:")
        print(f"  Total Trades:     {overview['total_trades']}")
        print(f"  Winning Trades:   {overview['winning_trades']} ({overview['win_rate']:.1%})")
        print(f"  Losing Trades:    {overview['losing_trades']} ({1-overview['win_rate']:.1%})")
        print(f"  Avg Win PnL:      {overview['avg_win_pnl']:.2%}")
        print(f"  Avg Loss PnL:     {overview['avg_loss_pnl']:.2%}")
        print(f"  Profit Factor:    {overview['profit_factor']:.2f}")
        print(f"  Expectancy:       {overview['expectancy']:.4f}\n")
        
        # Top discriminating features
        print("TOP DISCRIMINATING FEATURES (Wins vs Losses):")
        feature_comp = analysis['feature_comparison']
        
        # Sort by absolute difference
        sorted_features = sorted(
            feature_comp.items(),
            key=lambda x: abs(x[1]['difference']),
            reverse=True
        )[:10]
        
        for feat, stats in sorted_features:
            diff_pct = ((stats['wins_avg'] - stats['losses_avg']) / abs(stats['losses_avg']))
            direction = '↑' if stats['wins_better'] else '↓'
            
            print(f"  {feat:<30} | Wins: {stats['wins_avg']:>8.4f} | Loss: {stats['losses_avg']:>8.4f} | {direction}")
        
        print()
        
        # Statistically significant features
        print("STATISTICALLY SIGNIFICANT FEATURES (p < 0.05):")
        sig = analysis['statistical_significance']
        
        significant_feats = [
            (feat, data) for feat, data in sig.items()
            if data['is_significant']
        ]
        
        if significant_feats:
            for feat, data in sorted(significant_feats, key=lambda x: x[1]['p_value'])[:5]:
                print(f"  {feat:<30} | p-value: {data['p_value']:.6f}")
        else:
            print("  (No statistically significant features found)")
        
        print()
        
        # Trading rules
        if analysis['trading_rules']:
            print("EXTRACTED TRADING RULES:")
            for i, rule in enumerate(analysis['trading_rules'], 1):
                print(f"\n  Rule {i}: {rule['rule']}")
                print(f"    Recommendation: {rule['recommendation']}")
                if 'win_rate' in rule:
                    print(f"    Win Rate: {rule['win_rate']:.1%}")
        
        print("\n" + "="*70 + "\n")
    
    def get_winning_trade_profile(self) -> Dict:
        """Crea profilo del trade vincente ideale"""
        
        if self.trades_df is None:
            return {}
        
        wins = self.trades_df[self.trades_df[self.label_column] == 1]
        
        profile = {}
        
        for feat in self.feature_columns:
            if feat in wins.columns:
                vals = wins[feat].dropna()
                if len(vals) > 0:
                    profile[feat] = {
                        'mean': float(vals.mean()),
                        'median': float(vals.median()),
                        'std': float(vals.std()),
                        'min': float(vals.min()),
                        'max': float(vals.max()),
                        'q25': float(vals.quantile(0.25)),
                        'q75': float(vals.quantile(0.75)),
                    }
        
        return profile
