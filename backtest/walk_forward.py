"""
Walk-Forward Validation
Implementa ottimizzazione rolling window con train/test split.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

from backtest.engine import BacktestEngine, BacktestConfig


logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Finestra train/test."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_results: Optional[Dict] = None
    test_results: Optional[Dict] = None
    optimized_params: Optional[Dict] = None


class WalkForwardValidator:
    """
    Walk-forward validation con rolling windows.
    Simula ottimizzazione progressiva su dati storici.
    """
    
    def __init__(
        self,
        settings: Dict,
        symbol_configs: Dict,
        risk_config: Dict,
        wf_config: Optional[Dict] = None
    ):
        self.settings = settings
        self.symbol_configs = symbol_configs
        self.risk_config = risk_config
        
        # Config walk-forward (da settings.yaml backtest section)
        default_wf = {
            'train_months': 6,
            'test_months': 1,
            'anchored': False,  # rolling vs anchored
            'min_trades': 30,  # minimo trades in train per considerare valido
            'optimization_metric': 'sharpe_ratio'  # metrica da ottimizzare
        }
        self.wf_config = wf_config or self.settings.get('backtest', {}).get('walk_forward', default_wf)
        
        self.windows: List[WalkForwardWindow] = []
        self.combined_results: Optional[Dict] = None
        
        logger.info(f"WalkForwardValidator: train={self.wf_config['train_months']}m, test={self.wf_config['test_months']}m")
    
    def run(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: List[str],
        timeframes: List[str],
        initial_capital: float = 100000.0
    ) -> Dict:
        """
        Esegue walk-forward validation completa.
        
        Args:
            start_date: Inizio periodo totale
            end_date: Fine periodo totale
            symbols: Lista simboli
            timeframes: Lista timeframes
            initial_capital: Capitale iniziale
        
        Returns:
            Risultati combinati di tutte le windows
        """
        logger.info(f"Avvio walk-forward: {start_date} -> {end_date}")
        
        # 1. Crea windows train/test
        self._create_windows(start_date, end_date)
        
        if not self.windows:
            logger.error("Nessuna window generata")
            return {}
        
        logger.info(f"Generate {len(self.windows)} windows")
        
        # 2. Processa ogni window
        all_test_trades = []
        
        for i, window in enumerate(self.windows):
            logger.info(f"\n=== Window {i+1}/{len(self.windows)} ===")
            logger.info(f"Train: {window.train_start.date()} -> {window.train_end.date()}")
            logger.info(f"Test:  {window.test_start.date()} -> {window.test_end.date()}")
            
            # Train period (opzionale: ottimizzazione parametri)
            train_config = BacktestConfig(
                start_date=window.train_start,
                end_date=window.train_end,
                initial_capital=initial_capital,
                symbols=symbols,
                timeframes=timeframes,
                primary_timeframe=self.settings.get('trading', {}).get('primary_timeframe', 'H1')
            )
            
            train_engine = BacktestEngine(
                train_config,
                self.settings,
                self.symbol_configs,
                self.risk_config
            )
            
            train_results = train_engine.run()
            window.train_results = train_results
            
            # Verifica minimo trades
            n_train_trades = train_results.get('summary', {}).get('total_trades', 0)
            if n_train_trades < self.wf_config['min_trades']:
                logger.warning(f"Window {i+1}: solo {n_train_trades} trades in train (min {self.wf_config['min_trades']}), skip")
                continue
            
            # Ottimizzazione parametri (placeholder: qui si potrebbero ottimizzare params)
            # Per ora usa parametri di default
            optimized_params = self._optimize_parameters(train_results)
            window.optimized_params = optimized_params
            
            # Test period con parametri ottimizzati
            test_config = BacktestConfig(
                start_date=window.test_start,
                end_date=window.test_end,
                initial_capital=initial_capital,  # Reset capital ogni window
                symbols=symbols,
                timeframes=timeframes,
                primary_timeframe=self.settings.get('trading', {}).get('primary_timeframe', 'H1')
            )
            
            # Apply optimized params (placeholder)
            test_settings = self._apply_optimized_params(self.settings, optimized_params)
            
            test_engine = BacktestEngine(
                test_config,
                test_settings,
                self.symbol_configs,
                self.risk_config
            )
            
            test_results = test_engine.run()
            window.test_results = test_results
            
            # Accumula trades test
            test_trades = test_results.get('trades', [])
            all_test_trades.extend(test_trades)
            
            # Log performance
            train_return = train_results.get('summary', {}).get('total_return', 0)
            test_return = test_results.get('summary', {}).get('total_return', 0)
            train_sharpe = train_results.get('metrics', {}).get('ratios', {}).get('sharpe_ratio', 0)
            test_sharpe = test_results.get('metrics', {}).get('ratios', {}).get('sharpe_ratio', 0)
            
            logger.info(f"Train: Return={train_return:.2f}%, Sharpe={train_sharpe:.2f}, Trades={n_train_trades}")
            logger.info(f"Test:  Return={test_return:.2f}%, Sharpe={test_sharpe:.2f}, Trades={len(test_trades)}")
        
        # 3. Combina risultati di tutte le windows test
        self.combined_results = self._combine_results(all_test_trades, initial_capital)
        
        logger.info(f"\n=== Walk-Forward Completo ===")
        logger.info(f"Windows processate: {len(self.windows)}")
        logger.info(f"Total trades OOS: {len(all_test_trades)}")
        logger.info(f"Combined return: {self.combined_results.get('summary', {}).get('total_return', 0):.2f}%")
        
        return self.combined_results
    
    def _create_windows(self, start_date: datetime, end_date: datetime):
        """Crea finestre train/test."""
        train_months = self.wf_config['train_months']
        test_months = self.wf_config['test_months']
        anchored = self.wf_config.get('anchored', False)
        
        current_start = start_date
        
        while True:
            # Train period
            train_start = current_start
            train_end = train_start + timedelta(days=30 * train_months)
            
            # Test period
            test_start = train_end
            test_end = test_start + timedelta(days=30 * test_months)
            
            # Check se oltre end_date
            if test_end > end_date:
                break
            
            window = WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            )
            self.windows.append(window)
            
            # Next window
            if anchored:
                # Anchored: train_start fisso, espande train period
                current_start = start_date  # mantieni start
            else:
                # Rolling: avanza di test_months
                current_start = test_start
    
    def _optimize_parameters(self, train_results: Dict) -> Dict:
        """
        Ottimizza parametri su train period.
        
        Placeholder: implementa grid search, genetic algorithm, ecc.
        Per ora ritorna params di default.
        """
        # Estrai metriche train
        metrics = train_results.get('metrics', {})
        optimization_metric = self.wf_config.get('optimization_metric', 'sharpe_ratio')
        
        metric_value = metrics.get('ratios', {}).get(optimization_metric, 0)
        
        # Placeholder: qui potresti testare diverse combinazioni di parametri
        # e scegliere quella con migliore metric_value
        
        optimized_params = {
            'optimization_metric': optimization_metric,
            'metric_value': metric_value,
            # Aggiungi parametri ottimizzati qui
            # Es: 'rsi_period': 14, 'rsi_oversold': 30, ...
        }
        
        logger.debug(f"Params ottimizzati: {optimization_metric}={metric_value:.2f}")
        
        return optimized_params
    
    def _apply_optimized_params(self, settings: Dict, optimized_params: Dict) -> Dict:
        """
        Applica parametri ottimizzati a settings.
        
        Placeholder: modifica settings con params ottimizzati.
        """
        # Deep copy settings
        import copy
        new_settings = copy.deepcopy(settings)
        
        # Apply optimized params
        # Es: new_settings['indicators']['rsi']['period'] = optimized_params.get('rsi_period', 14)
        
        return new_settings
    
    def _combine_results(self, all_test_trades: List[Dict], initial_capital: float) -> Dict:
        """Combina risultati di tutte le windows test."""
        if not all_test_trades:
            return {
                'summary': {'total_trades': 0, 'total_return': 0},
                'metrics': {},
                'trades': []
            }
        
        # Calcola P&L totale
        total_pnl = sum(t.get('pnl', 0) for t in all_test_trades)
        final_capital = initial_capital + total_pnl
        total_return = (final_capital - initial_capital) / initial_capital * 100
        
        # Metriche aggregate
        from monitoring.metrics import MetricsCalculator
        calc = MetricsCalculator()
        metrics = calc.calculate_metrics(all_test_trades)
        
        # Summary per window
        window_summaries = []
        for i, window in enumerate(self.windows):
            if window.test_results:
                window_summaries.append({
                    'window': i + 1,
                    'test_period': f"{window.test_start.date()} to {window.test_end.date()}",
                    'trades': window.test_results.get('summary', {}).get('total_trades', 0),
                    'return': window.test_results.get('summary', {}).get('total_return', 0),
                    'sharpe': window.test_results.get('metrics', {}).get('ratios', {}).get('sharpe_ratio', 0),
                    'max_dd': window.test_results.get('metrics', {}).get('drawdown', {}).get('max_drawdown_pct', 0)
                })
        
        # Stabilità: std dei ritorni per window
        window_returns = [w['return'] for w in window_summaries if 'return' in w]
        stability_score = np.std(window_returns) if window_returns else 0
        
        combined = {
            'summary': {
                'initial_capital': initial_capital,
                'final_capital': final_capital,
                'total_return': total_return,
                'total_trades': len(all_test_trades),
                'num_windows': len(self.windows),
                'stability_score': stability_score  # Più basso = più stabile
            },
            'metrics': metrics,
            'trades': all_test_trades,
            'window_summaries': window_summaries,
            'config': {
                'train_months': self.wf_config['train_months'],
                'test_months': self.wf_config['test_months'],
                'anchored': self.wf_config.get('anchored', False),
                'optimization_metric': self.wf_config.get('optimization_metric', 'sharpe_ratio')
            }
        }
        
        return combined
    
    def export_results(self, output_dir: str = "walk_forward_results"):
        """Esporta risultati walk-forward."""
        import os
        import json
        
        if not self.combined_results:
            logger.warning("Nessun risultato da esportare")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON summary
        summary_path = os.path.join(output_dir, f"wf_summary_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump({
                'summary': self.combined_results['summary'],
                'metrics': self.combined_results['metrics'],
                'window_summaries': self.combined_results['window_summaries'],
                'config': self.combined_results['config']
            }, f, indent=2)
        
        # CSV trades
        trades_path = os.path.join(output_dir, f"wf_trades_{timestamp}.csv")
        df_trades = pd.DataFrame(self.combined_results['trades'])
        df_trades.to_csv(trades_path, index=False)
        
        # CSV window performance
        windows_path = os.path.join(output_dir, f"wf_windows_{timestamp}.csv")
        df_windows = pd.DataFrame(self.combined_results['window_summaries'])
        df_windows.to_csv(windows_path, index=False)
        
        logger.info(f"Walk-forward risultati esportati in {output_dir}")
        
        return {
            'summary': summary_path,
            'trades': trades_path,
            'windows': windows_path
        }
    
    def plot_results(self):
        """
        Genera plot walk-forward (opzionale).
        Richiede matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.combined_results:
                logger.warning("Nessun risultato da plottare")
                return
            
            window_summaries = self.combined_results.get('window_summaries', [])
            
            if not window_summaries:
                return
            
            # Plot per window
            windows = [w['window'] for w in window_summaries]
            returns = [w['return'] for w in window_summaries]
            sharpes = [w['sharpe'] for w in window_summaries]
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Returns
            axes[0].bar(windows, returns, color=['green' if r > 0 else 'red' for r in returns])
            axes[0].set_xlabel('Window')
            axes[0].set_ylabel('Return (%)')
            axes[0].set_title('Walk-Forward Returns per Window')
            axes[0].grid(True, alpha=0.3)
            axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Sharpe
            axes[1].plot(windows, sharpes, marker='o', color='blue')
            axes[1].set_xlabel('Window')
            axes[1].set_ylabel('Sharpe Ratio')
            axes[1].set_title('Walk-Forward Sharpe Ratio per Window')
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            plt.tight_layout()
            
            # Save
            import os
            os.makedirs("walk_forward_results", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = f"walk_forward_results/wf_plot_{timestamp}.png"
            plt.savefig(plot_path, dpi=150)
            logger.info(f"Plot salvato: {plot_path}")
            
            plt.close()
        
        except ImportError:
            logger.warning("matplotlib non disponibile, skip plot")
        except Exception as e:
            logger.error(f"Errore plot: {e}")


# Esempio usage
if __name__ == "__main__":
    import yaml
    
    logging.basicConfig(level=logging.INFO)
    
    # Carica config
    with open("config/settings.yaml") as f:
        settings = yaml.safe_load(f)
    
    with open("config/symbols.yaml") as f:
        symbol_configs = yaml.safe_load(f)
    
    with open("config/risk.yaml") as f:
        risk_config = yaml.safe_load(f)
    
    # Walk-forward
    validator = WalkForwardValidator(settings, symbol_configs, risk_config)
    
    results = validator.run(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 1, 1),
        symbols=['EURUSD', 'GBPUSD'],
        timeframes=['M15', 'H1', 'H4'],
        initial_capital=100000.0
    )
    
    # Export
    validator.export_results()
    validator.plot_results()
