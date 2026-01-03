"""
ML Pipeline Test & Example
Dimostra come usare l'intera pipeline ML per:
1. Addestrare modelli su dati storici
2. Analizzare il contesto (WHY trades win/lose)
3. Fare predizioni su nuovi trade
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.pipeline import MLPipeline
from ml.feature_engineer import FeatureEngineer
from ml.backtest_labeled import BacktestEngine
from ml.model_trainer import ModelTrainer
from ml.context_analyzer import ContextAnalyzer
from ml.prediction_engine import PredictionEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_bars: int = 500, symbol: str = 'EUR_USD') -> pd.DataFrame:
    """
    Genera dati OHLC sintetici per test.
    
    In produzione, questi sarebbero dati reali da Yahoo Finance / OANDA.
    """
    
    np.random.seed(42)
    
    # Trend casuale
    returns = np.random.randn(n_bars) * 0.001 + 0.0002
    prices = 1.0900 * np.exp(np.cumsum(returns))
    
    # OHLC
    opens = prices[:-1] if len(prices) > 1 else prices
    closes = prices[1:] if len(prices) > 1 else prices
    
    # Padding per match lengths
    if len(opens) != n_bars:
        opens = np.concatenate([opens, [prices[-1]]])
    
    highs = np.maximum(opens, closes) + np.random.uniform(0, 0.0005, n_bars)
    lows = np.minimum(opens, closes) - np.random.uniform(0, 0.0005, n_bars)
    
    volumes = np.random.uniform(100000, 1000000, n_bars)
    
    data = pd.DataFrame({
        'time': pd.date_range('2023-01-01', periods=n_bars, freq='1H'),
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
    })
    
    return data


def test_feature_engineering():
    """Test: Feature Engineering"""
    
    print("\n" + "="*70)
    print("TEST 1: Feature Engineering")
    print("="*70)
    
    data = generate_synthetic_data(n_bars=100)
    
    fe = FeatureEngineer()
    
    # Extract features da ultima candela
    features = fe.extract_all_features(
        data=data,
        symbol='EUR_USD',
        entry_price=data.iloc[-1]['close'],
        entry_side='BUY'
    )
    
    print(f"\n✓ Extracted {len(features)} features")
    print(f"  Features: {list(features.keys())[:5]}... ({len(features)} total)")
    
    # Sample feature values
    print(f"\n  Sample feature values:")
    for feat in ['rsi_current', 'macd_histogram', 'trend_type', 'signal_confluence']:
        if feat in features:
            print(f"    {feat}: {features[feat]}")
    
    return features


def test_backtest_engine():
    """Test: Backtest Engine with Label Generation"""
    
    print("\n" + "="*70)
    print("TEST 2: Backtest Engine with Labels")
    print("="*70)
    
    data = generate_synthetic_data(n_bars=200)
    
    be = BacktestEngine()
    
    # Simula alcuni trade
    n_trades = 5
    
    for i in range(50, 150, 20):
        entry_price = data.iloc[i]['close']
        
        trade = be.simulate_trade(
            symbol='EUR_USD',
            side='BUY',
            entry_price=entry_price,
            entry_time=data.iloc[i]['time'],
            stop_loss=entry_price - 0.005,
            take_profit=entry_price + 0.010,
            historical_data=data,
            starting_row_index=i,
            entry_features={
                'rsi_current': np.random.uniform(30, 70),
                'macd_histogram': np.random.randn() * 0.0001,
                'trend_type': 'UP',
                'volatility_atr': np.random.uniform(0.0005, 0.002),
                'signal_confluence': np.random.uniform(0.5, 1.0),
            }
        )
    
    trades_df = be.get_trades_dataframe()
    
    print(f"\n✓ Simulated {len(trades_df)} trades")
    
    if len(trades_df) > 0:
        metrics = be.get_performance_metrics()
        print(f"\n  Performance Metrics:")
        print(f"    Win Rate:     {metrics['win_rate']:.1%}")
        print(f"    Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"    Expectancy:   {metrics['expectancy']:.4f}")
        
        print(f"\n  Sample Trades:")
        for idx, trade in trades_df.head(3).iterrows():
            print(f"    {trade['symbol']} {trade['side']} @ {trade['entry_price']:.5f} → PnL: {trade['pnl_percent']:.2%} ({trade['label_exit_reason']})")
    
    return trades_df


def test_model_training():
    """Test: Model Training"""
    
    print("\n" + "="*70)
    print("TEST 3: Model Training")
    print("="*70)
    
    # Genera synthetic training data
    np.random.seed(42)
    n_samples = 100
    n_features = 30
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Y = semplice logica binaria basata su features
    y = pd.Series((X['feature_0'] > 0).astype(int))
    
    # Addestra modelli
    trainer = ModelTrainer(model_dir='ml/models')
    
    print(f"\nTraining on {len(X)} samples with {len(X.columns)} features...")
    
    results = trainer.train_all_models(X, y, test_size=0.2, cv_folds=3)
    
    print("\n✓ Models trained successfully\n")
    
    # Summary
    for model_name, metrics in results.items():
        print(f"  {model_name:<15} | Acc: {metrics['accuracy']:.3f} | F1: {metrics['f1']:.3f} | AUC: {metrics['roc_auc']:.3f}")
    
    return trainer, X, y


def test_prediction_engine(trainer, X_sample):
    """Test: Prediction Engine"""
    
    print("\n" + "="*70)
    print("TEST 4: Prediction Engine")
    print("="*70)
    
    fe = FeatureEngineer()
    pe = PredictionEngine(trainer, fe)
    
    # Dummy prediction (usa synthetic data)
    data = generate_synthetic_data(n_bars=100)
    
    pred = pe.predict_trade_outcome(
        symbol='EUR_USD',
        side='BUY',
        entry_price=1.0950,
        current_data=data,
        model_name='random_forest',
        confidence_threshold=0.55
    )
    
    print(f"\n✓ Prediction generated:\n")
    print(f"  Symbol:          {pred['symbol']} {pred['side']}")
    print(f"  Entry Price:     {pred['entry_price']:.5f}")
    print(f"  Win Probability: {pred['win_probability']:.1%}")
    print(f"  Confidence:      {pred['confidence']:.1%}")
    print(f"  Execute Trade:   {'YES ✓' if pred['should_trade'] else 'NO ✗'}")
    
    print(f"\n  Top Influential Features:")
    for feat, score in pred['top_influential_features'].items():
        print(f"    {feat:<30} {score:.4f}")
    
    return pred


def test_context_analysis(trades_df):
    """Test: Context Analysis"""
    
    print("\n" + "="*70)
    print("TEST 5: Context Analysis (WHY do trades win/lose?)")
    print("="*70)
    
    if len(trades_df) < 10:
        print("\n⚠ Not enough trades for analysis (need > 10)")
        return
    
    ca = ContextAnalyzer()
    
    analysis = ca.analyze_trades(trades_df)
    
    # Print analysis
    ca.print_analysis_report(analysis)
    
    return analysis


def test_full_pipeline():
    """Test: Complete ML Pipeline"""
    
    print("\n" + "="*70)
    print("TEST 6: Full ML Pipeline Integration")
    print("="*70)
    
    # Genera dati
    data = generate_synthetic_data(n_bars=300)
    
    print(f"\nRunning full pipeline on {len(data)} bars of data...")
    
    pipeline = MLPipeline()
    
    results = pipeline.run_full_pipeline(
        historical_data=data,
        symbol='EUR_USD',
        test_size=0.2,
        cv_folds=3
    )
    
    if results:
        print("\n✓ Pipeline completed successfully!")
        print(f"  - Features extracted: {len(results['features_df'].columns)}")
        print(f"  - Trades generated: {len(results['trades_df'])}")
        print(f"  - Models trained: {len(results['training_results'])}")
    
    return results


def main():
    """Esegui tutti i test"""
    
    print("\n" + "="*80)
    print("ML PIPELINE TEST SUITE")
    print("="*80)
    
    # Test 1: Feature Engineering
    try:
        features = test_feature_engineering()
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Backtest Engine
    try:
        trades_df = test_backtest_engine()
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        trades_df = None
    
    # Test 3: Model Training
    try:
        trainer, X, y = test_model_training()
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        trainer = None
    
    # Test 4: Prediction Engine
    if trainer is not None:
        try:
            test_prediction_engine(trainer, X.iloc[0:1])
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Test 5: Context Analysis
    if trades_df is not None and len(trades_df) > 5:
        try:
            test_context_analysis(trades_df)
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Test 6: Full Pipeline
    try:
        test_full_pipeline()
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("TEST SUITE COMPLETED")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
