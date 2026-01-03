# ML Pipeline - Trading System

## Overview

Questo è il **Machine Learning Pipeline** per il trading system. Permette di:

1. **Estrarre feature** da dati storici OHLCV (40+ indicatori tecnici)
2. **Fare backtest** su dati storici e generare label (win/loss)
3. **Addestrare modelli** (RandomForest, GradientBoosting, XGBoost, LightGBM)
4. **Analizzare il contesto** per capire **PERCHÉ** i trade vincono o perdono
5. **Predicare** l'outcome di nuovi trade prima di eseguirli

## Architecture

```
Historical Data (OHLCV)
        ↓
[Feature Engineering] → 40+ technical indicators
        ↓
[Backtesting Engine] → Simulate trades, generate labels (win/loss)
        ↓
[Model Training] → RandomForest, XGBoost, LightGBM
        ↓
[Context Analyzer] → Statistical analysis, trading rules
        ↓
[Prediction Engine] → Predict new trades, confidence scores
        ↓
[Integration] → Feed predictions into DecisionEngine
```

## Components

### 1. Feature Engineer (`feature_engineer.py`)

Estrae **40+ feature** da OHLCV data:

- **Price Action** (7): Distance to support/resistance, breakout strength, wicks
- **Technical Indicators** (15): RSI, MACD, Bollinger Bands, ATR, Stochastic
- **Momentum** (5): ROC, acceleration, consecutive closes
- **Volatility** (5): Std dev, range expansion, true range
- **Market Structure** (5): Higher Highs/Lows, swings
- **Trend & Regime** (6): Moving averages, trend strength, slope
- **Patterns** (5): Pin bars, engulfing, hammers, shooting stars
- **Edge Features** (4): Signal confluence, entry quality score

```python
from ml.feature_engineer import FeatureEngineer

fe = FeatureEngineer()
features = fe.extract_all_features(
    data=ohlcv_dataframe,
    symbol='EUR_USD',
    entry_price=1.0950,
    entry_side='BUY'
)
# Returns: Dict[str, float] with 40+ features
```

### 2. Backtest Engine with Labels (`backtest_labeled.py`)

Simula trade storici e genera **training labels**:

```python
from ml.backtest_labeled import BacktestEngine

be = BacktestEngine()

# Simula un trade
trade = be.simulate_trade(
    symbol='EUR_USD',
    side='BUY',
    entry_price=1.0950,
    entry_time=timestamp,
    stop_loss=1.0900,
    take_profit=1.1000,
    historical_data=ohlcv_df,
    starting_row_index=100,
    entry_features={...}
)

# Ottieni trades come DataFrame
trades_df = be.get_trades_dataframe()
# Columns: pnl, pnl_pips, label_win_loss (0/1), label_magnitude, exit_reason
```

**Output**: DataFrame con trades + labels per training

### 3. Model Trainer (`model_trainer.py`)

Addestra **4 modelli** con cross-validation e hyperparameter tuning:

- **RandomForest**: Buon balance tra interpretability e performance
- **GradientBoosting**: Molto accurato
- **XGBoost**: State-of-the-art
- **LightGBM**: Veloce, buon per dataset grandi

```python
from ml.model_trainer import ModelTrainer

trainer = ModelTrainer(model_dir='ml/models')

# Addestra tutti i modelli
results = trainer.train_all_models(X_features, y_labels)

# Usa per predire
predictions = trainer.predict(X_new, model_name='xgboost')
probabilities = trainer.predict_proba(X_new, model_name='xgboost')

# Feature importance
importance = trainer.get_feature_importance(model_name='xgboost', top_n=20)
```

### 4. Prediction Engine (`prediction_engine.py`)

Predice **outcome di nuovi trade** prima di eseguirli:

```python
from ml.prediction_engine import PredictionEngine

pe = PredictionEngine(model_trainer, feature_engineer)

# Predici outcome
prediction = pe.predict_trade_outcome(
    symbol='EUR_USD',
    side='BUY',
    entry_price=1.0950,
    current_data=last_50_bars,
    model_name='xgboost',
    confidence_threshold=0.55
)

# prediction['win_probability'] = 0.65 (65% probabilità di win)
# prediction['confidence'] = 0.30
# prediction['should_trade'] = True if win_prob > threshold

# Analizza la predizione
analysis = pe.analyze_prediction_context()
# Mostra quali feature hanno influenzato la decisione

# Report leggibile
print(pe.get_prediction_report())
```

### 5. Context Analyzer (`context_analyzer.py`)

Analizza il contesto per capire **PERCHÉ** i trade vincono:

```python
from ml.context_analyzer import ContextAnalyzer

ca = ContextAnalyzer()
analysis = ca.analyze_trades(trades_df)

# analysis contiene:
# - overview: win rate, avg win/loss, profit factor
# - feature_comparison: quali feature differenziano vincenti da perdenti
# - statistical_significance: t-test su features
# - market_conditions: trend, RSI, volatility analysis
# - trading_rules: regole estratte dai dati
# - exit_analysis: come escono i trade (TP/SL/timeout)

ca.print_analysis_report(analysis)

# Profilo del trade vincente ideale
profile = ca.get_winning_trade_profile()
```

### 6. ML Pipeline (`pipeline.py`)

Integra tutto in **un flusso end-to-end**:

```python
from ml.pipeline import MLPipeline

pipeline = MLPipeline()

# Esegui pipeline completo
results = pipeline.run_full_pipeline(
    historical_data=ohlcv_df,
    symbol='EUR_USD',
    test_size=0.2,
    cv_folds=5
)

# results contiene:
# - features_df: feature estratte
# - trades_df: trades simulati con labels
# - training_results: metriche di training
# - context_analysis: analisi del contesto

# Predici nuovo trade
prediction = pipeline.predict_new_trade(
    symbol='EUR_USD',
    side='BUY',
    entry_price=1.0950,
    current_data=current_ohlcv
)

# Stampa profilo del trade vincente
pipeline.print_winning_trade_profile()
```

## Usage Flow

### 1. Preparazione Dati

```python
from apis.data_fetcher_yahoo import YahooFinanceDataFetcher

# Carica dati storici
fetcher = YahooFinanceDataFetcher()
data = fetcher.fetch('EUR_USD', '2023-01-01', '2024-01-01')
# Returns DataFrame con: open, high, low, close, volume
```

### 2. Training

```python
pipeline = MLPipeline()
results = pipeline.run_full_pipeline(
    data,
    symbol='EUR_USD',
    test_size=0.2
)
```

Questo:
1. Estrae feature da tutti i bar storici
2. Simula trade su dati storici
3. Genera label (win/loss)
4. Addestra 4 modelli
5. Salva modelli in `ml/models/`
6. Stampa analisi contesto

### 3. Predizione su Nuovi Trade

```python
# Carica dati correnti (ultimi 50 bar)
current_data = fetcher.fetch_last_n_bars('EUR_USD', 50)

# Predici
prediction = pipeline.predict_new_trade(
    symbol='EUR_USD',
    side='BUY',
    entry_price=1.0950,
    current_data=current_data,
    confidence_threshold=0.55
)

if prediction['should_trade']:
    print(f"Execute trade! Win probability: {prediction['win_probability']:.1%}")
else:
    print("Skip trade - low confidence")
```

### 4. Integrazione con DecisionEngine

```python
from decision_engine import DecisionEngine
from ml.pipeline import MLPipeline

class EnhancedDecisionEngine(DecisionEngine):
    def __init__(self):
        super().__init__()
        self.ml_pipeline = MLPipeline()
        self.ml_pipeline.model_trainer.load_all_models()
    
    def decide_trade(self, symbol, data):
        # Logica ML
        ml_prediction = self.ml_pipeline.predict_new_trade(
            symbol=symbol,
            side='BUY',  # O logica più complessa
            entry_price=data[-1]['close'],
            current_data=data
        )
        
        # Filtra by confidence
        if ml_prediction['win_probability'] < 0.55:
            return False, "Low ML confidence"
        
        # Altrimenti usa logica tecnica
        return super().decide_trade(symbol, data)
```

## Testing

Esegui test suite:

```bash
python ml/test_pipeline.py
```

Test:
1. Feature Engineering
2. Backtest Engine
3. Model Training
4. Prediction Engine
5. Context Analysis
6. Full Pipeline Integration

## Output Files

Training genera:
- `ml/models/random_forest_model.pkl` - RandomForest model
- `ml/models/gradient_boosting_model.pkl` - GradientBoosting model
- `ml/models/xgboost_model.pkl` - XGBoost model
- `ml/models/lightgbm_model.pkl` - LightGBM model
- `ml/models/feature_names.json` - Feature column names
- `ml/models/training_metrics.json` - Training metrics

## Key Metrics

### Model Evaluation
- **Accuracy**: Percentuale predizioni corrette
- **Precision**: Quando predice WIN, quanto spesso è giusto?
- **Recall**: Quanti win effettivi trova?
- **F1 Score**: Balance tra precision e recall
- **ROC-AUC**: Area under ROC curve (0.5 = random, 1.0 = perfect)

### Trading Metrics
- **Win Rate**: % di trade vincenti
- **Profit Factor**: Gross profit / Gross loss
- **Expectancy**: (% win × avg win) - (% loss × avg loss)
- **Payoff Ratio**: Avg win / Avg loss

## Example Output

```
╔══════════════════════════════════════════════════════╗
║           TRADE PREDICTION ANALYSIS                  ║
╚══════════════════════════════════════════════════════╝

Trade Setup:
  Symbol:        EUR_USD
  Side:          BUY
  Entry Price:   1.09500

Prediction Results:
  Model:         XGBOOST
  Prediction:    WIN
  Win Prob:      65.3%
  Loss Prob:     34.7%
  Confidence:    30.6%
  
Decision:
  Execute Trade: ✓ YES
  Threshold:     55.0%

Top Influential Features:
  signal_confluence               0.2543
  entry_quality_score            0.1887
  rsi_current                    0.1654
  trend_strength                 0.0945
  volatility_atr                 0.0823

Market Context:
  Condition:     NEUTRAL
  Trend:         UP
```

## Requirements

```
pandas
numpy
scikit-learn
xgboost
lightgbm
scipy
```

Install:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm scipy
```

## Next Steps

1. **Collect historical data** from OANDA or Yahoo Finance
2. **Run training** with at least 500 trades
3. **Analyze results** - understand which indicators matter most
4. **Integrate with DecisionEngine** for live predictions
5. **Backtest integrated system** with mock broker
6. **Paper trade** to validate predictions on real data

## References

- Feature engineering best practices for time series
- Cross-validation for financial data (time-series aware)
- Interpretation of tree-based models (SHAP, feature importance)
- Risk management and position sizing
