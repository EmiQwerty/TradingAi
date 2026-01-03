# 🎯 Trading Strategies System

Sistema professionale di trading strategies con pattern recognition e ML filtering.

## 📋 Overview

Il sistema implementa 3 strategie professionali che possono essere usate singolarmente o combinate in ensemble:

1. **RSI Mean Reversion** - Trades su inversioni di trend
2. **Breakout Strategy** - Trades su rotture di livelli chiave
3. **Trend Following** - Trades su pullback in trend forti

Tutte le strategie includono:
- ✅ Chart pattern recognition (15+ patterns)
- ✅ Multiple confirmation filters
- ✅ ATR-based risk management
- ✅ Confidence scoring
- ✅ Metadata per ML analysis

## 🚀 Quick Start

### Uso Base

```python
from strategies.rsi_strategy import RSIMeanReversionStrategy
import pandas as pd

# Load data
data = pd.read_csv('EURUSD_H1.csv', index_col='time', parse_dates=True)

# Initialize strategy
strategy = RSIMeanReversionStrategy()

# Generate signals
signals = strategy.generate_signals(data)

print(f"Generated {len(signals)} trading signals")
for signal in signals[:5]:
    print(f"{signal.timestamp}: {signal.signal_type.value} @ {signal.entry_price:.5f}")
    print(f"  Reason: {signal.reason}")
    print(f"  Confidence: {signal.confidence:.2%}")
    print(f"  SL: {signal.stop_loss:.5f} | TP: {signal.take_profit:.5f}")
```

### Ensemble Mode

```python
from strategies.strategy_ensemble import StrategyEnsemble

# Initialize ensemble
ensemble = StrategyEnsemble(mode="WEIGHTED")

# Generate combined signals
signals = ensemble.generate_signals(data)

# Backtest
results = ensemble.backtest(data)
print(results)
```

## 📊 Strategies Detailed

### 1. RSI Mean Reversion Strategy

**Filosofia**: Compra oversold, vende overbought con conferme multiple.

**Entry LONG:**
- RSI < 30 (oversold)
- MACD histogram turning positive
- Price > EMA200 (trend filter)
- Volume spike > 1.5x average
- Optional: Bullish pattern (Hammer, Engulfing)

**Entry SHORT:**
- RSI > 70 (overbought)
- MACD histogram turning negative
- Price < EMA200
- Volume spike
- Optional: Bearish pattern (Shooting Star, Engulfing)

**Risk Management:**
- Stop Loss: 2x ATR
- Take Profit: 1:2 Risk/Reward
- Confidence: 0.7-0.85 (higher with patterns)

**Best For:**
- Range-bound markets
- H1 - H4 timeframes
- Major currency pairs

**Example:**
```python
strategy = RSIMeanReversionStrategy(
    rsi_oversold=25,    # More extreme
    rsi_overbought=75,
    ema_period=200,
    volume_threshold=1.8  # Stronger confirmation
)
```

---

### 2. Breakout Strategy

**Filosofia**: Cavalca momentum dopo rotture di livelli chiave.

**Entry LONG:**
- Price breaks above resistance (20-bar swing high)
- Volume > 1.5x average
- Bollinger Bands expanding (volatility increase)
- Optional: Triangle Ascending, Flag Bullish

**Entry SHORT:**
- Price breaks below support (20-bar swing low)
- Volume spike
- BB expansion
- Optional: Triangle Descending, Flag Bearish

**Risk Management:**
- Stop Loss: Below breakout level + 1.5x ATR buffer
- Take Profit: Measured move (range proiettato)
- Confidence: 0.7-0.85

**Best For:**
- Trending markets
- H1 - D1 timeframes
- High volatility instruments

**Example:**
```python
strategy = BreakoutStrategy(
    lookback_period=30,     # More stable levels
    volume_threshold=2.0,   # Strong confirmation
    atr_multiplier=1.0      # Tighter stops
)
```

---

### 3. Trend Following Strategy

**Filosofia**: Entra su pullback in trend stabiliti.

**Entry LONG:**
- Price > EMA50 > EMA200 (uptrend)
- EMA50 rising
- Pullback touches EMA20
- Bullish bounce candle (body > 50%)
- MACD > 0
- ADX > 25 (strong trend)
- Optional: Bullish Engulfing, Hammer

**Entry SHORT:**
- Price < EMA50 < EMA200 (downtrend)
- EMA50 falling
- Pullback reaches EMA20
- Bearish rejection candle
- MACD < 0
- ADX > 25
- Optional: Bearish Engulfing, Shooting Star

**Risk Management:**
- Stop Loss: Below/above EMA50 + 2x ATR
- Take Profit: 1:3 Risk/Reward
- Confidence: 0.75-0.88

**Best For:**
- Strong trending markets
- H4 - D1 timeframes
- Indices, major pairs

**Example:**
```python
strategy = TrendFollowingStrategy(
    ema_fast=20,
    ema_medium=50,
    ema_slow=200,
    adx_threshold=25,  # Minimum trend strength
    atr_multiplier=2.0
)
```

---

## 🎭 Ensemble Modes

### WEIGHTED Mode (Recommended)

Combines all strategies with weighted confidence.

**Features:**
- Averages entry/SL/TP from all agreeing strategies
- Weights confidence by strategy reliability
- Bonus confidence for multiple agreements (+5% per extra strategy)
- Best overall performance

**Example:**
```python
ensemble = StrategyEnsemble(
    mode="WEIGHTED",
    strategy_weights={
        'RSI': 1.0,
        'Breakout': 1.2,
        'TrendFollowing': 1.3  # Most reliable
    }
)
```

**Output:**
- 1 signal per bar if strategies agree
- Confidence: weighted average + agreement bonus
- Metadata includes all individual reasons

---

### VOTING Mode

Only generates signal if minimum N strategies agree.

**Features:**
- Requires democratic agreement
- Filters out weak signals
- Lower quantity, higher quality
- Good for conservative trading

**Example:**
```python
ensemble = StrategyEnsemble(
    mode="VOTING",
    min_votes=2  # Need 2+ strategies
)
```

**Best For:**
- Risk-averse traders
- High-noise markets
- When you want only strongest signals

---

### ALL Mode

Returns all signals from all strategies separately.

**Features:**
- Maximum signal count
- Can compare strategy performance
- Useful for ML training (more samples)
- Each signal tagged with strategy name

**Example:**
```python
ensemble = StrategyEnsemble(mode="ALL")
signals = ensemble.generate_signals(data)

# Check which strategy generated each
for sig in signals:
    print(f"{sig.metadata['strategy']}: {sig.confidence}")
```

**Best For:**
- ML training (want diverse samples)
- Strategy comparison/research
- Backtesting individual strategies

---

### SINGLE Mode

Uses only one strategy (set in TrainingManager initialization).

**Example:**
```python
# In dashboard or code
strategy_mode = "SINGLE"
selected_strategies = ["TrendFollowing"]
```

---

## 🎨 Chart Pattern Recognition

All strategies use `PatternRecognizer` for additional confirmation.

### Candlestick Patterns

**Bullish:**
- Hammer: Small body at top, long lower wick
- Bullish Engulfing: Large bullish body engulfs previous bearish
- Morning Star: 3-candle reversal pattern
- Doji: Indecision, can signal reversal

**Bearish:**
- Shooting Star: Small body at bottom, long upper wick
- Bearish Engulfing: Large bearish body engulfs previous bullish
- Evening Star: 3-candle reversal
- Doji (context-dependent)

### Chart Patterns

**Bullish:**
- Double Bottom: Two lows at same level
- Head & Shoulders Inverse
- Ascending Triangle
- Bullish Flag

**Bearish:**
- Double Top: Two highs at same level
- Head & Shoulders
- Descending Triangle
- Bearish Flag

**Usage:**
```python
from strategies.pattern_recognition import PatternRecognizer

recognizer = PatternRecognizer()
patterns = recognizer.scan_all_patterns(data)

for p in patterns:
    print(f"{p.pattern_type.value}: {p.direction} @ {p.start_index}-{p.end_index}")
    print(f"  Confidence: {p.confidence:.2%}")
```

---

## 🧪 Backtesting

### Individual Strategy

```python
strategy = RSIMeanReversionStrategy()
signals = strategy.generate_signals(data)

# Backtest built-in
results = strategy.backtest(data)
print(f"Win Rate: {results['win_rate']:.2%}")
print(f"Profit Factor: {results['profit_factor']:.2f}")
```

### Ensemble Backtest

```python
ensemble = StrategyEnsemble(mode="WEIGHTED")
results_df = ensemble.backtest(data)

# Analyze
print(f"Total Signals: {len(results_df)}")
print(f"Avg Confidence: {results_df['confidence'].mean():.2%}")

# Check agreement distribution
if 'votes' in results_df.columns:
    print("\nAgreement Distribution:")
    print(results_df['votes'].value_counts())
```

### Integration with ML Pipeline

```python
from ml.training_manager import TrainingManager

manager = TrainingManager(
    strategy_mode="WEIGHTED",
    selected_strategies=['RSI', 'Breakout', 'TrendFollowing']
)

success = manager.train(
    symbol='EUR_USD',
    historical_data=data,
    strategy_mode="WEIGHTED",
    selected_strategies=['RSI', 'Breakout', 'TrendFollowing']
)

# ML now learns which strategy signals work best!
```

---

## 📈 Performance Tips

### Data Requirements

**Minimum:**
- 500 bars (for pattern recognition)
- 1 year of data (H1-H4)

**Recommended:**
- 1000+ bars
- 1-2 years (H1-H4)
- 3-5 years (D1)

### Timeframe Selection

| Strategy | Best Timeframes |
|----------|----------------|
| RSI Mean Reversion | H1, H4 |
| Breakout | H1, H4, D1 |
| Trend Following | H4, D1 |
| Ensemble | H1, H4 |

### Expected Signal Count

| Mode | Signals/Year (H1) |
|------|------------------|
| RSI | 50-100 |
| Breakout | 30-60 |
| TrendFollowing | 20-40 |
| WEIGHTED | 40-80 |
| VOTING (2) | 20-50 |
| ALL | 100-200 |

---

## 🔧 Customization

### Create Your Own Strategy

```python
from strategies.strategy_base import StrategyBase, TradingSignal, SignalType

class MyCustomStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="MyStrategy")
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        # Add your indicators
        df['my_signal'] = ...
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        signals = []
        df = self.calculate_indicators(data)
        
        for i in range(50, len(df)):
            current = df.iloc[i]
            
            if current['my_signal'] > threshold:
                signal = TradingSignal(
                    timestamp=current.name,
                    signal_type=SignalType.BUY,
                    entry_price=current['close'],
                    stop_loss=self.calculate_stop_loss(...),
                    take_profit=self.calculate_take_profit(...),
                    confidence=0.8,
                    reason="My custom condition met"
                )
                signals.append(signal)
        
        return signals
    
    def calculate_stop_loss(self, entry_price, signal_type, atr, data, index):
        # Your SL logic
        return entry_price - (2 * atr)
    
    def calculate_take_profit(self, entry_price, signal_type, stop_loss, risk_reward_ratio=2.0):
        # Your TP logic
        risk = abs(entry_price - stop_loss)
        return entry_price + (risk * risk_reward_ratio)
```

### Add to Ensemble

```python
# In strategy_ensemble.py
self.strategies['MyStrategy'] = MyCustomStrategy()
self.weights['MyStrategy'] = 1.0
```

---

## 📊 ML Integration

### How ML Uses Strategies

1. **Signal Generation**: Strategies generate 20-200 quality signals per year
2. **Feature Extraction**: Each signal has 40+ technical indicators at entry
3. **Labeling**: Backtest simulates each signal → win/loss label
4. **Training**: ML learns which indicator combinations predict successful signals
5. **Prediction**: On new signals, ML predicts probability of success
6. **Filtering**: Only take signals ML rates as high probability

### Example Workflow

```python
# 1. Generate strategy signals
ensemble = StrategyEnsemble(mode="WEIGHTED")
signals = ensemble.generate_signals(historical_data)  # 50 signals

# 2. Backtest signals → labels
trades_df = backtest_signals(signals, historical_data)  # 50 trades with win/loss

# 3. Train ML on features
from ml.model_trainer import ModelTrainer
trainer = ModelTrainer()
results = trainer.train_all_models(trades_df)

# 4. Predict on new signals
new_signal = strategy.generate_signals(new_data)[-1]
prediction = trained_model.predict(new_signal.features)

if prediction > 0.7:  # High confidence
    execute_trade(new_signal)
```

---

## 🎯 Best Practices

### Do's ✅

- Use 1-2 years of data minimum
- Test on multiple timeframes
- Combine strategies in WEIGHTED mode
- Monitor agreement count (2-3 strategies = high confidence)
- Check strategy metadata for reasoning
- Validate with out-of-sample testing

### Don'ts ❌

- Don't use < 500 bars (not enough for patterns)
- Don't ignore pattern confirmations
- Don't use all strategies on M1 (too noisy)
- Don't blindly trust single strategy
- Don't overtrade - quality over quantity

---

## 📚 References

- `strategy_base.py` - Abstract base class
- `rsi_strategy.py` - RSI Mean Reversion implementation
- `breakout_strategy.py` - Breakout implementation
- `trend_following_strategy.py` - Trend Following implementation
- `pattern_recognition.py` - Chart patterns
- `strategy_ensemble.py` - Ensemble combiner

---

## 🆘 Troubleshooting

**Q: Strategies generate zero signals**

A: 
- Check data length (need 500+ bars)
- Check timeframe (H1/H4 work best)
- Relax thresholds (lower ADX, RSI limits)
- Check data quality (no gaps/NaN)

**Q: Too many signals (>500/year)**

A:
- Use VOTING mode with min_votes=2
- Increase confirmation thresholds
- Use higher timeframes (D1)

**Q: Low confidence signals**

A:
- Strategies don't agree → use VOTING mode
- Missing pattern confirmations
- Weak market conditions
- Consider skipping low-confidence signals

**Q: ML doesn't improve results**

A:
- Need 100+ trades minimum for training
- Check feature quality (no NaN/inf)
- Try longer data periods
- Validate on out-of-sample data

---

## 📞 Support

For issues or questions:
- Check dashboard Documentation tab
- Review this README
- Check training logs for errors
- Test with sample data first

---

**Version**: 2.0  
**Last Updated**: 2026-01-03  
**License**: MIT
