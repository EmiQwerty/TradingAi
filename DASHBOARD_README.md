# 📊 ML Trading Dashboard

**Dashboard Streamlit per il training e il monitoraggio del sistema ML di trading**

## 🚀 Quick Start

### Installazione delle dipendenze

```bash
# Dashboard + dipendenze aggiuntive
pip install -r requirements-dashboard.txt
```

### Avvio della dashboard

```bash
streamlit run ml/dashboard.py
```

La dashboard sarà disponibile su: `http://localhost:8501`

## 📋 Struttura della Dashboard

### Tab 1: 📊 Training Status
- **Progress Bar**: Visualizzazione in tempo reale del progresso
- **Metriche principali**: Status, percentuale, tempo elapsed, ETA
- **Progress Chart**: Grafico dell'avanzamento nel tempo
- **Status Message**: Messaggio dello stato attuale

### Tab 2: 📈 Metrics
- **Trading Performance**: Numero trades, win rate, profit factor
- **Model Performance**: Accuracy, F1 score, ROC-AUC
- **Feature Importance**: Chart dei 10 feature più importanti
- **Visualizzazioni**: Pie chart win/loss ratio

### Tab 3: 📋 Detailed Results
- **Trading Statistics**: Tabella completa delle metriche
- **Performance Visualization**: Grafici di P&L distribution
- **Model Metrics**: Dettagli precision, recall, F1 score

### Tab 4: 📜 Logs
- **Training Log**: Log cronologico di tutti i step
- **Export**: Download del log in CSV

### Tab 5: 📚 Documentation
- Guida all'uso della dashboard
- Spiegazione dei parametri
- Tips e best practices

## ⚙️ Configurazione

### Nella Sidebar
1. **Symbol**: Il pair da tradare (es. EUR_USD)
2. **Test Set Size**: % di dati per il test (0.1-0.5)
3. **CV Folds**: Numero di fold per cross-validation (3-10)
4. **Data Source**: 
   - Upload CSV con dati OHLCV
   - Oppure usa dati sintetici per test

### Formato CSV atteso
```
time,open,high,low,close,volume
2023-01-01 00:00,1.0850,1.0870,1.0840,1.0860,500000
2023-01-01 01:00,1.0860,1.0880,1.0850,1.0870,550000
...
```

## 🎯 Flusso di Training

La dashboard automaticamente:

1. **Load Data** (Step 1/6)
   - Carica i dati OHLCV
   - Valida il formato

2. **Feature Extraction** (Step 2/6)
   - Estrae 40+ indicatori tecnici
   - Calcola feature per ogni bar

3. **Backtesting** (Step 3/6)
   - Simula i trade storici
   - Genera le label (win/loss)

4. **Model Training** (Step 4/6)
   - Addestra 4 modelli ML
   - Iperparameter tuning
   - Cross-validation

5. **Context Analysis** (Step 5/6)
   - Analizza il contesto dei trades
   - Identifica feature discriminanti

6. **Results** (Step 6/6)
   - Salva i risultati
   - Visualizza nel grafico

## 📊 Interpretazione dei Risultati

### Trading Metrics

**Win Rate**: Percentuale di trades vincenti
- ✅ > 50% è buono
- ⚠️ 40-50% è borderline
- ❌ < 40% rivedere la strategia

**Profit Factor**: Gross Profit / Gross Loss
- ✅ > 2.0 è eccellente
- ✅ > 1.5 è buono
- ⚠️ 1.0-1.5 è okay
- ❌ < 1.0 è unprofitable

**Total Trades**: Numero di trades in backtest
- ✅ > 100 è un buon sample
- ⚠️ 30-100 è piccolo
- ❌ < 30 dati insufficienti

### Model Metrics

**Accuracy**: % di predizioni corrette
- ✅ > 60% è buono
- ⚠️ 50-60% borderline
- ❌ < 50% peggio del random

**F1 Score**: Balance tra precision e recall
- ✅ > 0.60 è buono
- ✅ > 0.50 è okay
- ❌ < 0.50 scadente

**ROC-AUC**: Area under ROC curve
- ✅ > 0.70 è buono
- ⚠️ 0.60-0.70 è okay
- ❌ < 0.60 scarso

### Feature Importance
Le feature più importanti nella parte superiore sono quelle che il modello usa più per discriminare tra win e loss.

💡 **Tip**: Se vedi feature non-sensate come "volume" in top 3, potrebbe significare che il modello sta overfitting.

## 🔧 Utility Functions (dashboard_utils.py)

### DataValidator
```python
from ml.dashboard_utils import DataValidator

# Validare OHLCV data
is_valid, message = DataValidator.validate_ohlcv(df)
if is_valid:
    df = DataValidator.prepare_dataframe(df)
```

### FileManager
```python
from ml.dashboard_utils import FileManager

# Caricare risultati precedenti
results = FileManager.load_results(Path("results"), "EUR_USD")

# Lista di training storici
history = FileManager.list_training_history(Path("results"), "EUR_USD")
```

### ChartBuilder
```python
from ml.dashboard_utils import ChartBuilder

# Equity curve
fig = ChartBuilder.build_equity_curve(trades_df)

# Drawdown chart
fig = ChartBuilder.build_drawdown_chart(trades_df)

# Returns distribution
fig = ChartBuilder.build_returns_distribution(trades_df)
```

### ProgressTracker
```python
from ml.dashboard_utils import ProgressTracker

# ETA calculation
eta = ProgressTracker.calculate_eta(start_time, step=2, total=6)

# Format duration
duration_str = ProgressTracker.format_duration(3661)  # "1h 1m"
```

## 📝 Training Manager API

```python
from ml.training_manager import TrainingManager

# Create manager
manager = TrainingManager()

# Register callbacks
def on_progress(progress):
    print(f"Progress: {progress.percentage:.1f}%")

def on_metrics(metrics):
    print(f"Win Rate: {metrics.win_rate:.1%}")

def on_error(error_type, message):
    print(f"Error: {error_type}: {message}")

manager.add_progress_callback(on_progress)
manager.add_metrics_callback(on_metrics)
manager.add_error_callback(on_error)

# Start training
success = manager.train(
    symbol='EUR_USD',
    historical_data=df,
    test_size=0.2,
    cv_folds=5
)

# Get state
progress = manager.get_progress()  # Dict
metrics = manager.get_metrics()    # Dict
history = manager.get_history()    # List[Dict]

# Control
manager.pause()
manager.resume()
manager.cancel()
```

## 🎨 Personalizzazioni

### Aggiungere un nuovo Tab

```python
# In dashboard.py, dopo la riga delle tabs:
with tab6:
    st.header("Mio Tab Custom")
    # Tuo contenuto qui
```

### Modificare i colori
I colori sono definiti nel CSS all'inizio di `dashboard.py`:
```python
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        ...
    }
</style>
""", unsafe_allow_html=True)
```

### Aggiungere notifiche custom
```python
from ml.dashboard_utils import NotificationManager

NotificationManager.show_success("Training completed!")
NotificationManager.show_error("Something went wrong")
NotificationManager.show_warning("Check your data")
NotificationManager.show_info("This is important")
```

## 🐛 Troubleshooting

### Dashboard non carica
```bash
# Controlla che Streamlit sia installato
pip install streamlit

# Controlla i requisiti
pip install -r requirements-dashboard.txt
```

### Training è lento
- Riduci il numero di dati
- Usa dataset più piccolo
- Aumenta il test_size per usare meno dati in training

### Metrics non si aggiornano
- Assicurati che i callback siano registrati PRIMA di startare il training
- Check i logs per errori

### CSV non carica
- Verifica il formato sia OHLCV
- Assicurati i nomi colonne siano: open, high, low, close, volume
- Controlla che non ci siano NaN values

## 📈 Best Practices

1. **Start with sample data**: Testa la dashboard con dati sintetici prima di usare veri dati
2. **Monitor profit factor**: È il KPI più importante per trading strategy
3. **Check feature importance**: Verifica che le top features abbiano senso economico
4. **Use sufficient data**: Almeno 100-200 trades per un buon modello
5. **Cross-validate**: Usa almeno 5 folds di CV per evitare overfitting
6. **Save results**: La dashboard salva automaticamente i risultati in `results/`

## 📚 File Correlati

- `ml/dashboard.py` - Main Streamlit app
- `ml/dashboard_utils.py` - Utility functions
- `ml/training_manager.py` - Training orchestration
- `ml/pipeline.py` - ML pipeline
- `requirements-dashboard.txt` - Dipendenze
- `results/` - Cartella automatica per i risultati

## 🤝 Integrazione con il Sistema

La dashboard si integra con:
- **FeatureEngineer**: Estrae feature nel Step 2
- **BacktestEngine**: Simula trades nel Step 3
- **ModelTrainer**: Addestra modelli nel Step 4
- **ContextAnalyzer**: Analizza features nel Step 5
- **PredictionEngine**: Usato per predizioni live

## 📞 Support

Per problemi o domande:
1. Controlla i logs nella directory (Training logs in Tab 4)
2. Verifica il format dei dati
3. Prova con dati sintetici
4. Check file ml/training_manager.py per il training logic
