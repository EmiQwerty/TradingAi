# 🚀 Dashboard Quick Start Guide

## ⚡ 30 Secondi per Iniziare

### 1. Installa Dipendenze (1 minuto)
```bash
cd /Users/emiliano/Desktop/Trading
pip install -r requirements-dashboard.txt
```

### 2. Lancia la Dashboard (1 secondo)
```bash
python launch_dashboard.py
```

La dashboard aprirà automaticamente su: **http://localhost:8501**

---

## 📊 Primo Training

### Step 1: Configure (nella Sidebar)
1. **Symbol**: Mantieni "EUR_USD" (o cambia se desideri)
2. **Test Set Size**: Mantieni 0.2 (20%)
3. **CV Folds**: Mantieni 5
4. **Data Source**: Seleziona "Use sample data"

### Step 2: Avvia Training
- Clicca il bottone verde **▶️ Start Training**

### Step 3: Monitora
- **Training Status Tab**: Vedi il progress bar e l'ETA
- **Metrics Tab**: Visualizza i risultati in tempo reale
- **Logs Tab**: Controlla i dettagli del training

### Step 4: Rivedi Risultati
- **Detailed Results Tab**: Vedi metriche complete e grafici

---

## 📁 Struttura dei File

```
Trading/
├── ml/
│   ├── dashboard.py              # 🎯 Main dashboard (START HERE)
│   ├── dashboard_utils.py        # 📚 Utility functions
│   ├── training_manager.py       # ⚙️ Training orchestration
│   ├── pipeline.py               # ML pipeline
│   ├── feature_engineer.py       # Feature extraction
│   ├── backtest_engine.py        # Trade simulation
│   ├── model_trainer.py          # ML model training
│   ├── context_analyzer.py       # Feature analysis
│   └── ... (altri moduli)
├── launch_dashboard.py           # 🚀 Launcher (USE THIS)
├── run_dashboard.sh              # 🐚 Bash launcher
├── test_dashboard.py             # 🧪 Test script
├── requirements-dashboard.txt    # 📦 Dipendenze
├── DASHBOARD_README.md           # 📖 Full documentation
├── DASHBOARD_FEATURES.md         # ✨ Feature list
└── QUICKSTART.md                 # ⚡ THIS FILE

data/                             # Input data folder
results/                          # Training results folder (auto-created)
```

---

## 🎯 Schermate Principali

### 📊 Training Status Tab
```
┌────────────────────────────────────────────┐
│         ML Trading Dashboard 🚀            │
├────────────────────────────────────────────┤
│  Status: MODEL_TRAINING  Progress: 65.2%  │
│  Elapsed: 45s  |  Remaining: 23s           │
├────────────────────────────────────────────┤
│  Training Progress ▰▰▰▰▰▱▱▱▱▱ 65%         │
│  📌 Training 4 ML models on 250 trades...  │
├────────────────────────────────────────────┤
│              Progress Chart                │
│     📈 [line chart showing 0→65%]          │
└────────────────────────────────────────────┘
```

### 📈 Metrics Tab
```
┌──────────────┬──────────────┬──────────────┐
│ Total Trades │ Win Rate     │ Profit Factor│
│     245      │   58.4%      │    2.15      │
└──────────────┴──────────────┴──────────────┘

┌──────────────┬──────────────┬──────────────┐
│   Accuracy   │   F1 Score   │   ROC-AUC    │
│    62.3%     │    0.618     │    0.71      │
└──────────────┴──────────────┴──────────────┘

Top Discriminating Features:
📊 RSI → 0.245
📊 MACD Signal → 0.198
📊 Volatility → 0.187
... (7 more)
```

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install -r requirements-dashboard.txt
```

### "Dashboard non carica"
```bash
# Verifica che porta 8501 sia libera
lsof -i :8501

# Se occupata, uccidi il processo
kill -9 <PID>

# Riavvia
python launch_dashboard.py
```

### "Training è lentissimo"
- Usa meno dati (nella config)
- Aumenta test_size a 0.3 per ridurre training set
- Usa meno CV folds (3 invece di 5)

### "Data non carica dal CSV"
Verifica il formato del file:
```
✅ Corretto:
time,open,high,low,close,volume
2023-01-01 00:00,1.0850,1.0870,1.0840,1.0860,500000

❌ Errato:
open,high,low,close               # Manca volume
2023-01-01,1.0850,1.0870,1.0840   # Manca close
```

---

## 💡 Tips

### 1. Inizia con Sample Data
Non caricare dati reali la prima volta. Usa "Use sample data" per:
- ✅ Test della dashboard
- ✅ Capire il flusso
- ✅ Verificare le metriche

### 2. Interpretazione dei Risultati
**Win Rate > 50%** ← Good  
**Profit Factor > 1.5** ← Good  
**Accuracy > 55%** ← Good (58% random)  

Se questi numeri sono BASSI con sample data, il sistema è ancora in development.

### 3. Salva i Risultati
Alla fine del training, risultati vengono salvati in:
```
results/EUR_USD_20240115_143022.json
```

Puoi scaricare i logs dal Tab "Logs" per analizzare offline.

### 4. Esperimenti
Prova diverse configurazioni:
- Symbol diversi
- Test size diversi
- CV folds diversi

E confronta i risultati!

---

## 📚 Documentazione Completa

Per approfondire:
- **DASHBOARD_README.md**: Guida completa e API
- **DASHBOARD_FEATURES.md**: Lista completa features
- **ml/training_manager.py**: Docstrings dei metodi
- **ml/dashboard_utils.py**: Utility functions

---

## 🎮 Primum Exemplum (First Example)

```python
# Se vuoi usare il training manager direttamente:

from ml.training_manager import TrainingManager
import pandas as pd

# Load data (o usa sample)
df = pd.DataFrame({...})  # OHLCV data

# Create manager
manager = TrainingManager()

# Register callbacks for real-time updates
def on_progress(progress):
    print(f"Progress: {progress.percentage:.1f}% - {progress.message}")

def on_metrics(metrics):
    print(f"Win Rate: {metrics.win_rate:.1%}, Profit Factor: {metrics.profit_factor:.2f}")

manager.add_progress_callback(on_progress)
manager.add_metrics_callback(on_metrics)

# Train
success = manager.train(
    symbol='EUR_USD',
    historical_data=df,
    test_size=0.2,
    cv_folds=5
)

# Get results
if success:
    metrics = manager.get_metrics()
    print(metrics)
```

---

## 🚀 Prossimi Step

### Opzione 1: Dashboard per Live Trading
Integra la dashboard con un sistema di live trading:
```python
from ml.prediction_engine import PredictionEngine
from ml.decision_engine import DecisionEngine

# Usa i modelli trainati per predizioni live
```

### Opzione 2: Automazione
Crea uno script che:
- Carica i dati ogni giorno
- Esegue il training
- Invia i risultati via email

### Opzione 3: Miglioramenti
- Aggiungi più ML models
- Tuning iper-parametri
- Validazione su new data

---

## ❓ FAQ

**Q: Quanto tempo impiega il training?**  
A: ~60-120 secondi con 500 bars di dati

**Q: Posso trainare mentre uso la dashboard?**  
A: Si! I dati si aggiornano in tempo reale

**Q: Dove vengono salvati i risultati?**  
A: In `results/` cartella, uno JSON file per session

**Q: Posso usare i miei dati?**  
A: Si! Upload un CSV con colonne: open,high,low,close,volume

**Q: Quale dataset size è ottimale?**  
A: 500-1000 bars per un buon training. Minimo 100.

---

## 📞 Support

Se hai problemi:
1. Esegui: `python test_dashboard.py`
2. Controlla i logs nella Dashboard
3. Verifica file format (OHLCV)
4. Riavvia con: `python launch_dashboard.py`

---

## 🎉 Hai Finito!

La dashboard è pronta all'uso!

```bash
python launch_dashboard.py
```

**Buon training! 🚀📊**
