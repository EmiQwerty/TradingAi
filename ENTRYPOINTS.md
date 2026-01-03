# 🚀 Sistema Trading - Entrypoint e Modalità di Avvio

## 📍 Entrypoint Principali

Il sistema ha **4 entrypoint principali** a seconda di cosa vuoi fare:

---

## 1. 🎯 **TRADING SYSTEM (main.py)** - Sistema Completo di Trading

**File**: `main.py`  
**Cosa fa**: Sistema di trading completo con decision engine, risk management, execution

### 🔴 **Modalità Demo Trading** (Consigliata per iniziare)
```bash
python main.py --mode demo
```

**Avvia**:
- ✅ Connessione a broker OANDA (account demo)
- ✅ Market feed real-time multi-simbolo/multi-timeframe
- ✅ Decision Engine con analisi H4→H1→M5
- ✅ Risk Manager con position sizing
- ✅ Order Executor (ordini demo)
- ✅ Monitoring e logging real-time
- ✅ State persistence

**Usa quando**: Vuoi fare paper trading con dati reali ma senza rischio

---

### 🟢 **Modalità Live Trading**
```bash
python main.py --mode live
```

**Avvia**:
- 🔴 **ATTENZIONE**: Usa account OANDA LIVE (soldi reali!)
- Tutto come demo mode ma con esecuzione reale

**Usa quando**: Sistema testato e pronto per trading con capitale reale

---

### 📊 **Modalità Backtest**
```bash
python main.py --mode backtest --start-date 2023-01-01 --end-date 2023-12-31 --capital 100000
```

**Avvia**:
- ✅ Backtesting su dati storici
- ✅ Simula tutte le decisioni del sistema
- ✅ Calcola performance metrics (win rate, profit factor, drawdown)
- ✅ Genera report dettagliato

**Usa quando**: Vuoi testare il sistema su dati storici prima di andare live

---

### 🔄 **Modalità Walk-Forward Validation**
```bash
python main.py --mode walk-forward --start-date 2023-01-01 --end-date 2023-12-31 --capital 100000
```

**Avvia**:
- ✅ Walk-forward optimization
- ✅ Training window + validation window
- ✅ Rolling forward nel tempo
- ✅ Valida robustezza del sistema

**Usa quando**: Vuoi validare che il sistema non sia overfitted

---

## 2. 📈 **ML TRAINING DASHBOARD** - Training e Monitoring ML

**File**: `launch_dashboard.py` o `ml/dashboard.py`  
**Cosa fa**: Dashboard Streamlit per training modelli ML e monitoring

### Avvio Dashboard
```bash
# Metodo 1: Script launcher
python launch_dashboard.py

# Metodo 2: Streamlit diretto
streamlit run ml/dashboard.py

# Metodo 3: Bash script (Unix/Mac)
./run_dashboard.sh
```

**Avvia**:
- ✅ Web UI su http://localhost:8501
- ✅ Upload dati storici (CSV)
- ✅ Training modelli ML (RF, GB, XGBoost, LightGBM)
- ✅ Real-time training progress
- ✅ Feature importance visualization
- ✅ Model comparison charts
- ✅ Test predictions live
- ✅ Download trained models

**Usa quando**: 
- Vuoi addestrare modelli ML su dati storici
- Vuoi visualizzare feature importance
- Vuoi testare predizioni in interfaccia grafica
- Vuoi confrontare performance di diversi modelli

---

## 3. 🧪 **TEST SUITE** - Testing e Validation

### Test ML Pipeline
```bash
python ml/test_pipeline.py
```

**Avvia**:
- ✅ Test Feature Engineering
- ✅ Test Backtest Engine
- ✅ Test Model Training
- ✅ Test Prediction Engine
- ✅ Test Context Analysis
- ✅ Test Full Pipeline Integration

**Usa quando**: Vuoi verificare che tutti i componenti ML funzionino

---

### Test Mock Broker
```bash
python test_mock_broker.py
```

**Avvia**:
- ✅ Test MockBroker con virtual account
- ✅ Test Yahoo Finance data fetcher
- ✅ Simula trade senza connessione broker
- ✅ Verifica PnL tracking

**Usa quando**: Vuoi testare il sistema senza connessione OANDA

---

### Test Dashboard
```bash
python test_dashboard.py
```

**Avvia**:
- ✅ Test componenti dashboard
- ✅ Test synthetic data generation
- ✅ Test training flow
- ✅ Test visualizations

**Usa quando**: Vuoi verificare che la dashboard funzioni

---

## 4. 🎓 **QUICKSTART & EXAMPLES** - Guide Rapide

### OANDA Quickstart
```bash
python OANDA_QUICKSTART.py
```

**Avvia**:
- ✅ Setup guidato OANDA API
- ✅ Test connessione
- ✅ Fetch dati real-time
- ✅ Place order demo

**Usa quando**: Prima volta con OANDA, vuoi verificare setup

---

### Mock Broker Quickstart
```bash
python MOCK_BROKER_QUICKSTART.py
```

**Avvia**:
- ✅ Setup MockBroker con $100k virtual
- ✅ Test trade simulation
- ✅ Test Yahoo Finance integration

**Usa quando**: Vuoi testare sistema senza OANDA

---

## 📊 Riepilogo Modalità

| Entrypoint | File | Cosa Fa | Quando Usare |
|------------|------|---------|--------------|
| **Demo Trading** | `main.py --mode demo` | Trading demo con dati reali | Paper trading |
| **Live Trading** | `main.py --mode live` | Trading reale con soldi veri | Produzione |
| **Backtest** | `main.py --mode backtest` | Test su dati storici | Validazione strategia |
| **Walk-Forward** | `main.py --mode walk-forward` | Validation robusta | Anti-overfitting |
| **ML Dashboard** | `launch_dashboard.py` | Training ML + UI | Addestrare modelli |
| **Test ML** | `ml/test_pipeline.py` | Test componenti ML | Verifica funzionamento |
| **Test Mock** | `test_mock_broker.py` | Test senza broker | Test offline |
| **OANDA Setup** | `OANDA_QUICKSTART.py` | Setup OANDA | Prima configurazione |

---

## 🔧 Configurazione

Prima di avviare, verifica configurazione:

### 1. File di Configurazione (config/)
```
config/
├── settings.yaml      # Impostazioni sistema
├── symbols.yaml       # Configurazione simboli
└── risk.yaml         # Risk management rules
```

### 2. Environment Variables (.env.oanda)
```bash
OANDA_ACCOUNT_ID=your-account-id
OANDA_API_TOKEN=your-api-token
OANDA_ENVIRONMENT=practice  # o 'live' per account reale
```

### 3. Modelli ML (ml/models/)
```
ml/models/
├── random_forest_model.pkl
├── xgboost_model.pkl
├── lightgbm_model.pkl
└── feature_names.json
```

Se non hai modelli, usa la **ML Dashboard** per addestrarne!

---

## 🚦 Flow Consigliato (Prima Volta)

### Step 1: Test Componenti
```bash
# 1. Test ML Pipeline
python ml/test_pipeline.py

# 2. Test Mock Broker
python test_mock_broker.py
```

### Step 2: Training ML
```bash
# Avvia dashboard per training
python launch_dashboard.py

# Poi:
# 1. Upload dati storici CSV
# 2. Click "Start Training"
# 3. Aspetta training completo
# 4. Download modelli addestrati
```

### Step 3: Backtest
```bash
# Test su dati storici
python main.py --mode backtest \
    --start-date 2023-06-01 \
    --end-date 2023-12-31 \
    --capital 100000

# Analizza report per verificare performance
```

### Step 4: Demo Trading
```bash
# Setup OANDA account demo
python OANDA_QUICKSTART.py

# Avvia demo trading
python main.py --mode demo

# Monitora per 1-2 settimane
```

### Step 5: Live Trading (Solo se Step 4 positivo!)
```bash
# ATTENZIONE: Usa capitale che puoi permetterti di perdere!
python main.py --mode live
```

---

## 📝 Argomenti Disponibili

### main.py
```bash
python main.py [OPTIONS]

Opzioni:
  --mode {demo,live,backtest,walk-forward}
                        Modalità operativa (default: demo)
  --config CONFIG       Directory configurazioni (default: config)
  --start-date YYYY-MM-DD
                        Data inizio (backtest/walk-forward)
  --end-date YYYY-MM-DD
                        Data fine (backtest/walk-forward)
  --capital FLOAT       Capitale iniziale (default: 100000.0)

Esempi:
  python main.py --mode demo
  python main.py --mode backtest --start-date 2023-01-01 --end-date 2023-12-31
  python main.py --mode live --config config_production
```

### launch_dashboard.py
```bash
python launch_dashboard.py

Nessuna opzione necessaria - avvia dashboard su http://localhost:8501
```

---

## 🔍 Logging e Output

### Log Files
```
logs/
├── trading_20240103.log    # Log sistema trading
├── ml_training.log         # Log training ML
└── dashboard.log           # Log dashboard
```

### Monitoring Real-Time

Durante demo/live trading, il sistema logga:
- ✅ Market feed updates
- ✅ Decision engine signals
- ✅ Trade executions
- ✅ PnL updates
- ✅ Risk metrics
- ✅ System events

Usa `tail -f logs/trading_YYYYMMDD.log` per monitoring real-time

---

## 🛑 Stop Sistema

### Stop Gracefully
```bash
# Premi Ctrl+C nel terminale
# Il sistema:
# 1. Chiude posizioni aperte (se configurato)
# 2. Salva stato corrente
# 3. Disconnette dal broker
# 4. Scrive report finale
```

### Force Stop (Emergenza)
```bash
# Unix/Mac
killall -9 python

# Windows
taskkill /F /IM python.exe
```

---

## 🆘 Troubleshooting

### "Cannot connect to broker"
```bash
# 1. Verifica .env.oanda
cat .env.oanda

# 2. Test connessione
python OANDA_QUICKSTART.py

# 3. Controlla account OANDA online
```

### "ModuleNotFoundError"
```bash
# Installa dipendenze
pip install -r requirements.txt
pip install -r requirements-dashboard.txt
```

### "No trained models found"
```bash
# Addestra modelli con dashboard
python launch_dashboard.py

# O usa test con synthetic data
python ml/test_pipeline.py
```

---

## 📚 Documentazione Aggiuntiva

- **Sistema Trading Completo**: [README.md](README.md)
- **ML Pipeline**: [ml/README.md](ml/README.md)
- **Dashboard Guide**: [README_DASHBOARD.md](README_DASHBOARD.md)
- **Quick Start ML**: [ml/GETTING_STARTED.md](ml/GETTING_STARTED.md)
- **Mock Broker**: [MOCK_BROKER_COMPLETE.py](MOCK_BROKER_COMPLETE.py)

---

## ✨ Quale Entrypoint Scegliere?

| Se vuoi... | Usa questo |
|-----------|------------|
| **Fare paper trading** | `python main.py --mode demo` |
| **Addestrare modelli ML** | `python launch_dashboard.py` |
| **Testare su dati storici** | `python main.py --mode backtest ...` |
| **Verificare componenti** | `python ml/test_pipeline.py` |
| **Setup prima volta** | `python OANDA_QUICKSTART.py` |
| **Trading con soldi reali** | `python main.py --mode live` ⚠️ |

---

**TL;DR**: 

- **Trading Demo**: `python main.py --mode demo`
- **ML Training**: `python launch_dashboard.py`
- **Backtest**: `python main.py --mode backtest --start-date ... --end-date ...`
- **Test**: `python ml/test_pipeline.py`

🎯 **Consiglio**: Inizia con ML Training Dashboard per addestrare modelli, poi Demo Trading!
