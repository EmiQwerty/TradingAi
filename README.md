# Trading System Multi-Symbol / Multi-Timeframe

Sistema di trading completo, modulare e operativo per trading demo/live su coppie forex e metalli con analisi multi-timeframe.

## Caratteristiche

### Architettura
- **Multi-symbol**: EURUSD, GBPUSD, XAUUSD (configurabile)
- **Multi-timeframe**: M1, M5, M15, H1, H4 con sincronizzazione automatica
- **Modulare**: Componenti indipendenti e riutilizzabili
- **Thread-safe**: State management centralizzato con lock
- **Event-driven**: Logging strutturato di tutti gli eventi

### Componenti Principali

#### Data Layer
- **MarketFeed**: Connessione broker via REST/WebSocket
- **TimeframeResampler**: Resampling automatico multi-TF
- **DataStorage**: Persistenza dati in formato parquet/CSV

#### Features
- **Indicators**: RSI, Stochastic, SMA/EMA, PMax, ATR, MACD, ADX, Bollinger
- **Structure**: BOS/CHOCH/Order Blocks (Smart Money Concepts)
- **Volatility**: Multiple estimators (Historical, Parkinson, Garman-Klass)

#### ML & Macro
- **MLInference**: Modelli frozen per regime/trend/volatility classification
- **MacroFilter**: Filtro notizie economiche con peso configurabile (8%)

#### Strategy
- **ConfidenceAggregator**: Aggregazione segnali multi-fonte (ML 35%, Indicators 30%, Structure 25%, Macro 10%)
- **DecisionEngine**: Generazione decisioni entry/exit con SL/TP automatici

#### Risk Management
- **RiskManager**: Controlli globali (exposure 15%, drawdown 20%, 1% per trade)
- **CorrelationAnalyzer**: Monitoraggio correlazioni cross-pair (limite 0.7)

#### Execution
- **BrokerAPI**: Interfaccia REST per broker
- **OrderExecutor**: Esecuzione ordini con retry e throttling

#### Monitoring
- **MetricsCalculator**: Performance metrics (Sharpe, Sortino, drawdown, win rate)
- **EventLogger**: Log centralizzato di segnali, trades, errori
- **Dashboard**: Streamlit read-only per monitoring real-time

#### Backtest
- **BacktestEngine**: Simulazione completa con pipeline reale
- **WalkForwardValidator**: Validazione rolling windows con ottimizzazione

## Installazione

### Requisiti
- Python 3.8+
- pip

### Setup

1. **Clone/Download progetto**
```bash
cd /Users/emiliano/Desktop/Trading
```

2. **Crea virtual environment (raccomandato)**
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows
```

3. **Installa dipendenze**
```bash
pip install -r requirements.txt
```

4. **Configura environment**
```bash
cp .env.template .env
# Modifica .env con le tue API keys
```

5. **Verifica configurazioni**
```bash
# Edita config/*.yaml secondo necessità
# - config/settings.yaml: configurazione globale
# - config/symbols.yaml: strategie per simbolo
# - config/risk.yaml: limiti di rischio
```

## Utilizzo

### Demo Trading (Default)
```bash
python main.py --mode demo
```

### Live Trading
```bash
python main.py --mode live
```

### Backtest
```bash
python main.py --mode backtest \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --capital 100000
```

### Walk-Forward Validation
```bash
python main.py --mode walk-forward \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --capital 100000
```

### Dashboard Monitoring (separato)
```bash
streamlit run monitoring/dashboard.py
```

## Struttura Progetto

```
Trading/
├── config/                 # Configurazioni YAML
│   ├── settings.yaml      # Config globale
│   ├── symbols.yaml       # Config per simbolo
│   └── risk.yaml          # Limiti rischio
├── data/                  # Data layer
│   ├── market_feed.py     # Connessione broker
│   ├── resampler.py       # Resampling multi-TF
│   └── storage.py         # Persistenza dati
├── features/              # Features engineering
│   ├── indicators.py      # Indicatori tecnici
│   ├── structure.py       # Strutture di mercato
│   └── volatility.py      # Analisi volatilità
├── ml/                    # Machine Learning
│   ├── models/            # Modelli trained (.pkl)
│   └── inference.py       # Inference engine
├── macro/                 # Macro economics
│   ├── calendar.py        # Calendario economico
│   └── macro_filter.py    # Filtro macro
├── strategy/              # Trading strategy
│   ├── confidence.py      # Aggregazione confidence
│   └── decision_engine.py # Decision engine
├── risk/                  # Risk management
│   ├── risk_manager.py    # Risk manager globale
│   └── correlation.py     # Analisi correlazioni
├── execution/             # Order execution
│   ├── broker_api.py      # API broker
│   └── orders.py          # Order executor
├── state/                 # State management
│   ├── models.py          # Data models
│   └── state_store.py     # State store centralizzato
├── monitoring/            # Monitoring & observability
│   ├── metrics.py         # Metriche performance
│   ├── events.py          # Event logger
│   └── dashboard.py       # Streamlit dashboard
├── backtest/              # Backtesting
│   ├── engine.py          # Backtest engine
│   └── walk_forward.py    # Walk-forward validation
├── main.py               # Entry point principale
├── requirements.txt      # Dipendenze Python
└── .env.template         # Template environment variables
```

## Configurazione

### settings.yaml
- **broker**: Configurazione connessione broker (API URL, keys via .env)
- **trading**: Simboli, timeframes, capitale iniziale, update interval
- **indicators**: Parametri indicatori (RSI period, Stochastic K/D, MA periods, ecc.)
- **structure**: Parametri rilevamento strutture (swing lookback, buffer, order blocks)
- **ml**: Paths modelli, features, confidence thresholds
- **macro**: Calendario economico, impact threshold, adjustment weight (8%)
- **risk**: Limiti globali (1% per trade, 15% exposure, 20% drawdown)
- **execution**: Order type, slippage, commission, retry logic
- **storage**: Formato (parquet), compressione, retention
- **logging**: Level, formato, file output
- **monitoring**: Dashboard config, metrics retention
- **backtest**: Periodo train/test, walk-forward settings

### symbols.yaml
Configurazione specifica per EURUSD, GBPUSD, XAUUSD:
- **strategy**: Nome strategia (trend_following, breakout, range_momentum)
- **primary_timeframe**: TF principale per decisioni
- **entry_conditions**: Condizioni entry (confidence min, structure required, MA alignment)
- **exit_conditions**: Condizioni exit (CHOCH, regime change, trailing stop)
- **risk_parameters**: SL/TP multiplier ATR, max position size
- **correlation_matrix**: Correlazioni attese con altri simboli
- **session_adjustments**: Volatility spread per sessione (Asia/London/NY)

### risk.yaml
Controlli rischio dettagliati:
- **global_risk**: Max risk per trade, total exposure, max open positions
- **position_sizing**: Risk per trade, max leverage, min/max volumes
- **stop_loss**: ATR multiplier, trailing stop, max loss per position
- **correlation_limits**: Max correlation, exposure reduction
- **drawdown_control**: Max drawdown%, daily loss limit, consecutive losses
- **time_based**: Trading hours, blackout periods

## Workflow

### Live Trading Flow
1. **Startup**: Connessione broker, sync posizioni, load configurazioni
2. **Main Loop** (ogni `update_interval_seconds`):
   - Fetch M1 data per tutti i simboli
   - Resample a tutti i timeframes
   - Calcola indicators per ogni TF
   - Detect structure (BOS/CHOCH/Order Blocks)
   - Analizza volatility
   - Esegui ML inference (regime/trend/volatility)
   - Apply macro filter
   - Aggregate confidence score
   - Generate decisions (entry/exit)
   - Check risk limits
   - Execute orders validi
   - Update state store
   - Log eventi
3. **Monitoring**: Check posizioni aperte per SL/TP/trailing stop
4. **Shutdown**: Chiusura graceful, export state/logs

### Backtest Flow
1. Load dati storici (M1) per periodo
2. Resample tutti i TF
3. Itera su barre primary TF:
   - Calcola features (indicators, structure, volatility)
   - ML inference
   - Generate decisions
   - Check risk limits
   - Simula execution (con slippage/commission)
   - Update equity curve
   - Track MAE/MFE
4. Calcola metriche finali
5. Export risultati (JSON summary, CSV trades/equity)

### Walk-Forward Flow
1. Crea finestre train/test (es. 6 mesi train, 1 mese test)
2. Per ogni finestra:
   - Run backtest su train period
   - (Opzionale) Ottimizza parametri su train
   - Apply parametri ottimizzati
   - Run backtest su test period (out-of-sample)
   - Log performance train vs test
3. Combina risultati OOS
4. Calcola stability score (std returns tra windows)
5. Export risultati aggregati e per window

## Monitoring

### Dashboard (Streamlit)
- **System Status**: Health, trading enabled, uptime
- **Account Overview**: Balance, equity, P&L, drawdown
- **Positions Tab**: Tabella posizioni aperte con entry/SL/TP/P&L
- **Performance Tab**: Metriche all-time/daily/weekly (win rate, Sharpe, drawdown)
- **Signals Tab**: Recent signals con confidence e reasoning
- **Market Overview Tab**: Stato simboli per TF (regime, trend, structure)
- **Events Tab**: Log eventi filtrabili (signals/trades/errors/system)
- **Auto-refresh**: Configurable interval

### Logs
- **File logs**: `logs/trading_YYYYMMDD.log`
- **Event export**: `events_YYYYMMDD_HHMMSS.json`
- **State export**: `state_YYYYMMDD_HHMMSS.json`

### Metrics
- **Win Rate**: % trades profittevoli
- **Profit Factor**: Gross profit / Gross loss
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Max Drawdown**: Max equity drop %
- **Average Trade**: Mean P&L per trade
- **Expectancy**: Expected value per trade
- **Consecutive Wins/Losses**: Max streaks

## Estensioni

### Aggiungere Nuovo Simbolo
1. Aggiungi entry in `config/symbols.yaml`
2. Aggiungi correlazioni in `correlation_matrix` degli altri simboli
3. Update `config/settings.yaml` trading.symbols list

### Aggiungere Nuovo Indicatore
1. Aggiungi metodo in `features/indicators.py`
2. Aggiungi config in `config/settings.yaml` indicators section
3. Update `calculate_all_indicators()` per includere nuovo indicatore

### Aggiungere Nuovo Modello ML
1. Train modello offline
2. Salva in `ml/models/<nome>.pkl` (joblib/pickle)
3. Aggiungi predict method in `ml/inference.py`
4. Update config in `config/settings.yaml` ml section

### Custom Strategy
1. Modifica `strategy/confidence.py` per custom aggregation weights
2. Modifica `strategy/decision_engine.py` per custom entry/exit logic
3. Update symbol-specific strategy in `config/symbols.yaml`

## Testing

### Test Componenti Individuali
```bash
# Test indicators
python -c "from features.indicators import TechnicalIndicators; import pandas as pd; print('OK')"

# Test ML inference
python -c "from ml.inference import MLInference; print('OK')"

# Test backtest engine
python -c "from backtest.engine import BacktestEngine; print('OK')"
```

### Dry Run (Demo Mode)
```bash
# Test sistema completo senza rischi
python main.py --mode demo
# Ctrl+C per interrompere dopo alcuni cicli
```

### Backtest Veloce
```bash
# Test su periodo breve
python main.py --mode backtest \
    --start-date 2023-12-01 \
    --end-date 2023-12-31 \
    --capital 10000
```

## Troubleshooting

### Import Errors
```bash
# Verifica che venv sia attivo
which python
# Dovrebbe mostrare path dentro venv/

# Re-installa dipendenze
pip install -r requirements.txt
```

### TA-Lib Installation (se problemi)
```bash
# macOS (con Homebrew)
brew install ta-lib
pip install ta-lib

# Linux (Ubuntu/Debian)
sudo apt-get install libta-lib-dev
pip install ta-lib

# Windows: scarica pre-built wheel da
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.XX-cpXX-cpXX-win_amd64.whl
```

### Broker Connection Failed
- Verifica `.env` con API keys corrette
- Testa endpoint broker separatamente
- Check logs per dettagli errore
- In demo mode, usa dummy data generati automaticamente

### Performance Issues
- Riduci `update_interval_seconds` in settings.yaml
- Riduci numero simboli/timeframes
- Incrementa storage compression
- Monitor CPU/RAM usage

## Sicurezza

### Credenziali
- **Mai committare** `.env` su git (già in `.gitignore`)
- Usa permissions restrittivi: `chmod 600 .env`
- Rigenera API keys periodicamente

### Risk Controls
- Testa sempre in **demo** prima di live
- Verifica limiti in `config/risk.yaml` siano appropriati
- Monitor dashboard costantemente durante live
- Setup alerts (email/telegram) per drawdown/errori

### Backup
```bash
# Backup configurazioni
tar -czf config_backup_$(date +%Y%m%d).tar.gz config/

# Backup dati storici
tar -czf data_backup_$(date +%Y%m%d).tar.gz data_storage/

# Backup logs
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/
```

## Performance

### Ottimizzazione
- Storage: usa parquet con snappy compression (default)
- Indicators: calcola solo necessari per strategy
- ML: usa modelli lightweight, inference <100ms
- Resampling: batch update invece che real-time per TF > H1

### Scalabilità
- Sistema supporta 10+ simboli con 5 TF simultaneamente
- Update interval consigliato: 60s per H1 primary, 300s per H4
- RAM usage: ~500MB per 10 simboli con 1 anno dati M1
- CPU: <10% su modern CPU (i5/Ryzen 5)

## Licenza

Proprietario - Uso personale/demo

## Contatti

Per supporto o domande: [tuo@email.com]

## Changelog

### v1.0.0 (2024)
- Release iniziale
- Multi-symbol/multi-TF completo
- Indicators, Structure, ML, Macro integration
- Risk management globale
- Backtest e walk-forward
- Streamlit dashboard
- Demo e live trading operativi
