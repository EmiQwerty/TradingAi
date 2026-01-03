"""
OANDA Demo Account Quick Start Guide
Sistema di trading pronto per girare su conto DEMO OANDA senza modifiche strutturali.
"""

# ============================================================================
# STEP 1: Setup Credenziali OANDA
# ============================================================================

# 1.1 Registra un conto DEMO gratuito su: https://www.oanda.com
# - Vai a https://www.oanda.com/register
# - Crea account gratuito (conto pratica)
# - Riceverai istantaneamente account ID e password

# 1.2 Ottieni API Key
# - Accedi al pannello: https://v20-demo.oanda.com
# - Menu > "Account Settings" > "API Access"
# - Clicca "Generate Token"
# - Copia il token (è la tua OANDA_API_KEY)

# 1.3 Identifica Account ID
# - Nel pannello principale, l'Account ID è visibile (es: 123456789-001)
# - Copia e conserva

# 1.4 Configura Environment
# Opzione A: File .env nella root del progetto
cp .env.template .env
# Aggiungi:
# OANDA_API_KEY=your_token_here
# OANDA_ACCOUNT_ID=123456789-001

# Opzione B: File .env.oanda
cp .env.oanda .env.oanda
# Compila il file .env.oanda

# Opzione C: Variabili di sistema
export OANDA_API_KEY="your_token_here"
export OANDA_ACCOUNT_ID="123456789-001"


# ============================================================================
# STEP 2: Configura Simboli Trading
# ============================================================================

# 2.1 Edita config/symbols.yaml
# Verifica che i simboli siano in formato OANDA standard:
# - EUR_USD (non EURUSD)
# - GBP_USD (non GBPUSD)
# - XAU_USD (non XAUUSD)

# 2.2 OANDA Strumenti disponibili (principali):
# Forex Pairs:
#   EUR_USD, GBP_USD, USD_JPY, USD_CHF, AUD_USD, NZD_USD, EUR_GBP,
#   EUR_JPY, GBP_JPY, ecc

# Commodities:
#   XAU_USD (Gold), XAG_USD (Silver)

# Indici:
#   SPX500_USD (S&P 500), NAS100_USD (Nasdaq)

# Bitcoin:
#   BTC_USD, ETH_USD

# 2.3 Configurazione per EUR_USD:
# symbol: EUR_USD
# pip_value: 0.0001           # OANDA standard
# contract_size: 100000       # 1 lot = 100k units
# min_lot: 0.01              # Minimo 0.01 lot
# max_lot: 100               # Massimo 100 lot
# lot_step: 0.01


# ============================================================================
# STEP 3: Avvia Sistema
# ============================================================================

# 3.1 Demo Mode (consigliato per prima volta)
python main.py --mode demo

# Output atteso:
# - "Connected to OANDA: Balance=100000.00"
# - "Main loop avviato"
# - "=== Status ===" ogni ciclo

# 3.2 Ctrl+C per fermare
# - Chiuderà gracefully posizioni aperte (opzionale)
# - Salverà logs e state

# 3.3 Modalità Live (con vero soldi - ATTENZIONE!)
# Assicurati:
# - Trading strategy testato in backtest
# - Risk management appropriato
# - Sono consapevole dei rischi

python main.py --mode live


# ============================================================================
# STEP 4: Monitoring
# ============================================================================

# 4.1 Dashboard Streamlit (in altro terminale)
streamlit run monitoring/dashboard.py

# Visualizzerai:
# - Account balance e equity in tempo reale
# - Posizioni aperte (entry, SL, TP, P&L)
# - Performance metrics
# - Segnali generati
# - Log eventi

# 4.2 Log Files
tail -f logs/trading_$(date +%Y%m%d).log

# 4.3 State File
cat data_storage/state_*.json


# ============================================================================
# STEP 5: Backtest Offline
# ============================================================================

# 5.1 Backtest storico (gennaio 2024)
python main.py --mode backtest \
    --start-date 2024-01-01 \
    --end-date 2024-02-01 \
    --capital 100000

# Output:
# - backtest_results/backtest_summary_*.json
# - backtest_results/backtest_trades_*.csv
# - backtest_results/backtest_equity_*.csv

# 5.2 Analizza risultati
cd backtest_results
ls -lah
cat backtest_summary_*.json | jq


# ============================================================================
# STEP 6: Walk-Forward Validation
# ============================================================================

# 6.1 Validazione rolling (6 mesi train + 1 mese test)
python main.py --mode walk-forward \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --capital 100000

# Output:
# - walk_forward_results/wf_summary_*.json
# - walk_forward_results/wf_trades_*.csv
# - walk_forward_results/wf_windows_*.csv
# - walk_forward_results/wf_plot_*.png


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

# Errore: "OANDA credentials not provided"
# -> Verifica che OANDA_API_KEY e OANDA_ACCOUNT_ID siano settati
# -> Esegui: echo $OANDA_API_KEY

# Errore: "Failed to connect to OANDA"
# -> Verifica token non scaduto (rigenera se necessario)
# -> Verifica Internet connection
# -> Verifica che is_live=false per conto DEMO

# Errore: "openTrades endpoint 404"
# -> Bug OANDA raro: usa /accounts/{id}/positions invece
# -> Contatta OANDA support se persiste

# Nessun segnale generato
# -> Verifica che regime != 'CHAOS'
# -> Verifica confidence score > 0.65 (threshold default)
# -> Analizza logs per vedere quale condition fallisce

# Dashboard non visibile
# -> Assicurati Streamlit installato: pip install streamlit>=1.28.0
# -> Default URL: http://localhost:8501


# ============================================================================
# COSA SUCCEDE AL STARTUP
# ============================================================================

# 1. Load configurazioni YAML
# 2. Setup logging
# 3. Connetti OANDA (verifica account info)
# 4. Sync posizioni aperte da broker
# 5. Main loop inizia:
#    - Fetch M1 data per EUR_USD, GBP_USD, XAU_USD
#    - Resample a M5, M15, H1, H4
#    - Calcola indicators (RSI, Stochastic, MA, ATR, ecc)
#    - Detect structure (BOS/CHOCH)
#    - Volatility analysis
#    - ML inference (regime, trend, volatility)
#    - Macro filter
#    - Aggregate confidence
#    - Decision Engine:
#       - Se regime=TREND: trend-following logic
#       - Se regime=RANGE: mean-reversion logic
#       - Se regime=CHAOS: no trading
#    - Check risk limits
#    - Execute market orders via OANDA API
#    - Update state store
#    - Log everything
#
# 6. Ciclo ripete ogni 60 secondi (configurable)
# 7. Ctrl+C: shutdown graceful


# ============================================================================
# DECISION ENGINE - LOGICA DETERMINISTICA
# ============================================================================

# TREND-FOLLOWING (quando ML regime = TREND):
# 1. H4 Confirmation:
#    - Controlla H4 trend strength > 0.5
#    - Estrai direzione da H4 RSI (>50=BUY, <50=SELL)
#
# 2. H1 Refinement:
#    - Verifica H1 direzione == H4 direzione
#    - Controlla BOS/CHOCH aligned
#    - Estrai confidence da H1 indicators
#
# 3. M5 Entry:
#    - Per BUY: RSI 30-55 + price > EMA9
#    - Per SELL: RSI 45-70 + price < EMA9
#
# 4. SL/TP Calculation:
#    - SL = entry ± 2xATR
#    - TP = entry ± 3xATR (1.5:1 risk/reward)
#
# 5. Position Sizing:
#    - Risk 1% del capitale per trade
#    - Size = risk_amount / (pips_to_sl * pip_value * contract_size)

# MEAN-REVERSION (quando ML regime = RANGE):
# 1. H1 RSI Extremes:
#    - RSI < 30: BUY (oversold bounce)
#    - RSI > 70: SELL (overbought bounce)
#
# 2. Bollinger Bands:
#    - SL al di là del band opposto
#    - TP al middle band
#
# 3. Size: 1% risk rule


# ============================================================================
# ESEMPIO FLUSSO REALE
# ============================================================================

# Terminal 1: Trading system
$ python main.py --mode demo

# Output:
# INFO:root:TradingSystem initialized: 3 symbols, 5 timeframes
# INFO:root:Avvio sistema in modalità DEMO
# INFO:root:Connected to OANDA: Balance=100000.00
# INFO:root:Main loop avviato
# DEBUG:root:Bar 100/5000: Equity=100000.00, Open=0
# ...
# INFO:strategy.decision_engine:DECISION: EUR_USD BUY 0.5 @ 1.10005 SL=1.09995 TP=1.10025 Conf=0.75
# INFO:execution.orders:Executing MARKET order: EUR_USD BUY 0.5 @ 1.10005
# INFO:monitoring.events:TRADE: EUR_USD BUY 0.5 @ 1.10005 SL=1.09995 TP=1.10025
# ...

# Terminal 2 (in parallelo): Dashboard
$ streamlit run monitoring/dashboard.py

# Visualizza in browser: http://localhost:8501
# - Account: Balance=100000, Equity=100012.50, P&L=+12.50
# - Positions: EUR_USD LONG 50k units @ 1.10005, unrealized P&L=+12.50
# - Performance: 1 trade, 100% win rate, +12.50 USD
# - Signals: [BUY EUR_USD @ 1.10005 H1 Conf=0.75]


# ============================================================================
# IMPORTANTE
# ============================================================================

# ⚠️ DISCLAIMER:
# - Questo sistema è per TESTING e EDUCATIONAL PURPOSES
# - Trading comporta rischi significativi
# - Testare SEMPRE su conto DEMO prima di live
# - Non usare se non capisci completamente il sistema
# - Risk management è CRITICO

# ✓ BEST PRACTICES:
# - Monitora dashboard mentre sistema gira
# - Imposta risk limits bassi per inizio
# - Backtest su dati storici
# - Walk-forward validate prima di live
# - Tieni log dettagliati di tutti i trades
# - Stop se equity cala > 20%

print("""
═══════════════════════════════════════════════════════════════════════════
 TRADING SYSTEM READY FOR OANDA DEMO ACCOUNT
═══════════════════════════════════════════════════════════════════════════

1. Compila OANDA credenziali in .env o .env.oanda
2. python main.py --mode demo
3. streamlit run monitoring/dashboard.py (altro terminale)
4. Monitora account mentre sistema opera
5. Controlla logs in logs/trading_*.log

Sistema pronto per OANDA v20 API senza modifiche strutturali.
═══════════════════════════════════════════════════════════════════════════
""")
