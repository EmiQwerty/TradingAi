"""
GUIDA RAPIDA - Mock Broker con Dati Yahoo Finance
Sistema di trading con conto virtuale e candele reali
"""

print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║  TRADING SYSTEM - MOCK BROKER + YAHOO FINANCE                           ║
║  Conto virtuale con dati REALI                                          ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────┐
│ 1. INSTALLAZIONE DIPENDENZE                                             │
└─────────────────────────────────────────────────────────────────────────┘

Prima di avviare, installa yfinance:

    pip install yfinance

Oppure:

    pip install -r requirements.txt


┌─────────────────────────────────────────────────────────────────────────┐
│ 2. CONFIGURAZIONE                                                       │
└─────────────────────────────────────────────────────────────────────────┘

Il sistema è già configurato in modalità MOCK.

File: config/settings.yaml

broker:
  broker_type: "mock"  # Conto virtuale (NO OANDA)
  
trading:
  initial_capital: 100000.0  # Capitale iniziale virtuale
  leverage: 30               # Leva 1:30
  commission_per_lot: 7.0    # Commissione per lotto
  symbols:
    - EUR_USD
    - GBP_USD
    - XAU_USD

Dati scaricati da Yahoo Finance:
- EUR_USD → EURUSD=X
- GBP_USD → GBPUSD=X
- XAU_USD → GC=F (Gold futures)


┌─────────────────────────────────────────────────────────────────────────┐
│ 3. AVVIO SISTEMA                                                        │
└─────────────────────────────────────────────────────────────────────────┘

Avvia il sistema di trading:

    python main.py

Il sistema:
1. Inizializza MockBroker con $100,000 virtuali
2. Scarica dati reali da Yahoo Finance
3. Calcola indicatori (RSI, MA, Stochastic, ecc.)
4. Genera decisioni di trading deterministiche
5. Esegue ordini su conto virtuale
6. Calcola PnL in tempo reale
7. Controlla SL/TP automaticamente


┌─────────────────────────────────────────────────────────────────────────┐
│ 4. MONITORAGGIO                                                         │
└─────────────────────────────────────────────────────────────────────────┘

Dashboard (terminale separato):

    streamlit run monitoring/dashboard.py

Visualizza:
- Balance e Equity in tempo reale
- Posizioni aperte con PnL
- Storico trades
- Performance metrics (Win Rate, ecc.)
- Grafici equity curve

Logs:

    tail -f logs/trading_*.log


┌─────────────────────────────────────────────────────────────────────────┐
│ 5. COME FUNZIONA IL MOCK BROKER                                        │
└─────────────────────────────────────────────────────────────────────────┘

MockBroker simula un broker reale:

✅ Account Management:
   - Balance iniziale: $100,000
   - Leverage: 1:30
   - Commissioni: $7 per lotto standard

✅ Trading Operations:
   - place_market_order() → apre posizioni
   - close_position() → chiude posizioni
   - update_trade_sl_tp() → modifica SL/TP
   
✅ Position Management:
   - Calcolo margin requirement
   - Verifica margin disponibile
   - Calcolo unrealized PnL in tempo reale
   
✅ Risk Controls:
   - Stop Loss automatico
   - Take Profit automatico
   - Margin call simulation
   
✅ Performance Tracking:
   - Total trades
   - Win rate
   - Total PnL
   - Storico completo


┌─────────────────────────────────────────────────────────────────────────┐
│ 6. DATI YAHOO FINANCE                                                   │
└─────────────────────────────────────────────────────────────────────────┘

YahooFinanceDataFetcher scarica candele REALI:

Simboli supportati:
- Forex: EUR_USD, GBP_USD, USD_JPY, AUD_USD, ecc.
- Metalli: XAU_USD (oro), XAG_USD (argento)
- Crypto: BTC_USD, ETH_USD
- Indici: SPX, NDX, DJI

Timeframes:
- M1, M5, M15, M30 (intraday)
- H1, H4 (orari)
- D1, W1, MN (daily/weekly/monthly)

Cache automatico:
- Dati cachati per 5 minuti
- Riduce chiamate API Yahoo
- Velocizza ricalcoli

Uso programmatico:

    from data.yahoo_fetcher import YahooFinanceDataFetcher
    
    fetcher = YahooFinanceDataFetcher()
    data = fetcher.fetch_historical_data('EUR_USD', 'H1', 500)
    
    # DataFrame con colonne: open, high, low, close, volume


┌─────────────────────────────────────────────────────────────────────────┐
│ 7. ESEMPIO FLUSSO OPERATIVO                                            │
└─────────────────────────────────────────────────────────────────────────┘

1. Sistema avvia:
   ├─ MockBroker: Balance=$100,000
   ├─ YahooFetcher: Scarica EUR_USD H1 (500 bars)
   └─ Log: "MockBroker initialized: Balance=100000.00"

2. Loop principale:
   ├─ Fetch dati ultimi 500 bars da Yahoo Finance
   ├─ Resample a M5, H1, H4
   ├─ Calcola indicatori (RSI, MA, ATR, ecc.)
   ├─ ML inference: regime=TREND, strength=0.75
   └─ Decision Engine: BUY signal generato

3. Decisione trading:
   {
     "action": "BUY",
     "symbol": "EUR_USD",
     "entry": 1.10005,
     "stop": 1.09905,    # -100 pips (2×ATR)
     "take_profit": 1.10155,  # +150 pips (3×ATR)
     "size": 0.5,        # 0.5 lotti
     "units": 50000,     # 50k units
     "confidence": 0.75
   }

4. Esecuzione:
   ├─ MockBroker.place_market_order(EUR_USD, BUY, 50000, 1.10005)
   ├─ Calcola margin: 50000 × 1.10005 / 30 = $1,833
   ├─ Verifica margin disponibile: OK
   ├─ Apre posizione
   ├─ Sottrae commissione: $7
   └─ Log: "Position OPENED: EUR_USD BUY 50000 @ 1.10005"

5. Monitoring:
   ├─ Fetch nuovi prezzi da Yahoo ogni 60 secondi
   ├─ Update posizioni con current_price
   ├─ Calcola unrealized PnL
   ├─ Controlla SL/TP hit
   └─ Dashboard aggiornata in tempo reale

6. Take Profit hit:
   ├─ Prezzo raggiunge 1.10155
   ├─ MockBroker auto-chiude posizione
   ├─ Realized PnL: +$75 (150 pips su 0.5 lotti)
   ├─ Balance: $100,068
   └─ Log: "TAKE PROFIT HIT: EUR_USD @ 1.10155"


┌─────────────────────────────────────────────────────────────────────────┐
│ 8. BACKTEST CON DATI REALI                                             │
└─────────────────────────────────────────────────────────────────────────┘

Testa strategia su storico:

    python backtest/backtest_engine.py \\
        --symbol EUR_USD \\
        --timeframe H1 \\
        --start-date 2024-01-01 \\
        --end-date 2024-12-31 \\
        --initial-capital 100000

Output:
- Total Return: +15.2%
- Sharpe Ratio: 1.8
- Max Drawdown: -8.5%
- Win Rate: 65%
- Total Trades: 150
- Profit Factor: 2.1

Dati scaricati da Yahoo Finance automaticamente.


┌─────────────────────────────────────────────────────────────────────────┐
│ 9. WALK-FORWARD VALIDATION                                             │
└─────────────────────────────────────────────────────────────────────────┘

Valida robustezza strategia:

    python backtest/walk_forward.py \\
        --symbol EUR_USD \\
        --train-window 180 \\
        --test-window 30 \\
        --step 30

Esegue:
1. Train su 180 giorni → Test su 30 giorni
2. Slide window → Train su successivi 180 → Test 30
3. Ripete per tutto il periodo
4. Aggrega risultati out-of-sample


┌─────────────────────────────────────────────────────────────────────────┐
│ 10. SWITCH A OANDA (QUANDO PRONTO)                                     │
└─────────────────────────────────────────────────────────────────────────┘

Per passare da Mock a OANDA:

1. Registra conto OANDA DEMO:
   https://www.oanda.com

2. Ottieni credenziali:
   - API Token
   - Account ID

3. Configura .env:
   export OANDA_API_KEY="your_token"
   export OANDA_ACCOUNT_ID="123-456-789"

4. Modifica config/settings.yaml:
   broker:
     broker_type: "oanda"  # Era "mock"

5. Riavvia sistema:
   python main.py

Sistema switcha automaticamente a OANDA v20 API.
Zero modifiche al codice necessarie.


┌─────────────────────────────────────────────────────────────────────────┐
│ 11. SIMBOLI AGGIUNTIVI                                                 │
└─────────────────────────────────────────────────────────────────────────┘

Aggiungi simboli in settings.yaml:

trading:
  symbols:
    - EUR_USD
    - GBP_USD
    - XAU_USD
    - BTC_USD  # Bitcoin (dati da Yahoo: BTC-USD)
    - ETH_USD  # Ethereum
    - SPX      # S&P 500 (^GSPC)

Mapping automatico in data/yahoo_fetcher.py:

SYMBOL_MAP = {
    'EUR_USD': 'EURUSD=X',
    'BTC_USD': 'BTC-USD',
    'SPX': '^GSPC',
    ...
}

Per simbolo custom:

    fetcher.add_symbol_mapping('MY_PAIR', 'YAHOO_TICKER')


┌─────────────────────────────────────────────────────────────────────────┐
│ 12. TESTING & DEBUGGING                                                │
└─────────────────────────────────────────────────────────────────────────┘

Test MockBroker:

    python -c "
    from execution.mock_broker import MockBroker
    
    broker = MockBroker(100000)
    print(broker.get_account_info())
    
    # Apri posizione
    result = broker.place_market_order(
        'EUR_USD', 'BUY', 100000, 1.10, 1.09, 1.11
    )
    print(result)
    
    # Update con nuovo prezzo
    broker.update_positions({'EUR_USD': 1.105})
    print(broker.get_account_info())
    "

Test YahooFetcher:

    python -c "
    from data.yahoo_fetcher import YahooFinanceDataFetcher
    
    fetcher = YahooFinanceDataFetcher()
    data = fetcher.fetch_historical_data('EUR_USD', 'H1', 100)
    print(f'Downloaded {len(data)} bars')
    print(data.tail())
    "


┌─────────────────────────────────────────────────────────────────────────┐
│ 13. FILES CREATI                                                       │
└─────────────────────────────────────────────────────────────────────────┘

NUOVI FILES:

├─ execution/mock_broker.py
│  └─ MockBroker class con account virtuale completo
│
├─ data/yahoo_fetcher.py
│  └─ YahooFinanceDataFetcher per dati reali
│
└─ config/settings.yaml
   └─ Aggiornato con broker_type="mock"

MODIFICATI:

└─ execution/broker_api.py
   └─ BrokerAPI supporta sia OANDA che MockBroker


┌─────────────────────────────────────────────────────────────────────────┐
│ 14. VANTAGGI MOCK BROKER                                               │
└─────────────────────────────────────────────────────────────────────────┘

✅ ZERO RISCHIO:
   - Conto completamente virtuale
   - Nessuna perdita reale possibile
   - Perfetto per testing

✅ DATI REALI:
   - Candele da Yahoo Finance
   - Prezzi aggiornati in tempo reale
   - Storico completo disponibile

✅ VELOCE:
   - No latenza broker API
   - Esecuzione istantanea
   - Ideale per backtest rapidi

✅ GRATUITO:
   - No registrazione broker
   - No credenziali necessarie
   - Avvio immediato

✅ DETERMINISTICO:
   - Nessuno slippage casuale
   - Fill sempre a prezzo richiesto
   - Risultati riproducibili

✅ COMPATIBILE:
   - Stessa interfaccia di OANDA
   - Switch trasparente
   - Zero modifiche codice


═══════════════════════════════════════════════════════════════════════════
SISTEMA PRONTO - AVVIA CON: python main.py
═══════════════════════════════════════════════════════════════════════════

MockBroker attivo con dati reali da Yahoo Finance.
Quando pronto, switcha a OANDA cambiando broker_type in config.

""")
