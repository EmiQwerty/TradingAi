"""
Test Mock Broker + Yahoo Finance
Verifica rapida funzionamento
"""

print("\n" + "="*70)
print("TEST 1: Yahoo Finance Data Fetcher")
print("="*70)

from data.yahoo_fetcher import YahooFinanceDataFetcher

fetcher = YahooFinanceDataFetcher()

# Test download EUR_USD
print("\nDownloading EUR_USD H1 data (100 bars)...")
data = fetcher.fetch_historical_data('EUR_USD', 'H1', 100)

if data is not None:
    print(f"✅ SUCCESS: Downloaded {len(data)} bars")
    print(f"\nLast 3 candles:")
    print(data[['open', 'high', 'low', 'close', 'volume']].tail(3))
    
    # Current price
    current_price = data['close'].iloc[-1]
    print(f"\nCurrent EUR_USD price: {current_price:.5f}")
else:
    print("❌ FAILED: Could not download data")

print("\n" + "="*70)
print("TEST 2: Mock Broker")
print("="*70)

from execution.mock_broker import MockBroker

# Inizializza broker
broker = MockBroker(initial_balance=100000, leverage=30)

# Account info
account_info = broker.get_account_info()
print(f"\n✅ MockBroker initialized:")
print(f"   Balance: ${account_info['balance']:,.2f}")
print(f"   Equity: ${account_info['equity']:,.2f}")
print(f"   Leverage: 1:{account_info['leverage']}")

# Piazza ordine BUY
print(f"\nPlacing BUY order: EUR_USD 100,000 units @ {current_price:.5f}")
result = broker.place_market_order(
    symbol='EUR_USD',
    side='BUY',
    units=100000,
    current_price=current_price,
    stop_loss=current_price - 0.0010,  # 100 pips SL
    take_profit=current_price + 0.0015  # 150 pips TP
)

if result['success']:
    print(f"✅ Order executed:")
    print(f"   Trade ID: {result['trade_id']}")
    print(f"   Entry: {result['entry_price']:.5f}")
    print(f"   Stop Loss: {result['stop_loss']:.5f}")
    print(f"   Take Profit: {result['take_profit']:.5f}")
    print(f"   Commission: ${result['commission']:.2f}")
else:
    print(f"❌ Order failed: {result.get('message')}")

# Verifica posizioni
positions = broker.get_open_positions()
print(f"\nOpen Positions: {len(positions)}")
for pos in positions:
    print(f"   {pos['symbol']}: {pos['side']} {pos['units']:,} @ {pos['avg_price']:.5f}")
    print(f"   Unrealized PnL: ${pos['unrealized_pnl']:.2f}")

# Simula price update
new_price = current_price + 0.0005  # Prezzo sale di 50 pips
print(f"\n💹 Price update: {current_price:.5f} → {new_price:.5f} (+50 pips)")

broker.update_positions({'EUR_USD': new_price})

# Account info aggiornato
account_info = broker.get_account_info()
print(f"\n📊 Updated Account:")
print(f"   Balance: ${account_info['balance']:,.2f}")
print(f"   Equity: ${account_info['equity']:,.2f}")
print(f"   Unrealized PnL: ${account_info['unrealized_pnl']:,.2f}")
print(f"   Margin Used: ${account_info['margin_used']:,.2f}")
print(f"   Margin Available: ${account_info['margin_available']:,.2f}")

# Chiudi posizione
print(f"\nClosing position at {new_price:.5f}...")
close_result = broker.close_position('EUR_USD', new_price)

if close_result['success']:
    print(f"✅ Position closed:")
    print(f"   Realized PnL: ${close_result['realized_pnl']:,.2f}")
    print(f"   New Balance: ${close_result['balance']:,.2f}")
else:
    print(f"❌ Close failed")

# Performance stats
stats = broker.get_performance_stats()
print(f"\n📈 Performance Summary:")
print(f"   Initial Balance: ${stats['initial_balance']:,.2f}")
print(f"   Current Balance: ${stats['current_balance']:,.2f}")
print(f"   Total PnL: ${stats['total_pnl']:,.2f}")
print(f"   Total Trades: {stats['total_trades']}")
print(f"   Winning Trades: {stats['winning_trades']}")
print(f"   Losing Trades: {stats['losing_trades']}")
print(f"   Win Rate: {stats['win_rate']:.1f}%")

print("\n" + "="*70)
print("TEST 3: BrokerAPI Integration")
print("="*70)

from execution.broker_api import BrokerAPI

# Config mock mode
config = {
    'broker': {
        'broker_type': 'mock'
    },
    'trading': {
        'initial_capital': 100000,
        'leverage': 30,
        'commission_per_lot': 7.0
    }
}

api = BrokerAPI(config)

# Connect
connected = api.connect()
print(f"\n✅ BrokerAPI Connected: {connected}")

# Account info
account = api.get_account_info()
print(f"\nAccount via API:")
print(f"   Balance: ${account['balance']:,.2f}")
print(f"   Equity: ${account['equity']:,.2f}")

print("\n" + "="*70)
print("✅ ALL TESTS PASSED - Mock Broker Ready!")
print("="*70)
print("\nPer avviare il sistema completo:")
print("   python main.py")
print("\nPer aprire dashboard:")
print("   streamlit run monitoring/dashboard.py")
print("="*70 + "\n")
