"""
Broker API - Supporta OANDA v20 API e MockBroker
Permette switching tra conto reale OANDA e conto finto mock per testing.
"""

import logging
import requests
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os

# Import implementazioni broker
from execution.mock_broker import MockBroker

logger = logging.getLogger(__name__)


class OANDAClient:
    """Client OANDA v20 API con tutte le operazioni supportate."""
    
    # Endpoints
    LIVE_URL = "https://api-fxtrade.oanda.com/v3"
    PRACTICE_URL = "https://api-fxpractice.oanda.com/v3"
    
    def __init__(self, api_key: str, account_id: str, is_live: bool = False):
        """
        Inizializza OANDA client.
        
        Args:
            api_key: OANDA API key
            account_id: Account ID OANDA
            is_live: True per live, False per demo
        """
        self.api_key = api_key
        self.account_id = account_id
        self.base_url = self.LIVE_URL if is_live else self.PRACTICE_URL
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept-Datetime-Format': 'UNIX'
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Tuple[int, Optional[Dict]]:
        """Esegue richiesta HTTP generica."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                response = self.session.get(url, timeout=10)
            elif method == 'POST':
                response = self.session.post(url, json=data, timeout=10)
            elif method == 'PUT':
                response = self.session.put(url, json=data, timeout=10)
            elif method == 'PATCH':
                response = self.session.patch(url, json=data, timeout=10)
            else:
                return 400, None
            
            if response.status_code >= 400:
                try:
                    error_data = response.json()
                    logger.error(f"OANDA API Error {response.status_code}: {error_data.get('errorMessage', response.text)}")
                except:
                    logger.error(f"OANDA API Error {response.status_code}: {response.text}")
                return response.status_code, None
            
            return response.status_code, response.json() if response.text else None
        
        except requests.exceptions.RequestException as e:
            logger.error(f"OANDA Request Error: {e}")
            return 500, None
    
    def get_account_info(self) -> Optional[Dict]:
        """Ottieni informazioni conto."""
        status, data = self._request('GET', f'/accounts/{self.account_id}')
        
        if status == 200 and data:
            account = data.get('account', {})
            return {
                'account_id': account.get('id'),
                'balance': float(account.get('balance', 0)),
                'equity': float(account.get('balance', 0)) + float(account.get('unrealizedPL', 0)),
                'margin_used': float(account.get('marginUsed', 0)),
                'margin_available': float(account.get('marginAvailable', 0)),
                'unrealized_pnl': float(account.get('unrealizedPL', 0)),
                'realized_pnl': float(account.get('financing', 0)),
                'open_positions': account.get('openPositionCount', 0),
                'open_trades': account.get('openTradeCount', 0)
            }
        
        return None
    
    def get_open_positions(self) -> Optional[List[Dict]]:
        """Ottieni tutte le posizioni aperte."""
        status, data = self._request('GET', f'/accounts/{self.account_id}/openPositions')
        
        if status == 200 and data:
            positions = []
            for pos in data.get('positions', []):
                if float(pos.get('long', {}).get('units', 0)) != 0:
                    side_data = pos['long']
                    side = 'BUY'
                elif float(pos.get('short', {}).get('units', 0)) != 0:
                    side_data = pos['short']
                    side = 'SELL'
                else:
                    continue
                
                positions.append({
                    'symbol': pos['instrument'],
                    'side': side,
                    'units': int(side_data['units']),
                    'avg_price': float(side_data['averagePrice']),
                    'unrealized_pnl': float(side_data.get('unrealizedPL', 0))
                })
            
            return positions
        
        return []
    
    def get_open_trades(self) -> Optional[List[Dict]]:
        """Ottieni tutti i trade aperti."""
        status, data = self._request('GET', f'/accounts/{self.account_id}/openTrades')
        
        if status == 200 and data:
            trades = []
            for trade in data.get('trades', []):
                trades.append({
                    'id': trade['id'],
                    'symbol': trade['instrument'],
                    'side': 'BUY' if int(trade['currentUnits']) > 0 else 'SELL',
                    'units': abs(int(trade['currentUnits'])),
                    'entry_price': float(trade['price']),
                    'unrealized_pnl': float(trade.get('unrealizedPL', 0)),
                    'open_time': trade['openTime']
                })
            
            return trades
        
        return []
    
    def get_pricing(self, symbols: List[str]) -> Optional[Dict[str, Dict]]:
        """Ottieni quotazioni bid/ask per simboli."""
        instruments = ','.join(symbols)
        status, data = self._request('GET', f'/accounts/{self.account_id}/pricing?instruments={instruments}')
        
        if status == 200 and data:
            prices = {}
            for price in data.get('prices', []):
                symbol = price['instrument']
                bid = float(price['bids'][0]['price']) if price.get('bids') else 0.0
                ask = float(price['asks'][0]['price']) if price.get('asks') else 0.0
                
                prices[symbol] = {
                    'bid': bid,
                    'ask': ask,
                    'mid': (bid + ask) / 2,
                    'time': price.get('time', '')
                }
            
            return prices
        
        return {}
    
    def place_market_order(
        self,
        symbol: str,
        units: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Tuple[bool, Optional[str], Dict]:
        """Piazza ordine market."""
        order_data = {
            'order': {
                'type': 'MARKET',
                'instrument': symbol,
                'units': str(units)
            }
        }
        
        if stop_loss:
            order_data['order']['stopLossOnFill'] = {'price': f'{stop_loss:.5f}'}
        
        if take_profit:
            order_data['order']['takeProfitOnFill'] = {'price': f'{take_profit:.5f}'}
        
        status, data = self._request('POST', f'/accounts/{self.account_id}/orders', order_data)
        
        if status in [200, 201] and data:
            fill = data.get('orderFillTransaction', {})
            return True, data.get('orderFillTransaction', {}).get('id'), {
                'fill_price': float(fill.get('price', 0)),
                'units_filled': int(fill.get('units', 0)),
                'time': fill.get('time', '')
            }
        
        return False, None, {}
    
    def close_trade(self, trade_id: str) -> Tuple[bool, Dict]:
        """Chiudi un trade specifico."""
        status, data = self._request('PUT', f'/accounts/{self.account_id}/trades/{trade_id}/close', {'units': 'ALL'})
        
        if status in [200, 204]:
            return True, data or {}
        
        return False, {}
    
    def close_all_trades(self) -> Tuple[bool, int]:
        """Chiudi tutti i trade aperti."""
        trades = self.get_open_trades()
        if not trades:
            return True, 0
        
        closed = 0
        for trade in trades:
            success, _ = self.close_trade(trade['id'])
            if success:
                closed += 1
        
        return True, closed
    
    def update_trade_sl_tp(
        self,
        trade_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Tuple[bool, Optional[Dict]]:
        """Modifica SL/TP di un trade."""
        trade_spec = {}
        
        if stop_loss:
            trade_spec['stopLoss'] = {'price': f'{stop_loss:.5f}'}
        
        if take_profit:
            trade_spec['takeProfit'] = {'price': f'{take_profit:.5f}'}
        
        status, data = self._request('PUT', f'/accounts/{self.account_id}/trades/{trade_id}/orders', trade_spec)
        
        if status in [200, 201]:
            return True, data
        
        return False, None


class BrokerAPI:
    """
    Wrapper API broker con supporto multi-broker.
    Supporta:
    - OANDA v20 API (demo/live)
    - MockBroker (testing con dati reali da Yahoo Finance)
    """
    
    def __init__(self, config: dict):
        """
        Inizializza broker API in base a configurazione.
        
        Args:
            config: Dict con broker_type ('oanda' o 'mock'), credenziali, ecc.
        """
        self.config = config
        broker_config = config.get('broker', {})
        self.broker_type = broker_config.get('broker_type', 'mock').lower()
        self.client = None
        
        logger.info(f"Initializing BrokerAPI with type: {self.broker_type}")
        
        if self.broker_type == 'oanda':
            # OANDA v20 API
            api_key = broker_config.get('oanda_api_key') or os.getenv('OANDA_API_KEY')
            account_id = broker_config.get('oanda_account_id') or os.getenv('OANDA_ACCOUNT_ID')
            is_live = broker_config.get('is_live', False)
            
            if api_key and account_id:
                try:
                    self.client = OANDAClient(api_key, account_id, is_live)
                    logger.info(f"OANDA client initialized: {'LIVE' if is_live else 'PRACTICE'}")
                except Exception as e:
                    logger.error(f"Failed to initialize OANDA client: {e}")
                    logger.warning("Falling back to MockBroker")
                    self._init_mock_broker(config)
            else:
                logger.warning("OANDA credentials missing, using MockBroker")
                self._init_mock_broker(config)
        
        elif self.broker_type == 'mock':
            # MockBroker con dati Yahoo Finance
            self._init_mock_broker(config)
        
        else:
            logger.error(f"Unknown broker_type: {self.broker_type}, using MockBroker")
            self._init_mock_broker(config)
    
    def _init_mock_broker(self, config: dict):
        """Inizializza MockBroker."""
        trading_config = config.get('trading', {})
        initial_balance = trading_config.get('initial_capital', 100000.0)
        leverage = trading_config.get('leverage', 30)
        commission = trading_config.get('commission_per_lot', 7.0)
        
        self.client = MockBroker(
            initial_balance=initial_balance,
            leverage=leverage,
            commission_per_lot=commission
        )
        self.broker_type = 'mock'
        logger.info(f"MockBroker initialized: Balance={initial_balance}, Leverage=1:{leverage}")
    
    def connect(self) -> bool:
        """
        Verifica connessione al broker.
        
        Returns:
            True se connesso, False altrimenti
        """
        try:
            account_info = self.get_account_info()
            if account_info:
                logger.info(f"Connected to {self.broker_type.upper()}: Balance={account_info.get('balance', 0):.2f}")
                return True
            return False
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def get_account_info(self) -> Optional[Dict]:
        """Ottieni informazioni conto."""
        if isinstance(self.client, MockBroker):
            return self.client.get_account_info()
        elif isinstance(self.client, OANDAClient):
            return self.client.get_account_info()
        else:
            # Dummy fallback
            return {
                'balance': 100000.0,
                'equity': 100000.0,
                'margin_used': 0.0,
                'margin_available': 100000.0,
                'unrealized_pnl': 0.0,
                'open_trades': 0
            }
    
    def get_open_positions(self) -> List[Dict]:
        """Ottieni posizioni aperte."""
        if isinstance(self.client, MockBroker):
            return self.client.get_open_positions()
        elif isinstance(self.client, OANDAClient):
            return self.client.get_open_positions() or []
        else:
            return []
    
    def get_open_trades(self) -> List[Dict]:
        """Ottieni trade aperti."""
        if isinstance(self.client, MockBroker):
            return self.client.get_open_trades()
        elif isinstance(self.client, OANDAClient):
            return self.client.get_open_trades() or []
        else:
            return []
    
    def get_pricing(self, symbols: List[str]) -> Dict[str, Dict]:
        """Ottieni quotazioni."""
        if isinstance(self.client, OANDAClient):
            return self.client.get_pricing(symbols) or {}
        else:
            # Per MockBroker, prezzi vengono da Yahoo Finance (gestito altrove)
            return {}
    
    def place_market_order(
        self,
        symbol: str,
        units: int,
        current_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict:
        """
        Piazza ordine market.
        
        Args:
            symbol: Simbolo (es. EUR_USD)
            units: Unità (positivo BUY, negativo SELL)
            current_price: Prezzo corrente (richiesto per MockBroker)
            stop_loss: Stop loss price
            take_profit: Take profit price
        
        Returns:
            Dict con risultato
        """
        side = 'BUY' if units > 0 else 'SELL'
        
        if isinstance(self.client, MockBroker):
            if current_price is None:
                logger.error("current_price required for MockBroker orders")
                return {'success': False, 'error': 'current_price missing'}
            
            return self.client.place_market_order(
                symbol, side, abs(units), current_price, stop_loss, take_profit
            )
        
        elif isinstance(self.client, OANDAClient):
            success, order_id, details = self.client.place_market_order(
                symbol, units, stop_loss, take_profit
            )
            
            return {
                'success': success,
                'order_id': order_id,
                **details
            }
        
        else:
            # Dummy
            logger.info(f"DUMMY: Order {symbol} {side} {abs(units)}")
            return {
                'success': True,
                'order_id': 'DUMMY_123',
                'fill_price': 1.1,
                'units_filled': abs(units)
            }
    
    def close_position(self, symbol: str, current_price: Optional[float] = None) -> Dict:
        """
        Chiudi posizione.
        
        Args:
            symbol: Simbolo
            current_price: Prezzo corrente (richiesto per MockBroker)
        
        Returns:
            Dict con risultato
        """
        if isinstance(self.client, MockBroker):
            if current_price is None:
                logger.error("current_price required for MockBroker close")
                return {'success': False, 'error': 'current_price missing'}
            
            return self.client.close_position(symbol, current_price)
        
        elif isinstance(self.client, OANDAClient):
            # Chiudi tutti i trade del simbolo
            trades = self.client.get_open_trades()
            closed = 0
            
            for trade in trades:
                if trade['symbol'] == symbol:
                    success, _ = self.client.close_trade(trade['id'])
                    if success:
                        closed += 1
            
            return {
                'success': True,
                'closed': closed,
                'message': f'Closed {closed} trades'
            }
        
        else:
            return {'success': True, 'message': 'DUMMY closed'}
    
    def close_all_trades(self) -> Dict:
        """Chiudi tutti i trade."""
        if isinstance(self.client, MockBroker):
            # Chiudi tutte le posizioni mock
            positions = self.client.get_open_positions()
            closed = 0
            
            for pos in positions:
                result = self.client.close_position(pos['symbol'], pos['current_price'])
                if result['success']:
                    closed += 1
            
            return {'success': True, 'closed': closed}
        
        elif isinstance(self.client, OANDAClient):
            success, closed = self.client.close_all_trades()
            return {'success': success, 'closed': closed}
        
        else:
            return {'success': True, 'closed': 0}
    
    def modify_position(
        self,
        symbol: str,
        trade_id: Optional[str] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict:
        """Modifica SL/TP di un trade."""
        if isinstance(self.client, MockBroker):
            if trade_id is None:
                # Trova trade_id dal simbolo
                trades = self.client.get_open_trades()
                for trade in trades:
                    if trade['symbol'] == symbol:
                        trade_id = trade['id']
                        break
            
            if trade_id:
                return self.client.update_trade_sl_tp(trade_id, stop_loss, take_profit)
            else:
                return {'success': False, 'error': 'Trade not found'}
        
        elif isinstance(self.client, OANDAClient):
            if trade_id:
                success, data = self.client.update_trade_sl_tp(trade_id, stop_loss, take_profit)
                return {'success': success, 'data': data}
            else:
                return {'success': False, 'error': 'trade_id required'}
        
        else:
            return {'success': True, 'data': {}}
    
    def update_positions(self, current_prices: Dict[str, float]) -> None:
        """
        Aggiorna posizioni con prezzi correnti.
        Importante per MockBroker per calcolare PnL e controllare SL/TP.
        
        Args:
            current_prices: Dict {symbol: price}
        """
        if isinstance(self.client, MockBroker):
            self.client.update_positions(current_prices)
    
    def get_performance_stats(self) -> Dict:
        """Ottieni statistiche performance."""
        if isinstance(self.client, MockBroker):
            return self.client.get_performance_stats()
        else:
            # Per OANDA, calcola da account info
            account_info = self.get_account_info()
            if account_info:
                return {
                    'current_balance': account_info['balance'],
                    'equity': account_info['equity'],
                    'total_pnl': account_info.get('unrealized_pnl', 0),
                    'open_positions': account_info.get('open_trades', 0)
                }
            return {}
