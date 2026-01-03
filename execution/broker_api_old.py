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
        status, data = self._request('GET', f'/accounts/{self.account_id}/positions')
        
        if status == 200 and data:
            positions = []
            
            for pos in data.get('positions', []):
                long_units = float(pos.get('long', {}).get('units', 0))
                short_units = float(pos.get('short', {}).get('units', 0))
                
                if long_units != 0:
                    positions.append({
                        'symbol': pos['instrument'],
                        'side': 'long',
                        'units': long_units,
                        'avg_price': float(pos['long'].get('averagePrice', 0)),
                        'unrealized_pnl': float(pos['long'].get('unrealizedPL', 0))
                    })
                
                if short_units != 0:
                    positions.append({
                        'symbol': pos['instrument'],
                        'side': 'short',
                        'units': abs(short_units),
                        'avg_price': float(pos['short'].get('averagePrice', 0)),
                        'unrealized_pnl': float(pos['short'].get('unrealizedPL', 0))
                    })
            
            return positions
        
        return None
    
    def get_open_trades(self) -> Optional[List[Dict]]:
        """Ottieni tutti i trade aperti."""
        status, data = self._request('GET', f'/accounts/{self.account_id}/openTrades')
        
        if status == 200 and data:
            trades = []
            
            for trade in data.get('trades', []):
                trades.append({
                    'id': trade['id'],
                    'symbol': trade['instrument'],
                    'side': 'long' if float(trade['initialUnits']) > 0 else 'short',
                    'units': abs(float(trade['initialUnits'])),
                    'entry_price': float(trade['price']),
                    'unrealized_pnl': float(trade['unrealizedPL']),
                    'open_time': trade['openTime']
                })
            
            return trades
        
        return None
    
    def get_pricing(self, symbols: List[str]) -> Optional[Dict[str, Dict]]:
        """Ottieni prezzi correnti per lista strumenti."""
        instruments = ','.join(symbols)
        status, data = self._request('GET', f'/accounts/{self.account_id}/pricing?instruments={instruments}')
        
        if status == 200 and data:
            pricing = {}
            
            for price in data.get('prices', []):
                instrument = price['instrument']
                bids = price.get('bids', [])
                asks = price.get('asks', [])
                
                if bids and asks:
                    bid_price = float(bids[0]['price'])
                    ask_price = float(asks[0]['price'])
                    
                    pricing[instrument] = {
                        'bid': bid_price,
                        'ask': ask_price,
                        'mid': (bid_price + ask_price) / 2,
                        'time': price['time']
                    }
            
            return pricing
        
        return None
    
    def place_market_order(self, symbol: str, units: int, 
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Piazza ordine market.
        
        Args:
            symbol: Strumento (es. EUR_USD)
            units: Quantità (positivo=buy, negativo=sell)
            stop_loss: SL price
            take_profit: TP price
        
        Returns:
            (success, order_id, details)
        """
        order = {
            'type': 'MARKET',
            'instrument': symbol,
            'units': int(units),
            'timeInForce': 'FOK'
        }
        
        if stop_loss:
            order['stopLossOnFill'] = {
                'price': str(round(stop_loss, 5))
            }
        
        if take_profit:
            order['takeProfitOnFill'] = {
                'price': str(round(take_profit, 5))
            }
        
        payload = {'order': order}
        status, data = self._request('POST', f'/accounts/{self.account_id}/orders', payload)
        
        if status in [200, 201] and data:
            fill_tx = data.get('orderFillTransaction', {})
            order_id = fill_tx.get('orderID', '')
            fill_price = float(fill_tx.get('price', 0))
            units_filled = int(fill_tx.get('units', 0))
            
            logger.info(f"Market order: {symbol} {units} @ {fill_price}")
            
            return True, order_id, {
                'order_id': order_id,
                'fill_price': fill_price,
                'units_filled': units_filled
            }
        else:
            logger.error(f"Market order failed: {data}")
            return False, None, {'error': 'Market order failed'}
    
    def close_trade(self, trade_id: str) -> Tuple[bool, Optional[Dict]]:
        """Chiudi un trade specifico."""
        status, data = self._request(
            'PUT',
            f'/accounts/{self.account_id}/trades/{trade_id}/close',
            {'units': 'ALL'}
        )
        
        if status in [200, 204]:
            logger.info(f"Trade {trade_id} closed")
            return True, data
        else:
            logger.error(f"Close trade failed: {data}")
            return False, None
    
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
        
        logger.info(f"Closed {closed}/{len(trades)} trades")
        return closed == len(trades), closed
    
    def update_trade_sl_tp(self, trade_id: str, stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None) -> Tuple[bool, Optional[Dict]]:
        """Aggiorna SL/TP di un trade."""
        data = {}
        
        if stop_loss is not None:
            data['stopLossOnFill'] = {
                'price': str(round(stop_loss, 5))
            }
        
        if take_profit is not None:
            data['takeProfitOnFill'] = {
                'price': str(round(take_profit, 5))
            }
        
        if not data:
            return True, None
        
        status, response = self._request(
            'PUT',
            f'/accounts/{self.account_id}/trades/{trade_id}/orders',
            data
        )
        
        if status in [200, 201]:
            logger.info(f"Trade {trade_id} SL/TP updated")
            return True, response
        else:
            logger.error(f"Update SL/TP failed: {response}")
            return False, None


class BrokerAPI:
class BrokerAPI:
    """
    Interfaccia di alto livello per OANDA.
    Wrapper che gestisce tutte le operazioni di trading.
    """
    
    def __init__(self, config: Dict):
        """
        Inizializza API broker.
        
        Args:
            config: Configurazione con oanda_api_key, oanda_account_id, is_live
        """
        self.config = config
        broker_config = config.get('broker', {})
        
        # Ottieni credenziali da config o environment
        api_key = broker_config.get('oanda_api_key') or os.getenv('OANDA_API_KEY', '')
        account_id = broker_config.get('oanda_account_id') or os.getenv('OANDA_ACCOUNT_ID', '')
        is_live = broker_config.get('is_live', False)
        
        if not api_key or not account_id:
            logger.warning("OANDA credentials not provided, using dummy mode")
            self.client = None
            self.connected = False
        else:
            self.client = OANDAClient(api_key, account_id, is_live)
            self.connected = False
        
        logger.info("BrokerAPI initialized (OANDA)")
    
    def connect(self) -> bool:
        """Verifica connessione."""
        if not self.client:
            logger.warning("No OANDA credentials, using dummy account")
            return True
        
        account_info = self.client.get_account_info()
        if account_info:
            self.connected = True
            logger.info(f"Connected to OANDA: Balance={account_info['balance']:.2f}")
            return True
        
        logger.error("Failed to connect to OANDA")
        return False
    
    def get_account_info(self) -> Optional[Dict]:
        """Ottieni info conto."""
        if not self.client:
            return {
                'account_id': 'DEMO_ACCOUNT',
                'balance': 100000.0,
                'equity': 100000.0,
                'margin_used': 0.0,
                'margin_available': 100000.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'open_positions': 0,
                'open_trades': 0
            }
        
        return self.client.get_account_info()
    
    def get_open_positions(self) -> Optional[List[Dict]]:
        """Ottieni posizioni aperte."""
        if not self.client:
            return []
        
        return self.client.get_open_positions()
    
    def get_open_trades(self) -> Optional[List[Dict]]:
        """Ottieni trade aperti."""
        if not self.client:
            return []
        
        return self.client.get_open_trades()
    
    def get_pending_orders(self) -> Optional[List[Dict]]:
        """Ottieni ordini pending."""
        if not self.client:
            return []
        
        status, data = self.client._request('GET', f'/accounts/{self.client.account_id}/orders')
        
        if status == 200 and data:
            orders = []
            for order in data.get('orders', []):
                if order.get('state') == 'PENDING':
                    orders.append({
                        'id': order['id'],
                        'symbol': order['instrument'],
                        'type': order['type'],
                        'units': int(order['units']),
                        'price': float(order.get('price', 0))
                    })
            return orders
        
        return []
    
    def get_pricing(self, symbols: List[str]) -> Optional[Dict[str, Dict]]:
        """Ottieni quotazioni."""
        if not self.client:
            # Dummy pricing
            return {s: {'bid': 1.1, 'ask': 1.1001, 'mid': 1.10005} for s in symbols}
        
        return self.client.get_pricing(symbols)
    
    def place_market_order(self, symbol: str, units: int,
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None) -> Dict:
        """Piazza ordine market."""
        if not self.client:
            logger.info(f"DUMMY: Market order {symbol} {units} SL={stop_loss} TP={take_profit}")
            return {
                'success': True,
                'order_id': 'DUMMY_ORDER_123',
                'fill_price': 1.1,
                'units_filled': units
            }
        
        success, order_id, details = self.client.place_market_order(
            symbol, units, stop_loss, take_profit
        )
        
        return {
            'success': success,
            'order_id': order_id,
            **details
        }
    
    def close_position(self, symbol: str) -> Dict:
        """Chiudi posizione (tutti i trade del simbolo)."""
        if not self.client:
            return {'success': True, 'message': 'DUMMY: Closed'}
        
        trades = self.client.get_open_trades()
        if not trades:
            return {'success': True, 'message': 'No trades to close'}
        
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
    
    def close_all_trades(self) -> Dict:
        """Chiudi tutti i trade."""
        if not self.client:
            return {'success': True, 'closed': 0}
        
        success, closed = self.client.close_all_trades()
        return {
            'success': success,
            'closed': closed
        }
    
    def modify_position(self, symbol: str, trade_id: str,
                       stop_loss: Optional[float] = None,
                       take_profit: Optional[float] = None) -> Dict:
        """Modifica SL/TP di un trade."""
        if not self.client:
            return {'success': True, 'data': {}}
        
        success, data = self.client.update_trade_sl_tp(trade_id, stop_loss, take_profit)
        
        return {
            'success': success,
            'data': data
        }
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancella ordine pending."""
        if not self.client:
            return {'success': True, 'data': {}}
        
        status, data = self.client._request(
            'PUT',
            f'/accounts/{self.client.account_id}/orders/{order_id}/cancel'
        )
        
        return {
            'success': status in [200, 204],
            'data': data
        }
