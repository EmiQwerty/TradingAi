"""
Mock Broker Implementation - Simula un conto trading con dati reali da Yahoo Finance
Mantiene stato del conto (balance, positions, orders) senza connessione a broker reale.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class MockBroker:
    """
    Broker simulato per testing con account virtuale.
    Simula tutte le operazioni di un broker reale.
    """
    
    def __init__(
        self, 
        initial_balance: float = 100000.0,
        leverage: int = 30,
        commission_per_lot: float = 7.0  # USD per lotto standard
    ):
        """
        Inizializza mock broker con account virtuale.
        
        Args:
            initial_balance: Capitale iniziale in USD
            leverage: Leva finanziaria (1:leverage)
            commission_per_lot: Commissione per lotto standard
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.leverage = leverage
        self.commission_per_lot = commission_per_lot
        
        # Account state
        self.open_positions: Dict[str, Dict] = {}  # symbol -> position info
        self.open_orders: Dict[str, Dict] = {}     # order_id -> order info
        self.trade_history: List[Dict] = []        # Storico trades chiusi
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        
        logger.info(f"MockBroker initialized: Balance={self.balance:.2f} USD, Leverage=1:{self.leverage}")
    
    def get_account_info(self) -> Dict:
        """
        Restituisce informazioni account come OANDA.
        
        Returns:
            Dict con balance, equity, margin, ecc.
        """
        # Calcola margin used
        margin_used = 0.0
        for symbol, pos in self.open_positions.items():
            # Margin = (units * price) / leverage
            margin_used += abs(pos['units']) * pos['entry_price'] / self.leverage
        
        # Calcola unrealized PnL
        unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.open_positions.values())
        
        # Equity = balance + unrealized PnL
        self.equity = self.balance + unrealized_pnl
        
        margin_available = self.equity - margin_used
        
        account_info = {
            'balance': self.balance,
            'equity': self.equity,
            'margin_used': margin_used,
            'margin_available': margin_available,
            'unrealized_pnl': unrealized_pnl,
            'open_trades': len(self.open_positions),
            'leverage': self.leverage,
            'currency': 'USD'
        }
        
        logger.debug(f"Account Info: Balance={self.balance:.2f}, Equity={self.equity:.2f}, Margin={margin_used:.2f}")
        
        return account_info
    
    def get_open_positions(self) -> List[Dict]:
        """
        Restituisce lista posizioni aperte.
        
        Returns:
            Lista di dict con info posizioni
        """
        positions = []
        for symbol, pos in self.open_positions.items():
            positions.append({
                'symbol': symbol,
                'side': pos['side'],
                'units': pos['units'],
                'avg_price': pos['entry_price'],
                'current_price': pos['current_price'],
                'unrealized_pnl': pos['unrealized_pnl'],
                'stop_loss': pos.get('stop_loss'),
                'take_profit': pos.get('take_profit'),
                'open_time': pos['open_time']
            })
        
        return positions
    
    def get_open_trades(self) -> List[Dict]:
        """
        Alias di get_open_positions per compatibilità OANDA.
        """
        trades = []
        for symbol, pos in self.open_positions.items():
            trades.append({
                'id': pos['trade_id'],
                'symbol': symbol,
                'side': pos['side'],
                'units': pos['units'],
                'entry_price': pos['entry_price'],
                'current_price': pos['current_price'],
                'unrealized_pnl': pos['unrealized_pnl'],
                'open_time': pos['open_time']
            })
        
        return trades
    
    def place_market_order(
        self,
        symbol: str,
        side: str,
        units: int,
        current_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict:
        """
        Piazza un ordine market simulato.
        
        Args:
            symbol: Simbolo (es. EUR_USD)
            side: 'BUY' o 'SELL'
            units: Numero di unità (positivo per BUY, negativo per SELL)
            current_price: Prezzo corrente per esecuzione
            stop_loss: Prezzo stop loss (opzionale)
            take_profit: Prezzo take profit (opzionale)
        
        Returns:
            Dict con risultato ordine
        """
        # Normalizza units in base a side
        if side == 'SELL' and units > 0:
            units = -units
        elif side == 'BUY' and units < 0:
            units = abs(units)
        
        # Calcola commissione
        lots = abs(units) / 100000  # Standard lot = 100k units
        commission = lots * self.commission_per_lot
        
        # Calcola margin richiesto
        margin_required = abs(units) * current_price / self.leverage
        
        # Controlla margin disponibile
        account_info = self.get_account_info()
        if margin_required > account_info['margin_available']:
            logger.error(f"Insufficient margin: Required={margin_required:.2f}, Available={account_info['margin_available']:.2f}")
            return {
                'success': False,
                'error': 'INSUFFICIENT_MARGIN',
                'message': f'Margin richiesto {margin_required:.2f} > disponibile {account_info["margin_available"]:.2f}'
            }
        
        # Genera trade ID
        trade_id = str(uuid.uuid4())[:8]
        
        # Crea posizione
        position = {
            'trade_id': trade_id,
            'symbol': symbol,
            'side': side,
            'units': units,
            'entry_price': current_price,
            'current_price': current_price,
            'unrealized_pnl': 0.0,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'commission': commission,
            'open_time': datetime.now().isoformat()
        }
        
        # Sottrai commissione dal balance
        self.balance -= commission
        
        # Aggiungi o aggiorna posizione
        if symbol in self.open_positions:
            # Aggiungi a posizione esistente (average price)
            existing = self.open_positions[symbol]
            total_units = existing['units'] + units
            
            if total_units == 0:
                # Posizione chiusa
                realized_pnl = self._calculate_pnl(existing, current_price)
                self.balance += realized_pnl
                self.total_pnl += realized_pnl
                
                # Aggiorna stats
                self.total_trades += 1
                if realized_pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                # Aggiungi a history
                self.trade_history.append({
                    'symbol': symbol,
                    'side': existing['side'],
                    'units': existing['units'],
                    'entry_price': existing['entry_price'],
                    'exit_price': current_price,
                    'pnl': realized_pnl,
                    'open_time': existing['open_time'],
                    'close_time': datetime.now().isoformat()
                })
                
                # Rimuovi posizione
                del self.open_positions[symbol]
                
                logger.info(f"Position CLOSED: {symbol} PnL={realized_pnl:.2f} USD")
                
                return {
                    'success': True,
                    'trade_id': trade_id,
                    'action': 'CLOSED',
                    'symbol': symbol,
                    'realized_pnl': realized_pnl,
                    'balance': self.balance
                }
            else:
                # Media ponderata entry price
                avg_price = (
                    (existing['units'] * existing['entry_price'] + units * current_price) 
                    / total_units
                )
                existing['units'] = total_units
                existing['entry_price'] = avg_price
                existing['current_price'] = current_price
                existing['commission'] += commission
                
                logger.info(f"Position UPDATED: {symbol} Units={total_units}, AvgPrice={avg_price:.5f}")
        else:
            # Nuova posizione
            self.open_positions[symbol] = position
            logger.info(f"Position OPENED: {symbol} {side} {units} @ {current_price:.5f}, SL={stop_loss}, TP={take_profit}")
        
        return {
            'success': True,
            'trade_id': trade_id,
            'action': 'OPENED',
            'symbol': symbol,
            'side': side,
            'units': units,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'commission': commission,
            'balance': self.balance
        }
    
    def close_trade(self, trade_id: str, current_price: float) -> Dict:
        """
        Chiude un trade specifico.
        
        Args:
            trade_id: ID del trade da chiudere
            current_price: Prezzo corrente per chiusura
        
        Returns:
            Dict con risultato
        """
        # Trova posizione per trade_id
        for symbol, pos in self.open_positions.items():
            if pos['trade_id'] == trade_id:
                # Chiudi posizione
                realized_pnl = self._calculate_pnl(pos, current_price)
                self.balance += realized_pnl
                self.total_pnl += realized_pnl
                
                # Stats
                self.total_trades += 1
                if realized_pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                # History
                self.trade_history.append({
                    'symbol': symbol,
                    'side': pos['side'],
                    'units': pos['units'],
                    'entry_price': pos['entry_price'],
                    'exit_price': current_price,
                    'pnl': realized_pnl,
                    'open_time': pos['open_time'],
                    'close_time': datetime.now().isoformat()
                })
                
                # Rimuovi
                del self.open_positions[symbol]
                
                logger.info(f"Trade CLOSED: {trade_id} {symbol} PnL={realized_pnl:.2f} USD")
                
                return {
                    'success': True,
                    'trade_id': trade_id,
                    'symbol': symbol,
                    'realized_pnl': realized_pnl,
                    'balance': self.balance
                }
        
        logger.error(f"Trade not found: {trade_id}")
        return {
            'success': False,
            'error': 'TRADE_NOT_FOUND',
            'message': f'Trade ID {trade_id} non trovato'
        }
    
    def close_position(self, symbol: str, current_price: float) -> Dict:
        """
        Chiude posizione per simbolo.
        
        Args:
            symbol: Simbolo da chiudere
            current_price: Prezzo corrente
        
        Returns:
            Dict con risultato
        """
        if symbol not in self.open_positions:
            return {
                'success': False,
                'error': 'NO_POSITION',
                'message': f'Nessuna posizione aperta su {symbol}'
            }
        
        pos = self.open_positions[symbol]
        return self.close_trade(pos['trade_id'], current_price)
    
    def update_trade_sl_tp(
        self,
        trade_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict:
        """
        Modifica SL/TP di un trade.
        
        Args:
            trade_id: ID del trade
            stop_loss: Nuovo stop loss
            take_profit: Nuovo take profit
        
        Returns:
            Dict con risultato
        """
        for symbol, pos in self.open_positions.items():
            if pos['trade_id'] == trade_id:
                if stop_loss is not None:
                    pos['stop_loss'] = stop_loss
                if take_profit is not None:
                    pos['take_profit'] = take_profit
                
                logger.info(f"Trade UPDATED: {trade_id} SL={stop_loss}, TP={take_profit}")
                
                return {
                    'success': True,
                    'trade_id': trade_id,
                    'symbol': symbol,
                    'stop_loss': pos['stop_loss'],
                    'take_profit': pos['take_profit']
                }
        
        return {
            'success': False,
            'error': 'TRADE_NOT_FOUND'
        }
    
    def update_positions(self, current_prices: Dict[str, float]) -> None:
        """
        Aggiorna prezzi correnti e unrealized PnL delle posizioni.
        Controlla anche SL/TP hit.
        
        Args:
            current_prices: Dict {symbol: current_price}
        """
        positions_to_close = []
        
        for symbol, pos in self.open_positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                pos['current_price'] = current_price
                
                # Calcola unrealized PnL
                pos['unrealized_pnl'] = self._calculate_pnl(pos, current_price)
                
                # Controlla SL/TP hit
                if pos['stop_loss'] is not None:
                    if (pos['side'] == 'BUY' and current_price <= pos['stop_loss']) or \
                       (pos['side'] == 'SELL' and current_price >= pos['stop_loss']):
                        logger.info(f"STOP LOSS HIT: {symbol} @ {current_price:.5f}")
                        positions_to_close.append((symbol, pos['stop_loss'], 'SL'))
                
                if pos['take_profit'] is not None:
                    if (pos['side'] == 'BUY' and current_price >= pos['take_profit']) or \
                       (pos['side'] == 'SELL' and current_price <= pos['take_profit']):
                        logger.info(f"TAKE PROFIT HIT: {symbol} @ {current_price:.5f}")
                        positions_to_close.append((symbol, pos['take_profit'], 'TP'))
        
        # Chiudi posizioni con SL/TP hit
        for symbol, exit_price, reason in positions_to_close:
            self.close_position(symbol, exit_price)
    
    def _calculate_pnl(self, position: Dict, exit_price: float) -> float:
        """
        Calcola PnL di una posizione.
        
        Args:
            position: Dict posizione
            exit_price: Prezzo di uscita
        
        Returns:
            PnL in USD
        """
        # PnL = (exit_price - entry_price) * units (per BUY)
        # PnL = (entry_price - exit_price) * units (per SELL, units è negativo)
        
        if position['side'] == 'BUY':
            pnl = (exit_price - position['entry_price']) * position['units']
        else:
            pnl = (position['entry_price'] - exit_price) * abs(position['units'])
        
        # Sottrai commissione già pagata all'apertura
        # (commissione chiusura è tipicamente inclusa nello spread)
        
        return pnl
    
    def get_performance_stats(self) -> Dict:
        """
        Restituisce statistiche performance.
        
        Returns:
            Dict con stats
        """
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0
        
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.balance,
            'equity': self.equity,
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'open_positions': len(self.open_positions)
        }
    
    def reset(self) -> None:
        """
        Reset completo del conto (per testing).
        """
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.open_positions.clear()
        self.open_orders.clear()
        self.trade_history.clear()
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        
        logger.info("MockBroker RESET: Account restored to initial state")
