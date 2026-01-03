"""
Macro Economic Calendar and News Filter
Downloads and parses economic calendar events and applies filter to trading decisions
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class MacroCalendar:
    """
    Manages economic calendar data from external API
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.macro_config = config.get('macro', {})
        
        self.enabled = self.macro_config.get('enabled', True)
        self.api_url = self.macro_config.get('api_url', '')
        self.api_key = self.macro_config.get('api_key', '')
        self.update_interval = self.macro_config.get('update_interval', 3600)
        
        self.events_cache = []
        self.last_update = None
        
        logger.info(f"MacroCalendar initialized (enabled: {self.enabled})")
    
    def fetch_events(
        self, 
        start_date: datetime = None,
        end_date: datetime = None,
        currencies: List[str] = None
    ) -> List[Dict]:
        """
        Fetch economic events from API
        
        Args:
            start_date: Start datetime (default: now)
            end_date: End datetime (default: +24 hours)
            currencies: List of currency codes (e.g., ['USD', 'EUR'])
        
        Returns:
            List of event dicts
        """
        if not self.enabled or not self.api_url:
            return []
        
        if start_date is None:
            start_date = datetime.utcnow()
        if end_date is None:
            end_date = start_date + timedelta(days=1)
        if currencies is None:
            currencies = ['USD', 'EUR', 'GBP']
        
        try:
            params = {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
                'currencies': ','.join(currencies),
                'api_key': self.api_key
            }
            
            response = requests.get(self.api_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                events = data.get('events', [])
                
                logger.info(f"Fetched {len(events)} macro events")
                return events
            else:
                logger.warning(f"API error {response.status_code}: {response.text}")
                return self._get_dummy_events(start_date, end_date)
        
        except Exception as e:
            logger.error(f"Error fetching macro events: {e}")
            return self._get_dummy_events(start_date, end_date)
    
    def _get_dummy_events(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dict]:
        """
        Generate dummy events for testing when API unavailable
        """
        dummy_events = [
            {
                'event': 'Non-Farm Payrolls',
                'currency': 'USD',
                'impact': 'high',
                'datetime': (start_date + timedelta(hours=8)).isoformat(),
                'forecast': '200K',
                'previous': '180K'
            },
            {
                'event': 'ECB Interest Rate Decision',
                'currency': 'EUR',
                'impact': 'high',
                'datetime': (start_date + timedelta(hours=12)).isoformat(),
                'forecast': '4.50%',
                'previous': '4.50%'
            },
            {
                'event': 'UK GDP',
                'currency': 'GBP',
                'impact': 'medium',
                'datetime': (start_date + timedelta(hours=6)).isoformat(),
                'forecast': '0.2%',
                'previous': '0.1%'
            }
        ]
        
        return [
            event for event in dummy_events 
            if start_date <= datetime.fromisoformat(event['datetime']) <= end_date
        ]
    
    def update_cache(self):
        """Update cached events"""
        now = datetime.utcnow()
        
        # Check if update needed
        if self.last_update and (now - self.last_update).seconds < self.update_interval:
            return
        
        # Fetch events for next 24 hours
        self.events_cache = self.fetch_events(now, now + timedelta(days=1))
        self.last_update = now
        
        logger.debug(f"Updated events cache: {len(self.events_cache)} events")
    
    def get_upcoming_events(
        self, 
        hours_ahead: int = 2,
        min_impact: str = 'medium'
    ) -> List[Dict]:
        """
        Get upcoming high-impact events
        
        Args:
            hours_ahead: Look ahead time window
            min_impact: Minimum impact level ('low', 'medium', 'high')
        
        Returns:
            List of relevant events
        """
        self.update_cache()
        
        now = datetime.utcnow()
        cutoff = now + timedelta(hours=hours_ahead)
        
        impact_levels = {'low': 0, 'medium': 1, 'high': 2}
        min_impact_level = impact_levels.get(min_impact, 1)
        
        relevant_events = []
        
        for event in self.events_cache:
            event_time = datetime.fromisoformat(event['datetime'])
            event_impact = impact_levels.get(event.get('impact', 'low'), 0)
            
            if now <= event_time <= cutoff and event_impact >= min_impact_level:
                relevant_events.append(event)
        
        return relevant_events


class MacroFilter:
    """
    Applies macro news filter to trading decisions
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.macro_config = config.get('macro', {})
        
        self.enabled = self.macro_config.get('enabled', True)
        self.weight = self.macro_config.get('weight', 0.08)
        
        self.impact_levels = self.macro_config.get('impact_levels', {
            'high': 0.9,
            'medium': 0.5,
            'low': 0.1
        })
        
        self.event_buffer = self.macro_config.get('event_buffer', {
            'before_minutes': 30,
            'after_minutes': 15
        })
        
        self.calendar = MacroCalendar(config)
        
        logger.info(f"MacroFilter initialized (weight: {self.weight})")
    
    def get_macro_adjustment(
        self, 
        symbol: str,
        current_time: datetime = None
    ) -> Dict[str, any]:
        """
        Calculate macro-based trading adjustment
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            current_time: Current datetime (default: now)
        
        Returns:
            Dict with adjustment factor and reason
        """
        if not self.enabled:
            return {
                'adjustment_factor': 1.0,
                'reason': 'macro_filter_disabled',
                'affected_by_events': []
            }
        
        if current_time is None:
            current_time = datetime.utcnow()
        
        # Extract currencies from symbol (e.g., EURUSD -> EUR, USD)
        currencies = self._extract_currencies(symbol)
        
        # Get upcoming events
        before_mins = self.event_buffer['before_minutes']
        after_mins = self.event_buffer['after_minutes']
        
        total_window = (before_mins + after_mins) / 60  # Convert to hours
        upcoming_events = self.calendar.get_upcoming_events(
            hours_ahead=total_window,
            min_impact='low'
        )
        
        # Filter events relevant to symbol currencies
        relevant_events = [
            event for event in upcoming_events
            if event.get('currency') in currencies
        ]
        
        if not relevant_events:
            return {
                'adjustment_factor': 1.0,
                'reason': 'no_relevant_events',
                'affected_by_events': []
            }
        
        # Calculate adjustment based on highest impact event
        max_impact_reduction = 0.0
        affecting_events = []
        
        for event in relevant_events:
            event_time = datetime.fromisoformat(event['datetime'])
            time_to_event = (event_time - current_time).total_seconds() / 60
            
            # Check if within buffer window
            if -after_mins <= time_to_event <= before_mins:
                impact = event.get('impact', 'low')
                reduction = self.impact_levels.get(impact, 0.1)
                
                if reduction > max_impact_reduction:
                    max_impact_reduction = reduction
                
                affecting_events.append({
                    'event': event['event'],
                    'impact': impact,
                    'time_to_event_minutes': time_to_event,
                    'currency': event['currency']
                })
        
        # Calculate final adjustment factor
        if max_impact_reduction > 0:
            adjustment_factor = 1.0 - (max_impact_reduction * self.weight)
            adjustment_factor = max(adjustment_factor, 0.1)  # Minimum 10% of normal size
            
            return {
                'adjustment_factor': adjustment_factor,
                'reason': f'high_impact_event_approaching',
                'affected_by_events': affecting_events,
                'max_impact_reduction': max_impact_reduction
            }
        
        return {
            'adjustment_factor': 1.0,
            'reason': 'events_outside_buffer',
            'affected_by_events': []
        }
    
    def _extract_currencies(self, symbol: str) -> List[str]:
        """
        Extract currency codes from symbol
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD', 'XAUUSD')
        
        Returns:
            List of currency codes
        """
        # Handle special cases
        if symbol.startswith('XAU'):  # Gold
            return ['USD']  # Gold mainly affected by USD events
        if symbol.startswith('XAG'):  # Silver
            return ['USD']
        
        # Forex pairs: extract first 3 and last 3 chars
        if len(symbol) >= 6:
            return [symbol[:3], symbol[3:6]]
        
        return []
    
    def should_trade(
        self, 
        symbol: str,
        min_adjustment: float = 0.5
    ) -> bool:
        """
        Determine if trading should proceed based on macro events
        
        Args:
            symbol: Trading symbol
            min_adjustment: Minimum acceptable adjustment factor
        
        Returns:
            True if safe to trade
        """
        adjustment = self.get_macro_adjustment(symbol)
        
        should_trade = adjustment['adjustment_factor'] >= min_adjustment
        
        if not should_trade:
            logger.info(
                f"Trading restricted for {symbol}: "
                f"adjustment={adjustment['adjustment_factor']:.2f}, "
                f"reason={adjustment['reason']}"
            )
        
        return should_trade
    
    def get_all_symbols_adjustments(
        self, 
        symbols: List[str]
    ) -> Dict[str, Dict]:
        """
        Get macro adjustments for all symbols
        
        Returns:
            Dict mapping symbol to adjustment info
        """
        adjustments = {}
        
        for symbol in symbols:
            adjustments[symbol] = self.get_macro_adjustment(symbol)
        
        return adjustments
