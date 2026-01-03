"""
Streamlit Dashboard
Real-time monitoring dashboard for trading system
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)


class TradingDashboard:
    """
    Real-time Streamlit dashboard for monitoring trading system
    Read-only view of system state
    """
    
    def __init__(self, state_store, metrics_calculator, event_logger):
        """
        Initialize dashboard
        
        Args:
            state_store: StateStore instance
            metrics_calculator: MetricsCalculator instance
            event_logger: EventLogger instance
        """
        self.state_store = state_store
        self.metrics = metrics_calculator
        self.events = event_logger
        
        logger.info("TradingDashboard initialized")
    
    def run(self, port: int = 8501):
        """
        Run Streamlit dashboard
        
        Args:
            port: Port number for dashboard
        """
        st.set_page_config(
            page_title="Trading System Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Title
        st.title("🤖 Multi-Timeframe Trading System")
        st.markdown("---")
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 60, 5)
        
        # Manual refresh button
        if st.sidebar.button("Refresh Now"):
            st.rerun()
        
        # System status
        self._render_system_status()
        
        # Account overview
        self._render_account_overview()
        
        st.markdown("---")
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Positions", 
            "📈 Performance", 
            "🎯 Signals", 
            "🌍 Market Overview",
            "⚠️ Events & Logs"
        ])
        
        with tab1:
            self._render_positions_tab()
        
        with tab2:
            self._render_performance_tab()
        
        with tab3:
            self._render_signals_tab()
        
        with tab4:
            self._render_market_overview_tab()
        
        with tab5:
            self._render_events_tab()
        
        # Auto refresh
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
    
    def _render_system_status(self):
        """Render system status header"""
        col1, col2, col3, col4 = st.columns(4)
        
        summary = self.state_store.get_state_summary()
        health = summary.get('system_health', 'UNKNOWN')
        
        # Health indicator
        health_colors = {
            'OK': '🟢',
            'WARNING': '🟡',
            'ERROR': '🔴',
            'CRITICAL': '🔴'
        }
        
        with col1:
            st.metric(
                "System Health",
                f"{health_colors.get(health, '⚪')} {health}",
                delta=None
            )
        
        with col2:
            trading_status = "Enabled" if summary['trading_enabled'] else "Disabled"
            st.metric("Trading Status", trading_status)
        
        with col3:
            st.metric("Symbols Tracked", summary['symbols_tracked'])
        
        with col4:
            last_update = summary.get('timestamp', 'N/A')
            st.metric("Last Update", last_update[:19] if isinstance(last_update, str) else 'N/A')
    
    def _render_account_overview(self):
        """Render account metrics"""
        st.subheader("💰 Account Overview")
        
        account = self.state_store.get_account_info()
        risk = self.state_store.get_risk_metrics()
        summary = self.state_store.get_state_summary()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        balance = account.get('balance', 0)
        equity = account.get('equity', 0)
        
        with col1:
            st.metric("Balance", f"${balance:,.2f}")
        
        with col2:
            st.metric("Equity", f"${equity:,.2f}")
        
        with col3:
            pnl = summary.get('total_position_pnl', 0)
            st.metric("Open P&L", f"${pnl:,.2f}", delta=f"{pnl:+.2f}")
        
        with col4:
            daily_pnl = risk.get('daily', {}).get('pnl', 0)
            st.metric("Daily P&L", f"${daily_pnl:,.2f}", delta=f"{daily_pnl:+.2f}")
        
        with col5:
            drawdown = risk.get('drawdown', {}).get('current_drawdown', 0) * 100
            st.metric("Drawdown", f"{drawdown:.2f}%")
    
    def _render_positions_tab(self):
        """Render positions tab"""
        st.subheader("Open Positions")
        
        positions = self.state_store.get_positions()
        
        if not positions:
            st.info("No open positions")
            return
        
        # Create positions table
        pos_data = []
        for pos in positions:
            pos_data.append({
                'Symbol': pos.symbol,
                'Direction': pos.direction.upper(),
                'Size': f"{pos.size:.2f}",
                'Entry': f"{pos.entry_price:.5f}",
                'Current': f"{pos.current_price:.5f}",
                'SL': f"{pos.stop_loss:.5f}" if pos.stop_loss else 'N/A',
                'TP': f"{pos.take_profit:.5f}" if pos.take_profit else 'N/A',
                'P&L': f"${pos.pnl:.2f}",
                'P&L %': f"{pos.pnl_pct:+.2f}%",
                'Duration': str(datetime.utcnow() - pos.entry_time).split('.')[0]
            })
        
        df = pd.DataFrame(pos_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    def _render_performance_tab(self):
        """Render performance metrics tab"""
        st.subheader("Performance Metrics")
        
        # Calculate metrics
        all_metrics = self.metrics.calculate_metrics()
        daily_metrics = self.metrics.get_daily_performance()
        weekly_metrics = self.metrics.get_weekly_performance()
        
        # Overall performance
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📊 All Time")
            st.metric("Total Trades", all_metrics['summary']['total_trades'])
            st.metric("Win Rate", f"{all_metrics['summary']['win_rate']*100:.1f}%")
            st.metric("Profit Factor", f"{all_metrics['ratios']['profit_factor']:.2f}")
        
        with col2:
            st.markdown("#### 📅 Today")
            st.metric("Trades", daily_metrics['summary']['total_trades'])
            st.metric("Win Rate", f"{daily_metrics['summary']['win_rate']*100:.1f}%")
            st.metric("P&L", f"${daily_metrics['summary']['total_pnl']:.2f}")
        
        with col3:
            st.markdown("#### 📆 This Week")
            st.metric("Trades", weekly_metrics['summary']['total_trades'])
            st.metric("Win Rate", f"{weekly_metrics['summary']['win_rate']*100:.1f}%")
            st.metric("P&L", f"${weekly_metrics['summary']['total_pnl']:.2f}")
        
        st.markdown("---")
        
        # Detailed metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Win/Loss Analysis")
            st.metric("Average Win", f"${all_metrics['wins_losses']['avg_win']:.2f}")
            st.metric("Average Loss", f"${all_metrics['wins_losses']['avg_loss']:.2f}")
            st.metric("Largest Win", f"${all_metrics['wins_losses']['largest_win']:.2f}")
            st.metric("Largest Loss", f"${all_metrics['wins_losses']['largest_loss']:.2f}")
        
        with col2:
            st.markdown("#### Risk Metrics")
            st.metric("Sharpe Ratio", f"{all_metrics['ratios']['sharpe_ratio']:.2f}")
            st.metric("Sortino Ratio", f"{all_metrics['ratios']['sortino_ratio']:.2f}")
            st.metric("Expectancy", f"${all_metrics['ratios']['expectancy']:.2f}")
            st.metric("Max Drawdown", f"{all_metrics['drawdown']['max_drawdown_pct']:.2f}%")
    
    def _render_signals_tab(self):
        """Render recent signals tab"""
        st.subheader("Recent Trading Signals")
        
        signals = self.events.get_recent_signals(20)
        
        if not signals:
            st.info("No recent signals")
            return
        
        signal_data = []
        for sig in reversed(signals):
            signal_data.append({
                'Time': sig['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'Symbol': sig['symbol'],
                'TF': sig['timeframe'],
                'Type': sig['signal_type'].upper(),
                'Direction': sig['direction'].upper(),
                'Confidence': f"{sig['confidence']:.2f}"
            })
        
        df = pd.DataFrame(signal_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    def _render_market_overview_tab(self):
        """Render market overview tab"""
        st.subheader("Market Overview")
        
        overview = self.state_store.get_market_overview()
        
        if not overview:
            st.info("No market data available")
            return
        
        market_data = []
        for symbol, data in overview.items():
            market_data.append({
                'Symbol': symbol,
                'Price': f"{data.get('price', 0):.5f}",
                'ML Regime': data.get('ml_regime', 'N/A'),
                'Direction': data.get('direction', 'neutral').upper(),
                'Confidence': f"{data.get('confidence', 0):.2f}",
                'Volatility': data.get('volatility', 'N/A')
            })
        
        df = pd.DataFrame(market_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    def _render_events_tab(self):
        """Render events and logs tab"""
        st.subheader("System Events & Logs")
        
        # Event statistics
        stats = self.events.get_event_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Events", stats['total_events'])
        with col2:
            st.metric("Signals", stats['signals'])
        with col3:
            st.metric("Trades", stats['trades'])
        with col4:
            st.metric("Errors", stats['errors'])
        
        st.markdown("---")
        
        # Filter events
        event_type = st.selectbox(
            "Filter by Type",
            ['All', 'Signal', 'Trade', 'Error', 'System']
        )
        
        # Get recent events
        if event_type == 'All':
            events = self.events.get_recent_events(50)
        else:
            events = self.events.get_recent_events(50, event_type=event_type.lower())
        
        if not events:
            st.info("No events to display")
            return
        
        # Display events
        for event in reversed(events):
            timestamp = event['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            event_type_str = event['type'].upper()
            
            if event['type'] == 'error':
                st.error(f"[{timestamp}] {event_type_str}: {event.get('message', '')}")
            elif event['type'] == 'warning':
                st.warning(f"[{timestamp}] {event_type_str}: {event.get('message', '')}")
            else:
                st.info(f"[{timestamp}] {event_type_str}: {event.get('message', '')}")


def run_dashboard(state_store, metrics_calculator, event_logger, port: int = 8501):
    """
    Run dashboard in standalone mode
    
    Args:
        state_store: StateStore instance
        metrics_calculator: MetricsCalculator instance
        event_logger: EventLogger instance
        port: Port number
    """
    dashboard = TradingDashboard(state_store, metrics_calculator, event_logger)
    dashboard.run(port)
