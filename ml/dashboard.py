"""
ML Trading Dashboard - Streamlit web interface for training management and monitoring
Run with: streamlit run ml/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
import threading
import logging

from ml.training_manager import TrainingManager, TrainingStatus
from ml.pipeline import MLPipeline
from data.yahoo_fetcher import YahooFinanceDataFetcher
from data.binance_fetcher import BinanceDataFetcher

logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="ML Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 14px;
        opacity: 0.9;
    }
    .status-running {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .status-failed {
        background-color: #f44336;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .status-completed {
        background-color: #2196F3;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'training_manager' not in st.session_state:
    st.session_state.training_manager = TrainingManager()
    st.session_state.progress_placeholder = None
    st.session_state.metrics_placeholder = None

if 'training_log' not in st.session_state:
    st.session_state.training_log = []

if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None

if 'metrics' not in st.session_state:
    st.session_state.metrics = {}

if 'last_error' not in st.session_state:
    st.session_state.last_error = None

if 'last_status' not in st.session_state:
    st.session_state.last_status = "Ready to train"


def progress_callback(progress):
    """Callback for progress updates"""
    try:
        if hasattr(st, 'session_state') and 'training_log' in st.session_state:
            st.session_state.training_log.append({
                'timestamp': datetime.now().isoformat(),
                'status': progress.status.value,
                'message': progress.message,
                'percentage': progress.percentage
            })
            st.session_state.last_status = progress.message
    except Exception as e:
        # Callback eseguito in thread separato, potrebbe non avere accesso a session_state
        logger.debug(f"Progress callback skipped: {e}")


def metrics_callback(metrics):
    """Callback for metrics updates"""
    try:
        if hasattr(st, 'session_state') and hasattr(st.session_state, '__contains__'):
            st.session_state.metrics = metrics.to_dict()
    except Exception as e:
        logger.debug(f"Metrics callback skipped: {e}")


def error_callback(error: str, details: str):
    """Callback for error notifications"""
    try:
        st.session_state.last_error = f"{error}: {details}"
        logger.error(f"Training error: {error} - {details}")
    except Exception as e:
        pass


# Header
st.title("🚀 ML Trading Dashboard")
st.markdown("Monitor and manage ML model training in real-time")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    symbol = st.text_input("Symbol", value="EUR_USD", help="Trading symbol (e.g., EUR_USD)")
    
    st.markdown("### 🎯 Strategy Configuration")
    strategy_mode = st.selectbox(
        "Strategy Mode",
        ["WEIGHTED", "VOTING", "ALL", "SINGLE"],
        help="WEIGHTED: combine with confidence weights | VOTING: require multiple strategies to agree | ALL: use all signals | SINGLE: use one strategy"
    )
    
    if strategy_mode == "SINGLE":
        selected_strategy = st.radio(
            "Select Strategy",
            ["RSI", "Breakout", "TrendFollowing"]
        )
        selected_strategies = [selected_strategy]
    else:
        selected_strategies = st.multiselect(
            "Select Strategies",
            ["RSI", "Breakout", "TrendFollowing"],
            default=["RSI", "Breakout", "TrendFollowing"]
        )
        
        if not selected_strategies:
            st.warning("⚠️ Select at least one strategy")
    
    if strategy_mode == "VOTING":
        min_votes = st.slider("Minimum Votes", 2, 3, 2, help="Minimum strategies that must agree")
    else:
        min_votes = 1
    
    st.markdown("---")
    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, help="Proportion of test set")
    cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5, help="Number of CV folds")
    
    st.markdown("---")
    st.subheader("📁 Data Source")
    data_source = st.radio(
        "Select data source", 
        ["Yahoo Finance", "Binance", "Upload CSV", "Sample Data"]
    )
    
    data_file = None
    historical_data = None
    
    if data_source == "Yahoo Finance":
        st.info("📊 Yahoo Finance - Forex, Stocks, Indices, Commodities")
        
        # Inizializza fetcher
        if 'yahoo_fetcher' not in st.session_state:
            st.session_state.yahoo_fetcher = YahooFinanceDataFetcher()
        
        # Input simbolo
        yahoo_symbol = st.text_input(
            "Symbol", 
            value="EURUSD=X",
            help="Examples: EURUSD=X (forex), BTC-USD (crypto), ^GSPC (S&P500)"
        )
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=700),  # 700 giorni per sicurezza
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now()
            )
        
        # Timeframe
        timeframe = st.selectbox(
            "Timeframe",
            options=['M5', 'M15', 'M30', 'H1', 'H4', 'D1'],
            index=3  # Default H1
        )
        
        # Validazione date range
        days_diff = (end_date - start_date).days
        max_days = 729 if timeframe in ['M5', 'M15', 'M30', 'H1'] else 3650  # 729 per sicurezza (bisestili)
        
        if days_diff > max_days:
            st.warning(f"⚠️ Yahoo Finance limits {timeframe} data to ~{max_days} days. Adjust date range or use D1 timeframe for longer history.")
        
        # Bottone download
        if st.button("📥 Download Data from Yahoo", key="yahoo_download"):
            # Validate date range
            days_diff = (end_date - start_date).days
            max_days = 729 if timeframe in ['M5', 'M15', 'M30', 'H1'] else 3650
            
            if days_diff > max_days:
                st.error(f"❌ Date range too large ({days_diff} days)! {timeframe} supports max {max_days} days. Please reduce range or use D1 timeframe.")
            else:
                with st.spinner(f"Downloading {yahoo_symbol}..."):
                    try:
                        # Calcola lookback bars approssimativo
                        if timeframe == 'H1':
                            lookback_bars = days_diff * 24
                        elif timeframe == 'H4':
                            lookback_bars = days_diff * 6
                        elif timeframe == 'D1':
                            lookback_bars = days_diff
                        elif timeframe == 'M30':
                            lookback_bars = days_diff * 48
                        elif timeframe == 'M15':
                            lookback_bars = days_diff * 96
                        else:  # M5
                            lookback_bars = days_diff * 288
                        
                        lookback_bars = min(lookback_bars, 5000)  # Limite massimo
                        
                        # Download tramite yfinance direttamente
                        import yfinance as yf
                        ticker = yf.Ticker(yahoo_symbol)
                        historical_data = ticker.history(
                            start=start_date,
                            end=end_date,
                            interval=st.session_state.yahoo_fetcher.TIMEFRAME_MAP.get(timeframe, '1h'),
                            auto_adjust=True
                        )
                        
                        if not historical_data.empty:
                            # Standardizza colonne
                            historical_data = historical_data.rename(columns={
                                'Open': 'open',
                                'High': 'high',
                                'Low': 'low',
                                'Close': 'close',
                                'Volume': 'volume'
                            })
                            
                            st.session_state.historical_data = historical_data
                            st.success(f"✅ Downloaded {len(historical_data)} bars from {historical_data.index[0].date()} to {historical_data.index[-1].date()}")
                            
                            # Mostra preview
                            st.dataframe(historical_data.head(), use_container_width=True)
                        else:
                            st.error("❌ No data received. Check symbol and date range.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        # Usa dati scaricati se disponibili
        if 'historical_data' in st.session_state:
            historical_data = st.session_state.historical_data
    
    elif data_source == "Binance":
        st.info("₿ Binance - Cryptocurrency spot trading")
        
        # Inizializza fetcher
        if 'binance_fetcher' not in st.session_state:
            st.session_state.binance_fetcher = BinanceDataFetcher()
        
        # Input simbolo
        binance_symbol = st.text_input(
            "Symbol",
            value="BTCUSDT",
            help="Examples: BTCUSDT, ETHUSDT, BNBUSDT"
        )
        
        # Numero bars
        lookback_bars = st.number_input(
            "Number of bars",
            min_value=100,
            max_value=1000,
            value=500,
            help="Max 1000 bars (Binance API limit)"
        )
        
        # Timeframe
        timeframe = st.selectbox(
            "Timeframe",
            options=['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'],
            index=4,  # Default H1
            key="binance_timeframe"
        )
        
        # Bottone download
        if st.button("📥 Download Data from Binance", key="binance_download"):
            with st.spinner(f"Downloading {binance_symbol}..."):
                try:
                    historical_data = st.session_state.binance_fetcher.fetch_historical_data(
                        binance_symbol,
                        timeframe,
                        lookback_bars,
                        use_cache=False
                    )
                    
                    if historical_data is not None:
                        st.session_state.historical_data = historical_data
                        st.success(f"✅ Downloaded {len(historical_data)} bars from {historical_data.index[0]} to {historical_data.index[-1]}")
                        
                        # Mostra preview
                        st.dataframe(historical_data.head(), use_container_width=True)
                    else:
                        st.error("❌ Failed to download data. Check symbol name.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Usa dati scaricati se disponibili
        if 'historical_data' in st.session_state:
            historical_data = st.session_state.historical_data
    
    elif data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload historical data (CSV)", type=['csv'])
        if uploaded_file:
            historical_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(historical_data)} bars")
    else:
        st.info("💡 Using synthetic sample data")
    
    st.markdown("---")
    
    # Data status indicator
    if 'historical_data' in st.session_state and st.session_state.historical_data is not None:
        data = st.session_state.historical_data
        st.success(f"✅ Data loaded: {len(data)} bars")
        st.caption(f"From {data.index[0]} to {data.index[-1]}")
    else:
        st.warning("⚠️ No data loaded - Download or upload data first")
    
    st.markdown("---")
    
    # Training controls
    st.subheader("🎯 Training Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_training = st.button("▶️ Start Training", key="train_btn")
    
    with col2:
        if st.session_state.training_manager.is_training:
            pause_training = st.button("⏸️ Pause", key="pause_btn")
        else:
            pause_training = False
    
    if start_training:
        # Validate data before starting
        if historical_data is None and 'historical_data' not in st.session_state:
            st.error("❌ Please download or upload data first!")
        elif historical_data is None:
            historical_data = st.session_state.get('historical_data')
        
        if historical_data is not None:
            # Clear old callbacks
            st.session_state.training_manager.progress_callbacks.clear()
            st.session_state.training_manager.metrics_callbacks.clear()
            st.session_state.training_manager.error_callbacks.clear()
            
            st.session_state.training_manager.add_progress_callback(progress_callback)
            st.session_state.training_manager.add_metrics_callback(metrics_callback)
            st.session_state.training_manager.add_error_callback(error_callback)
            
            # Run training in background
            thread = threading.Thread(
                target=st.session_state.training_manager.train,
                kwargs={
                    'symbol': symbol,
                    'historical_data': historical_data,
                    'data_file': data_file,
                    'test_size': test_size,
                    'cv_folds': cv_folds,
                    'strategy_mode': strategy_mode,
                    'selected_strategies': selected_strategies
                }
            )
            thread.daemon = True
            thread.start()
            st.success(f"✅ Training started with {len(historical_data)} bars!")
            st.info(f"🎯 Strategy Mode: {strategy_mode} | Strategies: {', '.join(selected_strategies)}")
            time.sleep(1)  # Give it a moment to start
            st.rerun()  # Refresh to show progress


# Main content area
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Training Status",
    "📈 Metrics",
    "📋 Details",
    "📜 Logs",
    "📚 Documentation"
])


# TAB 1: Training Status
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    
    manager = st.session_state.training_manager
    
    # Auto-refresh ogni 2 secondi se training è attivo
    if manager.is_training:
        time.sleep(0.1)  # Small delay
        st.rerun()
    
    with col1:
        st.metric("Status", manager.progress.status.value.upper())
    
    with col2:
        st.metric("Progress", f"{manager.progress.percentage:.1f}%")
    
    with col3:
        elapsed = manager.progress.elapsed_seconds
        st.metric("Elapsed", f"{int(elapsed // 60)}m {int(elapsed % 60)}s")
    
    with col4:
        if manager.progress.estimated_remaining_seconds:
            remaining = manager.progress.estimated_remaining_seconds
            st.metric("Remaining", f"{int(remaining // 60)}m {int(remaining % 60)}s")
        else:
            st.metric("Remaining", "—")
    
    st.markdown("---")
    
    # Progress bar
    st.subheader("Training Progress")
    progress_bar = st.progress(manager.progress.percentage / 100)
    
    # Status message
    st.info(f"📌 {manager.progress.message}")

    # Last error/status
    if st.session_state.get('last_error'):
        st.error(f"❌ {st.session_state.last_error}")
    else:
        st.caption(f"Status: {st.session_state.get('last_status', '...')}")
    
    # Progress visualization
    if manager.is_training or manager.progress.status == TrainingStatus.COMPLETED:
        st.subheader("Training Timeline")
        
        # Show key milestones
        milestones = [
            {"step": 0, "label": "Started", "percentage": 0},
            {"step": 1, "label": "Loading Data", "percentage": 16},
            {"step": 2, "label": "Feature Extraction", "percentage": 33},
            {"step": 3, "label": "Backtesting", "percentage": 50},
            {"step": 4, "label": "Training Models", "percentage": 66},
            {"step": 5, "label": "Context Analysis", "percentage": 83},
            {"step": 6, "label": "Completed", "percentage": 100},
        ]
        
        current_step = manager.progress.current_step
        
        cols = st.columns(7)
        for i, milestone in enumerate(milestones):
            with cols[i]:
                if milestone["step"] < current_step:
                    st.success(f"✅ {milestone['label']}")
                elif milestone["step"] == current_step:
                    st.info(f"⏳ {milestone['label']}")
                else:
                    st.text(f"⏸️ {milestone['label']}")


# TAB 2: Metrics
with tab2:
    if hasattr(st.session_state, 'metrics'):
        metrics = st.session_state.metrics
        
        # Trading metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Trades",
                metrics.get('total_trades', 0),
                delta=f"Win Rate: {metrics.get('win_rate', 0):.1%}"
            )
        
        with col2:
            st.metric(
                "Winning Trades",
                metrics.get('winning_trades', 0),
                delta=f"Avg Win: {metrics.get('avg_win_pnl', 0):.2%}"
            )
        
        with col3:
            st.metric(
                "Profit Factor",
                f"{metrics.get('profit_factor', 0):.2f}",
                delta="PF > 1.0 is good"
            )
        
        with col4:
            st.metric(
                "Best Model",
                metrics.get('best_model', 'N/A'),
                delta=f"F1: {metrics.get('model_f1', 0):.3f}"
            )
        
        st.markdown("---")
        
        # Model performance
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Accuracy", f"{metrics.get('model_accuracy', 0):.1%}")
        
        with col2:
            st.metric("Model F1 Score", f"{metrics.get('model_f1', 0):.3f}")
        
        with col3:
            st.metric("Model ROC-AUC", f"{metrics.get('model_roc_auc', 0):.3f}")
        
        st.markdown("---")
        
        # Feature importance
        if metrics.get('top_features'):
            st.subheader("Top Discriminating Features")
            
            features = metrics['top_features']
            feature_names = [f[0] for f in features]
            feature_scores = [f[1] for f in features]
            
            fig = go.Figure(data=[
                go.Bar(
                    y=feature_names,
                    x=feature_scores,
                    orientation='h',
                    marker=dict(color=feature_scores, colorscale='Viridis')
                )
            ])
            
            fig.update_layout(
                title="Feature Importance",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("⏳ Run training to see metrics")


# TAB 3: Detailed Results
with tab3:
    if hasattr(st.session_state, 'metrics'):
        metrics = st.session_state.metrics
        
        st.subheader("Trading Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            perf_data = {
                'Metric': ['Total Trades', 'Winning Trades', 'Losing Trades', 'Win Rate', 'Profit Factor', 'Avg Win %', 'Avg Loss %'],
                'Value': [
                    str(metrics.get('total_trades', 0)),
                    str(metrics.get('winning_trades', 0)),
                    str(metrics.get('total_trades', 0) - metrics.get('winning_trades', 0)),
                    f"{metrics.get('win_rate', 0):.1%}",
                    f"{metrics.get('profit_factor', 0):.2f}",
                    f"{metrics.get('avg_win_pnl', 0):.2%}",
                    f"{metrics.get('avg_loss_pnl', 0):.2%}"
                ]
            }
            
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df, use_container_width=True, hide_index=True)
        
        with col2:
            # Win rate pie chart
            winning = metrics.get('winning_trades', 0)
            losing = metrics.get('total_trades', 0) - winning
            
            fig = go.Figure(data=[go.Pie(
                labels=['Winning', 'Losing'],
                values=[winning, losing],
                marker=dict(colors=['#4CAF50', '#f44336'])
            )])
            
            fig.update_layout(title="Win/Loss Distribution", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Model Performance")
        
        model_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
            'Score': [
                f"{metrics.get('model_accuracy', 0):.1%}",
                f"{metrics.get('model_precision', 0):.1%}",
                f"{metrics.get('model_recall', 0):.1%}",
                f"{metrics.get('model_f1', 0):.3f}",
                f"{metrics.get('model_roc_auc', 0):.3f}"
            ]
        }
        
        model_df = pd.DataFrame(model_data)
        st.dataframe(model_df, use_container_width=True, hide_index=True)
    else:
        st.info("⏳ Run training to see detailed results")


# TAB 4: Training Logs
with tab4:
    st.subheader("Training Log")
    if st.session_state.training_log:
        log_df = pd.DataFrame(st.session_state.training_log)
        st.dataframe(log_df, use_container_width=True, hide_index=True)
        csv = log_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Log",
            data=csv,
            file_name=f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("📝 Training logs will appear here")


# TAB 5: Documentation
with tab5:
    st.markdown("""
    # 📚 Dashboard Guide
    
    ## Overview
    This dashboard allows you to train ML models for trading with real-time monitoring.
    
    ## Features
    
    ### 📊 Training Status
    - **Progress Bar**: Visual representation of training progress
    - **Step Counter**: Current step and total steps
    - **Elapsed Time**: How long training has been running
    - **Estimated Time**: How long until training completes
    - **Status Messages**: Real-time updates on what's happening
    
    ### 📈 Metrics
    - **Trading Performance**: Win rate, profit factor, average P&L
    - **Model Performance**: Accuracy, F1 score, ROC-AUC
    - **Feature Importance**: Which indicators matter most
    
    ### 📋 Detailed Results
    - **Trading Statistics**: Comprehensive trading metrics
    - **Performance Visualization**: Charts and tables
    - **Model Metrics**: Detailed evaluation scores
    
    ### 📜 Logs
    - **Training Log**: Step-by-step log of training process
    - **Export**: Download logs for analysis
    
    ## Configuration
    
    ### Symbol
    The trading symbol (e.g., EUR_USD, BTC_USD)
    
    ### Test Set Size
    Proportion of data used for testing (0.1 - 0.5)
    - Smaller = more training data, potentially overfit
    - Larger = more test data, less training data
    - Default: 0.2 (20% test, 80% train)
    
    ### Cross-Validation Folds
    Number of folds for cross-validation (3-10)
    - More folds = more computation but better validation
    - Default: 5-fold
    
    ### 🎯 Strategy Configuration
    
    **Strategy Modes:**
    - **WEIGHTED**: Combines signals with weighted average based on confidence
    - **VOTING**: Only generates signal if N strategies agree (min_votes)
    - **ALL**: Uses all signals from all strategies separately
    - **SINGLE**: Uses only one strategy
    
    **Available Strategies:**
    - **RSI Mean Reversion**: Trades oversold/overbought with MACD confirmation
    - **Breakout**: Trades resistance/support breaks with volume confirmation
    - **Trend Following**: Trades pullbacks in strong trends (ADX > 25)
    
    All strategies include:
    - Chart pattern recognition (Hammer, Engulfing, Doji, Double Top/Bottom, H&S)
    - Professional entry logic with multiple confirmations
    - ATR-based stop loss
    - Risk/Reward ratios (1:2 or 1:3)
    - Confidence scoring
    
    ### Data Source
    - **Yahoo Finance**: Download forex, stocks, indices, commodities
      - Examples: EURUSD=X, BTC-USD, ^GSPC, GC=F
      - Date range selection
      - Multiple timeframes (M5 to D1)
    - **Binance**: Download cryptocurrency spot data
      - Examples: BTCUSDT, ETHUSDT, BNBUSDT
      - Up to 1000 bars
      - Multiple timeframes (M1 to D1)
    - **Upload CSV**: Upload your own OHLCV data
    - **Sample Data**: Use synthetic data for testing
    
    ## Training Process
    
    1. **Data Loading**: Load historical OHLCV data
    2. **Feature Extraction**: Extract 40+ technical indicators
    3. **STRATEGY-DRIVEN Backtesting**: Professional strategies generate signals
    4. **Model Training**: Train 4 models (RF, GB, XGBoost, LightGBM) to predict which signals work
    5. **Context Analysis**: Analyze which features predict wins
    6. **Results**: Save and display training results
    
    ### Why Strategy-Driven?
    
    ❌ OLD approach: Simulate trades on EVERY bar → produces zero valid trades
    
    ✅ NEW approach: Professional strategies generate 20-200 signals/year → ML learns which signals work best
    
    ## Tips
    
    💡 **Use at least 1-2 years of data** for strategies to generate enough signals
    
    💡 **Try WEIGHTED mode first** - combines all strategies intelligently
    
    💡 **Check strategy agreement** - signals with 2-3 strategies have higher confidence
    
    💡 **Monitor "Trades Count"** - expect 20-200 trades depending on timeframe
    
    💡 **Review strategy reasons** to understand why trades were taken
    
    ## Next Steps
    
    1. Select strategy mode and strategies
    2. Download historical data (1-2 years recommended)
    3. Click "Start Training"
    4. Monitor progress - strategies will generate signals
    5. Review which strategy signals the ML identifies as best
    6. Use predictions to filter future signals
    
    ## Data Format
    
    CSV file should have these columns:
    ```
    time,open,high,low,close,volume
    2023-01-01 00:00,1.0850,1.0870,1.0840,1.0860,500000
    2023-01-01 01:00,1.0860,1.0880,1.0850,1.0870,550000
    ...
    ```
    """)


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 12px;">
    ML Trading Dashboard v2.0 - Strategy-Driven | Powered by Streamlit
</div>
""", unsafe_allow_html=True)
