"""
Dashboard Utilities - Helper functions for the Streamlit dashboard
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import streamlit as st


class DataValidator:
    """Validate and prepare data for training"""
    
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate OHLCV data format
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            (is_valid, message)
        """
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        
        df_cols = set([col.lower() for col in df.columns])
        
        if not required_cols.issubset(df_cols):
            missing = required_cols - df_cols
            return False, f"Missing columns: {missing}"
        
        if len(df) < 50:
            return False, f"Need at least 50 bars, got {len(df)}"
        
        # Check for NaN values
        if df[['open', 'high', 'low', 'close', 'volume']].isnull().any().any():
            return False, "Data contains NaN values"
        
        # Check for logical OHLC relationships
        invalid = ((df['high'] < df['low']) | 
                   (df['high'] < df['close']) | 
                   (df['low'] > df['close'])).sum()
        
        if invalid > 0:
            return False, f"{invalid} bars have invalid OHLC relationships"
        
        return True, "Valid OHLCV data"
    
    @staticmethod
    def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for training
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Prepared DataFrame
        """
        df = df.copy()
        
        # Standardize column names to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        
        # Select only required columns
        df = df[required].copy()
        
        # Convert to numeric
        for col in required:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove NaN rows
        df = df.dropna()
        
        # Reset index
        df.reset_index(drop=True, inplace=True)
        
        return df


class ResultsFormatter:
    """Format training results for display"""
    
    @staticmethod
    def format_metrics(metrics: Dict) -> Dict:
        """Format metrics for display"""
        return {
            'Total Trades': metrics.get('total_trades', 0),
            'Winning Trades': metrics.get('winning_trades', 0),
            'Win Rate': f"{metrics.get('win_rate', 0):.1%}",
            'Profit Factor': f"{metrics.get('profit_factor', 0):.2f}",
            'Model Accuracy': f"{metrics.get('model_accuracy', 0):.1%}",
            'Model F1 Score': f"{metrics.get('model_f1', 0):.3f}",
            'Best Model': metrics.get('best_model', 'N/A'),
            'Avg Win PnL': f"{metrics.get('avg_win_pnl', 0):.2%}",
            'Avg Loss PnL': f"{metrics.get('avg_loss_pnl', 0):.2%}",
        }
    
    @staticmethod
    def format_feature_importance(features: List[Tuple[str, float]], top_n: int = 10) -> pd.DataFrame:
        """Format feature importance for display"""
        top_features = features[:top_n]
        df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
        df = df.sort_values('Importance', ascending=True)
        return df


class FileManager:
    """Manage file operations for dashboard"""
    
    @staticmethod
    def load_results(results_dir: Path, symbol: str) -> Optional[Dict]:
        """Load latest training results for symbol"""
        results_dir = Path(results_dir)
        
        if not results_dir.exists():
            return None
        
        # Find latest results file
        pattern = f"{symbol}_*.json"
        files = sorted(results_dir.glob(pattern))
        
        if not files:
            return None
        
        latest = files[-1]
        
        try:
            with open(latest, 'r') as f:
                return json.load(f)
        except Exception as e:
            return None
    
    @staticmethod
    def list_training_history(results_dir: Path, symbol: str = None) -> List[Dict]:
        """List all training results"""
        results_dir = Path(results_dir)
        
        if not results_dir.exists():
            return []
        
        history = []
        
        if symbol:
            pattern = f"{symbol}_*.json"
            files = results_dir.glob(pattern)
        else:
            pattern = "*_*.json"
            files = results_dir.glob(pattern)
        
        for file in sorted(files):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    history.append({
                        'file': file.name,
                        'timestamp': data.get('timestamp', 'Unknown'),
                        'symbol': data.get('symbol', 'Unknown'),
                        'trades': data.get('metrics', {}).get('total_trades', 0),
                        'win_rate': data.get('metrics', {}).get('win_rate', 0),
                        'profit_factor': data.get('metrics', {}).get('profit_factor', 0),
                    })
            except Exception:
                continue
        
        return history


class ChartBuilder:
    """Build charts for dashboard"""
    
    @staticmethod
    def build_equity_curve(trades_df: pd.DataFrame):
        """Build equity curve from trades"""
        if trades_df.empty:
            return None
        
        import plotly.graph_objects as go
        
        # Calculate cumulative returns
        trades_df = trades_df.copy()
        trades_df['pnl'] = trades_df.get('pnl_percent', 0)
        trades_df['cumulative_return'] = (1 + trades_df['pnl']).cumprod()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trades_df.index,
            y=trades_df['cumulative_return'],
            mode='lines',
            name='Equity',
            line=dict(color='#667eea', width=2),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.1)'
        ))
        
        fig.update_layout(
            title='Equity Curve',
            xaxis_title='Trade Number',
            yaxis_title='Cumulative Return',
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    @staticmethod
    def build_drawdown_chart(trades_df: pd.DataFrame):
        """Build drawdown chart"""
        if trades_df.empty:
            return None
        
        import plotly.graph_objects as go
        
        trades_df = trades_df.copy()
        trades_df['pnl'] = trades_df.get('pnl_percent', 0)
        trades_df['cumulative_return'] = (1 + trades_df['pnl']).cumprod()
        
        # Calculate running max and drawdown
        running_max = trades_df['cumulative_return'].expanding().max()
        drawdown = (trades_df['cumulative_return'] - running_max) / running_max
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=trades_df.index,
            y=drawdown,
            name='Drawdown',
            marker=dict(color=drawdown, colorscale='Reds', showscale=True)
        ))
        
        fig.update_layout(
            title='Drawdown Chart',
            xaxis_title='Trade Number',
            yaxis_title='Drawdown %',
            height=400
        )
        
        return fig
    
    @staticmethod
    def build_returns_distribution(trades_df: pd.DataFrame):
        """Build returns distribution histogram"""
        if trades_df.empty:
            return None
        
        import plotly.graph_objects as go
        
        trades_df = trades_df.copy()
        pnl = trades_df.get('pnl_percent', [])
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=pnl,
            nbinsx=30,
            name='Returns',
            marker=dict(color='#667eea')
        ))
        
        fig.update_layout(
            title='Returns Distribution',
            xaxis_title='PnL %',
            yaxis_title='Frequency',
            hovermode='x unified',
            height=400
        )
        
        return fig


class ProgressTracker:
    """Track and visualize training progress"""
    
    @staticmethod
    def calculate_eta(start_time: datetime, current_step: int, total_steps: int) -> Optional[datetime]:
        """Calculate estimated completion time"""
        if current_step == 0:
            return None
        
        elapsed = (datetime.now() - start_time).total_seconds()
        avg_per_step = elapsed / current_step
        remaining = avg_per_step * (total_steps - current_step)
        
        return datetime.now() + pd.Timedelta(seconds=remaining)
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration as human-readable string"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{int(minutes)}m {int(seconds % 60)}s"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{int(hours)}h {int(minutes)}m"


class NotificationManager:
    """Manage notifications for dashboard"""
    
    @staticmethod
    def show_success(message: str, icon: str = "✅"):
        """Show success notification"""
        st.success(f"{icon} {message}")
    
    @staticmethod
    def show_error(message: str, icon: str = "❌"):
        """Show error notification"""
        st.error(f"{icon} {message}")
    
    @staticmethod
    def show_warning(message: str, icon: str = "⚠️"):
        """Show warning notification"""
        st.warning(f"{icon} {message}")
    
    @staticmethod
    def show_info(message: str, icon: str = "ℹ️"):
        """Show info notification"""
        st.info(f"{icon} {message}")
