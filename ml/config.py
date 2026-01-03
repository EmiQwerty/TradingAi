"""
ML Pipeline Configuration
Default settings for training, prediction, and backtesting
"""

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'price_action': {
        'enabled': True,
        'lookback': 20,
    },
    'technical_indicators': {
        'enabled': True,
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'bb_std': 2,
        'atr_period': 14,
        'stoch_period': 14,
        'stoch_smooth': 3,
    },
    'momentum': {
        'enabled': True,
        'roc_period': 12,
    },
    'volatility': {
        'enabled': True,
        'std_period': 20,
        'true_range_period': 14,
    },
    'market_structure': {
        'enabled': True,
        'swing_lookback': 10,
    },
    'trend_regime': {
        'enabled': True,
        'ma_fast': 9,
        'ma_mid': 21,
        'ma_slow': 50,
    },
    'patterns': {
        'enabled': True,
        'lookback': 5,
    },
    'edge': {
        'enabled': True,
    }
}

# Backtest Configuration
BACKTEST_CONFIG = {
    'commission_rate': 0.0001,  # 0.01% per trade
    'position_size': 100000,    # Micro lot = 0.01 lot = $1 per pip
    'pip_value': 0.0001,        # EUR_USD pip value
    'max_trade_duration_bars': 100,  # Max 100 bars per trade
    'stop_loss_pips': 50,       # Default SL = 50 pips
    'take_profit_pips': 100,    # Default TP = 100 pips (2:1 ratio)
}

# Model Training Configuration
TRAINING_CONFIG = {
    'test_size': 0.2,           # 20% test set
    'cv_folds': 5,              # 5-fold cross-validation
    'random_state': 42,         # Reproducibility
    'min_samples_for_training': 50,  # Min trades to train
    
    'random_forest': {
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    },
    
    'gradient_boosting': {
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
    },
    
    'xgboost': {
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
    },
    
    'lightgbm': {
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [10, 31, 50],
        }
    }
}

# Prediction Configuration
PREDICTION_CONFIG = {
    'default_model': 'xgboost',           # Model to use for predictions
    'default_confidence_threshold': 0.55,  # Only trade if win_prob > this
    'min_confidence_to_trade': 0.30,      # Model must be at least this certain
    'high_confidence_threshold': 0.60,    # Consider "high confidence" if > this
}

# Context Analysis Configuration
CONTEXT_CONFIG = {
    'significance_level': 0.05,  # p-value threshold for t-test
    'feature_comparison_top_n': 20,  # Show top 20 discriminating features
    'min_trades_for_analysis': 10,  # Min trades to do analysis
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': ['console', 'file'],  # Both console and file
    'log_file': 'logs/ml_pipeline.log',
}

# Data Configuration
DATA_CONFIG = {
    'lookback_bars': 50,  # Min bars needed for feature extraction
    'required_history': 100,  # Min history for backtesting
    'ohlcv_columns': ['open', 'high', 'low', 'close', 'volume'],
}

# Model Directory Configuration
MODEL_CONFIG = {
    'model_directory': 'ml/models',
    'save_models': True,
    'load_models_if_exist': True,
}

# Feature Importance Configuration
FEATURE_IMPORTANCE_CONFIG = {
    'top_n': 20,  # Show top 20 features
    'min_importance': 0.01,  # Filter features below this importance
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'min_win_rate': 0.50,  # Minimum acceptable win rate (50%)
    'min_profit_factor': 1.0,  # Minimum acceptable profit factor
    'min_roc_auc': 0.60,  # Minimum model ROC-AUC (0.5 = random)
    'min_f1_score': 0.50,  # Minimum F1 score
}

# Risk Management
RISK_CONFIG = {
    'max_loss_per_trade_percent': 0.02,  # Max 2% loss per trade
    'max_consecutive_losses': 3,  # Stop after 3 losses
    'position_size_multiplier': 1.0,  # Can reduce if losses
    'correlation_threshold': 0.7,  # Skip trades with high correlation
}

# Feature Selection
FEATURE_SELECTION_CONFIG = {
    'use_all_features': True,  # Use all 40+ features
    'select_top_n': None,  # Or select top N features only
    'remove_low_importance': False,  # Remove features below min_importance
    'remove_correlated': False,  # Remove highly correlated features (correlation > 0.9)
}

def get_config():
    """Return full configuration as dict"""
    return {
        'features': FEATURE_CONFIG,
        'backtest': BACKTEST_CONFIG,
        'training': TRAINING_CONFIG,
        'prediction': PREDICTION_CONFIG,
        'context': CONTEXT_CONFIG,
        'logging': LOGGING_CONFIG,
        'data': DATA_CONFIG,
        'models': MODEL_CONFIG,
        'feature_importance': FEATURE_IMPORTANCE_CONFIG,
        'performance_thresholds': PERFORMANCE_THRESHOLDS,
        'risk': RISK_CONFIG,
        'feature_selection': FEATURE_SELECTION_CONFIG,
    }

def update_config(key: str, value: dict):
    """Update configuration at runtime"""
    config_map = {
        'features': FEATURE_CONFIG,
        'backtest': BACKTEST_CONFIG,
        'training': TRAINING_CONFIG,
        'prediction': PREDICTION_CONFIG,
        'context': CONTEXT_CONFIG,
        'logging': LOGGING_CONFIG,
        'data': DATA_CONFIG,
        'models': MODEL_CONFIG,
        'feature_importance': FEATURE_IMPORTANCE_CONFIG,
        'performance_thresholds': PERFORMANCE_THRESHOLDS,
        'risk': RISK_CONFIG,
        'feature_selection': FEATURE_SELECTION_CONFIG,
    }
    
    if key in config_map:
        config_map[key].update(value)
        return True
    return False

if __name__ == '__main__':
    import json
    config = get_config()
    print(json.dumps(config, indent=2, default=str))
