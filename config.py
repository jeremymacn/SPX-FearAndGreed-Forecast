# --- Global Configuration ---

# Data Gathering
DATA_CONFIG = {
    "years": 10
}

# Feature Engineering
FEATURE_ENGINEERING_CONFIG = {
    "lag_features": ['SMA_200', 'SMA_50', 'Bollinger_Upper', 'Bollinger_Lower', 'VIXCLS'],
    "lag_periods": [1, 2, 3, 5, 10],
    "rolling_periods": [5, 10]
}

# Preprocessing
PREPROCESSING_CONFIG = {
    "features": [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_Signal',
        'Bollinger_Upper', 'Bollinger_Lower', 'VIXCLS', 'DGS10',
        'value' # Fear & Greed Index value
    ],
    "target_periods": [1, 5, 10, 21]
}

# Backtesting
BACKTEST_CONFIG = {
    "model_params": {
        "n_estimators": 100,
    },
    "holding_period": 40,
    "window_type": "rolling",
    "target_column": "Target_21d",
    "training_window": 3,
    "testing_window": 1,
    "step": 1,
    "transaction_cost": 0.001,
    "initial_capital": 10000
}
