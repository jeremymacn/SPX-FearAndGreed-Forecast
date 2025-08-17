# --- Global Configuration ---

# Data Gathering
DATA_CONFIG = {
    "years": 10
}

# Feature Engineering
FEATURE_ENGINEERING_CONFIG = {
    "lag_features": ['SMA_200', 'SMA_50', 'Bollinger_Upper', 'Bollinger_Lower', 'VIXCLS'],
    "lag_periods": [1, 2, 3, 5, 10],
    "rolling_periods": [5, 10],
    "normalization_strategies": {
        "log": ["M1SL", "M2SL"],
        "min_max": [
            "new_high_1y", "new_low_1y", "new_high_6m", "new_low_6m",
            "new_high_3m", "new_low_3m", "new_high_1m", "new_low_1m",
            "new_high_2w", "new_low_2w", "new_high_1w", "new_low_1w"
        ],
        "z_score": ["FEDFUNDS", "VIXCLS", "VVIX", "DGS10"]
    },
    "feature_selection_k": 20
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
        "RandomForestClassifier": {
            "n_estimators": 100,
        },
        "GradientBoostingClassifier": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3
        }
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
