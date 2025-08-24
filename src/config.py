# --- Global Configuration ---

# Data Gathering
DATA_CONFIG = {
    "years": 1
}

# Feature Engineering
FEATURE_ENGINEERING_CONFIG = {
    "lag_features": ['SMA_200', 'SMA_50', 'Bollinger_Upper', 'Bollinger_Lower', 'VIXCLS'],
    "lag_periods": [1, 2, 3, 5, 10],
    "rolling_periods": [5, 10],
    "normalization_strategies": {
        "log": ["M1SL", "M2SL"],
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
        'A/D', 'ADX', 'Aroon_Down', 'Aroon_Up', 'Stoch_SlowK', 'Stoch_SlowD', 'OBV',
        'value' # Fear & Greed Index value
    ],
    "target_periods": [1, 3, 5, 10, 21]
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
        },
        "LGBMClassifier": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31
        },
        "DecisionTreeClassifier": {
            "max_depth": 5
        },
        "XGBClassifier": {
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
    "initial_capital": 10000,
    "strategy": "buy_and_hold_parking",
    "parking_threshold": 2
}
