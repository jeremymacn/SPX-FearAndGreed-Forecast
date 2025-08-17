import pandas as pd
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

def engineer_features(df, config):
    """
    Engineers new features from the existing data.
    """
    logging.info("Engineering new features...")
    
    # Lagged Features
    lag_features = config.get("lag_features", [])
    lag_periods = config.get("lag_periods", [])
    for feature in lag_features:
        if feature in df.columns:
            for period in lag_periods:
                df[f'{feature}_lag_{period}'] = df[feature].shift(period)
            
    # Rolling Statistics
    rolling_periods = config.get("rolling_periods", [])
    for period in rolling_periods:
        df[f'Close_rolling_std_{period}'] = df['Close'].rolling(window=period).std()
        
    # Interaction Features
    if 'Close' in df.columns and 'SMA_200' in df.columns:
        df['Close_vs_SMA200'] = df['Close'] - df['SMA_200']
    if 'Bollinger_Upper' in df.columns and 'Bollinger_Lower' in df.columns:
        df['Bollinger_Band_Width'] = df['Bollinger_Upper'] - df['Bollinger_Lower']
    
    # Drop NaNs created by lagging and rolling stats
    df.dropna(inplace=True)
    
    # Normalization
    if "normalization_strategies" in config:
        logging.info("Applying normalization strategies...")
        strategies = config["normalization_strategies"]

        # Log transform
        for col in strategies.get("log", []):
            if col in df.columns:
                df[col] = np.log(df[col] + 1e-6)

        # Min-Max scaling
        min_max_cols = [col for col in strategies.get("min_max", []) if col in df.columns]
        if min_max_cols:
            scaler = MinMaxScaler()
            df[min_max_cols] = scaler.fit_transform(df[min_max_cols])

        # Z-score scaling
        z_score_cols = [col for col in strategies.get("z_score", []) if col in df.columns]
        if z_score_cols:
            scaler = StandardScaler()
            df[z_score_cols] = scaler.fit_transform(df[z_score_cols])

    # Feature Selection
    if "feature_selection_k" in config:
        logging.info(f"Selecting top {config['feature_selection_k']} features...")
        target_col = [col for col in df.columns if 'Target' in col]
        if not target_col:
            logging.error("No target column found for feature selection.")
            return df

        X = df.drop(columns=target_col)
        y = df[target_col[0]] # Assuming single target

        selector = SelectKBest(f_classif, k=config['feature_selection_k'])
        selector.fit(X, y)

        selected_features = X.columns[selector.get_support()].tolist()

        # Ensure 'Close' is always included for backtesting calculations
        if 'Close' not in selected_features:
            selected_features.append('Close')

        df = df[selected_features + target_col]

    logging.info(f"Shape after feature engineering: {df.shape}")
    
    return df
