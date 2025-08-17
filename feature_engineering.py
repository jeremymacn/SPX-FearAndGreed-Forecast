import pandas as pd
import logging

def engineer_features(df, config):
    """
    Engineers new features from the existing data.
    """
    logging.info("Engineering new features...")
    
    # Lagged Features
    lag_features = config["lag_features"]
    lag_periods = config["lag_periods"]
    for feature in lag_features:
        for period in lag_periods:
            df[f'{feature}_lag_{period}'] = df[feature].shift(period)
            
    # Rolling Statistics
    rolling_periods = config["rolling_periods"]
    for period in rolling_periods:
        df[f'Close_rolling_std_{period}'] = df['Close'].rolling(window=period).std()
        
    # Interaction Features
    df['Close_vs_SMA200'] = df['Close'] - df['SMA_200']
    df['Bollinger_Band_Width'] = df['Bollinger_Upper'] - df['Bollinger_Lower']
    
    # Drop NaNs created by lagging and rolling stats
    df.dropna(inplace=True)
    
    logging.info(f"Shape after feature engineering: {df.shape}")
    
    return df
