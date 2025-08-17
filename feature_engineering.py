import pandas as pd

def engineer_features(df):
    """
    Engineers new features from the existing data.
    """
    print("Engineering new features...")
    
    # Lagged Features
    lag_features = ['SMA_200', 'SMA_50', 'Bollinger_Upper', 'Bollinger_Lower', 'VIXCLS']
    lag_periods = [1, 2, 3, 5, 10]
    for feature in lag_features:
        for period in lag_periods:
            df[f'{feature}_lag_{period}'] = df[feature].shift(period)
            
    # Rolling Statistics
    rolling_periods = [5, 10]
    for period in rolling_periods:
        df[f'Close_rolling_std_{period}'] = df['Close'].rolling(window=period).std()
        
    # Interaction Features
    df['Close_vs_SMA200'] = df['Close'] - df['SMA_200']
    df['Bollinger_Band_Width'] = df['Bollinger_Upper'] - df['Bollinger_Lower']
    
    # Drop NaNs created by lagging and rolling stats
    df.dropna(inplace=True)
    
    print(f"Shape after feature engineering: {df.shape}")
    
    return df
