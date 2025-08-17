import pandas as pd

def create_target_labels(df, periods):
    """
    Creates target labels for different time periods.
    """
    for period in periods:
        df[f'Target_{period}d'] = (df['Close'].shift(-period) > df['Close']).astype(int)
    return df

def preprocess_data(df):
    """
    Performs preprocessing on the data.
    """
    # Select features
    features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_Signal',
        'Bollinger_Upper', 'Bollinger_Lower', 'VIXCLS', 'DGS10',
        'value' # Fear & Greed Index value
    ]
    df = df[features].copy()

    print(f"Shape after feature selection: {df.shape}")

    # Handle missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    initial_rows = df.shape[0]
    df.dropna(inplace=True)
    final_rows = df.shape[0]

    dropped_rows = initial_rows - final_rows
    if initial_rows > 0:
        dropped_percentage = (dropped_rows / initial_rows) * 100
        print(f"Dropped {dropped_rows} rows ({dropped_percentage:.2f}%) due to missing values.")
        if dropped_percentage > 90:
            print("WARNING: More than 90% of rows were dropped. You should investigate the data sources.")

    print(f"Shape after handling missing values: {df.shape}")

    # Create the target variables
    periods = [1, 5, 10, 21] # 1 day, 1 week, 2 weeks, 1 month
    df = create_target_labels(df, periods)
    df.dropna(inplace=True)
    print(f"Shape after creating targets: {df.shape}")

    return df
