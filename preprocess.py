import pandas as pd
import logging

def create_target_labels(df, periods):
    """
    Creates target labels for different time periods.
    """
    for period in periods:
        df[f'Target_{period}d'] = (df['Close'].shift(-period) > df['Close']).astype(int)
    return df

def preprocess_data(df, config):
    """
    Performs preprocessing on the data.
    """
    # Select features
    features = config["features"]
    df = df[features].copy()

    logging.info(f"Shape after feature selection: {df.shape}")

    # Handle missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    initial_rows = df.shape[0]
    df.dropna(inplace=True)
    final_rows = df.shape[0]

    dropped_rows = initial_rows - final_rows
    if initial_rows > 0:
        dropped_percentage = (dropped_rows / initial_rows) * 100
        logging.info(f"Dropped {dropped_rows} rows ({dropped_percentage:.2f}%) due to missing values.")
        if dropped_percentage > 90:
            logging.warning("More than 90% of rows were dropped. You should investigate the data sources.")

    logging.info(f"Shape after handling missing values: {df.shape}")

    # Create the target variables
    periods = config["target_periods"]
    df = create_target_labels(df, periods)
    df.dropna(inplace=True)
    logging.info(f"Shape after creating targets: {df.shape}")

    return df
