import logging
from sp500_indicators import gather_all_data
from preprocess import preprocess_data
from feature_engineering import engineer_features
from walk_forward_backtest import run_walk_forward_backtest
import config

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to run the ML pipeline.
    """
    # --- Pipeline ---

    # Step 1: Data Collection
    logging.info("Gathering data...")
    raw_df = gather_all_data(years=config.DATA_CONFIG["years"])
    logging.info("Data gathering complete.")

    # Step 2: Data Preprocessing
    logging.info("Preprocessing data...")
    preprocessed_df = preprocess_data(raw_df.copy(), config.PREPROCESSING_CONFIG)
    logging.info("Data preprocessing complete.")

    # Step 3: Feature Engineering
    logging.info("Engineering features...")
    engineered_df = engineer_features(preprocessed_df.copy(), config.FEATURE_ENGINEERING_CONFIG)
    logging.info("Feature engineering complete.")

    # Step 4: Walk-Forward Backtest
    logging.info("Running Walk-Forward Backtest...")
    run_walk_forward_backtest(engineered_df, **config.BACKTEST_CONFIG)
    logging.info("Walk-forward backtest complete.")


if __name__ == '__main__':
    main()