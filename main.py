import logging
from sp500_indicators import gather_all_data
from preprocess import preprocess_data
from feature_engineering import engineer_features
from walk_forward_backtest import run_walk_forward_backtest
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
    models_to_run = {
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier
    }

    for model_name, model_class in models_to_run.items():
        logging.info(f"--- Running backtest for {model_name} ---")

        backtest_config = config.BACKTEST_CONFIG.copy()
        backtest_config["model_class"] = model_class
        backtest_config["model_params"] = config.BACKTEST_CONFIG["model_params"][model_name]

        run_walk_forward_backtest(engineered_df.copy(), **backtest_config)
        logging.info(f"--- Backtest for {model_name} complete ---")


if __name__ == '__main__':
    main()