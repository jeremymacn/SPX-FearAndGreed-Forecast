from sp500_indicators import gather_all_data
from preprocess import preprocess_data
from feature_engineering import engineer_features
from tuning import tune_hyperparameters
from walk_forward_backtest import run_walk_forward_backtest

def main():
    """
    Main function to run the ML pipeline.
    """
    # --- Configuration ---
    config = {
        "data": {
            "years": 10
        },
        "backtest": {
            "model_params": None, # Will be set by tuning
            "holding_period": 40, # New optimal holding period
            "window_type": "rolling",
            "target_column": "Target_21d", # Still predicting the 21-day outcome
            "training_window": 3,
            "testing_window": 1,
            "step": 1,
            "transaction_cost": 0.001,
            "initial_capital": 10000
        }
    }

    # --- Pipeline ---

    # Step 1: Data Collection
    print("Gathering data...")
    raw_df = gather_all_data(years=config["data"]["years"])
    print("Data gathering complete.")

    # Step 2: Data Preprocessing
    print("\nPreprocessing data...")
    preprocessed_df = preprocess_data(raw_df.copy())
    print("Data preprocessing complete.")

    # Step 3: Feature Engineering
    print("\nEngineering features...")
    engineered_df = engineer_features(preprocessed_df.copy())
    print("Feature engineering complete.")

    # Step 4: Hyperparameter Tuning
    print("\nRunning Hyperparameter Tuning...")
    best_params = tune_hyperparameters(engineered_df, target_column=config["backtest"]["target_column"])
    print("\nHyperparameter Tuning complete.")

    # Step 5: Walk-Forward Backtest with Tuned Model and Optimal Holding Period
    if best_params:
        config["backtest"]["model_params"] = best_params
        run_walk_forward_backtest(engineered_df, **config["backtest"])
        print("\nWalk-forward backtest with tuned model and optimal holding period complete.")


if __name__ == '__main__':
    main()