import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

def tune_hyperparameters(df, target_column='Target_21d'):
    """
    Performs hyperparameter tuning for the RandomForestClassifier with a proper hold-out set.
    """
    print("\n--- Hyperparameter Tuning with Hold-Out Set ---")

    # Prepare data
    X = df.drop(columns=[col for col in df.columns if 'Target' in col])
    y = df[target_column]

    # Split data into training and hold-out set (80/20 split)
    split_index = int(len(X) * 0.8)
    X_train, X_holdout = X[:split_index], X[split_index:]
    y_train, y_holdout = y[:split_index], y[split_index:]

    print(f"Training set size: {len(X_train)}")
    print(f"Hold-out set size: {len(X_holdout)}")

    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10]
    }

    # Initialize the model and TimeSeriesSplit
    model = RandomForestClassifier(random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=2)

    # Fit the grid search on the training data
    grid_search.fit(X_train, y_train)

    # Print the best parameters
    print("\nBest parameters found:")
    print(grid_search.best_params_)

    # Evaluate the best model on the hold-out set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_holdout)
    accuracy = accuracy_score(y_holdout, y_pred)

    print(f"\nAccuracy of the best model on the hold-out set: {accuracy:.2f}")
    print("\nClassification Report on the hold-out set:")
    print(classification_report(y_holdout, y_pred))

    return grid_search.best_params_