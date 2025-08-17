import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def run_simple_backtest(df, target_column='Target_21d'):
    """
    Performs a simple backtest by training the model yearly and predicting the next year.
    """
    years = sorted(df.index.year.unique())
    
    for i in range(len(years) - 1):
        train_year_end = years[i]
        test_year = years[i+1]
        
        # Split data into training and testing sets
        train_df = df[df.index.year <= train_year_end]
        test_df = df[df.index.year == test_year]
        
        if len(train_df) == 0 or len(test_df) == 0:
            continue
            
        print(f"\n--- Training on data up to {train_year_end}, Testing on {test_year} ---")
        
        # Select features and target
        X_train = train_df.drop(columns=[col for col in df.columns if 'Target' in col])
        y_train = train_df[target_column]
        X_test = test_df.drop(columns=[col for col in df.columns if 'Target' in col])
        y_test = test_df[target_column]
        
        # Initialize and train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {test_year}: {accuracy:.2f}")

