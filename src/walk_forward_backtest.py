import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import logging

def run_walk_forward_backtest(df, model_class, model_params, holding_period=21, window_type='rolling', target_column='Target_21d', training_window=3, testing_window=1, step=1, transaction_cost=0.001, initial_capital=10000, strategy='long_only', parking_threshold=2, output_dir='reports/images'):
    """
    Performs a walk-forward backtest of the trading strategy with rolling or expanding windows.
    """
    years = sorted(df.index.year.unique())
    all_equity = [initial_capital]
    trade_log = []

    if strategy == 'buy_and_hold':
        model_name = "BuyAndHold"
        logging.info(f"--- Running Backtest for {model_name} ---")
        all_equity, trade_log = run_buy_and_hold_backtest(df, initial_capital)
    else:
        model_name = model_class.__name__
        logging.info(f"--- Running {window_type.capitalize()} Window Backtest with {model_name} and {strategy} strategy ---")
        if strategy == 'trend_following':
            all_equity, trade_log = run_trend_following_backtest(df, model_class, model_params, holding_period, window_type, target_column, training_window, testing_window, step, transaction_cost, initial_capital)
        elif strategy == 'buy_and_hold_parking':
            all_equity, trade_log = run_buy_and_hold_parking_backtest(df, model_class, model_params, window_type, target_column, training_window, testing_window, step, transaction_cost, initial_capital, parking_threshold)
        else:
            raise ValueError(f"Strategy '{strategy}' not supported for model-based backtesting.")

    # Final performance metrics
    if len(all_equity) > 1:
        equity_series = pd.Series(all_equity)
        total_return = (equity_series.iloc[-1] / initial_capital) - 1
        daily_returns = equity_series.pct_change().dropna()
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) != 0 else 0

        # Calculate Buy & Hold return for the entire period
        buy_and_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1

        peak = equity_series.expanding(min_periods=1).max()
        drawdown = (equity_series/peak) - 1
        max_drawdown = drawdown.min()

        logging.info("--- Backtest Performance ---")
        logging.info(f"Total Return: {total_return:.2%}")
        logging.info(f"Buy & Hold Return: {buy_and_hold_return:.2%}")
        logging.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logging.info(f"Maximum Drawdown: {max_drawdown:.2%}")

        # Plot equity curve
        plt.figure(figsize=(12, 8))
        equity_series.plot()
        plt.title(f'Equity Curve - {model_name} - {window_type.capitalize()} Window')
        plt.xlabel('Trade Number')
        plt.ylabel('Equity')
        plt.grid(True)
        output_path = f"{output_dir}/equity_curve_{model_name}_{window_type}.png"
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Equity curve plot saved to {output_path}")

        return {
            "Total Return": total_return,
            "Buy & Hold Return": buy_and_hold_return,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown
        }
    else:
        logging.info("No trades were made during the backtest.")
        return {
            "Total Return": 0,
            "Buy & Hold Return": 0,
            "Sharpe Ratio": 0,
            "Max Drawdown": 0
        }

def run_trend_following_backtest(df, model_class, model_params, holding_period, window_type, target_column, training_window, testing_window, step, transaction_cost, initial_capital):
    years = sorted(df.index.year.unique())
    all_equity = [initial_capital]
    trade_log = []
    in_trade = False

    for i in range(0, len(years) - training_window - testing_window + 1, step):
        train_start_year = years[i]
        train_end_year = years[i + training_window - 1]
        test_start_year = years[i + training_window]
        test_end_year = years[i + training_window + testing_window - 1]

        if window_type == 'rolling':
            train_df = df[(df.index.year >= train_start_year) & (df.index.year <= train_end_year)]
        else: # expanding
            train_df = df[df.index.year <= train_end_year]

        test_df = df[(df.index.year >= test_start_year) & (df.index.year <= test_end_year)]

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        X_train = train_df.drop(columns=[col for col in df.columns if 'Target' in col])
        y_train = train_df[target_column]
        X_test = test_df.drop(columns=[col for col in df.columns if 'Target' in col])

        model = model_class(**model_params, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        for j in range(len(test_df)):
            current_price = test_df['Close'].iloc[j]
            if j > 0:
                prev_price = test_df['Close'].iloc[j-1]
                daily_return = (current_price / prev_price) - 1
                if in_trade:
                    all_equity.append(all_equity[-1] * (1 + daily_return))
                else:
                    all_equity.append(all_equity[-1])

            # Buy signal
            if not in_trade and predictions[j] >= 4:
                in_trade = True
                all_equity[-1] *= (1 - transaction_cost)
                trade_log.append(f"{test_df.index[j].date()}: Buy at {current_price:.2f}")

            # Sell signal
            elif in_trade and predictions[j] <= 2:
                in_trade = False
                all_equity[-1] *= (1 - transaction_cost)
                trade_log.append(f"{test_df.index[j].date()}: Sell at {current_price:.2f}")

    return all_equity, trade_log

def run_buy_and_hold_parking_backtest(df, model_class, model_params, window_type, target_column, training_window, testing_window, step, transaction_cost, initial_capital, parking_threshold):
    years = sorted(df.index.year.unique())
    all_equity = [initial_capital]
    trade_log = []
    is_parked = False

    for i in range(0, len(years) - training_window - testing_window + 1, step):
        train_start_year = years[i]
        train_end_year = years[i + training_window - 1]
        test_start_year = years[i + training_window]
        test_end_year = years[i + training_window + testing_window - 1]

        if window_type == 'rolling':
            train_df = df[(df.index.year >= train_start_year) & (df.index.year <= train_end_year)]
        else: # expanding
            train_df = df[df.index.year <= train_end_year]
        
        test_df = df[(df.index.year >= test_start_year) & (df.index.year <= test_end_year)]

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        X_train = train_df.drop(columns=[col for col in df.columns if 'Target' in col])
        y_train = train_df[target_column]
        X_test = test_df.drop(columns=[col for col in df.columns if 'Target' in col])

        model = model_class(**model_params, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        for j in range(len(test_df)):
            current_price = test_df['Close'].iloc[j]

            if j > 0:
                prev_price = test_df['Close'].iloc[j-1]
                daily_return = (current_price / prev_price) - 1
                if not is_parked:
                    all_equity.append(all_equity[-1] * (1 + daily_return))
                else:
                    all_equity.append(all_equity[-1])

            if predictions[j] <= parking_threshold and not is_parked:
                is_parked = True
                # Sell signal, apply transaction cost
                all_equity[-1] *= (1 - transaction_cost)
                trade_log.append(f"{test_df.index[j].date()}: Parked at {current_price:.2f}")

            elif predictions[j] > parking_threshold and is_parked:
                is_parked = False
                # Buy signal, apply transaction cost
                all_equity[-1] *= (1 - transaction_cost)
                trade_log.append(f"{test_df.index[j].date()}: Re-entered at {current_price:.2f}")

    return all_equity, trade_log

def run_buy_and_hold_backtest(df, initial_capital):
    """
    Runs a simple buy-and-hold backtest.
    """
    trade_log = [f"{df.index[0].date()}: Buy and Hold at {df['Close'].iloc[0]:.2f}"]

    # Calculate daily returns
    daily_returns = df['Close'].pct_change()

    # Calculate equity curve
    all_equity = [initial_capital]
    for daily_return in daily_returns[1:]: # first return is NaN
        all_equity.append(all_equity[-1] * (1 + daily_return))

    trade_log.append(f"{df.index[-1].date()}: Final equity {all_equity[-1]:.2f}")

    return all_equity, trade_log