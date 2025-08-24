# S&P 500 Predictive Model

## Project Goal

The primary goal of this project is to develop and evaluate machine learning models to predict the movement of the S&P 500 index.

## Project Structure

The project has been refactored into a more organized structure:

- `src/`: Contains all the Python source code.
  - `main.py`: The main entry point to run the pipeline.
  - `data_sources.py`: Functions for gathering data from various sources.
  - `preprocess.py`: Data preprocessing and target label creation.
  - `feature_engineering.py`: Feature engineering and selection.
  - `technical_indicators.py`: Calculation of technical indicators.
  - `walk_forward_backtest.py`: The backtesting engine.
  - `reporting.py`: Generation of results matrices.
  - `config.py`: All configuration for the project.
- `data/`: Caching directory for downloaded data.
- `reports/`: Output directory for reports and plots.
  - `images/`: Output directory for plots.
- `requirements.txt`: Python dependencies.

## Models

The model portfolio has been expanded to include a wider range of classifiers:

- **RandomForestClassifier**
- **GradientBoostingClassifier**
- **LGBMClassifier**
- **DecisionTreeClassifier**
- **XGBClassifier**

## Backtesting Strategies

The backtesting framework now supports multiple strategies to evaluate the models:

- **Buy and Hold**: A baseline strategy that buys at the beginning of the period and holds until the end.
- **Trend Following**: A strategy that buys when the model predicts an uptrend and sells when a downtrend is predicted.
- **Buy and Hold with Parking**: A strategy that stays invested but moves to cash ("parks") when the model predicts a significant downturn.

## Workflow

The project follows a standard machine learning pipeline:

1.  **Data Collection**: Gathers S&P 500 price data, economic indicators, and sentiment data.
2.  **Preprocessing**: Cleans the data and creates granular target labels for future price movements.
3.  **Feature Engineering**: Creates new features from the data, such as lagged features and rolling statistics.
4.  **Walk-Forward Backtesting**: Evaluates each model with each strategy using a robust walk-forward methodology.
5.  **Reporting**: Generates a results matrix comparing the performance of all model-strategy combinations.

## How to Run

1.  Install the dependencies: `pip install -r requirements.txt`
2.  Run the main pipeline: `python -m src.main`

The results will be saved in the `reports/` directory.
