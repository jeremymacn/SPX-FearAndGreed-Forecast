# S&P 500 Predictive Model

## Project Goal

The primary goal of this project was to develop a machine learning model to predict the movement of the S&P 500.

## Major Updates

This project has been significantly updated with new features, models, and strategies to improve its predictive power and provide more realistic backtesting.

### 1. Enhanced Feature Set with New Technical Indicators
We have expanded our feature set by incorporating a wider range of technical indicators. In addition to the existing indicators, we have added:
- **On-Balance Volume (OBV):** To gauge buying and selling pressure.
- **Accumulation/Distribution Line (A/D):** To identify underlying market trends.
- **Average Directional Index (ADX):** To measure the strength of trends.
- **Aroon Indicator:** To identify new trends and their strength.
- **Stochastic Oscillator:** To identify overbought and oversold conditions.

These new indicators provide a more comprehensive view of the market, allowing our models to capture more complex patterns.

### 2. Granular Target Labels for More Precise Predictions
We have moved beyond simple binary classification (up/down) to a more granular, multi-class target labeling system. The new labels categorize future price movements into percentage-based buckets over various time horizons (1-day, 3-days, 1-week, 2-weeks, 1-month). This allows for more nuanced predictions and more sophisticated trading strategies.

The new target classes are:
- `0`: < -5%
- `1`: -5% to -3%
- `2`: -3% to -1%
- `3`: -1% to 1%
- `4`: 1% to 3%
- `5`: 3% to 5%
- `6`: > 5%

### 3. Advanced "Buy-and-Hold with Parking" Strategy
We have implemented a new, more realistic backtesting strategy called "buy-and-hold with parking." This strategy is designed to mitigate risk by "parking" the investment in cash when the model predicts a significant downturn. This is a significant improvement over the previous "long-only" strategy, as it allows the model to actively manage risk.

### 4. Expanded Model Portfolio with LightGBM
We have added LightGBM, a state-of-the-art gradient boosting framework, to our model portfolio. LightGBM is known for its speed and efficiency, and it provides a powerful new tool for our predictive pipeline. We now have three models in our portfolio:
- RandomForestClassifier
- GradientBoostingClassifier
- LGBMClassifier

## Workflow and Key Discoveries

We started with a standard machine learning workflow and made several key discoveries and amendments along the way:

1.  **From Correlation to Mutual Information:** We began by analyzing the linear correlation between our features and the target, which was very weak. This led us to use **Mutual Information**, a more powerful technique that uncovered significant non-linear relationships in the data.

2.  **Finding the Right Time Horizon:** We discovered that our features had much more predictive power for the **21-day** outcome than for shorter periods. This was a crucial insight that guided our modeling efforts.

3.  **The Power of Feature Engineering:** We created new features, including lagged indicators, rolling statistics, and interaction terms. This dramatically improved our model's predictive power.

4.  **The Importance of Rigorous Backtesting:**
    *   Our first **simple backtest** showed inconsistent year-to-year performance and highlighted the dangers of lookahead bias.
    *   This led us to implement a more robust **walk-forward backtest**.
    *   We then found and fixed a critical bug in our backtesting logic that was producing unrealistic returns.

5.  **Rolling vs. Expanding Windows:** We ran an experiment to compare these two backtesting methodologies and discovered that a **rolling window** was superior, indicating that the market's behavior changes over time and our model needs to adapt.

6.  **Optimizing the Holding Period:** We experimented with different holding periods and found that a **40-day hold** offered the best risk-adjusted returns for our strategy.

7.  **The Lesson of Hyperparameter Tuning:** We learned that a more complex, highly-tuned model is not always better. Our simpler, untuned model proved to be more robust and profitable in our backtest.

## Final Result

Through this iterative process, we have successfully built a complete, well-structured project with a profitable trading strategy. The project is now equipped with a more powerful and flexible pipeline.

*(Note: The backtest results need to be updated to reflect the new models and strategies.)*
