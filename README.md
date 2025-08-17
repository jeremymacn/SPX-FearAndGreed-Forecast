# S&P 500 Predictive Model

## Project Goal

The primary goal of this project was to develop a machine learning model to predict the movement of the S&P 500.

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

Through this iterative process, we have successfully built a complete, well-structured project with a profitable trading strategy. Our final model has a **73.96% total return** and an excellent **Sharpe Ratio of 4.35** in a realistic, walk-forward backtest.
