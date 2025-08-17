# S&P 500 Predictive Indicators Checklist

This checklist will be used to track the implementation of data retrieval for various indicators that may have predictive power on the S&P 500.

## Economic Indicators


- [x] GDP (Gross Domestic Product)
- [x] Inflation (CPI, PPI, PCE)
- [x] Labor Market Data (Nonfarm Payrolls, Unemployment Rate)
- [x] Durable Goods Orders
- [x] Yield Curve (e.g., 10-Year vs. 2-Year Treasury Yield Spread)
- [x] Housing Starts
- [x] Retail Sales
- [x] Industrial Production
- [x] Consumer Confidence Index (CCI)
- [x] CBOE Volatility Index (VIX)
- [x] Federal Funds Rate
- [x] M1, M2 Money Supply

## Technical Indicators

- [x] VVIX
- [x] Number of S&P 500 stocks hitting new highs/lows
- [x] Moving Averages (e.g., 50-day, 200-day SMA)
- [x] Relative Strength Index (RSI)
- [x] Moving Average Convergence Divergence (MACD)
- [x] Bollinger Bands
- [x] On-Balance Volume (OBV)
- [x] Put/Call Ratio

## Sentiment Indicators

- [x] Fear & Greed Index
- [ ] AAII Sentiment Survey
- [x] Commitments of Traders (COT) Report

## Valuation Indicators

- [x] P/E Ratio (Price-to-Earnings)
- [x] Shiller P/E Ratio (CAPE)
- [x] Dividend Yield
- [x] Market Capitalization to GDP Ratio (Buffett Indicator)

## Skipped Indicators

- **AAII Sentiment Survey**: Requires manual download of an Excel file from the AAII website. No direct API or reliable library found.
- **Commitments of Traders (COT) Report**: The `cot_reports` library was not working reliably. Further investigation into the CFTC API is needed for a stable solution.
- **Purchasing Managers' Index (PMI)**: The FRED series `NAPM` and `MANPMI` are no longer available. Scraping from YCharts and ISM website was unreliable.
- **Shiller P/E Ratio (CAPE)**: Scraping from multpl.com was unreliable due to dynamic content loading. The Quandl API for this data also failed.
- **Leverage among S&P 500 companies**: No readily available free data source found.
- **Growth among S&P 500 industries**: No readily available free data source found.