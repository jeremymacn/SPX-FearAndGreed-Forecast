import pandas as pd
import datetime
from .data_sources import get_sp500_data, get_fred_data, get_fear_and_greed_index, get_yfinance_data, get_sp500_tickers
from .technical_indicators import calculate_moving_average, calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_obv, calculate_ad, calculate_adx, calculate_aroon, calculate_stochastic_oscillator
import logging

# def calculate_new_high_lows(start_date, end_date, index_df):
#     """
#     Calculates the number of S&P 500 stocks hitting new highs or lows.
#     """
#     logging.info("Calculating new highs and lows for S&P 500 stocks...")
#     tickers = get_sp500_tickers()
#     if not tickers:
#         return pd.DataFrame()

#     periods = {
#         '1y': 252,
#         '6m': 126,
#         '3m': 63,
#         '1m': 21,
#         '2w': 10,
#         '1w': 5
#     }

#     high_low_df = pd.DataFrame(index=index_df.index)

#     for period_name, period_days in periods.items():
#         high_low_df[f'new_high_{period_name}'] = 0
#         high_low_df[f'new_low_{period_name}'] = 0

#     for i, ticker in enumerate(tickers):
#         logging.info(f"Processing ticker {i+1}/{len(tickers)}: {ticker}")
#         try:
#             ticker_data = get_yfinance_data(ticker, start_date, end_date)
#             if not ticker_data.empty:
#                 ticker_data.index = ticker_data.index.tz_localize(None)

#                 for period_name, period_days in periods.items():
#                     rolling_high = ticker_data['High'].rolling(window=period_days).max()
#                     rolling_low = ticker_data['Low'].rolling(window=period_days).min()

#                     new_highs = (ticker_data['High'] >= rolling_high).astype(int)
#                     new_lows = (ticker_data['Low'] <= rolling_low).astype(int)

#                     # Align with the main dataframe's index
#                     aligned_highs = new_highs.reindex(high_low_df.index).fillna(0)
#                     aligned_lows = new_lows.reindex(high_low_df.index).fillna(0)

#                     high_low_df[f'new_high_{period_name}'] += aligned_highs
#                     high_low_df[f'new_low_{period_name}'] += aligned_lows
#         except Exception as e:
#             logging.warning(f"Could not process ticker {ticker}: {e}")
#             continue

#     return high_low_df

def gather_all_data(years=10):
    """
    Gathers all the data into a single DataFrame.
    """
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365 * years)

    # Get S&P 500 data
    sp500_df = get_sp500_data(start_date, end_date)
    if sp500_df.empty:
        raise ConnectionError("Failed to download S&P 500 data. The pipeline cannot continue.")

    # Convert index to timezone-naive
    sp500_df.index = sp500_df.index.tz_localize(None)

    # Create the main DataFrame
    main_df = pd.DataFrame(index=sp500_df.index)
    main_df = main_df.join(sp500_df[['Open', 'High', 'Low', 'Close', 'Volume']])

    # --- Add Technical Indicators ---
    main_df['SMA_50'] = calculate_moving_average(main_df, 50)
    main_df['SMA_200'] = calculate_moving_average(main_df, 200)
    main_df['RSI'] = calculate_rsi(main_df)
    main_df['MACD'], main_df['MACD_Signal'] = calculate_macd(main_df)
    main_df['Bollinger_Upper'], main_df['Bollinger_Lower'] = calculate_bollinger_bands(main_df)
    main_df['OBV'] = calculate_obv(main_df)
    main_df['A/D'] = calculate_ad(main_df)
    main_df['ADX'] = calculate_adx(main_df)
    main_df['Aroon_Down'], main_df['Aroon_Up'] = calculate_aroon(main_df)
    main_df['Stoch_SlowK'], main_df['Stoch_SlowD'] = calculate_stochastic_oscillator(main_df)

    # --- Add Economic Indicators ---
    economic_indicators = {
        "10-Year Treasury Yield": "DGS10",
        "VIX": "VIXCLS",
        "Federal Funds Rate": "FEDFUNDS",
        "M1": "M1SL",
        "M2": "M2SL",
    }

    for name, series_id in economic_indicators.items():
        df = get_fred_data(series_id, start_date, end_date)
        if not df.empty:
            main_df = main_df.join(df)

    # --- Add Other Indicators ---
    vvix_df = get_yfinance_data('^VVIX', start_date, end_date)
    if not vvix_df.empty:
        vvix_df.index = vvix_df.index.tz_localize(None)
        main_df = main_df.join(vvix_df[['Close']].rename(columns={'Close': 'VVIX'}), rsuffix='_vvix')

    # --- Add New High/Low Indicators (Commented Out) ---
    # new_high_low_df = calculate_new_high_lows(start_date, end_date, main_df)
    # if not new_high_low_df.empty:
    #     main_df = main_df.join(new_high_low_df)

    # --- Add Sentiment Indicators ---
    fng_df = get_fear_and_greed_index()
    if not fng_df.empty:
        main_df = main_df.join(fng_df, rsuffix='_fng')

    # Forward-fill missing values
    main_df.ffill(inplace=True)

    return main_df
