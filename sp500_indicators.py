import pandas as pd
import datetime
from data_sources import get_sp500_data, get_fred_data, get_fear_and_greed_index
from technical_indicators import calculate_moving_average, calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_obv

def gather_all_data(years=10):
    """
    Gathers all the data into a single DataFrame.
    """
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365 * years)

    # Get S&P 500 data
    sp500_df = get_sp500_data(start_date, end_date)
    if sp500_df.empty:
        return pd.DataFrame()

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

    # --- Add Economic Indicators ---
    economic_indicators = {
        "10-Year Treasury Yield": "DGS10",
        "VIX": "VIXCLS",
    }

    for name, series_id in economic_indicators.items():
        df = get_fred_data(series_id, start_date, end_date)
        if not df.empty:
            main_df = main_df.join(df)

    # --- Add Sentiment Indicators ---
    fng_df = get_fear_and_greed_index()
    if not fng_df.empty:
        main_df = main_df.join(fng_df, rsuffix='_fng')

    # Forward-fill missing values
    main_df.ffill(inplace=True)

    return main_df
