import pandas as pd
import requests

import pandas_datareader.data as web
import datetime
import yfinance as yf

# --- SENTIMENT INDICATORS ---

def get_fear_and_greed_index():
    """
    Retrieves the Fear & Greed Index data and returns it as a pandas DataFrame.
    """
    url = "https://api.alternative.me/fng/?limit=365"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()['data']
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('timestamp')
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Fear & Greed Index: {e}")
        return pd.DataFrame()

# --- ECONOMIC INDICATORS ---

def get_fred_data(series_id, start_date, end_date):
    """
    Retrieves data from FRED using pandas-datareader.
    """
    try:
        return web.DataReader(series_id, 'fred', start_date, end_date)
    except Exception as e:
        print(f"Error fetching {series_id} from FRED: {e}")
        return pd.DataFrame()

# --- TECHNICAL INDICATORS ---

def get_sp500_data(start_date, end_date):
    """
    Retrieves S&P 500 historical data from Yahoo Finance.
    """
    try:
        sp500 = yf.Ticker("^GSPC")
        return sp500.history(start=start_date, end=end_date)
    except Exception as e:
        print(f"Error fetching S&P 500 data: {e}")
        return pd.DataFrame()

def calculate_moving_average(data, window):
    """
    Calculates the moving average for a given dataset.
    """
    return data['Close'].rolling(window=window).mean()

def calculate_rsi(data, window=14):
    """
    Calculates the Relative Strength Index (RSI).
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, slow=26, fast=12, signal=9):
    """
    Calculates the Moving Average Convergence Divergence (MACD).
    """
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    """
    Calculates Bollinger Bands.
    """
    sma = data['Close'].rolling(window=window).mean()
    std_dev = data['Close'].rolling(window=window).std()
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return upper_band, lower_band

def calculate_obv(data):
    """
    Calculates On-Balance Volume (OBV).
    """
    obv = (data['Volume'] * (~data['Close'].diff().le(0) * 2 - 1)).cumsum()
    return obv

def get_put_call_ratio():
    """
    Calculates the Put/Call ratio for SPY.
    """
    try:
        spy = yf.Ticker("SPY")
        options = spy.option_chain(spy.options[0])
        puts = options.puts
        calls = options.calls
        put_call_ratio = puts['openInterest'].sum() / calls['openInterest'].sum()
        return put_call_ratio
    except Exception as e:
        print(f"Error fetching Put/Call Ratio: {e}")
        return None

# --- VALUATION INDICATORS ---

def get_pe_ratio(ticker_symbol):
    """
    Gets the P/E ratio for a given ticker.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        return ticker.info['trailingPE']
    except Exception as e:
        print(f"Error fetching P/E ratio for {ticker_symbol}: {e}")
        return None

def get_dividend_yield(ticker_symbol):
    """
    Gets the dividend yield for a given ticker.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        return ticker.info['dividendYield']
    except Exception as e:
        print(f"Error fetching dividend yield for {ticker_symbol}: {e}")
        return None

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
        "GDP": "GDP",
        "CPI": "CPIAUCSL",
        "PPI": "PPIACO",
        "PCE": "PCEPI",
        "Nonfarm Payrolls": "PAYEMS",
        "Unemployment Rate": "UNRATE",
        "Durable Goods Orders": "DGORDER",
        "10-Year Treasury Yield": "DGS10",
        "2-Year Treasury Yield": "DGS2",
        "Housing Starts": "HOUST",
        "Retail Sales": "RSAFS",
        "Industrial Production": "INDPRO",
        "Consumer Confidence Index": "UMCSENT",
        "VIX": "VIXCLS",
        "Market Capitalization to GDP Ratio": "DDDM01USA156NWDB"
    }

    for name, series_id in economic_indicators.items():
        df = get_fred_data(series_id, start_date, end_date)
        if not df.empty:
            main_df = main_df.join(df, rsuffix=f"_{name}")

    # --- Add Sentiment Indicators ---
    fng_df = get_fear_and_greed_index()
    if not fng_df.empty:
        main_df = main_df.join(fng_df, rsuffix='_fng')

    # --- Add Valuation Indicators ---
    main_df['Put_Call_Ratio'] = get_put_call_ratio()
    main_df['PE_Ratio'] = get_pe_ratio("SPY")
    main_df['Dividend_Yield'] = get_dividend_yield("SPY")

    # Forward-fill missing values
    main_df.ffill(inplace=True)

    return main_df


