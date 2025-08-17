import pandas as pd
import requests
import pandas_datareader.data as web
import yfinance as yf
import logging
import time

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
        logging.error(f"Error fetching Fear & Greed Index: {e}")
        return pd.DataFrame()

def get_fred_data(series_id, start_date, end_date):
    """
    Retrieves data from FRED using pandas-datareader.
    """
    try:
        return web.DataReader(series_id, 'fred', start_date, end_date)
    except Exception as e:
        logging.error(f"Error fetching {series_id} from FRED: {e}")
        return pd.DataFrame()

def get_sp500_data(start_date, end_date):
    """
    Retrieves S&P 500 historical data from Yahoo Finance.
    """
    return get_yfinance_data("^GSPC", start_date, end_date)

def get_yfinance_data(ticker, start_date, end_date, retries=3, backoff_factor=0.5):
    """
    Retrieves historical data for a given ticker from Yahoo Finance with retry logic.
    """
    for i in range(retries):
        try:
            data = yf.Ticker(ticker)
            history = data.history(start=start_date, end=end_date)
            if not history.empty:
                return history
        except Exception as e:
            logging.warning(f"Error fetching {ticker} (attempt {i+1}/{retries}): {e}")
            if i < retries - 1:
                sleep_time = backoff_factor * (2 ** i)
                logging.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logging.error(f"Failed to fetch {ticker} after {retries} attempts.")
                return pd.DataFrame()
    return pd.DataFrame()

def get_sp500_tickers():
    """
    Retrieves the list of S&P 500 tickers from Wikipedia.
    """
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        table = pd.read_html(url)
        tickers = table[0]['Symbol'].tolist()
        return tickers
    except Exception as e:
        logging.error(f"Error fetching S&P 500 tickers: {e}")
        return []
