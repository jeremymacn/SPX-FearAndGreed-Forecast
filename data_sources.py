import pandas as pd
import requests
import pandas_datareader.data as web
import yfinance as yf
import logging

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
    try:
        sp500 = yf.Ticker("^GSPC")
        return sp500.history(start=start_date, end=end_date)
    except Exception as e:
        logging.error(f"Error fetching S&P 500 data: {e}")
        return pd.DataFrame()
