import yfinance as yf
import pandas as pd

def get_stock_data(ticker_symbol, time_period="1y", time_interval="1d"):
    """
    Fetches historical stock data.
    time_period: '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'
    time_interval: '1d' (daily), '1wk' (weekly), '1mo' (monthly)
    """
    stock = yf.Ticker(ticker_symbol)
    df = stock.history(period=time_period, interval=time_interval)
    return df