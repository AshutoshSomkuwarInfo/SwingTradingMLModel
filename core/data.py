import yfinance as yf
import ta
import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    data = yf.download(ticker, period="2y", progress=False)
    if data.empty:
        return None

    # Calculate technical indicators
    close_prices = data[("Close", ticker)]  # This is a pandas Series already
    data["RSI"] = ta.momentum.RSIIndicator(close_prices).rsi().squeeze()
    data["EMA_10"] = ta.trend.EMAIndicator(close_prices, window=10).ema_indicator().squeeze()
    data["EMA_20"] = ta.trend.EMAIndicator(close_prices, window=20).ema_indicator().squeeze()
    macd = ta.trend.MACD(close_prices)
    data["MACD"] = macd.macd().squeeze()
    data["MACD_Signal"] = macd.macd_signal().squeeze()

    data = data.dropna()

    # Create labels for 15-day future returns for swing trading signals
    data["Future_Close"] = close_prices.shift(-15)
    data["Return_15d"] = (data["Future_Close"] - close_prices) / close_prices * 100
    data["Signal"] = data["Return_15d"].apply(lambda x: "BUY" if x > 5 else ("SELL" if x < -5 else "HOLD"))
    data = data.dropna()
    print(data)
    return data