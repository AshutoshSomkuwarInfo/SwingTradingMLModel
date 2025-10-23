import pandas as pd
import yfinance as yf
from .data import get_stock_data
from .model import train_model

def backtest_simple(stock_list):
    portfolio_returns = pd.Series(dtype=float)
    for stock in stock_list:
        data = get_stock_data(stock)
        if data is None or data.empty:
            continue
        model = train_model(data)
        preds = model.predict(data[["RSI","EMA_10","EMA_20","MACD"]])
        data["Pred_Signal"] = preds
        data["Strategy_Returns"] = 0.0
        for i in range(len(data)-15):
            if data["Pred_Signal"].iloc[i] == 2:
                data.at[data.index[i], "Strategy_Returns"] = (data["Close"].iloc[i+15]-data["Close"].iloc[i])/data["Close"].iloc[i]
            elif data["Pred_Signal"].iloc[i] == 0:
                data.at[data.index[i], "Strategy_Returns"] = (data["Close"].iloc[i]-data["Close"].iloc[i+15])/data["Close"].iloc[i]
        portfolio_returns = portfolio_returns.add(data["Strategy_Returns"], fill_value=0)
    return (1 + portfolio_returns.fillna(0)).cumprod() * 100

def backtest_realistic(stock_list, initial_capital=100000, position_size=0.2, cost=0.002):
    capital = initial_capital
    portfolio_history = []
    trades = []
    for stock in stock_list:
        data = get_stock_data(stock)
        if data is None or data.empty:
            continue
        model = train_model(data)
        preds = model.predict(data[["RSI","EMA_10","EMA_20","MACD"]])
        data["Pred_Signal"] = preds
        i = 0
        while i < len(data)-15:
            signal = data["Pred_Signal"].iloc[i]
            entry_price = data["Close"].iloc[i]
            exit_price = data["Close"].iloc[i+15]
            if signal in [0,2]:
                trade_capital = capital * position_size
                if signal == 2:
                    gross_return = (exit_price - entry_price) / entry_price
                    signal_str = "BUY"
                else:
                    gross_return = (entry_price - exit_price) / entry_price
                    signal_str = "SELL"
                net_return = gross_return - (2 * cost)
                capital += trade_capital * net_return
                trades.append({"Stock": stock, "Date": data.index[i], "Signal": signal_str, "Entry": round(entry_price,2),
                               "Exit": round(exit_price,2), "Return%": round(net_return*100,2)})
                i += 15
            else:
                i += 1
            portfolio_history.append({"Date": data.index[i], "Capital": capital})
    return pd.DataFrame(portfolio_history).set_index("Date"), trades

def get_nifty50_benchmark():
    nifty = yf.download("^NSEI", period="2y", progress=False)
    if nifty.empty:
        return None
    nifty_returns = nifty["Close"].pct_change().fillna(0)
    return (1 + nifty_returns).cumprod() * 100
