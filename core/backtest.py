import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier
from .data import get_stock_data
from .model import train_model

def backtest_simple(stock_list, period="5y"):
    """Run a simple strategy backtest over provided stocks and return cumulative portfolio growth and diagnostics.

    Returns:
        growth_series (pd.Series): cumulative returns series
        diagnostics (dict): {'test_slice_length': int, 'predicted_signal_counts': {'BUY':n,'HOLD':n,'SELL':n}}
    """
    portfolio_returns = pd.Series(dtype=float)
    overall_diagnostics = {"test_slice_length": 0, "predicted_signal_counts": {"BUY": 0, "HOLD": 0, "SELL": 0}}

    for stock in stock_list:
        data = get_stock_data(stock, period=period)
        if data is None or data.empty:
            continue

        # split into train/test to avoid lookahead: train on first 80%, trade on last 20%
        if len(data) < 40:
            # not enough data to split meaningfully
            continue
        split = int(len(data) * 0.8)
        train = data.iloc[:split]
        test = data.iloc[split:].copy()

        features = ["RSI", "EMA_10", "EMA_20", "MACD"]
        label_mapping = {"SELL": 0, "HOLD": 1, "BUY": 2}

        X_train = train[features].astype(float)
        y_train = train["Signal"].map(label_mapping)
        model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", verbosity=0)
        model.fit(X_train, y_train)

        preds = model.predict(test[features])
        test["Pred_Signal"] = preds

        # accumulate diagnostic counts for this test slice
        overall_diagnostics["test_slice_length"] += len(test)
        overall_diagnostics["predicted_signal_counts"]["BUY"] += int((preds == 2).sum())
        overall_diagnostics["predicted_signal_counts"]["HOLD"] += int((preds == 1).sum())
        overall_diagnostics["predicted_signal_counts"]["SELL"] += int((preds == 0).sum())

        test["Strategy_Returns"] = 0.0

        # compute returns using forward 15-day exit inside the test period
        for i in range(len(test) - 15):
            if test["Pred_Signal"].iloc[i] == 2:
                test.iat[i, test.columns.get_loc("Strategy_Returns")] = (
                    (test["Close"].iloc[i + 15] - test["Close"].iloc[i]) / test["Close"].iloc[i]
                )
            elif test["Pred_Signal"].iloc[i] == 0:
                test.iat[i, test.columns.get_loc("Strategy_Returns")] = (
                    (test["Close"].iloc[i] - test["Close"].iloc[i + 15]) / test["Close"].iloc[i]
                )

        portfolio_returns = portfolio_returns.add(test["Strategy_Returns"].reindex(portfolio_returns.index).fillna(0), fill_value=0)

    return (1 + portfolio_returns.fillna(0)).cumprod() * 100, overall_diagnostics

def backtest_realistic(stock_list, initial_capital=100000, position_size=0.2, cost=0.002, period="5y"):
    """Run a more realistic backtest (position sizing, costs) and return portfolio history, trades, and diagnostics.

    Returns:
        portfolio_df (pd.DataFrame), trades (list), diagnostics (dict)
    """
    capital = initial_capital
    portfolio_history = []
    trades = []
    overall_diagnostics = {"test_slice_length": 0, "predicted_signal_counts": {"BUY": 0, "HOLD": 0, "SELL": 0}}

    for stock in stock_list:
        data = get_stock_data(stock, period=period)
        if data is None or data.empty:
            continue

        # split into train/test to avoid lookahead
        if len(data) < 40:
            continue
        split = int(len(data) * 0.8)
        train = data.iloc[:split]
        test = data.iloc[split:].copy().reset_index()

        features = ["RSI", "EMA_10", "EMA_20", "MACD"]
        label_mapping = {"SELL": 0, "HOLD": 1, "BUY": 2}

        X_train = train[features].astype(float)
        y_train = train["Signal"].map(label_mapping)
        model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", verbosity=0)
        model.fit(X_train, y_train)

        preds = model.predict(test[features])
        test["Pred_Signal"] = preds

        overall_diagnostics["test_slice_length"] += len(test)
        overall_diagnostics["predicted_signal_counts"]["BUY"] += int((preds == 2).sum())
        overall_diagnostics["predicted_signal_counts"]["HOLD"] += int((preds == 1).sum())
        overall_diagnostics["predicted_signal_counts"]["SELL"] += int((preds == 0).sum())

        i = 0
        while i < len(test) - 15:
            signal = test["Pred_Signal"].iloc[i]
            entry_price = test["Close"].iloc[i]
            exit_price = test["Close"].iloc[i + 15]
            if signal in [0, 2]:
                trade_capital = capital * position_size
                if signal == 2:
                    gross_return = (exit_price - entry_price) / entry_price
                    signal_str = "BUY"
                else:
                    gross_return = (entry_price - exit_price) / entry_price
                    signal_str = "SELL"
                net_return = gross_return - (2 * cost)
                capital += trade_capital * net_return
                trades.append({
                    "Stock": stock,
                    "Date": test.loc[i, "Date"],
                    "Signal": signal_str,
                    "Entry": round(entry_price, 2),
                    "Exit": round(exit_price, 2),
                    "Return%": round(net_return * 100, 2),
                })
                i += 15
            else:
                i += 1
            # append current capital snapshot
            portfolio_history.append({"Date": test.loc[min(i, len(test)-1), "Date"], "Capital": capital})

    return pd.DataFrame(portfolio_history).set_index("Date"), trades, overall_diagnostics

def get_nifty50_benchmark(period="5y"):
    """Download NIFTY50 (^NSEI) benchmark for the given period and return cumulative returns scaled to 100.

    Args:
        period (str): yfinance period string, e.g. '1y', '5y', 'max'. Defaults to '5y'.
    """
    nifty = yf.download("^NSEI", period=period, progress=False)
    if nifty.empty:
        return None
    nifty_returns = nifty["Close"].pct_change().fillna(0)
    return (1 + nifty_returns).cumprod() * 100
