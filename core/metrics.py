import pandas as pd
import numpy as np

def calculate_metrics(returns_series, freq=252):
    if returns_series is None or len(returns_series) == 0:
        return {"Total Return (%)":0,"CAGR (%)":0,"Sharpe Ratio":0,"Max Drawdown (%)":0}
    total_return = (returns_series.add(1).prod() - 1) * 100
    periods = len(returns_series)
    cagr = ((returns_series.add(1).prod()) ** (freq / periods) - 1) * 100 if periods > 0 else 0
    sharpe = np.sqrt(freq) * returns_series.mean() / returns_series.std() if returns_series.std()!=0 else 0
    cumulative = (1 + returns_series).cumprod()
    drawdown = (cumulative / cumulative.cummax() - 1).min() * 100
    return {"Total Return (%)":round(total_return,2),"CAGR (%)":round(cagr,2),"Sharpe Ratio":round(sharpe,2),"Max Drawdown (%)":round(drawdown,2)}

def analyze_trades(trades):
    df = pd.DataFrame(trades)
    if df.empty:
        return {"Total Trades":0}, df
    df["Return%"] = pd.to_numeric(df["Return%"], errors="coerce")
    df = df.dropna(subset=["Return%"])
    total_trades = len(df)
    buy_trades = len(df[df["Signal"]=="BUY"])
    sell_trades = len(df[df["Signal"]=="SELL"])
    win_rate = round((len(df[df["Return%"]>0]) / total_trades) * 100, 2) if total_trades>0 else 0
    avg_gain = round(df[df["Return%"]>0]["Return%"].mean(), 2) if len(df[df["Return%"]>0])>0 else 0
    avg_loss = round(df[df["Return%"]<0]["Return%"].mean(), 2) if len(df[df["Return%"]<0])>0 else 0
    best_trade = round(df["Return%"].max(), 2)
    worst_trade = round(df["Return%"].min(), 2)
    summary = {"Total Trades": total_trades,"BUY Trades": buy_trades,"SELL Trades": sell_trades,"Win Rate (%)": win_rate,
               "Avg Gain (%)": avg_gain,"Avg Loss (%)": avg_loss,"Best Trade (%)": best_trade,"Worst Trade (%)": worst_trade}
    return summary, df

def clean_series(s):
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    if isinstance(s, pd.DataFrame):
        s = s.select_dtypes(include=[np.number]).iloc[:,0]
    s = pd.Series(s.values, index=s.index)
    s = s.dropna()
    return s.astype(float)
