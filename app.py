import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.data import get_stock_data
from core.model import train_model, predict_signal
from core.charts import plot_chart
from core.backtest import backtest_simple, backtest_realistic, get_nifty50_benchmark
from core.metrics import calculate_metrics, analyze_trades, clean_series

# Streamlit UI setup
st.set_page_config(layout="wide")
st.title("📊 Swing Trading Dashboard (15–20 Days)")

stock_list = ["INFY.NS","TCS.NS","RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","HINDUNILVR.NS"]

st.sidebar.header("Settings")
selected_stock = st.sidebar.selectbox("Choose a Stock", stock_list)

# Predictions for all stocks
signals = []
for stock in stock_list:
    data = get_stock_data(stock)
    if data is not None and not data.empty:
        model = train_model(data)
        signal = predict_signal(model, data)
        signals.append({"Stock": stock, "Signal": signal})
signals_df = pd.DataFrame(signals)

st.subheader("📌 Latest Predictions")
st.dataframe(signals_df)

# Technical Chart
st.subheader(f"📈 Technical Chart for {selected_stock}")
data = get_stock_data(selected_stock)
if data is not None:
    fig = plot_chart(data, selected_stock)
    st.plotly_chart(fig, use_container_width=True)

# Backtesting
st.subheader("💰 Portfolio Backtest vs NIFTY50")
simple_growth = backtest_simple(stock_list)
realistic_growth, trades = backtest_realistic(stock_list)
nifty_growth = get_nifty50_benchmark()

fig2 = go.Figure()
if simple_growth is not None and not simple_growth.empty:
    fig2.add_trace(go.Scatter(x=simple_growth.index, y=simple_growth, mode="lines", name="Simple Backtest"))
if realistic_growth is not None and not realistic_growth.empty:
    fig2.add_trace(go.Scatter(x=realistic_growth.index, y=realistic_growth["Capital"]/1000, mode="lines", name="Realistic Backtest (Capital/1000)"))
if nifty_growth is not None and not nifty_growth.empty:
    fig2.add_trace(go.Scatter(x=nifty_growth.index, y=nifty_growth, mode="lines", name="NIFTY50 Benchmark"))
fig2.update_layout(title="Portfolio Growth Comparison", xaxis_title="Date", yaxis_title="Value")
st.plotly_chart(fig2, use_container_width=True)

# Performance metrics
st.subheader("📊 Performance Metrics")
simple_daily = clean_series(simple_growth.pct_change()) if simple_growth is not None else pd.Series(dtype=float)
realistic_daily = clean_series(realistic_growth["Capital"].pct_change()) if realistic_growth is not None else pd.Series(dtype=float)
nifty_daily = clean_series(nifty_growth.pct_change()) if nifty_growth is not None else pd.Series(dtype=float)

simple_metrics = calculate_metrics(simple_daily)
realistic_metrics = calculate_metrics(realistic_daily)
nifty_metrics = calculate_metrics(nifty_daily)
metrics_df = pd.DataFrame([simple_metrics, realistic_metrics, nifty_metrics],
                          index=["Simple Backtest", "Realistic Backtest", "NIFTY50"])
st.table(metrics_df)

# Trade analysis
st.subheader("📋 Trade Report (Realistic Backtest)")
trade_summary, trade_details = analyze_trades(trades)
st.json(trade_summary)
st.dataframe(trade_details)
