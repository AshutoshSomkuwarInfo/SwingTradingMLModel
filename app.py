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

stock_list = [
    "ADANIPORTS.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS",
    "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFC.NS",
    "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS",
    "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
    "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBIN.NS", "SHREECEM.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
    "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "UPL.NS",
    "WIPRO.NS", "ZEEL.NS", "DIVISLAB.NS", "JSWSTEEL.NS",
    "BPCL.NS"
]


st.sidebar.header("Settings")
# show user-friendly labels (hide the .NS suffix) but keep underlying tickers for calculations
selected_stock = st.sidebar.selectbox("Choose a Stock", stock_list, format_func=lambda x: x.replace('.NS', ''))

# a cleaned display name for UI elements (no .NS)
display_stock = selected_stock.replace('.NS', '')

# Session cache for per-stock results to avoid recomputing on reruns
if "signals_cache" not in st.session_state:
    st.session_state["signals_cache"] = {}


def compute_signal_for_stock(stock: str):
    """Compute and cache signal for a single stock, with spinner."""
    if stock in st.session_state["signals_cache"]:
        return st.session_state["signals_cache"][stock]

    # use cleaned label in spinner
    label = stock.replace('.NS', '')
    with st.spinner(f"Loading signal for {label}..."):
        data = get_stock_data(stock)
        signal = None
        if data is not None and not data.empty:
            model = train_model(data)
            signal = predict_signal(model, data)
    st.session_state["signals_cache"][stock] = signal
    return signal


# Only compute signal for the currently selected stock (initially first in list)
current_signal = compute_signal_for_stock(selected_stock)
signals_df = pd.DataFrame([{"Stock": display_stock, "Signal": current_signal}])

st.subheader("📌 Latest Predictions")
st.dataframe(signals_df)

# Technical Chart
st.subheader(f"📈 Technical Chart for {display_stock}")
with st.spinner(f"Loading chart for {display_stock}..."):
    data = get_stock_data(selected_stock)
    if data is not None:
        fig = plot_chart(data, selected_stock)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Last 15 Days Data", expanded=True):
            st.write(data.head(15))
    else:
        # show last data error if available
        last_err = None
        try:
            last_err = st.session_state.get("last_data_error")
        except Exception:
            last_err = None
        if last_err:
            st.error(last_err)
        else:
            st.warning("No data available for the selected stock.")

# Backtesting
st.subheader("💰 Portfolio Backtest vs NIFTY50")
# Backtests can be expensive — run on demand
if "backtests" not in st.session_state:
    st.session_state["backtests"] = None

if st.button("Run Backtests"):
    with st.spinner("Running backtests (this may take a while)..."):
        simple_growth = backtest_simple(stock_list)
        realistic_growth, trades = backtest_realistic(stock_list)
        nifty_growth = get_nifty50_benchmark()
        st.session_state["backtests"] = (simple_growth, realistic_growth, nifty_growth, trades)

if st.session_state["backtests"] is None:
    st.info("Backtests not run. Click 'Run Backtests' to execute backtesting and fetch benchmark.")
    simple_growth = realistic_growth = nifty_growth = trades = None
else:
    simple_growth, realistic_growth, nifty_growth, trades = st.session_state["backtests"]

fig2 = go.Figure()
if simple_growth is not None and not getattr(simple_growth, "empty", False):
    fig2.add_trace(go.Scatter(x=simple_growth.index, y=simple_growth, mode="lines", name="Simple Backtest"))
if realistic_growth is not None and getattr(realistic_growth, "shape", (None,))[0] != 0:
    # realistic_growth expected to be a DataFrame with "Capital"
    if "Capital" in realistic_growth.columns:
        fig2.add_trace(go.Scatter(x=realistic_growth.index, y=realistic_growth["Capital"] / 1000, mode="lines", name="Realistic Backtest (Capital/1000)"))
if nifty_growth is not None and not getattr(nifty_growth, "empty", False):
    fig2.add_trace(go.Scatter(x=nifty_growth.index, y=nifty_growth, mode="lines", name="NIFTY50 Benchmark"))

if len(fig2.data) > 0:
    fig2.update_layout(title="Portfolio Growth Comparison", xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig2, use_container_width=True)

# Performance metrics
st.subheader("📊 Performance Metrics")
simple_daily = clean_series(simple_growth.pct_change()) if simple_growth is not None else pd.Series(dtype=float)
realistic_daily = clean_series(realistic_growth["Capital"].pct_change()) if (realistic_growth is not None and "Capital" in getattr(realistic_growth, 'columns', [])) else pd.Series(dtype=float)
nifty_daily = clean_series(nifty_growth.pct_change()) if nifty_growth is not None else pd.Series(dtype=float)

simple_metrics = calculate_metrics(simple_daily)
realistic_metrics = calculate_metrics(realistic_daily)
nifty_metrics = calculate_metrics(nifty_daily)
metrics_df = pd.DataFrame([simple_metrics, realistic_metrics, nifty_metrics],
                          index=["Simple Backtest", "Realistic Backtest", "NIFTY50"])
st.table(metrics_df)

# Trade analysis
st.subheader("📋 Trade Report (Realistic Backtest)")
if trades is not None:
    trade_summary, trade_details = analyze_trades(trades)
    st.json(trade_summary)
    st.dataframe(trade_details)
else:
    st.info("No trade data available. Run backtests to generate trade report.")
