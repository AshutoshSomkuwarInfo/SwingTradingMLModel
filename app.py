import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.data import get_stock_data
from core.model import train_model, predict_signal, predict_latest_signal
from core.charts import plot_chart
# Using fixed backtest module with proper position tracking and no look-ahead bias
from core.backtest_fixed import backtest_simple_fixed as backtest_simple, backtest_realistic_fixed as backtest_realistic, get_nifty50_benchmark
from core.backtest_simple_signal import simple_signal_backtest, get_signal_summary
from core.metrics import calculate_metrics, analyze_trades, clean_series
from core.paper_trading import PaperTradingSystem
from datetime import datetime

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

# history length selector
history_period = st.sidebar.selectbox("History", ["1y", "2y", "5y", "max"], index=2)

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
        data = get_stock_data(stock, period=history_period)
        signal = None
        if data is not None and not data.empty:
            # train on historical data up to the last row and predict the latest row to avoid leakage
            try:
                signal = predict_latest_signal(data)
            except Exception:
                # fallback to previous behavior if helper fails
                try:
                    model, _ = train_model(data)
                    signal = predict_signal(model, data)
                except Exception:
                    signal = None
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
    data = get_stock_data(selected_stock, period=history_period)
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

# Simple Signal Backtest Section
st.header("🎯 Simple Signal Accuracy Test")
st.info("✅ This simple test directly follows your model's BUY/HOLD/SELL signals and measures accuracy vs actual results")

if st.button("🔄 Run Simple Signal Test"):
    with st.spinner("Running simple signal backtest..."):
        results = simple_signal_backtest([selected_stock], period=history_period)
        
        st.session_state['simple_signal_results'] = results
        
        # Display results
        st.subheader("📊 Signal Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Prediction Accuracy", f"{results['accuracy']:.2f}%")
        
        with col2:
            st.metric("Correct Predictions", results['correct_predictions'])
        
        with col3:
            st.metric("Wrong Predictions", results['wrong_predictions'])
        
        with col4:
            st.metric("Total Signals", results['total_signals'])
        
        st.divider()
        
        # Signal Breakdown
        st.subheader("📈 Signal Breakdown")
        signal_df = pd.DataFrame({
            'Signal Type': ['BUY', 'HOLD', 'SELL'],
            'Count': [results['buy_signals'], results['hold_signals'], results['sell_signals']]
        })
        st.bar_chart(signal_df.set_index('Signal Type'))
        
        st.divider()
        
        # Performance Summary
        st.subheader("💰 Performance Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Profit", f"₹{results['total_profit']:,.2f}")
            st.metric("Total Loss", f"₹{results['total_loss']:,.2f}")
            st.metric("Net P&L", f"₹{results['net_pnl']:,.2f}", 
                     f"{results['total_return_pct']:.2f}%")
        
        with col2:
            st.metric("Total Return", f"{results['total_return_pct']:.2f}%")
            st.metric("NIFTY50 Return", f"{results['nifty_return_pct']:.2f}%")
            st.metric("vs NIFTY50", f"{results['vs_nifty']:.2f}%")
        
        st.divider()
        
        # Trade Details
        if results['trade_details']:
            st.subheader("📋 Trade Details")
            trades_df = pd.DataFrame(results['trade_details'])
            st.dataframe(trades_df, use_container_width=True)
            
            # Download button
            csv = trades_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Trade Details",
                data=csv,
                file_name=f"signal_backtest_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No trades to display. HOLD signals don't generate trades.")
        
        # Insights
        st.divider()
        st.subheader("💡 Insights")
        
        if results['accuracy'] >= 60:
            st.success(f"✅ Great! Your model has {results['accuracy']:.2f}% accuracy")
        elif results['accuracy'] >= 50:
            st.info(f"⚡ Good! Your model has {results['accuracy']:.2f}% accuracy")
        else:
            st.warning(f"⚠️ Model accuracy is {results['accuracy']:.2f}%. Consider training with more data.")
        
        if results['vs_nifty'] > 0:
            st.success(f"✅ Outperformed NIFTY50 by {results['vs_nifty']:.2f}%")
        else:
            st.warning(f"⚠️ Underperformed NIFTY50 by {abs(results['vs_nifty']):.2f}%")
        
        st.info(f"📌 Model predicted {results['buy_signals']} BUY, {results['hold_signals']} HOLD, and {results['sell_signals']} SELL signals")

elif 'simple_signal_results' in st.session_state:
    # Show cached results
    results = st.session_state['simple_signal_results']
    
    st.info("👆 Click 'Run Simple Signal Test' to test signal accuracy")
    st.expander("📊 Previously Run Results", expanded=False).json(results)