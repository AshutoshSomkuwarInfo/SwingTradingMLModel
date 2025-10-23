import yfinance as yf
import ta
import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    try:
        data = yf.download(ticker, period="2y", progress=False)
    except Exception as e:
        try:
            st.session_state["last_data_error"] = f"Error downloading {ticker}: {e}"
        except Exception:
            pass
        return None

    if data is None or data.empty:
        try:
            st.session_state["last_data_error"] = f"No data returned for {ticker} (empty dataframe)"
        except Exception:
            pass
        return None

    # If yfinance returned multi-index columns (e.g. (Ticker, Field)), flatten to field names
    if isinstance(data.columns, pd.MultiIndex):
        # prefer any element that matches a known field name (Open/High/Low/Close/Adj Close/Volume)
        known_fields = {"open", "high", "low", "close", "adj close", "volume"}
        new_cols = []
        for col in data.columns:
            field = None
            if isinstance(col, tuple):
                # try to find a tuple element that looks like a field name
                for part in col:
                    try:
                        if isinstance(part, str) and part.strip().lower() in known_fields:
                            field = part.strip()
                            break
                    except Exception:
                        continue
                # if none matched, pick the element that doesn't look like a ticker (heuristic)
                if field is None:
                    # choose the first element that does not contain a dot/uppercase ticker-like pattern
                    picked = None
                    for part in col:
                        if isinstance(part, str) and ("." not in part and not part.isupper()):
                            picked = part
                            break
                    if picked is None:
                        # fallback to first non-empty string
                        for part in col:
                            if isinstance(part, str) and part not in ("", None):
                                picked = part
                                break
                    field = picked if picked is not None else col[0]
            else:
                field = col
            new_cols.append(field)
        data.columns = new_cols

    # Ensure price columns exist
    required = ["Open", "High", "Low", "Close"]
    if not all(col in data.columns for col in required):
        # Try to discover columns by substring (best-effort)
        found = {}
        for col in data.columns:
            name = str(col).lower()
            for req in required:
                if req.lower() in name and req not in found:
                    found[req] = col
        if len(found) == 4:
            data = data.rename(columns={found[r]: r for r in required})
        else:
            # missing price columns — cannot proceed
            try:
                st.session_state["last_data_error"] = f"Missing price columns for {ticker}. Available columns: {list(data.columns)}"
            except Exception:
                pass
            return None

    # Work with Close series
    close_prices = data["Close"]

    # Create labels for 15-day future returns for swing trading signals BEFORE dropping NaNs
    data["Future_Close"] = close_prices.shift(-15)
    data["Return_15d"] = (data["Future_Close"] - close_prices) / close_prices * 100
    data["Signal"] = data["Return_15d"].apply(lambda x: "BUY" if x > 5 else ("SELL" if x < -5 else "HOLD"))

    # Calculate technical indicators (do not squeeze — leave as Series)
    data["RSI"] = ta.momentum.RSIIndicator(close_prices).rsi()
    data["EMA_10"] = ta.trend.EMAIndicator(close_prices, window=10).ema_indicator()
    data["EMA_20"] = ta.trend.EMAIndicator(close_prices, window=20).ema_indicator()
    macd = ta.trend.MACD(close_prices)
    data["MACD"] = macd.macd()
    data["MACD_Signal"] = macd.macd_signal()

    # Drop rows with NaNs introduced by indicators or shift
    data = data.dropna()

    # clear last error when data successfully prepared
    try:
        st.session_state["last_data_error"] = None
    except Exception:
        pass

    return data