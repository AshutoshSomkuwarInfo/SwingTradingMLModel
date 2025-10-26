import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_chart(data, ticker):
    # Check which features are available
    has_extended_features = all(col in data.columns for col in ["BB_Upper", "BB_Lower", "Stoch", "ATR"])
    
    if has_extended_features:
        # Enhanced chart with new indicators
        fig = make_subplots(
            rows=5, cols=1, 
            shared_xaxes=True, 
            row_heights=[0.35, 0.15, 0.15, 0.15, 0.15], 
            vertical_spacing=0.04,
            subplot_titles=("Price with Bollinger Bands", "RSI (14)", "MACD", "Stochastic Oscillator", "ATR")
        )
        
        # Row 1: Price with Bollinger Bands
        fig.add_trace(go.Candlestick(
            x=data.index, 
            open=data["Open"], 
            high=data["High"], 
            low=data["Low"], 
            close=data["Close"], 
            name="Candlesticks"
        ), row=1, col=1)
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data["BB_Upper"], 
            line=dict(color="gray", width=1, dash="dash"), 
            name="BB Upper"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data["BB_Lower"], 
            line=dict(color="gray", width=1, dash="dash"), 
            fill="tonexty",
            fillcolor="rgba(128,128,128,0.1)",
            name="BB Lower"
        ), row=1, col=1)
        
        # EMA lines on price chart
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data["EMA_10"], 
            line=dict(color="orange", width=1), 
            name="EMA 10"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data["EMA_20"], 
            line=dict(color="blue", width=1), 
            name="EMA 20"
        ), row=1, col=1)
        
        # Row 2: RSI
        fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], line=dict(color="blue", width=2), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
        
        # Row 3: MACD
        fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], line=dict(color="purple", width=2), name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["MACD_Signal"], line=dict(color="orange", width=2), name="Signal"), row=3, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)
        
        # Row 4: Stochastic Oscillator
        fig.add_trace(go.Scatter(x=data.index, y=data["Stoch"], line=dict(color="green", width=2), name="Stoch"), row=4, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=4, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=4, col=1)
        
        # Row 5: ATR (Average True Range)
        fig.add_trace(go.Scatter(x=data.index, y=data["ATR"], line=dict(color="red", width=2), fill="tozeroy", 
                                  fillcolor="rgba(255,0,0,0.2)", name="ATR"), row=5, col=1)
        
        fig.update_layout(
            xaxis_rangeslider_visible=False, 
            showlegend=False, 
            height=1200
        )
        
    else:
        # Original chart for backward compatibility
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6,0.2,0.2], vertical_spacing=0.05,
                            subplot_titles=("Price","RSI (14)","MACD"))
        fig.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"], name="Candlesticks"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], line=dict(color="blue"), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], line=dict(color="purple"), name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["MACD_Signal"], line=dict(color="orange"), name="Signal"), row=3, col=1)
        fig.update_layout(xaxis_rangeslider_visible=False, showlegend=False, height=800)
    
    return fig
