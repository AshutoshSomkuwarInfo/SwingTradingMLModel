import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_chart(data, ticker):
    # display a cleaned ticker (hide .NS) in the chart titles
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
