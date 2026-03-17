import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_candlestick_chart(df, ticker):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, 
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=(f"{ticker.upper()} Price & Moving Averages", "Volume", "MACD", "RSI & Stochastic")
    )

    # ROW 1: Price, SMA, and Bollinger Bands (Group: Price & Trend)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
        name="Price", legendgroup="Trend", legendgrouptitle_text="Price & Trend"
    ), row=1, col=1)
    
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='blue', width=1), name='SMA 20', legendgroup="Trend"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='SMA 50', legendgroup="Trend"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], line=dict(color='gray', width=1, dash='dot'), name='Upper BB', legendgroup="Trend"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], line=dict(color='gray', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', name='Lower BB', legendgroup="Trend"), row=1, col=1)

    # ROW 2: Volume (Usually doesn't need a legend, so we keep it hidden)
    colors = ['green' if row['Close'] >= row['Open'] else 'red' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name="Volume", showlegend=False), row=2, col=1)

    # ROW 3: MACD (Group: MACD)
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue', width=1), name='MACD Line', legendgroup="MACD", legendgrouptitle_text="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='orange', width=1), name='Signal Line', legendgroup="MACD"), row=3, col=1)
        macd_colors = ['green' if val >= 0 else 'red' for val in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=macd_colors, name='Histogram', legendgroup="MACD"), row=3, col=1)

    # ROW 4: RSI and Stochastic (Group: Momentum)
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1), name='RSI', legendgroup="Momentum", legendgrouptitle_text="Momentum"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], line=dict(color='cyan', width=1), name='Stoch %K', legendgroup="Momentum"), row=4, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="gray", row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="gray", row=4, col=1)

    # Clean up the layout
    fig.update_layout(
        template="plotly_dark",
        height=900,
        showlegend=True,
        # Let Plotly handle the legend vertically on the right side by default!
        margin=dict(l=20, r=20, t=60, b=20), 
    )
    fig.update_xaxes(rangeslider_visible=False)

    return fig