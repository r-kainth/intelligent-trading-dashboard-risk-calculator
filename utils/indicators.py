import pandas as pd
import numpy as np

def add_technical_indicators(df):
    data = df.copy()
    
    # 1. Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # 2. Bollinger Bands (Volatility)
    data['BB_std'] = data['Close'].rolling(window=20).std()
    data['BB_upper'] = data['SMA_20'] + (data['BB_std'] * 2)
    data['BB_lower'] = data['SMA_20'] - (data['BB_std'] * 2)
    
    # 3. RSI (Momentum)
    delta = data['Close'].diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gain = gains.rolling(window=14, min_periods=1).mean()
    avg_loss = losses.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. MACD (Trend Following)
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    # 5. Stochastic Oscillator (Momentum)
    low_14 = data['Low'].rolling(window=14).min()
    high_14 = data['High'].rolling(window=14).max()
    data['Stoch_K'] = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
    data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()
    
    return data

def get_rsi_metrics(rsi_value):
    """Evaluates RSI and returns the State, Color, and Arrow direction."""
    # 1. Handle the NaN case immediately (Early Return)
    if pd.isna(rsi_value):
        return 0, "Neutral", "primary", "off"
    
    # 2. Evaluate from top to bottom
    if rsi_value > 70:
        return rsi_value, "Overbought", "red", "up"
    if rsi_value >= 60:
        return rsi_value, "Approaching Overbought", "red", "up"
    if rsi_value < 30:
        return rsi_value, "Oversold", "green", "down"
    if rsi_value <= 40:
        return rsi_value, "Approaching Oversold", "green", "down"
        
    # 3. Default fallback
    return rsi_value, "Neutral", "yellow", "off"