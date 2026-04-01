import pandas as pd
import numpy as np

def generate_ml_features(df):
    """
    Takes raw price data and engineers advanced mathematical features 
    specifically formatted for Machine Learning models.
    """
    # Create a copy so we don't mess up our charting data
    data = df.copy()

    # ==========================================
    # 1. TREND FEATURES
    # ==========================================
    # Exponential Moving Averages (More weight to recent prices)
    data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['EMA_21'] = data['Close'].ewm(span=21, adjust=False).mean()

    # Average Directional Index (ADX) - Measures the strength of a trend
    # Step 1: Calculate +DM, -DM, and True Range (TR)
    plus_dm = data['High'].diff()
    minus_dm = data['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    # Isolate true directional movement
    data['+DM'] = np.where((plus_dm > abs(minus_dm)), plus_dm, 0)
    data['-DM'] = np.where((abs(minus_dm) > plus_dm), abs(minus_dm), 0)
    
    # We use the ATR we calculate later in the Volatility section, so let's calculate TR here
    tr1 = pd.DataFrame(data['High'] - data['Low'])
    tr2 = pd.DataFrame(abs(data['High'] - data['Close'].shift(1)))
    tr3 = pd.DataFrame(abs(data['Low'] - data['Close'].shift(1)))
    frames = [tr1, tr2, tr3]
    data['TR'] = pd.concat(frames, axis=1, join='inner').max(axis=1)
    
    # Step 2: Smooth them over 14 days
    smoothed_plus_dm = data['+DM'].rolling(14).sum()
    smoothed_minus_dm = data['-DM'].rolling(14).sum()
    smoothed_tr = data['TR'].rolling(14).sum()
    
    # Step 3: Calculate +DI and -DI
    data['+DI_14'] = 100 * (smoothed_plus_dm / smoothed_tr)
    data['-DI_14'] = 100 * (smoothed_minus_dm / smoothed_tr)
    
    # Step 4: Calculate DX and ADX
    di_diff = abs(data['+DI_14'] - data['-DI_14'])
    di_sum = data['+DI_14'] + data['-DI_14']
    dx = 100 * (di_diff / di_sum)
    data['ADX_14'] = dx.rolling(14).mean()
    
    # Clean up the intermediate columns so we don't feed junk to the AI
    data.drop(columns=['+DM', '-DM', 'TR', '+DI_14', '-DI_14'], inplace=True)

    # ==========================================
    # 2. MOMENTUM FEATURES
    # ==========================================
    # Rate of Change (ROC) - How much has the price jumped in % over X days?
    data['ROC_5'] = data['Close'].pct_change(periods=5) * 100
    data['ROC_10'] = data['Close'].pct_change(periods=10) * 100

    # ==========================================
    # 3. VOLATILITY FEATURES
    # ==========================================
    # Average True Range (ATR) - The average dollar amount the stock moves per day
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    
    # Concatenate the three ranges and find the max to get the "True Range"
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    data['ATR_14'] = true_range.rolling(14).mean()

    # Bollinger %B (Where is the price relative to the bands? 0=Bottom, 1=Top)
    sma_20 = data['Close'].rolling(window=20).mean()
    std_20 = data['Close'].rolling(window=20).std()
    upper_bb = sma_20 + (std_20 * 2)
    lower_bb = sma_20 - (std_20 * 2)
    data['BB_PctB'] = (data['Close'] - lower_bb) / (upper_bb - lower_bb)

    # ==========================================
    # 4. VOLUME / LIQUIDITY FEATURES
    # ==========================================
    # On-Balance Volume (OBV) - Adds volume on green days, subtracts on red days
    obv_changes = np.where(data['Close'] > data['Close'].shift(1), data['Volume'], 
                  np.where(data['Close'] < data['Close'].shift(1), -data['Volume'], 0))
    data['OBV'] = pd.Series(obv_changes, index=data.index).cumsum()

    # Volume Weighted Average Price (VWAP) - The institutional benchmark
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()

    # ==========================================
    # 5. LAG FEATURES (THE MEMORY)
    # ==========================================
    # Daily Percentage Return
    data['Daily_Return'] = data['Close'].pct_change()
    
    # What did the stock do 1, 2, and 3 days ago? (Helps AI find repeating patterns)
    data['Lag_1_Return'] = data['Daily_Return'].shift(1)
    data['Lag_2_Return'] = data['Daily_Return'].shift(2)
    data['Lag_3_Return'] = data['Daily_Return'].shift(3)

    # ==========================================
    # CLEANUP
    # ==========================================
    # Machine Learning models CRASH if you feed them "NaN" (Not a Number) values.
    # Because a 20-day moving average needs 20 days to calculate, the first 19 days 
    # of our dataset will be NaN. We drop those rows entirely to protect the AI.
    data.dropna(inplace=True)

    return data