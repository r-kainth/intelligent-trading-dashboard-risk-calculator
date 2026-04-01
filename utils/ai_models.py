from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import numpy as np

def train_and_predict(df):
    data = df.copy()
    data['Target'] = data['Close'].shift(-1)
    
    # Isolate the final row for our actual prediction
    today_features = data.iloc[-1:].drop(columns=['Target', 'Close', 'Open', 'High', 'Low', 'Volume'])
    
    data.dropna(inplace=True)
    
    X = data.drop(columns=['Target', 'Close', 'Open', 'High', 'Low', 'Volume'])
    y = data['Target']
    
    # Train/Test Split
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Calculate Accuracy (100% - Error%)
    test_predictions = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, test_predictions)
    accuracy = (1 - mape) * 100
    
    # Extract Feature Importances (What did the AI care about most?)
    importances = model.feature_importances_
    feature_names = X.columns
    # Create a dataframe of the top 10 features
    feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_df = feature_df.sort_values(by='Importance', ascending=False).head(10)
    
    # Predict the next candle
    model.fit(X, y)
    next_prediction = model.predict(today_features)[0]
    
    return next_prediction, accuracy, feature_df

def generate_analyst_briefing(analyzed_data, predicted_price):
    """
    Acts as a pseudo-LLM. Reads the technical indicators and generates 
    a human-readable Bull Case, Bear Case, and Final Verdict.
    """
    latest = analyzed_data.iloc[-1]
    close = latest['Close']
    
    bull_points = []
    bear_points = []
    
    # 1. Trend Analysis (Price vs SMAs)
    if close > latest['SMA_20'] and close > latest['SMA_50']:
        bull_points.append(f"Price is showing strong upward momentum, trading firmly above both the 20-day (${latest['SMA_20']:.2f}) and 50-day moving averages.")
    elif close < latest['SMA_20'] and close < latest['SMA_50']:
        bear_points.append(f"Stock is locked in a downtrend, facing heavy resistance below both key moving averages.")
    
    # 2. Momentum Analysis (MACD & RSI)
    if latest['MACD_Hist'] > 0 and latest['MACD_Hist'] > analyzed_data['MACD_Hist'].iloc[-2]:
        bull_points.append("MACD histogram is expanding positively, indicating accelerating buyer momentum.")
    elif latest['MACD_Hist'] < 0:
        bear_points.append("MACD is currently negative, suggesting bears remain in control of the short-term trend.")
        
    if latest['RSI'] < 35:
        bull_points.append("RSI is deeply oversold. A technical bounce is highly probable as sellers become exhausted.")
    elif latest['RSI'] > 65:
        bear_points.append("RSI is approaching overbought territory. Risk of a sudden pullback or consolidation is high.")
        
    # 3. Prediction Analysis (Safeguarded!)
    if predicted_price is not None:
        predicted_pct_change = ((predicted_price - close) / close) * 100
        if predicted_pct_change > 0:
            bull_points.append(f"Our Machine Learning model forecasts a bullish continuation with a target of ${predicted_price:.2f}.")
        else:
            bear_points.append(f"Our Machine Learning model detects weakness, projecting a near-term dip toward ${predicted_price:.2f}.")
    # If predicted_price is None, it simply skips adding an ML bullet point!

    # 4. Generate the Final Verdict
    bull_score = len(bull_points)
    bear_score = len(bear_points)
    
    if bull_score > bear_score + 1:
        verdict = "🟢 **BULLISH BIAS:** The technical structure heavily favors the upside. Pullbacks should be viewed as potential buying opportunities."
    elif bear_score > bull_score + 1:
        verdict = "🔴 **BEARISH BIAS:** The chart looks technically broken. Caution is advised, and long positions carry significant risk."
    else:
        verdict = "🟡 **NEUTRAL / MIXED:** The market is giving conflicting signals. Wait for a clearer breakout or breakdown before committing heavy capital."

    # Fallbacks just in case
    if not bull_points: bull_points.append("No significant bullish catalysts detected at this time.")
    if not bear_points: bear_points.append("No immediate bearish threats identified in the current setup.")

    return bull_points, bear_points, verdict