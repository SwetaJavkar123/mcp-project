import pandas as pd

def generate_signals(df, strategy_type="combined"):
    """
    Generate Buy/Sell/Hold signals based on technical indicators.
    :param df: DataFrame with technical indicators (RSI, MACD, Bollinger Bands, etc.)
    :param strategy_type: Type of strategy ("combined", "momentum", "mean_reversion", "trend_following")
    :return: DataFrame with Signal and Confidence columns
    """
    df = df.copy()
    df["Signal"] = "HOLD"
    df["Confidence"] = 0.0
    
    if strategy_type == "combined":
        df = _combined_strategy(df)
    elif strategy_type == "momentum":
        df = _momentum_strategy(df)
    elif strategy_type == "mean_reversion":
        df = _mean_reversion_strategy(df)
    elif strategy_type == "trend_following":
        df = _trend_following_strategy(df)
    
    return df

def _combined_strategy(df):
    """
    Combined strategy using RSI, MACD, and Bollinger Bands.
    - BUY: RSI < 30 AND price < Lower BB AND MACD > Signal
    - SELL: RSI > 70 AND price > Upper BB AND MACD < Signal
    """
    for i in range(1, len(df)):
        rsi = df["RSI"].iloc[i] if "RSI" in df.columns else 50
        macd = df["MACD"].iloc[i] if "MACD" in df.columns else 0
        macd_signal = df["MACD_Signal"].iloc[i] if "MACD_Signal" in df.columns else 0
        close = df["Close"].iloc[i]
        bb_upper = df["BB_Upper"].iloc[i] if "BB_Upper" in df.columns else close + 10
        bb_lower = df["BB_Lower"].iloc[i] if "BB_Lower" in df.columns else close - 10
        
        # Buy signal
        if rsi < 30 and close < bb_lower and macd > macd_signal:
            df.loc[df.index[i], "Signal"] = "BUY"
            df.loc[df.index[i], "Confidence"] = min(100, (30 - rsi) + (bb_lower - close) * 10)
        
        # Sell signal
        elif rsi > 70 and close > bb_upper and macd < macd_signal:
            df.loc[df.index[i], "Signal"] = "SELL"
            df.loc[df.index[i], "Confidence"] = min(100, (rsi - 70) + (close - bb_upper) * 10)
    
    return df

def _momentum_strategy(df):
    """
    Momentum strategy using MACD and RSI.
    - BUY: MACD > Signal AND RSI > 50
    - SELL: MACD < Signal AND RSI < 50
    """
    for i in range(1, len(df)):
        rsi = df["RSI"].iloc[i] if "RSI" in df.columns else 50
        macd = df["MACD"].iloc[i] if "MACD" in df.columns else 0
        macd_signal = df["MACD_Signal"].iloc[i] if "MACD_Signal" in df.columns else 0
        
        if macd > macd_signal and rsi > 50:
            df.loc[df.index[i], "Signal"] = "BUY"
            df.loc[df.index[i], "Confidence"] = min(100, (rsi - 50) + abs(macd - macd_signal) * 100)
        
        elif macd < macd_signal and rsi < 50:
            df.loc[df.index[i], "Signal"] = "SELL"
            df.loc[df.index[i], "Confidence"] = min(100, (50 - rsi) + abs(macd - macd_signal) * 100)
    
    return df

def _mean_reversion_strategy(df):
    """
    Mean reversion strategy using Bollinger Bands and RSI.
    - BUY: Price < Lower BB AND RSI < 30
    - SELL: Price > Upper BB AND RSI > 70
    """
    for i in range(1, len(df)):
        rsi = df["RSI"].iloc[i] if "RSI" in df.columns else 50
        close = df["Close"].iloc[i]
        bb_upper = df["BB_Upper"].iloc[i] if "BB_Upper" in df.columns else close + 10
        bb_lower = df["BB_Lower"].iloc[i] if "BB_Lower" in df.columns else close - 10
        bb_ma = df["BB_MA"].iloc[i] if "BB_MA" in df.columns else close
        
        # Buy signal - price below lower band
        if close < bb_lower and rsi < 30:
            df.loc[df.index[i], "Signal"] = "BUY"
            df.loc[df.index[i], "Confidence"] = min(100, (30 - rsi) + (bb_lower - close) / bb_ma * 100)
        
        # Sell signal - price above upper band
        elif close > bb_upper and rsi > 70:
            df.loc[df.index[i], "Signal"] = "SELL"
            df.loc[df.index[i], "Confidence"] = min(100, (rsi - 70) + (close - bb_upper) / bb_ma * 100)
    
    return df

def _trend_following_strategy(df):
    """
    Trend following strategy using MACD, ADX, and Stochastic.
    - BUY: ADX > 25 AND MACD > Signal AND Stoch_K < 20
    - SELL: ADX > 25 AND MACD < Signal AND Stoch_K > 80
    """
    for i in range(1, len(df)):
        adx = df["ADX"].iloc[i] if "ADX" in df.columns else 20
        macd = df["MACD"].iloc[i] if "MACD" in df.columns else 0
        macd_signal = df["MACD_Signal"].iloc[i] if "MACD_Signal" in df.columns else 0
        stoch_k = df["Stoch_K"].iloc[i] if "Stoch_K" in df.columns else 50
        
        # Buy signal - strong uptrend
        if adx > 25 and macd > macd_signal and stoch_k < 20:
            df.loc[df.index[i], "Signal"] = "BUY"
            df.loc[df.index[i], "Confidence"] = min(100, (adx - 25) + abs(macd - macd_signal) * 100)
        
        # Sell signal - strong downtrend
        elif adx > 25 and macd < macd_signal and stoch_k > 80:
            df.loc[df.index[i], "Signal"] = "SELL"
            df.loc[df.index[i], "Confidence"] = min(100, (adx - 25) + abs(macd - macd_signal) * 100)
    
    return df

def get_strategy_description(strategy_type):
    """Return a description of the strategy."""
    descriptions = {
        "combined": "Combined strategy: RSI + Bollinger Bands + MACD. Best for volatile markets.",
        "momentum": "Momentum strategy: MACD + RSI. Best for trending markets.",
        "mean_reversion": "Mean reversion strategy: Bollinger Bands + RSI. Best for range-bound markets.",
        "trend_following": "Trend following strategy: MACD + ADX + Stochastic. Best for strong trends.",
    }
    return descriptions.get(strategy_type, "Unknown strategy")
