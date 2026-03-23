import pandas as pd
from agents.market_data_agent import get_data

def fetch_and_process_data(symbol, start, end):
    """
    Fetch stock data and process it.
    :param symbol: Stock ticker symbol
    :param start: Start date (YYYY-MM-DD)
    :param end: End date (YYYY-MM-DD)
    :return: Processed DataFrame or None if no data found
    """
    df = get_data(symbol, start=start, end=end)
    if df.empty:
        return None
    # Flatten multi-level columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def calculate_moving_average(df, window):
    """
    Calculate moving average for the Close price.
    :param df: DataFrame with stock data
    :param window: Moving average window size
    :return: DataFrame with MA column added
    """
    df["MA"] = df["Close"].rolling(window=window).mean()
    df["MA"] = df["MA"].fillna(method="bfill")
    return df

def get_basic_statistics(df):
    """
    Calculate basic statistics for the Close price.
    :param df: DataFrame with stock data
    :return: Dictionary of statistics
    """
    stats = {
        "Mean Close": round(df["Close"].mean(), 2),
        "Min Close": round(df["Close"].min(), 2),
        "Max Close": round(df["Close"].max(), 2),
        "Volatility (std dev)": round(df["Close"].std(), 2),
    }
    return stats

def get_popular_symbols():
    """Return a list of popular stock symbols."""
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "UNH"]

def calculate_rsi(df, window=14):
    """
    Calculate Relative Strength Index (RSI).
    :param df: DataFrame with stock data
    :param window: RSI window size (default 14)
    :return: DataFrame with RSI column added
    """
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def calculate_bollinger_bands(df, window=20, num_std=2):
    """
    Calculate Bollinger Bands.
    :param df: DataFrame with stock data
    :param window: Moving average window size (default 20)
    :param num_std: Number of standard deviations (default 2)
    :return: DataFrame with upper and lower bands added
    """
    df["BB_MA"] = df["Close"].rolling(window=window).mean()
    bb_std = df["Close"].rolling(window=window).std()
    df["BB_Upper"] = df["BB_MA"] + (bb_std * num_std)
    df["BB_Lower"] = df["BB_MA"] - (bb_std * num_std)
    return df

def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence).
    :param df: DataFrame with stock data
    :param fast: Fast EMA window (default 12)
    :param slow: Slow EMA window (default 26)
    :param signal: Signal line EMA window (default 9)
    :return: DataFrame with MACD, Signal, and Histogram added
    """
    ema_fast = df["Close"].ewm(span=fast).mean()
    ema_slow = df["Close"].ewm(span=slow).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]
    return df

def calculate_stochastic(df, window=14, smooth_k=3, smooth_d=3):
    """
    Calculate Stochastic Oscillator.
    :param df: DataFrame with stock data
    :param window: Lookback window (default 14)
    :param smooth_k: K smoothing period (default 3)
    :param smooth_d: D smoothing period (default 3)
    :return: DataFrame with %K and %D added
    """
    low_min = df["Low"].rolling(window=window).min()
    high_max = df["High"].rolling(window=window).max()
    df["Stoch_K"] = 100 * ((df["Close"] - low_min) / (high_max - low_min))
    df["Stoch_D"] = df["Stoch_K"].rolling(window=smooth_d).mean()
    df["Stoch_K"] = df["Stoch_K"].rolling(window=smooth_k).mean()
    return df

def calculate_atr(df, window=14):
    """
    Calculate Average True Range (ATR) for volatility.
    :param df: DataFrame with stock data
    :param window: ATR window (default 14)
    :return: DataFrame with ATR added
    """
    df["TR"] = df[["High", "Close"]].max(axis=1) - df[["Low", "Close"]].min(axis=1)
    df["ATR"] = df["TR"].rolling(window=window).mean()
    return df

def calculate_adx(df, window=14):
    """
    Calculate ADX (Average Directional Index) for trend strength.
    :param df: DataFrame with stock data
    :param window: ADX window (default 14)
    :return: DataFrame with ADX added
    """
    df["TR"] = df[["High", "Close"]].max(axis=1) - df[["Low", "Close"]].min(axis=1)
    df["Plus_DM"] = df["High"].diff()
    df["Minus_DM"] = -df["Low"].diff()
    df.loc[df["Plus_DM"] < 0, "Plus_DM"] = 0
    df.loc[df["Minus_DM"] < 0, "Minus_DM"] = 0
    
    tr_sum = df["TR"].rolling(window=window).sum()
    plus_di = 100 * (df["Plus_DM"].rolling(window=window).sum() / tr_sum)
    minus_di = 100 * (df["Minus_DM"].rolling(window=window).sum() / tr_sum)
    
    di_sum = (plus_di + minus_di).abs()
    adx = (plus_di - minus_di).abs() / di_sum * 100
    df["ADX"] = adx.rolling(window=window).mean()
    return df
