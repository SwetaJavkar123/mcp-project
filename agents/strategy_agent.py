"""
StrategyAgent
==============
Generates BUY / SELL / HOLD signals with a 0-100 confidence score.

Confidence scoring guide
------------------------
  0-29  = weak signal   (backtest skips trades below 30)
  30-59 = moderate
  60-79 = strong
  80-100 = very strong

Every row gets a confidence:
  - HOLD rows receive 50 (neutral).
  - BUY/SELL rows are scored 30-100 based on how strongly the
    indicators agree.  Scores are always clamped to [0, 100].
"""

import numpy as np
import pandas as pd


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp a value between lo and hi."""
    return max(lo, min(hi, value))


def _normalise_macd(macd_diff: float, atr: float) -> float:
    """
    Normalise MACD histogram to 0-1 scale using ATR.

    Raw MACD diff is price-dependent (AAPL ~2, penny stock ~0.002).
    Dividing by ATR makes the signal comparable across instruments.
    """
    if atr <= 0:
        return 0.0
    return min(abs(macd_diff) / atr, 1.0)


def generate_signals(df, strategy_type="combined"):
    """
    Generate Buy/Sell/Hold signals based on technical indicators.

    Parameters
    ----------
    df : DataFrame with indicator columns (RSI, MACD, BB, Stoch, ATR, ADX).
    strategy_type : "combined" | "momentum" | "mean_reversion" | "trend_following"

    Returns
    -------
    DataFrame with Signal (str) and Confidence (float 0-100) columns.
    """
    df = df.copy()
    df["Signal"] = "HOLD"
    df["Confidence"] = 50.0  # neutral baseline for HOLD

    if strategy_type == "combined":
        df = _combined_strategy(df)
    elif strategy_type == "momentum":
        df = _momentum_strategy(df)
    elif strategy_type == "mean_reversion":
        df = _mean_reversion_strategy(df)
    elif strategy_type == "trend_following":
        df = _trend_following_strategy(df)

    return df


# ---------------------------------------------------------------------------
# Combined: RSI + Bollinger + MACD  (2-of-3 required, 3-of-3 = higher conf)
# ---------------------------------------------------------------------------

def _combined_strategy(df):
    """
    Combined strategy — relaxed to 2-of-3 indicator agreement.

    Indicators scored:
      - RSI:  oversold (<35) / overbought (>65)  → 0-40 pts
      - BB:   price outside band                  → 0-30 pts
      - MACD: histogram direction                 → 0-30 pts

    Requires at least 2 of the 3 to agree for a signal.
    """
    has_rsi = "RSI" in df.columns
    has_macd = "MACD" in df.columns and "MACD_Signal" in df.columns
    has_bb = "BB_Upper" in df.columns and "BB_Lower" in df.columns
    has_atr = "ATR" in df.columns

    for i in range(1, len(df)):
        rsi = float(df["RSI"].iloc[i]) if has_rsi else 50.0
        macd = float(df["MACD"].iloc[i]) if has_macd else 0.0
        macd_sig = float(df["MACD_Signal"].iloc[i]) if has_macd else 0.0
        close = float(df["Close"].iloc[i])
        bb_upper = float(df["BB_Upper"].iloc[i]) if has_bb else close + 1
        bb_lower = float(df["BB_Lower"].iloc[i]) if has_bb else close - 1
        bb_ma = float(df["BB_MA"].iloc[i]) if "BB_MA" in df.columns else close
        atr = float(df["ATR"].iloc[i]) if has_atr else 1.0

        macd_diff = macd - macd_sig
        macd_norm = _normalise_macd(macd_diff, atr)

        # --- BUY scoring ---
        buy_votes = 0
        buy_score = 0.0

        if rsi < 35:
            buy_votes += 1
            buy_score += _clamp((35 - rsi) / 35 * 40, 0, 40)  # 0-40 pts

        if close < bb_lower and bb_ma > 0:
            buy_votes += 1
            buy_score += _clamp((bb_lower - close) / bb_ma * 300, 0, 30)  # 0-30 pts

        if macd_diff > 0:
            buy_votes += 1
            buy_score += _clamp(macd_norm * 30, 0, 30)  # 0-30 pts

        if buy_votes >= 2:
            df.loc[df.index[i], "Signal"] = "BUY"
            df.loc[df.index[i], "Confidence"] = _clamp(buy_score)
            continue

        # --- SELL scoring ---
        sell_votes = 0
        sell_score = 0.0

        if rsi > 65:
            sell_votes += 1
            sell_score += _clamp((rsi - 65) / 35 * 40, 0, 40)

        if close > bb_upper and bb_ma > 0:
            sell_votes += 1
            sell_score += _clamp((close - bb_upper) / bb_ma * 300, 0, 30)

        if macd_diff < 0:
            sell_votes += 1
            sell_score += _clamp(macd_norm * 30, 0, 30)

        if sell_votes >= 2:
            df.loc[df.index[i], "Signal"] = "SELL"
            df.loc[df.index[i], "Confidence"] = _clamp(sell_score)

    return df


# ---------------------------------------------------------------------------
# Momentum: MACD + RSI
# ---------------------------------------------------------------------------

def _momentum_strategy(df):
    """
    Momentum strategy — MACD cross + RSI direction.

    Confidence = RSI component (0-60) + normalised MACD component (0-40).
    """
    has_rsi = "RSI" in df.columns
    has_macd = "MACD" in df.columns and "MACD_Signal" in df.columns
    has_atr = "ATR" in df.columns

    for i in range(1, len(df)):
        rsi = float(df["RSI"].iloc[i]) if has_rsi else 50.0
        macd = float(df["MACD"].iloc[i]) if has_macd else 0.0
        macd_sig = float(df["MACD_Signal"].iloc[i]) if has_macd else 0.0
        atr = float(df["ATR"].iloc[i]) if has_atr else 1.0

        macd_diff = macd - macd_sig
        macd_norm = _normalise_macd(macd_diff, atr)

        # BUY: MACD bullish cross + RSI > 50
        if macd_diff > 0 and rsi > 50:
            rsi_score = _clamp((rsi - 50) / 50 * 60, 0, 60)
            macd_score = _clamp(macd_norm * 40, 0, 40)
            df.loc[df.index[i], "Signal"] = "BUY"
            df.loc[df.index[i], "Confidence"] = _clamp(rsi_score + macd_score)

        # SELL: MACD bearish cross + RSI < 50
        elif macd_diff < 0 and rsi < 50:
            rsi_score = _clamp((50 - rsi) / 50 * 60, 0, 60)
            macd_score = _clamp(macd_norm * 40, 0, 40)
            df.loc[df.index[i], "Signal"] = "SELL"
            df.loc[df.index[i], "Confidence"] = _clamp(rsi_score + macd_score)

    return df


# ---------------------------------------------------------------------------
# Mean Reversion: Bollinger + RSI
# ---------------------------------------------------------------------------

def _mean_reversion_strategy(df):
    """
    Mean reversion — price vs Bollinger Bands + RSI.

    Relaxed thresholds: RSI < 35 (buy) / RSI > 65 (sell).
    Confidence = RSI component (0-50) + BB deviation component (0-50).
    BB deviation normalised by band width to keep scores consistent.
    """
    has_rsi = "RSI" in df.columns
    has_bb = "BB_Upper" in df.columns and "BB_Lower" in df.columns

    for i in range(1, len(df)):
        rsi = float(df["RSI"].iloc[i]) if has_rsi else 50.0
        close = float(df["Close"].iloc[i])
        bb_upper = float(df["BB_Upper"].iloc[i]) if has_bb else close + 1
        bb_lower = float(df["BB_Lower"].iloc[i]) if has_bb else close - 1
        bb_ma = float(df["BB_MA"].iloc[i]) if "BB_MA" in df.columns else close

        band_width = bb_upper - bb_lower
        if band_width <= 0:
            band_width = 1.0

        # BUY: price near/below lower band + RSI oversold
        if close < bb_lower and rsi < 35:
            rsi_score = _clamp((35 - rsi) / 35 * 50, 0, 50)
            # How far below the lower band, relative to band width
            bb_score = _clamp((bb_lower - close) / band_width * 100, 0, 50)
            df.loc[df.index[i], "Signal"] = "BUY"
            df.loc[df.index[i], "Confidence"] = _clamp(rsi_score + bb_score)

        # SELL: price near/above upper band + RSI overbought
        elif close > bb_upper and rsi > 65:
            rsi_score = _clamp((rsi - 65) / 35 * 50, 0, 50)
            bb_score = _clamp((close - bb_upper) / band_width * 100, 0, 50)
            df.loc[df.index[i], "Signal"] = "SELL"
            df.loc[df.index[i], "Confidence"] = _clamp(rsi_score + bb_score)

    return df


# ---------------------------------------------------------------------------
# Trend Following: MACD + ADX + Stochastic
# ---------------------------------------------------------------------------

def _trend_following_strategy(df):
    """
    Trend following — requires strong trend (ADX) + MACD direction.

    Relaxed: ADX > 20 (was 25), Stochastic < 30 / > 70 (was 20/80).
    Confidence = ADX component (0-40) + MACD component (0-30) + Stoch component (0-30).
    """
    has_adx = "ADX" in df.columns
    has_macd = "MACD" in df.columns and "MACD_Signal" in df.columns
    has_stoch = "Stoch_K" in df.columns
    has_atr = "ATR" in df.columns

    for i in range(1, len(df)):
        adx = float(df["ADX"].iloc[i]) if has_adx else 15.0
        macd = float(df["MACD"].iloc[i]) if has_macd else 0.0
        macd_sig = float(df["MACD_Signal"].iloc[i]) if has_macd else 0.0
        stoch_k = float(df["Stoch_K"].iloc[i]) if has_stoch else 50.0
        atr = float(df["ATR"].iloc[i]) if has_atr else 1.0

        macd_diff = macd - macd_sig
        macd_norm = _normalise_macd(macd_diff, atr)

        # Need at least a moderate trend
        if adx <= 20:
            continue

        adx_score = _clamp((adx - 20) / 30 * 40, 0, 40)  # 0-40 pts

        # BUY: uptrend + stoch oversold
        if macd_diff > 0 and stoch_k < 30:
            macd_score = _clamp(macd_norm * 30, 0, 30)
            stoch_score = _clamp((30 - stoch_k) / 30 * 30, 0, 30)
            df.loc[df.index[i], "Signal"] = "BUY"
            df.loc[df.index[i], "Confidence"] = _clamp(adx_score + macd_score + stoch_score)

        # SELL: downtrend + stoch overbought
        elif macd_diff < 0 and stoch_k > 70:
            macd_score = _clamp(macd_norm * 30, 0, 30)
            stoch_score = _clamp((stoch_k - 70) / 30 * 30, 0, 30)
            df.loc[df.index[i], "Signal"] = "SELL"
            df.loc[df.index[i], "Confidence"] = _clamp(adx_score + macd_score + stoch_score)

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
