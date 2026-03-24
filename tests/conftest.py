"""
Shared fixtures for all tests.
Provides sample DataFrames that mimic real market data so we never
need to hit yfinance during the test run.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_prices():
    """100-day OHLCV DataFrame with realistic-looking synthetic data."""
    np.random.seed(42)
    dates = pd.bdate_range("2025-01-01", periods=100)
    close = 150 + np.cumsum(np.random.randn(100) * 1.5)
    high = close + np.abs(np.random.randn(100))
    low = close - np.abs(np.random.randn(100))
    open_ = close + np.random.randn(100) * 0.5
    volume = np.random.randint(10_000_000, 50_000_000, size=100)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def sample_returns(sample_prices):
    """Daily returns derived from sample_prices."""
    return sample_prices["Close"].pct_change().dropna()


@pytest.fixture
def enriched_df(sample_prices):
    """sample_prices with all technical indicators already computed."""
    from utils_marketdata import (
        calculate_rsi,
        calculate_bollinger_bands,
        calculate_macd,
        calculate_stochastic,
        calculate_atr,
        calculate_adx,
        calculate_moving_average,
    )

    df = sample_prices.copy()
    df = calculate_moving_average(df, window=20)
    df = calculate_rsi(df, window=14)
    df = calculate_bollinger_bands(df, window=20, num_std=2)
    df = calculate_macd(df, fast=12, slow=26, signal=9)
    df = calculate_stochastic(df, window=14)
    df = calculate_atr(df, window=14)
    df = calculate_adx(df, window=14)
    return df
