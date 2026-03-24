"""Tests for utils_marketdata.py — technical indicator calculations."""

import pandas as pd
import numpy as np
import pytest

from utils_marketdata import (
    calculate_moving_average,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_stochastic,
    calculate_atr,
    calculate_adx,
    get_basic_statistics,
    get_popular_symbols,
)


class TestMovingAverage:
    def test_adds_ma_column(self, sample_prices):
        df = calculate_moving_average(sample_prices.copy(), window=20)
        assert "MA" in df.columns

    def test_ma_length_matches(self, sample_prices):
        df = calculate_moving_average(sample_prices.copy(), window=20)
        assert len(df["MA"]) == len(df)

    def test_ma_values_reasonable(self, sample_prices):
        df = calculate_moving_average(sample_prices.copy(), window=20)
        assert df["MA"].iloc[-1] > 0


class TestRSI:
    def test_adds_rsi_column(self, sample_prices):
        df = calculate_rsi(sample_prices.copy(), window=14)
        assert "RSI" in df.columns

    def test_rsi_bounded_0_to_100(self, sample_prices):
        df = calculate_rsi(sample_prices.copy(), window=14)
        valid = df["RSI"].dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_custom_window(self, sample_prices):
        df = calculate_rsi(sample_prices.copy(), window=7)
        assert "RSI" in df.columns


class TestBollingerBands:
    def test_adds_columns(self, sample_prices):
        df = calculate_bollinger_bands(sample_prices.copy())
        assert "BB_MA" in df.columns
        assert "BB_Upper" in df.columns
        assert "BB_Lower" in df.columns

    def test_upper_above_lower(self, sample_prices):
        df = calculate_bollinger_bands(sample_prices.copy())
        valid = df.dropna(subset=["BB_Upper", "BB_Lower"])
        assert (valid["BB_Upper"] >= valid["BB_Lower"]).all()

    def test_ma_between_bands(self, sample_prices):
        df = calculate_bollinger_bands(sample_prices.copy())
        valid = df.dropna(subset=["BB_Upper", "BB_Lower", "BB_MA"])
        assert (valid["BB_MA"] <= valid["BB_Upper"]).all()
        assert (valid["BB_MA"] >= valid["BB_Lower"]).all()


class TestMACD:
    def test_adds_columns(self, sample_prices):
        df = calculate_macd(sample_prices.copy())
        assert "MACD" in df.columns
        assert "MACD_Signal" in df.columns
        assert "MACD_Histogram" in df.columns

    def test_histogram_is_difference(self, sample_prices):
        df = calculate_macd(sample_prices.copy())
        diff = (df["MACD"] - df["MACD_Signal"]).dropna()
        hist = df["MACD_Histogram"].dropna()
        pd.testing.assert_series_equal(diff, hist, check_names=False)


class TestStochastic:
    def test_adds_columns(self, sample_prices):
        df = calculate_stochastic(sample_prices.copy())
        assert "Stoch_K" in df.columns
        assert "Stoch_D" in df.columns

    def test_stoch_bounded(self, sample_prices):
        df = calculate_stochastic(sample_prices.copy())
        valid_k = df["Stoch_K"].dropna()
        assert valid_k.min() >= -0.1  # small tolerance for smoothing
        assert valid_k.max() <= 100.1


class TestATR:
    def test_adds_columns(self, sample_prices):
        df = calculate_atr(sample_prices.copy())
        assert "ATR" in df.columns
        assert "TR" in df.columns

    def test_atr_positive(self, sample_prices):
        df = calculate_atr(sample_prices.copy())
        valid = df["ATR"].dropna()
        assert (valid >= 0).all()


class TestADX:
    def test_adds_column(self, sample_prices):
        df = calculate_adx(sample_prices.copy())
        assert "ADX" in df.columns

    def test_adx_positive(self, sample_prices):
        df = calculate_adx(sample_prices.copy())
        valid = df["ADX"].dropna()
        assert (valid >= 0).all()


class TestBasicStatistics:
    def test_returns_dict(self, sample_prices):
        stats = get_basic_statistics(sample_prices)
        assert isinstance(stats, dict)
        assert "Mean Close" in stats
        assert "Min Close" in stats
        assert "Max Close" in stats
        assert "Volatility (std dev)" in stats

    def test_min_less_than_max(self, sample_prices):
        stats = get_basic_statistics(sample_prices)
        assert stats["Min Close"] <= stats["Max Close"]


class TestPopularSymbols:
    def test_returns_list(self):
        symbols = get_popular_symbols()
        assert isinstance(symbols, list)
        assert len(symbols) > 0

    def test_contains_aapl(self):
        assert "AAPL" in get_popular_symbols()
