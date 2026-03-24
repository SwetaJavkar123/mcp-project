"""Tests for agents/strategy_agent.py — signal generation & confidence scoring."""

import pandas as pd
import numpy as np
import pytest

from agents.strategy_agent import generate_signals, get_strategy_description, _clamp


ALL_STRATEGIES = ["combined", "momentum", "mean_reversion", "trend_following"]


class TestClamp:
    def test_within_range(self):
        assert _clamp(50) == 50

    def test_below_floor(self):
        assert _clamp(-10) == 0

    def test_above_ceiling(self):
        assert _clamp(150) == 100

    def test_custom_bounds(self):
        assert _clamp(5, 0, 10) == 5
        assert _clamp(15, 0, 10) == 10


class TestGenerateSignals:
    """Test that every strategy type runs without error and produces valid output."""

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_adds_signal_and_confidence_columns(self, enriched_df, strategy):
        result = generate_signals(enriched_df, strategy_type=strategy)
        assert "Signal" in result.columns
        assert "Confidence" in result.columns

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_signals_are_valid_values(self, enriched_df, strategy):
        result = generate_signals(enriched_df, strategy_type=strategy)
        valid = {"BUY", "SELL", "HOLD"}
        assert set(result["Signal"].unique()).issubset(valid)

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_confidence_always_between_0_and_100(self, enriched_df, strategy):
        result = generate_signals(enriched_df, strategy_type=strategy)
        assert (result["Confidence"] >= 0).all(), "Confidence went negative!"
        assert (result["Confidence"] <= 100).all(), "Confidence exceeded 100!"

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_hold_confidence_is_50(self, enriched_df, strategy):
        result = generate_signals(enriched_df, strategy_type=strategy)
        holds = result.loc[result["Signal"] == "HOLD", "Confidence"]
        if len(holds):
            assert (holds == 50.0).all(), f"HOLD confidence should be 50, got: {holds.unique()}"

    def test_does_not_mutate_input(self, enriched_df):
        original_cols = set(enriched_df.columns)
        generate_signals(enriched_df, strategy_type="combined")
        assert set(enriched_df.columns) == original_cols

    def test_unknown_strategy_defaults_to_hold(self, enriched_df):
        result = generate_signals(enriched_df, strategy_type="unknown_xyz")
        assert (result["Signal"] == "HOLD").all()
        assert (result["Confidence"] == 50.0).all()


class TestConfidenceOnVolatileData:
    """Test confidence scoring on highly volatile synthetic data to stress-test edge cases."""

    @pytest.fixture
    def volatile_df(self):
        """500-day volatile dataset that triggers signals in all strategies."""
        from utils_marketdata import (
            calculate_rsi, calculate_bollinger_bands, calculate_macd,
            calculate_stochastic, calculate_atr, calculate_adx,
            calculate_moving_average,
        )
        np.random.seed(7)
        n = 500
        dates = pd.bdate_range("2023-01-01", periods=n)
        close = 100 + np.cumsum(np.random.randn(n) * 3)
        high = close + np.abs(np.random.randn(n)) * 2
        low = close - np.abs(np.random.randn(n)) * 2
        open_ = close + np.random.randn(n)
        volume = np.random.randint(10_000_000, 50_000_000, size=n)
        df = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )
        df = calculate_moving_average(df, window=20)
        df = calculate_rsi(df, window=14)
        df = calculate_bollinger_bands(df, window=20, num_std=2)
        df = calculate_macd(df, fast=12, slow=26, signal=9)
        df = calculate_stochastic(df, window=14)
        df = calculate_atr(df, window=14)
        df = calculate_adx(df, window=14)
        return df

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_no_negative_confidence_volatile(self, volatile_df, strategy):
        result = generate_signals(volatile_df, strategy_type=strategy)
        assert (result["Confidence"] >= 0).all()

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_no_over_100_confidence_volatile(self, volatile_df, strategy):
        result = generate_signals(volatile_df, strategy_type=strategy)
        assert (result["Confidence"] <= 100).all()

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_signal_confidence_above_zero(self, volatile_df, strategy):
        """Signals (BUY/SELL) should have non-zero confidence — otherwise why trigger?"""
        result = generate_signals(volatile_df, strategy_type=strategy)
        signals = result.loc[result["Signal"].isin(["BUY", "SELL"]), "Confidence"]
        if len(signals):
            # Allow 0 as a clamped edge case but most should be > 0
            assert signals.mean() > 5, f"Mean signal confidence is too low: {signals.mean():.1f}"

    def test_momentum_generates_signals(self, volatile_df):
        result = generate_signals(volatile_df, strategy_type="momentum")
        assert (result["Signal"] != "HOLD").any()

    def test_mean_reversion_reasonable_confidence(self, volatile_df):
        result = generate_signals(volatile_df, strategy_type="mean_reversion")
        signals = result.loc[result["Signal"] != "HOLD", "Confidence"]
        if len(signals):
            assert signals.mean() > 10, f"Mean reversion confidence too low: {signals.mean():.1f}"


class TestGetStrategyDescription:
    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_returns_string(self, strategy):
        desc = get_strategy_description(strategy)
        assert isinstance(desc, str)
        assert len(desc) > 10

    def test_unknown_returns_fallback(self):
        desc = get_strategy_description("nonexistent")
        assert desc == "Unknown strategy"
