"""Tests for agents/strategy_agent.py — signal generation."""

import pandas as pd
import numpy as np
import pytest

from agents.strategy_agent import generate_signals, get_strategy_description


class TestGenerateSignals:
    """Test that every strategy type runs without error and produces valid output."""

    @pytest.mark.parametrize(
        "strategy",
        ["combined", "momentum", "mean_reversion", "trend_following"],
    )
    def test_adds_signal_and_confidence_columns(self, enriched_df, strategy):
        result = generate_signals(enriched_df, strategy_type=strategy)
        assert "Signal" in result.columns
        assert "Confidence" in result.columns

    @pytest.mark.parametrize(
        "strategy",
        ["combined", "momentum", "mean_reversion", "trend_following"],
    )
    def test_signals_are_valid_values(self, enriched_df, strategy):
        result = generate_signals(enriched_df, strategy_type=strategy)
        valid = {"BUY", "SELL", "HOLD"}
        assert set(result["Signal"].unique()).issubset(valid)

    @pytest.mark.parametrize(
        "strategy",
        ["combined", "momentum", "mean_reversion", "trend_following"],
    )
    def test_confidence_non_negative(self, enriched_df, strategy):
        result = generate_signals(enriched_df, strategy_type=strategy)
        assert (result["Confidence"] >= 0).all()

    def test_does_not_mutate_input(self, enriched_df):
        original_cols = set(enriched_df.columns)
        generate_signals(enriched_df, strategy_type="combined")
        assert set(enriched_df.columns) == original_cols

    def test_unknown_strategy_defaults_to_hold(self, enriched_df):
        result = generate_signals(enriched_df, strategy_type="unknown_xyz")
        assert (result["Signal"] == "HOLD").all()


class TestGetStrategyDescription:
    @pytest.mark.parametrize(
        "strategy",
        ["combined", "momentum", "mean_reversion", "trend_following"],
    )
    def test_returns_string(self, strategy):
        desc = get_strategy_description(strategy)
        assert isinstance(desc, str)
        assert len(desc) > 10

    def test_unknown_returns_fallback(self):
        desc = get_strategy_description("nonexistent")
        assert desc == "Unknown strategy"
