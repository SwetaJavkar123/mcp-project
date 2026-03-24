"""Tests for agents/backtest_agent.py — backtesting engine."""

import pandas as pd
import numpy as np
import pytest

from agents.backtest_agent import (
    run_backtest,
    BacktestConfig,
    Trade,
    trades_to_dataframe,
)
from agents.strategy_agent import generate_signals


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_signalled_df(enriched_df, strategy="momentum"):
    """Return an enriched DataFrame with signals attached."""
    return generate_signals(enriched_df, strategy_type=strategy)


# ── Tests ─────────────────────────────────────────────────────────────────


class TestRunBacktest:
    def test_returns_expected_keys(self, enriched_df):
        df = _make_signalled_df(enriched_df)
        result = run_backtest(df, symbol="TEST")
        assert "summary" in result
        assert "equity_curve" in result
        assert "trades" in result
        assert "monthly_returns" in result

    def test_equity_curve_length(self, enriched_df):
        df = _make_signalled_df(enriched_df)
        result = run_backtest(df)
        assert len(result["equity_curve"]) == len(df)

    def test_initial_capital_in_summary(self, enriched_df):
        df = _make_signalled_df(enriched_df)
        cfg = BacktestConfig(initial_capital=50_000)
        result = run_backtest(df, config=cfg)
        assert result["summary"]["initial_capital"] == 50_000

    def test_final_equity_positive(self, enriched_df):
        df = _make_signalled_df(enriched_df)
        result = run_backtest(df)
        assert result["summary"]["final_equity"] > 0

    def test_win_rate_bounded(self, enriched_df):
        df = _make_signalled_df(enriched_df)
        result = run_backtest(df)
        wr = result["summary"]["win_rate"]
        assert 0 <= wr <= 100

    def test_raises_without_signal_column(self, sample_prices):
        with pytest.raises(ValueError, match="Signal"):
            run_backtest(sample_prices)

    def test_all_hold_produces_no_trades(self, enriched_df):
        df = enriched_df.copy()
        df["Signal"] = "HOLD"
        df["Confidence"] = 50.0
        result = run_backtest(df)
        assert result["summary"]["total_trades"] == 0


class TestBacktestConfig:
    def test_defaults(self):
        cfg = BacktestConfig()
        assert cfg.initial_capital == 100_000
        assert cfg.stop_loss_pct == 0.05

    def test_custom_values(self):
        cfg = BacktestConfig(initial_capital=200_000, stop_loss_pct=0.03)
        assert cfg.initial_capital == 200_000
        assert cfg.stop_loss_pct == 0.03


class TestTrade:
    def test_dataclass_fields(self):
        t = Trade(
            symbol="AAPL",
            entry_date="2025-01-01",
            entry_price=150,
            exit_date="2025-01-10",
            exit_price=160,
            shares=10,
            pnl=100,
            pnl_pct=0.066,
            exit_reason="SIGNAL_SELL",
        )
        assert t.symbol == "AAPL"
        assert t.pnl == 100


class TestTradesToDataframe:
    def test_returns_dataframe(self):
        trades = [
            Trade("AAPL", "2025-01-01", 150, "2025-01-10", 160, 10, pnl=100, pnl_pct=0.066, exit_reason="SELL"),
        ]
        df = trades_to_dataframe(trades)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "Symbol" in df.columns

    def test_empty_list(self):
        df = trades_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
