"""Tests for agents/risk_agent.py — risk metrics, sizing, and validation."""

import numpy as np
import pandas as pd
import pytest

from agents.risk_agent import (
    Position,
    Portfolio,
    TradeProposal,
    calculate_var,
    calculate_cvar,
    calculate_sharpe,
    calculate_sortino,
    calculate_max_drawdown,
    calculate_risk_metrics,
    calculate_position_size,
    validate_trade,
    calculate_portfolio_summary,
)


# ── Data classes ──────────────────────────────────────────────────────────


class TestPosition:
    def test_market_value(self):
        p = Position(symbol="AAPL", shares=10, avg_cost=100, current_price=120)
        assert p.market_value == 1200

    def test_pnl(self):
        p = Position(symbol="AAPL", shares=10, avg_cost=100, current_price=120)
        assert p.pnl == 200

    def test_pnl_pct(self):
        p = Position(symbol="AAPL", shares=10, avg_cost=100, current_price=120)
        assert p.pnl_pct == pytest.approx(0.20)

    def test_pnl_pct_zero_cost(self):
        p = Position(symbol="X", shares=10, avg_cost=0, current_price=50)
        assert p.pnl_pct == 0.0


class TestPortfolio:
    def test_total_value_cash_only(self):
        pf = Portfolio(cash=100_000)
        assert pf.total_value == 100_000

    def test_total_value_with_positions(self):
        pf = Portfolio(
            cash=50_000,
            positions=[Position("AAPL", 100, 150, 160)],
        )
        assert pf.total_value == 50_000 + 100 * 160

    def test_position_for_found(self):
        pos = Position("AAPL", 10, 100, 110)
        pf = Portfolio(positions=[pos])
        assert pf.position_for("AAPL") is pos

    def test_position_for_not_found(self):
        pf = Portfolio()
        assert pf.position_for("AAPL") is None


# ── Risk metrics ──────────────────────────────────────────────────────────


class TestVaR:
    def test_returns_float(self, sample_returns):
        result = calculate_var(sample_returns)
        assert isinstance(result, float)

    def test_var_is_negative(self, sample_returns):
        # 95% VaR should be a loss (negative number)
        assert calculate_var(sample_returns, 0.95) < 0

    def test_empty_returns_zero(self):
        assert calculate_var(pd.Series(dtype=float)) == 0.0


class TestCVaR:
    def test_cvar_worse_than_var(self, sample_returns):
        var = calculate_var(sample_returns)
        cvar = calculate_cvar(sample_returns)
        assert cvar <= var  # CVaR is deeper into the tail


class TestSharpe:
    def test_returns_float(self, sample_returns):
        assert isinstance(calculate_sharpe(sample_returns), float)

    def test_zero_std_returns_zero(self):
        flat = pd.Series([0.0] * 100)
        assert calculate_sharpe(flat) == 0.0


class TestSortino:
    def test_returns_float(self, sample_returns):
        assert isinstance(calculate_sortino(sample_returns), float)


class TestMaxDrawdown:
    def test_drawdown_negative(self, sample_prices):
        dd = calculate_max_drawdown(sample_prices["Close"])
        assert dd <= 0

    def test_monotonic_up_has_zero_drawdown(self):
        prices = pd.Series([1, 2, 3, 4, 5])
        assert calculate_max_drawdown(prices) == 0.0


class TestRiskMetrics:
    def test_returns_all_keys(self, sample_returns, sample_prices):
        metrics = calculate_risk_metrics(sample_returns, sample_prices["Close"])
        expected_keys = {
            "daily_mean_return", "daily_volatility",
            "annualised_return", "annualised_volatility",
            "var_95", "cvar_95",
            "sharpe_ratio", "sortino_ratio",
            "max_drawdown",
        }
        assert expected_keys.issubset(metrics.keys())


# ── Position sizing ───────────────────────────────────────────────────────


class TestPositionSize:
    def test_returns_expected_keys(self):
        result = calculate_position_size(100_000, 150)
        for key in ("shares", "dollar_amount", "pct_of_portfolio", "stop_loss_price", "take_profit_price"):
            assert key in result

    def test_shares_not_negative(self):
        result = calculate_position_size(100_000, 150)
        assert result["shares"] >= 0

    def test_zero_price(self):
        result = calculate_position_size(100_000, 0)
        assert result["shares"] == 0

    def test_respects_max_position(self):
        result = calculate_position_size(100_000, 10, max_position_pct=0.10)
        assert result["dollar_amount"] <= 100_000 * 0.10


# ── Trade validation ─────────────────────────────────────────────────────


class TestValidateTrade:
    def test_approved_buy(self):
        pf = Portfolio(cash=100_000)
        proposal = TradeProposal("AAPL", "BUY", 10, 150, confidence=60)
        result = validate_trade(proposal, pf)
        assert result["approved"] is True

    def test_rejected_insufficient_cash(self):
        pf = Portfolio(cash=100)
        proposal = TradeProposal("AAPL", "BUY", 100, 150, confidence=60)
        result = validate_trade(proposal, pf)
        assert result["approved"] is False
        assert any("cash" in r.lower() for r in result["reasons"])

    def test_rejected_sell_no_position(self):
        pf = Portfolio(cash=100_000)
        proposal = TradeProposal("AAPL", "SELL", 10, 150, confidence=60)
        result = validate_trade(proposal, pf)
        assert result["approved"] is False

    def test_rejected_concentration(self):
        pf = Portfolio(cash=100_000, max_position_pct=0.10)
        # 50 shares × $250 = $12,500 > 10% of $100k
        proposal = TradeProposal("AAPL", "BUY", 50, 250, confidence=60)
        result = validate_trade(proposal, pf)
        assert result["approved"] is False

    def test_low_confidence_warning(self):
        pf = Portfolio(cash=100_000)
        proposal = TradeProposal("AAPL", "BUY", 5, 150, confidence=20)
        result = validate_trade(proposal, pf)
        assert len(result["warnings"]) > 0


# ── Portfolio summary ─────────────────────────────────────────────────────


class TestPortfolioSummary:
    def test_returns_expected_keys(self):
        pf = Portfolio(
            cash=80_000,
            positions=[Position("AAPL", 100, 150, 160)],
        )
        summary = calculate_portfolio_summary(pf)
        assert "cash" in summary
        assert "total_value" in summary
        assert "positions" in summary
        assert len(summary["positions"]) == 1
