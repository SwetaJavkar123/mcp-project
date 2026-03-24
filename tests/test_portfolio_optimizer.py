"""Tests for agents/portfolio_optimizer.py — portfolio optimisation."""

import numpy as np
import pandas as pd
import pytest

from agents.portfolio_optimizer import (
    OptimiserConfig,
    efficient_frontier,
    risk_parity,
    black_litterman,
    rebalance_portfolio,
    optimise_portfolio,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def multi_asset_returns():
    """Synthetic daily returns for 4 assets, 500 days."""
    np.random.seed(42)
    n = 500
    dates = pd.bdate_range("2024-01-01", periods=n)
    data = {
        "AAPL": np.random.randn(n) * 0.015 + 0.0005,
        "MSFT": np.random.randn(n) * 0.012 + 0.0004,
        "GOOGL": np.random.randn(n) * 0.018 + 0.0003,
        "AMZN": np.random.randn(n) * 0.020 + 0.0006,
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def market_caps():
    return {"AAPL": 3e12, "MSFT": 2.8e12, "GOOGL": 1.8e12, "AMZN": 1.6e12}


# ── Efficient Frontier ────────────────────────────────────────────────────


class TestEfficientFrontier:
    def test_returns_expected_keys(self, multi_asset_returns):
        result = efficient_frontier(multi_asset_returns)
        assert "frontier" in result
        assert "max_sharpe" in result
        assert "min_volatility" in result
        assert "symbols" in result

    def test_frontier_is_dataframe(self, multi_asset_returns):
        result = efficient_frontier(multi_asset_returns)
        assert isinstance(result["frontier"], pd.DataFrame)
        assert len(result["frontier"]) > 0

    def test_max_sharpe_has_weights(self, multi_asset_returns):
        result = efficient_frontier(multi_asset_returns)
        ms = result["max_sharpe"]
        assert "weights" in ms
        assert "return" in ms
        assert "volatility" in ms
        assert "sharpe" in ms

    def test_weights_sum_to_one(self, multi_asset_returns):
        result = efficient_frontier(multi_asset_returns)
        weights = list(result["max_sharpe"]["weights"].values())
        assert abs(sum(weights) - 1.0) < 0.05  # allow small tolerance

    def test_min_vol_lower_than_average(self, multi_asset_returns):
        result = efficient_frontier(multi_asset_returns)
        min_vol = result["min_volatility"]["volatility"]
        avg_vol = result["frontier"]["volatility"].mean()
        assert min_vol <= avg_vol

    def test_custom_config(self, multi_asset_returns):
        cfg = OptimiserConfig(num_portfolios=100, max_weight=0.50)
        result = efficient_frontier(multi_asset_returns, config=cfg)
        assert len(result["frontier"]) == 100

    def test_symbols_match(self, multi_asset_returns):
        result = efficient_frontier(multi_asset_returns)
        assert set(result["symbols"]) == {"AAPL", "MSFT", "GOOGL", "AMZN"}


# ── Risk Parity ───────────────────────────────────────────────────────────


class TestRiskParity:
    def test_returns_expected_keys(self, multi_asset_returns):
        result = risk_parity(multi_asset_returns)
        assert "weights" in result
        assert "return" in result
        assert "volatility" in result
        assert "sharpe" in result
        assert "risk_contributions" in result

    def test_weights_sum_to_one(self, multi_asset_returns):
        result = risk_parity(multi_asset_returns)
        weights = list(result["weights"].values())
        assert abs(sum(weights) - 1.0) < 0.01

    def test_risk_contributions_roughly_equal(self, multi_asset_returns):
        result = risk_parity(multi_asset_returns)
        rcs = list(result["risk_contributions"].values())
        # All risk contributions should be somewhat similar
        assert max(rcs) - min(rcs) < 0.10  # within 10%

    def test_all_weights_positive(self, multi_asset_returns):
        result = risk_parity(multi_asset_returns)
        for w in result["weights"].values():
            assert w >= 0

    def test_volatility_positive(self, multi_asset_returns):
        result = risk_parity(multi_asset_returns)
        assert result["volatility"] > 0


# ── Black-Litterman ───────────────────────────────────────────────────────


class TestBlackLitterman:
    def test_without_views(self, multi_asset_returns, market_caps):
        result = black_litterman(multi_asset_returns, market_caps)
        assert "weights" in result
        assert "implied_returns" in result
        assert "posterior_returns" in result
        weights = list(result["weights"].values())
        assert abs(sum(weights) - 1.0) < 0.05

    def test_with_views(self, multi_asset_returns, market_caps):
        views = [
            {"assets": ["AAPL"], "weights": [1.0], "return": 0.15, "confidence": 0.7},
        ]
        result = black_litterman(multi_asset_returns, market_caps, views=views)
        assert "weights" in result
        # With bullish view on AAPL, weight should be notable
        assert result["weights"]["AAPL"] > 0

    def test_relative_view(self, multi_asset_returns, market_caps):
        views = [
            {"assets": ["AAPL", "MSFT"], "weights": [1, -1], "return": 0.05, "confidence": 0.6},
        ]
        result = black_litterman(multi_asset_returns, market_caps, views=views)
        assert "weights" in result

    def test_implied_returns_count(self, multi_asset_returns, market_caps):
        result = black_litterman(multi_asset_returns, market_caps)
        assert len(result["implied_returns"]) == 4


# ── Rebalancing ───────────────────────────────────────────────────────────


class TestRebalance:
    def test_no_trades_when_aligned(self):
        current = {"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.25, "AMZN": 0.25}
        target = {"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.25, "AMZN": 0.25}
        prices = {"AAPL": 200, "MSFT": 400, "GOOGL": 170, "AMZN": 180}
        result = rebalance_portfolio(current, target, 100_000, prices)
        assert result["num_trades"] == 0

    def test_generates_trades_on_drift(self):
        current = {"AAPL": 0.40, "MSFT": 0.10, "GOOGL": 0.25, "AMZN": 0.25}
        target = {"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.25, "AMZN": 0.25}
        prices = {"AAPL": 200, "MSFT": 400, "GOOGL": 170, "AMZN": 180}
        result = rebalance_portfolio(current, target, 100_000, prices)
        assert result["num_trades"] > 0

    def test_trade_actions_correct(self):
        current = {"AAPL": 0.40, "MSFT": 0.10}
        target = {"AAPL": 0.25, "MSFT": 0.25}
        prices = {"AAPL": 200, "MSFT": 400}
        result = rebalance_portfolio(current, target, 100_000, prices)
        actions = {t["symbol"]: t["action"] for t in result["trades"]}
        assert actions.get("AAPL") == "SELL"
        assert actions.get("MSFT") == "BUY"

    def test_total_cost_nonnegative(self):
        current = {"AAPL": 0.50, "MSFT": 0.50}
        target = {"AAPL": 0.30, "MSFT": 0.70}
        prices = {"AAPL": 200, "MSFT": 400}
        result = rebalance_portfolio(current, target, 100_000, prices)
        assert result["total_cost"] >= 0

    def test_skips_tiny_rebalances(self):
        current = {"AAPL": 0.251, "MSFT": 0.749}
        target = {"AAPL": 0.250, "MSFT": 0.750}
        prices = {"AAPL": 200, "MSFT": 400}
        result = rebalance_portfolio(current, target, 100_000, prices)
        assert result["num_trades"] == 0


# ── Consolidated optimise_portfolio ───────────────────────────────────────


class TestOptimisePortfolio:
    def test_returns_mv_and_rp(self, multi_asset_returns):
        result = optimise_portfolio(multi_asset_returns)
        assert "mean_variance" in result
        assert "risk_parity" in result

    def test_includes_bl_when_caps_given(self, multi_asset_returns, market_caps):
        result = optimise_portfolio(multi_asset_returns, market_caps=market_caps)
        assert "black_litterman" in result

    def test_includes_rebalance(self, multi_asset_returns):
        current = {"AAPL": 0.40, "MSFT": 0.30, "GOOGL": 0.20, "AMZN": 0.10}
        prices = {"AAPL": 200, "MSFT": 400, "GOOGL": 170, "AMZN": 180}
        result = optimise_portfolio(
            multi_asset_returns,
            current_weights=current,
            prices=prices,
        )
        assert "rebalance" in result
