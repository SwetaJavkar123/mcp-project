"""Tests for agents/execution_agent.py — paper trading engine."""

import pytest

from agents.execution_agent import (
    ExecutionAgent,
    PaperBroker,
    Order,
    OrderType,
    OrderStatus,
    TradingMode,
)


class TestPaperBroker:
    def test_initial_state(self):
        broker = PaperBroker(initial_cash=50_000)
        assert broker.cash == 50_000
        assert broker.get_positions() == {}
        assert broker.get_orders() == []

    def test_market_buy(self):
        broker = PaperBroker(initial_cash=100_000)
        order = broker.submit_order("AAPL", "buy", 10, current_price=150.0)
        assert order.status == OrderStatus.FILLED
        assert order.filled_price > 0
        assert broker.cash < 100_000
        assert "AAPL" in broker.get_positions()
        assert broker.get_positions()["AAPL"]["shares"] == 10

    def test_market_sell(self):
        broker = PaperBroker(initial_cash=100_000)
        broker.submit_order("AAPL", "buy", 10, current_price=150.0)
        order = broker.submit_order("AAPL", "sell", 5, current_price=160.0)
        assert order.status == OrderStatus.FILLED
        assert broker.get_positions()["AAPL"]["shares"] == 5

    def test_sell_all_removes_position(self):
        broker = PaperBroker(initial_cash=100_000)
        broker.submit_order("AAPL", "buy", 10, current_price=150.0)
        broker.submit_order("AAPL", "sell", 10, current_price=160.0)
        assert "AAPL" not in broker.get_positions()

    def test_insufficient_cash_rejected(self):
        broker = PaperBroker(initial_cash=100)
        order = broker.submit_order("AAPL", "buy", 10, current_price=150.0)
        assert order.status == OrderStatus.REJECTED
        assert "cash" in order.notes.lower()

    def test_insufficient_shares_rejected(self):
        broker = PaperBroker(initial_cash=100_000)
        order = broker.submit_order("AAPL", "sell", 10, current_price=150.0)
        assert order.status == OrderStatus.REJECTED
        assert "shares" in order.notes.lower()

    def test_limit_buy_triggered(self):
        broker = PaperBroker(initial_cash=100_000)
        order = broker.submit_order(
            "AAPL", "buy", 10,
            order_type=OrderType.LIMIT,
            limit_price=155.0,
            current_price=150.0,  # price is below limit → triggers
        )
        assert order.status == OrderStatus.FILLED

    def test_limit_buy_not_triggered(self):
        broker = PaperBroker(initial_cash=100_000)
        order = broker.submit_order(
            "AAPL", "buy", 10,
            order_type=OrderType.LIMIT,
            limit_price=140.0,
            current_price=150.0,  # price is above limit → pending
        )
        assert order.status == OrderStatus.PENDING

    def test_order_history(self):
        broker = PaperBroker(initial_cash=100_000)
        broker.submit_order("AAPL", "buy", 10, current_price=150.0)
        broker.submit_order("MSFT", "buy", 5, current_price=400.0)
        history = broker.get_order_history()
        assert len(history) == 2
        assert all(isinstance(h, dict) for h in history)

    def test_portfolio_value(self):
        broker = PaperBroker(initial_cash=100_000)
        broker.submit_order("AAPL", "buy", 10, current_price=150.0)
        value = broker.get_portfolio_value(prices={"AAPL": 160.0})
        assert value > 0

    def test_filter_orders_by_symbol(self):
        broker = PaperBroker(initial_cash=100_000)
        broker.submit_order("AAPL", "buy", 10, current_price=150.0)
        broker.submit_order("MSFT", "buy", 5, current_price=400.0)
        aapl_orders = broker.get_orders("AAPL")
        assert len(aapl_orders) == 1
        assert aapl_orders[0].symbol == "AAPL"


class TestExecutionAgent:
    def test_paper_mode(self):
        agent = ExecutionAgent(mode=TradingMode.PAPER)
        order = agent.execute("AAPL", "BUY", 10, current_price=150.0)
        assert order.status == OrderStatus.FILLED

    def test_get_positions(self):
        agent = ExecutionAgent(mode=TradingMode.PAPER, initial_cash=100_000)
        agent.execute("AAPL", "BUY", 10, current_price=150.0)
        positions = agent.get_positions()
        assert "AAPL" in positions

    def test_get_portfolio_value(self):
        agent = ExecutionAgent(mode=TradingMode.PAPER, initial_cash=100_000)
        value = agent.get_portfolio_value()
        assert value == 100_000

    def test_live_mode_raises(self):
        with pytest.raises(NotImplementedError):
            ExecutionAgent(mode=TradingMode.LIVE)
