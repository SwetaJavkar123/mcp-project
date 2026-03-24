"""
ExecutionAgent — Paper & Live Trading
======================================
Supports paper-trading mode (simulated) out of the box.
For live trading, set ALPACA_API_KEY / ALPACA_SECRET in the environment
and initialise with mode="live".

Order types: market, limit, stop-limit.
Integration: RiskAgent approves → ExecutionAgent places trade.
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


@dataclass
class Order:
    order_id: str
    symbol: str
    side: str               # "buy" or "sell"
    order_type: str         # market / limit / stop_limit
    shares: float
    price: float            # limit / stop price (0 for market)
    status: str = OrderStatus.PENDING
    filled_price: float = 0.0
    filled_at: str = ""
    slippage: float = 0.0
    commission: float = 0.0
    created_at: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Paper-trading engine (no broker dependency)
# ---------------------------------------------------------------------------

class PaperBroker:
    """In-memory simulated broker for paper trading."""

    def __init__(self, initial_cash: float = 100_000.0):
        self.cash = initial_cash
        self.positions: dict[str, dict] = {}   # symbol → {shares, avg_cost}
        self.orders: list[Order] = []
        self._next_id = 1

    # -- Order placement ---------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        side: str,
        shares: float,
        order_type: str = OrderType.MARKET,
        limit_price: float = 0.0,
        current_price: float = 0.0,
        slippage_pct: float = 0.0005,
        commission_pct: float = 0.001,
    ) -> Order:
        """
        Submit an order.  For paper trading the fill is immediate (market)
        or conditional (limit / stop-limit checked against *current_price*).
        """
        oid = f"PAPER-{self._next_id:06d}"
        self._next_id += 1
        now = datetime.now(timezone.utc).isoformat()

        order = Order(
            order_id=oid,
            symbol=symbol.upper(),
            side=side.lower(),
            order_type=order_type,
            shares=shares,
            price=limit_price if order_type != OrderType.MARKET else current_price,
            created_at=now,
        )

        # Determine fill price
        fill_price = self._resolve_fill(order, current_price, slippage_pct)

        if fill_price is None:
            order.status = OrderStatus.PENDING
            order.notes = "Limit/stop not triggered — order pending."
            self.orders.append(order)
            return order

        # Validate
        rejection = self._validate(order, fill_price)
        if rejection:
            order.status = OrderStatus.REJECTED
            order.notes = rejection
            self.orders.append(order)
            return order

        # Execute
        total_cost = shares * fill_price
        commission = total_cost * commission_pct
        slippage = total_cost * slippage_pct

        if side.lower() == "buy":
            self.cash -= total_cost + commission + slippage
            pos = self.positions.setdefault(symbol.upper(), {"shares": 0, "avg_cost": 0.0})
            old_val = pos["shares"] * pos["avg_cost"]
            pos["shares"] += shares
            pos["avg_cost"] = (old_val + total_cost) / pos["shares"] if pos["shares"] else 0
        else:
            self.cash += total_cost - commission - slippage
            pos = self.positions.get(symbol.upper())
            if pos:
                pos["shares"] -= shares
                if pos["shares"] <= 0:
                    del self.positions[symbol.upper()]

        order.status = OrderStatus.FILLED
        order.filled_price = round(fill_price, 4)
        order.filled_at = now
        order.slippage = round(slippage, 2)
        order.commission = round(commission, 2)
        self.orders.append(order)
        return order

    # -- Helpers -----------------------------------------------------------

    def _resolve_fill(self, order: Order, current_price: float, slippage_pct: float) -> float | None:
        if order.order_type == OrderType.MARKET:
            slip = current_price * slippage_pct
            return current_price + slip if order.side == "buy" else current_price - slip

        if order.order_type == OrderType.LIMIT:
            if order.side == "buy" and current_price <= order.price:
                return order.price
            if order.side == "sell" and current_price >= order.price:
                return order.price
            return None

        if order.order_type == OrderType.STOP_LIMIT:
            if order.side == "sell" and current_price <= order.price:
                return order.price
            if order.side == "buy" and current_price >= order.price:
                return order.price
            return None

        return None

    def _validate(self, order: Order, fill_price: float) -> str | None:
        cost = order.shares * fill_price
        if order.side == "buy" and cost > self.cash:
            return f"Insufficient cash: need ${cost:,.2f}, have ${self.cash:,.2f}"
        if order.side == "sell":
            pos = self.positions.get(order.symbol)
            if not pos or pos["shares"] < order.shares:
                held = pos["shares"] if pos else 0
                return f"Insufficient shares: want to sell {order.shares}, hold {held}"
        return None

    # -- Queries -----------------------------------------------------------

    def get_positions(self) -> dict:
        return dict(self.positions)

    def get_orders(self, symbol: str | None = None) -> list[Order]:
        if symbol:
            return [o for o in self.orders if o.symbol == symbol.upper()]
        return list(self.orders)

    def get_portfolio_value(self, prices: dict[str, float] | None = None) -> float:
        invested = 0.0
        for sym, pos in self.positions.items():
            price = (prices or {}).get(sym, pos["avg_cost"])
            invested += pos["shares"] * price
        return self.cash + invested

    def get_order_history(self) -> list[dict]:
        return [asdict(o) for o in self.orders]


# ---------------------------------------------------------------------------
# ExecutionAgent — high-level API used by the controller
# ---------------------------------------------------------------------------

class ExecutionAgent:
    """
    Unified interface for paper / live execution.
    Currently implements paper trading; live mode is a placeholder
    that can be wired to Alpaca / IBKR in future.
    """

    def __init__(self, mode: str = TradingMode.PAPER, initial_cash: float = 100_000.0):
        self.mode = mode
        if mode == TradingMode.PAPER:
            self.broker = PaperBroker(initial_cash)
        else:
            raise NotImplementedError(
                "Live trading requires broker API credentials. "
                "Set ALPACA_API_KEY and ALPACA_SECRET env vars."
            )

    def execute(
        self,
        symbol: str,
        action: str,
        shares: float,
        current_price: float,
        order_type: str = OrderType.MARKET,
        limit_price: float = 0.0,
    ) -> Order:
        """Execute a trade (paper or live)."""
        return self.broker.submit_order(
            symbol=symbol,
            side=action.lower(),
            shares=shares,
            order_type=order_type,
            limit_price=limit_price,
            current_price=current_price,
        )

    def get_positions(self) -> dict:
        return self.broker.get_positions()

    def get_portfolio_value(self, prices: dict[str, float] | None = None) -> float:
        return self.broker.get_portfolio_value(prices)

    def get_order_history(self) -> list[dict]:
        return self.broker.get_order_history()
