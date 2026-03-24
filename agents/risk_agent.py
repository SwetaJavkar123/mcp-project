"""
RiskAgent
Validates proposed trades against portfolio-level risk constraints.
Calculates VaR, Sharpe, max drawdown, position sizing, and stop-loss levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data classes for portfolio / trade representation
# ---------------------------------------------------------------------------

@dataclass
class Position:
    symbol: str
    shares: float
    avg_cost: float
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def pnl(self) -> float:
        return (self.current_price - self.avg_cost) * self.shares

    @property
    def pnl_pct(self) -> float:
        if self.avg_cost == 0:
            return 0.0
        return (self.current_price - self.avg_cost) / self.avg_cost


@dataclass
class Portfolio:
    cash: float = 100_000.0
    positions: list[Position] = field(default_factory=list)
    max_position_pct: float = 0.20       # max 20% of portfolio in one name
    max_portfolio_risk_pct: float = 0.02  # max 2% daily VaR
    max_correlation: float = 0.85         # max correlation between positions
    stop_loss_pct: float = 0.05           # 5% stop loss per position

    @property
    def total_value(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions)

    def position_for(self, symbol: str) -> Position | None:
        for p in self.positions:
            if p.symbol == symbol:
                return p
        return None


@dataclass
class TradeProposal:
    symbol: str
    action: str              # BUY or SELL
    shares: float
    price: float
    confidence: float = 50.0
    strategy: str = ""


# ---------------------------------------------------------------------------
# Risk metrics calculated on a price DataFrame
# ---------------------------------------------------------------------------

def calculate_portfolio_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Daily returns from a DataFrame of closing prices (columns = symbols)."""
    return prices_df.pct_change().dropna()


def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value-at-Risk at the given confidence level."""
    if returns.empty:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR (expected shortfall)."""
    var = calculate_var(returns, confidence)
    return float(returns[returns <= var].mean()) if (returns <= var).any() else var


def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.04) -> float:
    """Annualised Sharpe ratio (assumes 252 trading days)."""
    if returns.std() == 0:
        return 0.0
    excess = returns.mean() - risk_free_rate / 252
    return float(excess / returns.std() * np.sqrt(252))


def calculate_sortino(returns: pd.Series, risk_free_rate: float = 0.04) -> float:
    """Annualised Sortino ratio."""
    downside = returns[returns < 0]
    if downside.std() == 0:
        return 0.0
    excess = returns.mean() - risk_free_rate / 252
    return float(excess / downside.std() * np.sqrt(252))


def calculate_max_drawdown(prices: pd.Series) -> float:
    """Maximum drawdown as a negative fraction (e.g. -0.15 = 15% drawdown)."""
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return float(drawdown.min())


def calculate_risk_metrics(returns: pd.Series, prices: pd.Series | None = None) -> dict:
    """Bundle all risk metrics into a dict."""
    metrics = {
        "daily_mean_return": float(returns.mean()),
        "daily_volatility": float(returns.std()),
        "annualised_return": float(returns.mean() * 252),
        "annualised_volatility": float(returns.std() * np.sqrt(252)),
        "var_95": calculate_var(returns, 0.95),
        "cvar_95": calculate_cvar(returns, 0.95),
        "sharpe_ratio": calculate_sharpe(returns),
        "sortino_ratio": calculate_sortino(returns),
    }
    if prices is not None:
        metrics["max_drawdown"] = calculate_max_drawdown(prices)
    return metrics


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def calculate_position_size(
    portfolio_value: float,
    price: float,
    risk_per_trade: float = 0.02,
    stop_loss_pct: float = 0.05,
    max_position_pct: float = 0.20,
) -> dict:
    """
    Kelly-inspired position sizing.
    Returns {shares, dollar_amount, pct_of_portfolio, stop_loss_price}.
    """
    risk_amount = portfolio_value * risk_per_trade
    shares_from_risk = int(risk_amount / (price * stop_loss_pct)) if price > 0 else 0
    max_shares = int(portfolio_value * max_position_pct / price) if price > 0 else 0
    shares = min(shares_from_risk, max_shares)

    return {
        "shares": shares,
        "dollar_amount": round(shares * price, 2),
        "pct_of_portfolio": round(shares * price / portfolio_value * 100, 2) if portfolio_value else 0,
        "stop_loss_price": round(price * (1 - stop_loss_pct), 2),
        "take_profit_price": round(price * (1 + stop_loss_pct * 2), 2),  # 2:1 reward-risk
    }


# ---------------------------------------------------------------------------
# Trade validation
# ---------------------------------------------------------------------------

def validate_trade(
    proposal: TradeProposal,
    portfolio: Portfolio,
    returns: pd.Series | None = None,
) -> dict:
    """
    Validate a trade proposal against portfolio risk rules.

    Returns:
        {
            approved: bool,
            reasons: list[str],          # rejection reasons (if any)
            warnings: list[str],
            position_size: dict,
            risk_metrics: dict | None,
        }
    """
    reasons: list[str] = []
    warnings: list[str] = []

    # 1. Concentration check
    proposed_value = proposal.shares * proposal.price
    pct_of_portfolio = proposed_value / portfolio.total_value if portfolio.total_value else 0
    if pct_of_portfolio > portfolio.max_position_pct:
        reasons.append(
            f"Position size ({pct_of_portfolio:.1%}) exceeds max allowed ({portfolio.max_position_pct:.0%})"
        )

    # 2. Cash check for buys
    if proposal.action == "BUY" and proposed_value > portfolio.cash:
        reasons.append(
            f"Insufficient cash: need ${proposed_value:,.0f}, have ${portfolio.cash:,.0f}"
        )

    # 3. Shares check for sells
    if proposal.action == "SELL":
        pos = portfolio.position_for(proposal.symbol)
        if pos is None:
            reasons.append(f"No existing position in {proposal.symbol} to sell")
        elif proposal.shares > pos.shares:
            reasons.append(
                f"Cannot sell {proposal.shares} shares – only hold {pos.shares}"
            )

    # 4. Confidence threshold
    if proposal.confidence < 30:
        warnings.append(f"Low signal confidence ({proposal.confidence:.0f}%)")

    # 5. Portfolio-level VaR
    risk_metrics = None
    if returns is not None and not returns.empty:
        risk_metrics = calculate_risk_metrics(returns)
        if abs(risk_metrics["var_95"]) > portfolio.max_portfolio_risk_pct:
            warnings.append(
                f"Portfolio daily VaR ({risk_metrics['var_95']:.2%}) exceeds limit ({portfolio.max_portfolio_risk_pct:.0%})"
            )

    # 6. Recommended position size
    sizing = calculate_position_size(
        portfolio.total_value,
        proposal.price,
        stop_loss_pct=portfolio.stop_loss_pct,
        max_position_pct=portfolio.max_position_pct,
    )

    approved = len(reasons) == 0
    return {
        "approved": approved,
        "reasons": reasons,
        "warnings": warnings,
        "position_size": sizing,
        "risk_metrics": risk_metrics,
    }


def calculate_portfolio_summary(portfolio: Portfolio) -> dict:
    """Human-readable portfolio snapshot."""
    positions_data = []
    for p in portfolio.positions:
        positions_data.append({
            "symbol": p.symbol,
            "shares": p.shares,
            "avg_cost": p.avg_cost,
            "current_price": p.current_price,
            "market_value": p.market_value,
            "pnl": round(p.pnl, 2),
            "pnl_pct": f"{p.pnl_pct:.2%}",
        })

    return {
        "cash": portfolio.cash,
        "invested": sum(p.market_value for p in portfolio.positions),
        "total_value": portfolio.total_value,
        "num_positions": len(portfolio.positions),
        "positions": positions_data,
    }
