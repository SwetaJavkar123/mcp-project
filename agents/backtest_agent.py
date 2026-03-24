"""
BacktestAgent
Simulates a trading strategy on historical price data and returns
performance metrics + trade log.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from agents.risk_agent import calculate_max_drawdown, calculate_sharpe, calculate_sortino


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001          # 0.1% per trade (round-trip)
    slippage_pct: float = 0.0005           # 0.05% slippage per trade
    position_size_pct: float = 0.10        # use 10% of capital per trade
    stop_loss_pct: float = 0.05            # 5% stop loss
    take_profit_pct: float = 0.10          # 10% take profit
    max_open_positions: int = 5


@dataclass
class Trade:
    symbol: str
    entry_date: str
    entry_price: float
    exit_date: str | None = None
    exit_price: float | None = None
    shares: float = 0.0
    side: str = "LONG"
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""


def run_backtest(
    df: pd.DataFrame,
    symbol: str = "STOCK",
    config: BacktestConfig | None = None,
) -> dict:
    """
    Run a backtest on a DataFrame that already contains 'Close', 'Signal',
    and 'Confidence' columns (output of StrategyAgent).

    Returns:
        {
            summary: dict,         # performance summary
            equity_curve: pd.Series,
            trades: list[Trade],
            monthly_returns: pd.Series,
        }
    """
    if config is None:
        config = BacktestConfig()

    df = df.copy()
    if "Signal" not in df.columns:
        raise ValueError("DataFrame must contain a 'Signal' column")

    capital = config.initial_capital
    equity = capital
    position: Trade | None = None
    trades: list[Trade] = []
    equity_curve = []

    for i in range(len(df)):
        date = df.index[i] if isinstance(df.index[i], str) else str(df.index[i].date()) if hasattr(df.index[i], 'date') else str(df.index[i])
        close = float(df["Close"].iloc[i])
        signal = df["Signal"].iloc[i]
        confidence = float(df["Confidence"].iloc[i]) if "Confidence" in df.columns else 50.0

        # -- Check stop-loss / take-profit on open position --
        if position is not None:
            pnl_pct = (close - position.entry_price) / position.entry_price
            if position.side == "SHORT":
                pnl_pct = -pnl_pct

            # Stop loss
            if pnl_pct <= -config.stop_loss_pct:
                _close_position(position, close, date, "STOP_LOSS", config)
                capital += position.shares * close * (1 - config.commission_pct)
                trades.append(position)
                position = None

            # Take profit
            elif pnl_pct >= config.take_profit_pct:
                _close_position(position, close, date, "TAKE_PROFIT", config)
                capital += position.shares * close * (1 - config.commission_pct)
                trades.append(position)
                position = None

        # -- Act on signals --
        if signal == "BUY" and position is None and confidence >= 30:
            # Open long
            trade_capital = capital * config.position_size_pct
            shares = int(trade_capital / (close * (1 + config.commission_pct + config.slippage_pct)))
            if shares > 0:
                cost = shares * close * (1 + config.commission_pct + config.slippage_pct)
                capital -= cost
                position = Trade(
                    symbol=symbol,
                    entry_date=date,
                    entry_price=close,
                    shares=shares,
                    side="LONG",
                )

        elif signal == "SELL" and position is not None:
            # Close long
            _close_position(position, close, date, "SIGNAL_SELL", config)
            capital += position.shares * close * (1 - config.commission_pct)
            trades.append(position)
            position = None

        # Track equity
        pos_value = position.shares * close if position else 0.0
        equity = capital + pos_value
        equity_curve.append(equity)

    # Close any open position at end
    if position is not None:
        close = float(df["Close"].iloc[-1])
        date = str(df.index[-1].date()) if hasattr(df.index[-1], 'date') else str(df.index[-1])
        _close_position(position, close, date, "END_OF_DATA", config)
        capital += position.shares * close * (1 - config.commission_pct)
        trades.append(position)

    equity_series = pd.Series(equity_curve, index=df.index)
    returns = equity_series.pct_change().dropna()

    summary = _build_summary(trades, equity_series, returns, config)
    monthly_returns = _monthly_returns(equity_series)

    return {
        "summary": summary,
        "equity_curve": equity_series,
        "trades": trades,
        "monthly_returns": monthly_returns,
    }


def _close_position(position: Trade, close: float, date: str, reason: str, config: BacktestConfig):
    """Fill exit fields on a Trade object."""
    position.exit_date = date
    position.exit_price = close
    position.pnl = (close - position.entry_price) * position.shares
    if position.side == "SHORT":
        position.pnl = -position.pnl
    position.pnl -= position.shares * close * config.commission_pct  # exit commission
    position.pnl_pct = position.pnl / (position.entry_price * position.shares) if position.entry_price else 0
    position.exit_reason = reason


def _build_summary(trades: list[Trade], equity: pd.Series, returns: pd.Series, config: BacktestConfig) -> dict:
    """Aggregate performance metrics."""
    winning = [t for t in trades if t.pnl > 0]
    losing = [t for t in trades if t.pnl <= 0]

    total_pnl = sum(t.pnl for t in trades)
    gross_profit = sum(t.pnl for t in winning)
    gross_loss = sum(t.pnl for t in losing)

    return {
        "initial_capital": config.initial_capital,
        "final_equity": round(float(equity.iloc[-1]), 2) if len(equity) else config.initial_capital,
        "total_return_pct": round((float(equity.iloc[-1]) / config.initial_capital - 1) * 100, 2) if len(equity) else 0,
        "total_pnl": round(total_pnl, 2),
        "total_trades": len(trades),
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "win_rate": round(len(winning) / len(trades) * 100, 2) if trades else 0,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(gross_profit / abs(gross_loss), 2) if gross_loss else float("inf"),
        "avg_win": round(gross_profit / len(winning), 2) if winning else 0,
        "avg_loss": round(gross_loss / len(losing), 2) if losing else 0,
        "largest_win": round(max((t.pnl for t in trades), default=0), 2),
        "largest_loss": round(min((t.pnl for t in trades), default=0), 2),
        "max_drawdown": round(calculate_max_drawdown(equity) * 100, 2) if len(equity) > 1 else 0,
        "sharpe_ratio": round(calculate_sharpe(returns), 2) if len(returns) > 1 else 0,
        "sortino_ratio": round(calculate_sortino(returns), 2) if len(returns) > 1 else 0,
    }


def _monthly_returns(equity: pd.Series) -> pd.Series:
    """Resample equity curve to monthly returns."""
    try:
        monthly = equity.resample("ME").last()
        return monthly.pct_change().dropna()
    except Exception:
        return pd.Series(dtype=float)


def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    """Convert list of Trade objects to a DataFrame."""
    rows = []
    for t in trades:
        rows.append({
            "Symbol": t.symbol,
            "Side": t.side,
            "Entry Date": t.entry_date,
            "Entry Price": round(t.entry_price, 2),
            "Exit Date": t.exit_date,
            "Exit Price": round(t.exit_price, 2) if t.exit_price else None,
            "Shares": t.shares,
            "P&L ($)": round(t.pnl, 2),
            "P&L (%)": f"{t.pnl_pct:.2%}",
            "Exit Reason": t.exit_reason,
        })
    return pd.DataFrame(rows)
