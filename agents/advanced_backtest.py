"""
Advanced Backtesting Agent
===========================
Extends the core BacktestAgent with:

  - Walk-forward analysis (rolling train / test windows)
  - Monte Carlo simulation of strategy returns
  - Parameter optimisation (grid search)
  - Multi-strategy portfolio backtesting
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from itertools import product
from typing import Callable

from agents.backtest_agent import run_backtest, BacktestConfig, Trade
from agents.risk_agent import calculate_sharpe, calculate_max_drawdown


# ---------------------------------------------------------------------------
# Walk-Forward Analysis
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardConfig:
    train_window: int = 252       # trading days for in-sample
    test_window: int = 63         # trading days for out-of-sample (~1 quarter)
    step_size: int = 63           # days to advance between folds
    strategy_type: str = "combined"


def walk_forward_analysis(
    df: pd.DataFrame,
    symbol: str = "STOCK",
    wf_config: WalkForwardConfig | None = None,
    bt_config: BacktestConfig | None = None,
    signal_fn: Callable | None = None,
) -> dict:
    """
    Run walk-forward analysis: train on in-sample, test on out-of-sample,
    roll forward.

    Parameters
    ----------
    df          : OHLCV DataFrame with indicator columns.
    signal_fn   : Optional callable(df, strategy_type) → df with Signal column.
                  Defaults to agents.strategy_agent.generate_signals.

    Returns
    -------
    {
        folds: list of {train_period, test_period, test_summary, test_trades},
        combined_equity: pd.Series,
        combined_summary: dict,
    }
    """
    if wf_config is None:
        wf_config = WalkForwardConfig()
    if bt_config is None:
        bt_config = BacktestConfig()
    if signal_fn is None:
        from agents.strategy_agent import generate_signals
        signal_fn = lambda d, st=wf_config.strategy_type: generate_signals(d, strategy_type=st)

    n = len(df)
    folds: list[dict] = []
    combined_equities: list[pd.Series] = []
    all_trades: list[Trade] = []
    capital = bt_config.initial_capital

    start = 0
    while start + wf_config.train_window + wf_config.test_window <= n:
        train_end = start + wf_config.train_window
        test_end = min(train_end + wf_config.test_window, n)

        train_df = df.iloc[start:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()

        if len(test_df) < 5:
            break

        # Generate signals on full in-sample + out-of-sample, then slice test
        try:
            full_slice = df.iloc[start:test_end].copy()
            full_signals = signal_fn(full_slice)
            test_signals = full_signals.iloc[train_end - start:].copy()
        except Exception:
            start += wf_config.step_size
            continue

        if "Signal" not in test_signals.columns:
            start += wf_config.step_size
            continue

        # Backtest on out-of-sample
        fold_config = BacktestConfig(
            initial_capital=capital,
            commission_pct=bt_config.commission_pct,
            slippage_pct=bt_config.slippage_pct,
            position_size_pct=bt_config.position_size_pct,
            stop_loss_pct=bt_config.stop_loss_pct,
            take_profit_pct=bt_config.take_profit_pct,
            max_open_positions=bt_config.max_open_positions,
        )

        try:
            result = run_backtest(test_signals, symbol=symbol, config=fold_config)
        except Exception:
            start += wf_config.step_size
            continue

        train_start_date = str(train_df.index[0].date()) if hasattr(train_df.index[0], "date") else str(train_df.index[0])
        train_end_date = str(train_df.index[-1].date()) if hasattr(train_df.index[-1], "date") else str(train_df.index[-1])
        test_start_date = str(test_df.index[0].date()) if hasattr(test_df.index[0], "date") else str(test_df.index[0])
        test_end_date = str(test_df.index[-1].date()) if hasattr(test_df.index[-1], "date") else str(test_df.index[-1])

        folds.append({
            "fold": len(folds) + 1,
            "train_period": f"{train_start_date} → {train_end_date}",
            "test_period": f"{test_start_date} → {test_end_date}",
            "test_summary": result["summary"],
            "num_test_trades": result["summary"]["total_trades"],
        })

        combined_equities.append(result["equity_curve"])
        all_trades.extend(result["trades"])
        capital = result["summary"]["final_equity"]

        start += wf_config.step_size

    # Combine equity curves
    if combined_equities:
        combined_equity = pd.concat(combined_equities)
        combined_returns = combined_equity.pct_change().dropna()
        combined_summary = {
            "num_folds": len(folds),
            "initial_capital": bt_config.initial_capital,
            "final_equity": round(float(combined_equity.iloc[-1]), 2),
            "total_return_pct": round((float(combined_equity.iloc[-1]) / bt_config.initial_capital - 1) * 100, 2),
            "total_trades": len(all_trades),
            "sharpe_ratio": round(calculate_sharpe(combined_returns), 2) if len(combined_returns) > 1 else 0,
            "max_drawdown": round(calculate_max_drawdown(combined_equity) * 100, 2) if len(combined_equity) > 1 else 0,
        }
    else:
        combined_equity = pd.Series(dtype=float)
        combined_summary = {
            "num_folds": 0,
            "initial_capital": bt_config.initial_capital,
            "final_equity": bt_config.initial_capital,
            "total_return_pct": 0.0,
            "total_trades": 0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }

    return {
        "folds": folds,
        "combined_equity": combined_equity,
        "combined_summary": combined_summary,
        "all_trades": all_trades,
    }


# ---------------------------------------------------------------------------
# Monte Carlo Simulation
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloConfig:
    num_simulations: int = 1_000
    num_days: int = 252            # 1 year forward projection
    initial_capital: float = 100_000.0
    confidence_levels: list[float] = field(default_factory=lambda: [0.05, 0.25, 0.50, 0.75, 0.95])


def monte_carlo_simulation(
    returns: pd.Series,
    config: MonteCarloConfig | None = None,
) -> dict:
    """
    Monte Carlo simulation of future portfolio value based on historical return
    distribution.

    Parameters
    ----------
    returns : Historical daily returns (pd.Series).

    Returns
    -------
    {
        simulations: np.ndarray (num_simulations x num_days),
        percentiles: DataFrame (day x percentile),
        final_values: dict of percentile → final portfolio value,
        statistics: {mean, median, std, best, worst, prob_profit, prob_loss},
    }
    """
    if config is None:
        config = MonteCarloConfig()

    if returns.empty:
        return {
            "simulations": np.array([]),
            "percentiles": pd.DataFrame(),
            "final_values": {},
            "statistics": {},
        }

    mu = returns.mean()
    sigma = returns.std()

    # Simulate paths
    sims = np.zeros((config.num_simulations, config.num_days))
    rng = np.random.default_rng(seed=42)

    for i in range(config.num_simulations):
        daily_returns = rng.normal(mu, sigma, config.num_days)
        price_path = config.initial_capital * np.cumprod(1 + daily_returns)
        sims[i] = price_path

    # Percentile bands
    pct_data = {}
    for p in config.confidence_levels:
        pct_data[f"p{int(p*100):02d}"] = np.percentile(sims, p * 100, axis=0)
    percentiles = pd.DataFrame(pct_data, index=range(1, config.num_days + 1))
    percentiles.index.name = "day"

    # Final value distribution
    finals = sims[:, -1]
    final_values = {
        f"p{int(p*100):02d}": round(float(np.percentile(finals, p * 100)), 2)
        for p in config.confidence_levels
    }

    statistics = {
        "mean": round(float(finals.mean()), 2),
        "median": round(float(np.median(finals)), 2),
        "std": round(float(finals.std()), 2),
        "best_case": round(float(finals.max()), 2),
        "worst_case": round(float(finals.min()), 2),
        "prob_profit": round(float((finals > config.initial_capital).mean()) * 100, 2),
        "prob_loss": round(float((finals < config.initial_capital).mean()) * 100, 2),
        "expected_return_pct": round(float((finals.mean() / config.initial_capital - 1) * 100), 2),
    }

    return {
        "simulations": sims,
        "percentiles": percentiles,
        "final_values": final_values,
        "statistics": statistics,
    }


# ---------------------------------------------------------------------------
# Parameter Optimisation (Grid Search)
# ---------------------------------------------------------------------------

@dataclass
class ParamGrid:
    """Define parameter ranges for grid search."""
    stop_loss_pct: list[float] = field(default_factory=lambda: [0.03, 0.05, 0.07])
    take_profit_pct: list[float] = field(default_factory=lambda: [0.06, 0.10, 0.15])
    position_size_pct: list[float] = field(default_factory=lambda: [0.05, 0.10, 0.15])


def optimise_parameters(
    df: pd.DataFrame,
    symbol: str = "STOCK",
    param_grid: ParamGrid | None = None,
    base_config: BacktestConfig | None = None,
    optimise_for: str = "sharpe_ratio",
) -> dict:
    """
    Grid search over backtest parameters to find the best combination.

    Parameters
    ----------
    df            : DataFrame with Signal and Confidence columns.
    param_grid    : Parameter ranges to search.
    optimise_for  : Metric to maximise: 'sharpe_ratio', 'total_return_pct',
                    'profit_factor', 'win_rate'.

    Returns
    -------
    {
        best_params: dict,
        best_result: backtest summary dict,
        all_results: DataFrame of all parameter combos tested,
    }
    """
    if param_grid is None:
        param_grid = ParamGrid()
    if base_config is None:
        base_config = BacktestConfig()

    combos = list(product(
        param_grid.stop_loss_pct,
        param_grid.take_profit_pct,
        param_grid.position_size_pct,
    ))

    records: list[dict] = []

    for sl, tp, ps in combos:
        cfg = BacktestConfig(
            initial_capital=base_config.initial_capital,
            commission_pct=base_config.commission_pct,
            slippage_pct=base_config.slippage_pct,
            position_size_pct=ps,
            stop_loss_pct=sl,
            take_profit_pct=tp,
            max_open_positions=base_config.max_open_positions,
        )
        try:
            result = run_backtest(df, symbol=symbol, config=cfg)
            summary = result["summary"]
            records.append({
                "stop_loss_pct": sl,
                "take_profit_pct": tp,
                "position_size_pct": ps,
                **summary,
            })
        except Exception:
            continue

    if not records:
        return {"best_params": {}, "best_result": {}, "all_results": pd.DataFrame()}

    results_df = pd.DataFrame(records)

    # Find the best
    best_idx = results_df[optimise_for].idxmax()
    best_row = results_df.loc[best_idx]

    best_params = {
        "stop_loss_pct": float(best_row["stop_loss_pct"]),
        "take_profit_pct": float(best_row["take_profit_pct"]),
        "position_size_pct": float(best_row["position_size_pct"]),
    }

    return {
        "best_params": best_params,
        "best_result": best_row.to_dict(),
        "all_results": results_df,
        "total_combos_tested": len(records),
        "optimised_for": optimise_for,
    }


# ---------------------------------------------------------------------------
# Multi-Strategy Portfolio Backtesting
# ---------------------------------------------------------------------------

def multi_strategy_backtest(
    df: pd.DataFrame,
    symbol: str = "STOCK",
    strategies: list[str] | None = None,
    config: BacktestConfig | None = None,
    allocation: dict[str, float] | None = None,
) -> dict:
    """
    Run multiple strategies on the same data and combine results.

    Parameters
    ----------
    strategies  : List of strategy names to test.
    allocation  : {strategy_name: weight} — how to split capital.
                  Defaults to equal weight.

    Returns
    -------
    {
        individual: {strategy_name: backtest_result},
        comparison: DataFrame comparing metrics across strategies,
        combined_equity: pd.Series (weighted combination),
        combined_summary: dict,
        best_strategy: str,
    }
    """
    if strategies is None:
        strategies = ["combined", "momentum", "mean_reversion", "trend_following"]
    if config is None:
        config = BacktestConfig()

    from agents.strategy_agent import generate_signals

    n_strategies = len(strategies)
    if allocation is None:
        allocation = {s: 1.0 / n_strategies for s in strategies}

    individual_results: dict[str, dict] = {}
    comparison_rows: list[dict] = []

    for strat in strategies:
        try:
            strat_df = generate_signals(df.copy(), strategy_type=strat)
            # Allocate proportional capital
            weight = allocation.get(strat, 1.0 / n_strategies)
            strat_config = BacktestConfig(
                initial_capital=config.initial_capital * weight,
                commission_pct=config.commission_pct,
                slippage_pct=config.slippage_pct,
                position_size_pct=config.position_size_pct,
                stop_loss_pct=config.stop_loss_pct,
                take_profit_pct=config.take_profit_pct,
                max_open_positions=config.max_open_positions,
            )
            result = run_backtest(strat_df, symbol=symbol, config=strat_config)
            individual_results[strat] = result
            comparison_rows.append({"strategy": strat, **result["summary"]})
        except Exception:
            continue

    if not individual_results:
        return {
            "individual": {},
            "comparison": pd.DataFrame(),
            "combined_equity": pd.Series(dtype=float),
            "combined_summary": {},
            "best_strategy": "",
        }

    comparison_df = pd.DataFrame(comparison_rows)

    # Combined weighted equity
    equity_curves = []
    for strat, result in individual_results.items():
        eq = result["equity_curve"]
        weight = allocation.get(strat, 1.0 / n_strategies)
        # Normalise to weight-proportional contribution
        equity_curves.append(eq)

    combined_equity = sum(equity_curves)

    if len(combined_equity) > 0:
        combined_returns = combined_equity.pct_change().dropna()
        combined_summary = {
            "initial_capital": config.initial_capital,
            "final_equity": round(float(combined_equity.iloc[-1]), 2),
            "total_return_pct": round((float(combined_equity.iloc[-1]) / config.initial_capital - 1) * 100, 2),
            "sharpe_ratio": round(calculate_sharpe(combined_returns), 2) if len(combined_returns) > 1 else 0,
            "max_drawdown": round(calculate_max_drawdown(combined_equity) * 100, 2) if len(combined_equity) > 1 else 0,
            "num_strategies": len(individual_results),
        }
    else:
        combined_summary = {}

    # Best single strategy by Sharpe
    best = comparison_df.loc[comparison_df["sharpe_ratio"].idxmax(), "strategy"] if len(comparison_df) else ""

    return {
        "individual": individual_results,
        "comparison": comparison_df,
        "combined_equity": combined_equity,
        "combined_summary": combined_summary,
        "best_strategy": best,
    }
