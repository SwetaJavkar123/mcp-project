"""Tests for agents/advanced_backtest.py — walk-forward, Monte Carlo, param opt, multi-strategy."""

import numpy as np
import pandas as pd
import pytest

from agents.advanced_backtest import (
    WalkForwardConfig,
    walk_forward_analysis,
    MonteCarloConfig,
    monte_carlo_simulation,
    ParamGrid,
    optimise_parameters,
    multi_strategy_backtest,
)
from agents.backtest_agent import BacktestConfig
from agents.strategy_agent import generate_signals


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_signalled_df(enriched_df, strategy="momentum"):
    return generate_signals(enriched_df.copy(), strategy_type=strategy)


# ── Walk-Forward ──────────────────────────────────────────────────────────


class TestWalkForward:
    def test_returns_expected_keys(self, enriched_df):
        wf = WalkForwardConfig(train_window=40, test_window=20, step_size=20)
        result = walk_forward_analysis(enriched_df, wf_config=wf)
        assert "folds" in result
        assert "combined_equity" in result
        assert "combined_summary" in result
        assert "all_trades" in result

    def test_produces_folds(self, enriched_df):
        wf = WalkForwardConfig(train_window=40, test_window=20, step_size=20)
        result = walk_forward_analysis(enriched_df, wf_config=wf)
        assert result["combined_summary"]["num_folds"] >= 1

    def test_fold_has_periods(self, enriched_df):
        wf = WalkForwardConfig(train_window=40, test_window=20, step_size=20)
        result = walk_forward_analysis(enriched_df, wf_config=wf)
        if result["folds"]:
            fold = result["folds"][0]
            assert "train_period" in fold
            assert "test_period" in fold
            assert "test_summary" in fold

    def test_final_equity_positive(self, enriched_df):
        wf = WalkForwardConfig(train_window=40, test_window=20, step_size=20)
        result = walk_forward_analysis(enriched_df, wf_config=wf)
        assert result["combined_summary"]["final_equity"] > 0

    def test_too_short_data_returns_zero_folds(self, enriched_df):
        tiny = enriched_df.iloc[:10]
        wf = WalkForwardConfig(train_window=100, test_window=50, step_size=50)
        result = walk_forward_analysis(tiny, wf_config=wf)
        assert result["combined_summary"]["num_folds"] == 0


# ── Monte Carlo ───────────────────────────────────────────────────────────


class TestMonteCarlo:
    def test_returns_expected_keys(self, sample_returns):
        result = monte_carlo_simulation(sample_returns)
        assert "simulations" in result
        assert "percentiles" in result
        assert "final_values" in result
        assert "statistics" in result

    def test_simulation_shape(self, sample_returns):
        cfg = MonteCarloConfig(num_simulations=100, num_days=50)
        result = monte_carlo_simulation(sample_returns, config=cfg)
        assert result["simulations"].shape == (100, 50)

    def test_percentiles_dataframe(self, sample_returns):
        cfg = MonteCarloConfig(num_simulations=200, num_days=60)
        result = monte_carlo_simulation(sample_returns, config=cfg)
        assert isinstance(result["percentiles"], pd.DataFrame)
        assert len(result["percentiles"]) == 60

    def test_final_values_has_percentiles(self, sample_returns):
        result = monte_carlo_simulation(sample_returns)
        assert "p05" in result["final_values"]
        assert "p50" in result["final_values"]
        assert "p95" in result["final_values"]

    def test_statistics_has_prob_profit(self, sample_returns):
        result = monte_carlo_simulation(sample_returns)
        stats = result["statistics"]
        assert "prob_profit" in stats
        assert "prob_loss" in stats
        assert 0 <= stats["prob_profit"] <= 100
        assert 0 <= stats["prob_loss"] <= 100

    def test_median_positive(self, sample_returns):
        result = monte_carlo_simulation(sample_returns)
        assert result["statistics"]["median"] > 0

    def test_empty_returns(self):
        result = monte_carlo_simulation(pd.Series(dtype=float))
        assert result["simulations"].size == 0
        assert result["statistics"] == {}

    def test_worst_case_below_best(self, sample_returns):
        result = monte_carlo_simulation(sample_returns)
        assert result["statistics"]["worst_case"] <= result["statistics"]["best_case"]


# ── Parameter Optimisation ────────────────────────────────────────────────


class TestParamOptimisation:
    def test_returns_expected_keys(self, enriched_df):
        df = _make_signalled_df(enriched_df)
        grid = ParamGrid(
            stop_loss_pct=[0.03, 0.05],
            take_profit_pct=[0.06, 0.10],
            position_size_pct=[0.10],
        )
        result = optimise_parameters(df, param_grid=grid)
        assert "best_params" in result
        assert "best_result" in result
        assert "all_results" in result

    def test_tests_all_combos(self, enriched_df):
        df = _make_signalled_df(enriched_df)
        grid = ParamGrid(
            stop_loss_pct=[0.03, 0.05],
            take_profit_pct=[0.10],
            position_size_pct=[0.10],
        )
        result = optimise_parameters(df, param_grid=grid)
        assert result["total_combos_tested"] == 2  # 2 x 1 x 1

    def test_best_params_valid(self, enriched_df):
        df = _make_signalled_df(enriched_df)
        grid = ParamGrid(
            stop_loss_pct=[0.03, 0.05],
            take_profit_pct=[0.06, 0.10],
            position_size_pct=[0.10],
        )
        result = optimise_parameters(df, param_grid=grid)
        bp = result["best_params"]
        assert bp["stop_loss_pct"] in [0.03, 0.05]
        assert bp["take_profit_pct"] in [0.06, 0.10]

    def test_optimise_for_return(self, enriched_df):
        df = _make_signalled_df(enriched_df)
        grid = ParamGrid(
            stop_loss_pct=[0.05],
            take_profit_pct=[0.10],
            position_size_pct=[0.05, 0.10],
        )
        result = optimise_parameters(df, param_grid=grid, optimise_for="total_return_pct")
        assert result["optimised_for"] == "total_return_pct"


# ── Multi-Strategy Backtesting ────────────────────────────────────────────


class TestMultiStrategy:
    def test_returns_expected_keys(self, enriched_df):
        result = multi_strategy_backtest(
            enriched_df,
            strategies=["momentum", "mean_reversion"],
        )
        assert "individual" in result
        assert "comparison" in result
        assert "combined_equity" in result
        assert "combined_summary" in result
        assert "best_strategy" in result

    def test_individual_results(self, enriched_df):
        strats = ["momentum", "mean_reversion"]
        result = multi_strategy_backtest(enriched_df, strategies=strats)
        for s in strats:
            assert s in result["individual"]

    def test_comparison_dataframe(self, enriched_df):
        strats = ["momentum", "combined"]
        result = multi_strategy_backtest(enriched_df, strategies=strats)
        assert isinstance(result["comparison"], pd.DataFrame)
        assert len(result["comparison"]) == 2

    def test_best_strategy_in_list(self, enriched_df):
        strats = ["momentum", "mean_reversion", "trend_following"]
        result = multi_strategy_backtest(enriched_df, strategies=strats)
        assert result["best_strategy"] in strats

    def test_combined_equity_positive(self, enriched_df):
        result = multi_strategy_backtest(
            enriched_df,
            strategies=["momentum", "combined"],
        )
        if len(result["combined_equity"]) > 0:
            assert float(result["combined_equity"].iloc[-1]) > 0

    def test_custom_allocation(self, enriched_df):
        alloc = {"momentum": 0.7, "mean_reversion": 0.3}
        result = multi_strategy_backtest(
            enriched_df,
            strategies=["momentum", "mean_reversion"],
            allocation=alloc,
        )
        assert "individual" in result
        assert len(result["individual"]) == 2
