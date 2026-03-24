"""
Hedge Fund Controller — Unified Orchestrator
=============================================
Connects every agent into a single end-to-end pipeline:

  MarketData → Indicators → Strategy Signals
       ↓                         ↓
  CompanyResearch          Risk Validation
       ↓                         ↓
  News / Sentiment         Position Sizing
       ↓                         ↓
  ─────────── Backtest ──────────

Each stage enriches a shared AnalysisResult dict so that any downstream
consumer (Streamlit UI, CLI, notebook) can pick the pieces it needs.
"""

from __future__ import annotations

import pandas as pd
from datetime import datetime

from utils_marketdata import (
    fetch_and_process_data,
    calculate_moving_average,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_stochastic,
    calculate_atr,
    calculate_adx,
    get_basic_statistics,
)
from agents.strategy_agent import generate_signals, get_strategy_description
from agents.company_research_agent import generate_research_report
from agents.risk_agent import (
    Portfolio,
    Position,
    TradeProposal,
    validate_trade,
    calculate_risk_metrics,
    calculate_position_size,
    calculate_portfolio_summary,
)
from agents.backtest_agent import run_backtest, BacktestConfig, trades_to_dataframe
from agents.news_sentiment_agent import get_sentiment_report
from agents.llm_research_agent import generate_llm_summary
from agents.execution_agent import ExecutionAgent, TradingMode, OrderType
from agents.alternative_data_agent import get_alternative_data_report
from agents.portfolio_optimizer import optimise_portfolio, OptimiserConfig
from agents.advanced_backtest import (
    walk_forward_analysis, WalkForwardConfig,
    monte_carlo_simulation, MonteCarloConfig,
    optimise_parameters, ParamGrid,
    multi_strategy_backtest,
)


# ---------------------------------------------------------------------------
# Helper: calculate all technical indicators at once
# ---------------------------------------------------------------------------

def _enrich_with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add every indicator the platform supports."""
    df = calculate_moving_average(df, window=20)
    df = calculate_rsi(df, window=14)
    df = calculate_bollinger_bands(df, window=20, num_std=2)
    df = calculate_macd(df, fast=12, slow=26, signal=9)
    df = calculate_stochastic(df, window=14)
    df = calculate_atr(df, window=14)
    df = calculate_adx(df, window=14)
    return df


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def run_full_analysis(
    symbol: str,
    start_date: str,
    end_date: str,
    strategy_type: str = "combined",
    portfolio: Portfolio | None = None,
    backtest: bool = True,
    backtest_config: BacktestConfig | None = None,
    include_sentiment: bool = True,
    include_llm_summary: bool = True,
    include_alt_data: bool = False,
    include_advanced_backtest: bool = False,
    verbose: bool = False,
) -> dict | None:
    """
    Execute the complete hedge-fund analysis pipeline.

    Parameters
    ----------
    symbol : str            Ticker (e.g. "AAPL").
    start_date, end_date :  Date strings "YYYY-MM-DD".
    strategy_type :         One of combined / momentum / mean_reversion / trend_following.
    portfolio :             Optional Portfolio object for risk checks.
    backtest :              Whether to run backtesting.
    backtest_config :       Custom backtest parameters.
    verbose :               Print progress to stdout.

    Returns
    -------
    dict with keys:
        symbol, period, strategy,
        data (DataFrame), statistics,
        signals_summary, latest,
        research_report,
        risk_validation, portfolio_summary,
        backtest_results
    """

    _log = print if verbose else (lambda *a, **k: None)

    # ── 1. Fetch market data ────────────────────────────────────────────────
    _log(f"\n{'='*60}")
    _log(f"  Hedge Fund Analysis: {symbol}  ({strategy_type})")
    _log(f"  Period: {start_date} → {end_date}")
    _log(f"{'='*60}\n")

    _log("📊  Fetching market data …")
    df = fetch_and_process_data(symbol, start_date, end_date)
    if df is None or df.empty:
        _log("❌  No data returned.")
        return None
    _log(f"✅  {len(df)} trading days loaded.\n")

    # ── 2. Technical indicators ─────────────────────────────────────────────
    _log("📈  Calculating technical indicators …")
    df = _enrich_with_indicators(df)
    stats = get_basic_statistics(df)
    _log("✅  Indicators calculated.\n")

    # ── 3. Strategy signals ─────────────────────────────────────────────────
    _log("🎯  Generating strategy signals …")
    df = generate_signals(df, strategy_type=strategy_type)
    buy_n = int((df["Signal"] == "BUY").sum())
    sell_n = int((df["Signal"] == "SELL").sum())
    hold_n = int((df["Signal"] == "HOLD").sum())
    _log(f"✅  BUY={buy_n}  SELL={sell_n}  HOLD={hold_n}\n")

    latest_price = float(df["Close"].iloc[-1])
    latest_signal = df["Signal"].iloc[-1]
    latest_confidence = float(df["Confidence"].iloc[-1])

    signals_summary = {"buy": buy_n, "sell": sell_n, "hold": hold_n}
    latest = {
        "price": latest_price,
        "signal": latest_signal,
        "confidence": latest_confidence,
        "rsi": float(df["RSI"].iloc[-1]) if "RSI" in df.columns else None,
        "macd": float(df["MACD"].iloc[-1]) if "MACD" in df.columns else None,
        "adx": float(df["ADX"].iloc[-1]) if "ADX" in df.columns else None,
    }

    # ── 4. Company research ─────────────────────────────────────────────────
    _log("🔍  Running company research …")
    try:
        research = generate_research_report(symbol)
    except Exception as exc:
        _log(f"⚠️  Research failed: {exc}")
        research = None
    _log("✅  Research report ready.\n" if research else "")

    # ── 5. Risk validation ──────────────────────────────────────────────────
    if portfolio is None:
        portfolio = Portfolio()  # default $100k paper portfolio

    risk_result = None
    if latest_signal in ("BUY", "SELL"):
        _log("🛡️  Validating trade against risk rules …")
        sizing = calculate_position_size(
            portfolio.total_value, latest_price,
            stop_loss_pct=portfolio.stop_loss_pct,
            max_position_pct=portfolio.max_position_pct,
        )
        proposal = TradeProposal(
            symbol=symbol,
            action=latest_signal,
            shares=sizing["shares"],
            price=latest_price,
            confidence=latest_confidence,
            strategy=strategy_type,
        )
        returns = df["Close"].pct_change().dropna()
        risk_result = validate_trade(proposal, portfolio, returns)
        _log(f"✅  Trade {'APPROVED' if risk_result['approved'] else 'REJECTED'}.\n")

    port_summary = calculate_portfolio_summary(portfolio)

    # ── 6. Backtest ─────────────────────────────────────────────────────────
    bt_result = None
    if backtest:
        _log("⏪  Running backtest …")
        try:
            bt_result = run_backtest(df, symbol=symbol, config=backtest_config)
            s = bt_result["summary"]
            _log(f"✅  Backtest done — Return: {s['total_return_pct']}%  "
                 f"Sharpe: {s['sharpe_ratio']}  MaxDD: {s['max_drawdown']}%\n")
        except Exception as exc:
            _log(f"⚠️  Backtest failed: {exc}")

    # ── Package result ──────────────────────────────────────────────────────

    # ── 7. News sentiment ───────────────────────────────────────────────────
    sentiment_report = None
    if include_sentiment:
        _log("📰  Analysing news sentiment …")
        try:
            sentiment_report = get_sentiment_report(symbol)
            overall = sentiment_report["summary"]["overall"]
            _log(f"✅  Sentiment: {overall} "
                 f"({sentiment_report['summary']['bullish']}↑ "
                 f"{sentiment_report['summary']['bearish']}↓)\n")
        except Exception as exc:
            _log(f"⚠️  Sentiment analysis failed: {exc}")

    # ── 8. LLM research summary ────────────────────────────────────────────
    llm_summary = None
    if include_llm_summary and research:
        _log("🤖  Generating AI research summary …")
        try:
            llm_summary = generate_llm_summary(research)
            _log("✅  AI summary ready.\n")
        except Exception as exc:
            _log(f"⚠️  LLM summary failed: {exc}")

    # ── 9. Alternative data ─────────────────────────────────────────────────
    alt_data = None
    if include_alt_data:
        _log("📂  Fetching alternative data …")
        try:
            alt_data = get_alternative_data_report(symbol)
            _log("✅  Alternative data ready.\n")
        except Exception as exc:
            _log(f"⚠️  Alternative data failed: {exc}")

    # ── 10. Advanced backtesting (walk-forward + Monte Carlo) ───────────
    advanced_bt = None
    if include_advanced_backtest and bt_result is not None:
        _log("🔬  Running advanced backtesting …")
        try:
            returns = df["Close"].pct_change().dropna()

            # Walk-forward
            wf_config = WalkForwardConfig(
                train_window=min(60, len(df) // 3),
                test_window=min(20, len(df) // 6),
                step_size=min(20, len(df) // 6),
                strategy_type=strategy_type,
            )
            wf_result = walk_forward_analysis(df, symbol=symbol, wf_config=wf_config, bt_config=backtest_config)

            # Monte Carlo
            mc_result = monte_carlo_simulation(returns, MonteCarloConfig(
                num_simulations=500, num_days=252, initial_capital=100_000.0,
            ))

            # Parameter optimisation
            signalled_df = generate_signals(df.copy(), strategy_type=strategy_type)
            param_result = optimise_parameters(signalled_df, symbol=symbol, param_grid=ParamGrid(
                stop_loss_pct=[0.03, 0.05, 0.07],
                take_profit_pct=[0.06, 0.10, 0.15],
                position_size_pct=[0.05, 0.10],
            ))

            # Multi-strategy comparison
            multi_strat = multi_strategy_backtest(df, symbol=symbol)

            advanced_bt = {
                "walk_forward": wf_result,
                "monte_carlo": {
                    "statistics": mc_result["statistics"],
                    "final_values": mc_result["final_values"],
                },
                "param_optimisation": {
                    "best_params": param_result["best_params"],
                    "best_result": param_result.get("best_result", {}),
                    "total_combos": param_result.get("total_combos_tested", 0),
                },
                "multi_strategy": {
                    "comparison": multi_strat["comparison"],
                    "best_strategy": multi_strat["best_strategy"],
                    "combined_summary": multi_strat["combined_summary"],
                },
            }
            _log(f"✅  Advanced backtesting done — "
                 f"WF folds: {wf_result['combined_summary']['num_folds']}, "
                 f"MC prob profit: {mc_result['statistics'].get('prob_profit', 'N/A')}%\n")
        except Exception as exc:
            _log(f"⚠️  Advanced backtesting failed: {exc}")

    result = {
        "symbol": symbol,
        "period": {"start": start_date, "end": end_date},
        "strategy": strategy_type,
        "strategy_description": get_strategy_description(strategy_type),
        "data": df,
        "statistics": stats,
        "signals_summary": signals_summary,
        "latest": latest,
        "research_report": research,
        "risk_validation": risk_result,
        "portfolio_summary": port_summary,
        "backtest_results": bt_result,
        "sentiment_report": sentiment_report,
        "llm_summary": llm_summary,
        "alternative_data": alt_data,
        "advanced_backtest": advanced_bt,
    }

    _log(f"{'='*60}")
    _log(f"  Analysis complete for {symbol}")
    _log(f"{'='*60}\n")
    return result


# ---------------------------------------------------------------------------
# Quick CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_full_analysis(
        symbol="AAPL",
        start_date="2024-01-01",
        end_date="2024-12-31",
        strategy_type="combined",
        verbose=True,
    )
    if result:
        print("\n📋 LATEST SNAPSHOT")
        for k, v in result["latest"].items():
            print(f"  {k}: {v}")
        if result["backtest_results"]:
            print("\n📊 BACKTEST SUMMARY")
            for k, v in result["backtest_results"]["summary"].items():
                print(f"  {k}: {v}")
