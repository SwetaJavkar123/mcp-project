# Implementation Plan: Hedge Fund Multi-Agent Platform

## Vision

Build a modular, multi-agent Python platform that mimics a real hedge fund's
analysis workflow — from raw market data to risk-checked, backtested trading
signals — with clear agent orchestration and a polished dashboard.

---

## Phase 1 ✅  Foundation

| Step | Deliverable | Status |
|------|-------------|--------|
| 1.1 | **MarketDataAgent** — fetch OHLCV via yfinance | ✅ Done |
| 1.2 | **utils_marketdata** — indicator library (RSI, MACD, BB, Stoch, ATR, ADX) | ✅ Done |
| 1.3 | **Streamlit UI** — basic technical analysis tabs | ✅ Done |
| 1.4 | **controller_strategy.py** — data → indicators → signals pipeline | ✅ Done |

## Phase 2 ✅  Strategy Signals

| Step | Deliverable | Status |
|------|-------------|--------|
| 2.1 | **StrategyAgent** — combined, momentum, mean-reversion, trend-following | ✅ Done |
| 2.2 | Confidence scoring for every signal | ✅ Done |
| 2.3 | Strategy tab in Streamlit | ✅ Done |

## Phase 3 ✅  Company Research

| Step | Deliverable | Status |
|------|-------------|--------|
| 3.1 | **CompanyResearchAgent** — fundamentals, scoring, peers, news | ✅ Done |
| 3.2 | Fundamental score (0-100) with strengths/risks | ✅ Done |
| 3.3 | Sector peer comparison table | ✅ Done |
| 3.4 | Research tab in Streamlit | ✅ Done |

## Phase 4 ✅  Risk Management

| Step | Deliverable | Status |
|------|-------------|--------|
| 4.1 | **RiskAgent** — VaR, CVaR, Sharpe, Sortino, max drawdown | ✅ Done |
| 4.2 | Position sizing (Kelly-inspired) | ✅ Done |
| 4.3 | Trade validation against portfolio rules | ✅ Done |
| 4.4 | Risk tab in Streamlit | ✅ Done |

## Phase 5 ✅  Backtesting

| Step | Deliverable | Status |
|------|-------------|--------|
| 5.1 | **BacktestAgent** — historical simulation engine | ✅ Done |
| 5.2 | Commission, slippage, stop-loss, take-profit | ✅ Done |
| 5.3 | Equity curve, monthly returns, trade log | ✅ Done |
| 5.4 | Backtest tab in Streamlit | ✅ Done |

## Phase 6 ✅  Unified Controller & Dashboard

| Step | Deliverable | Status |
|------|-------------|--------|
| 6.1 | **controller_hedge_fund.py** — end-to-end orchestrator | ✅ Done |
| 6.2 | **app_hedge_fund.py** — 7-tab Streamlit dashboard | ✅ Done |
| 6.3 | Multi-symbol comparison with normalised returns & correlation | ✅ Done |
| 6.4 | Updated README, arch.md, plan.md, futurework.md | ✅ Done |

## Phase 7 ✅  Advanced Agents & Infrastructure

| Step | Deliverable | Status |
|------|-------------|--------|
| 7.1 | **ExecutionAgent** — paper / live trading (market, limit, stop-limit orders) | ✅ Done |
| 7.2 | **NewsSentimentAgent** — FinBERT-powered sentiment on financial news | ✅ Done |
| 7.3 | Multi-asset support (crypto, forex, indices via yfinance + symbol resolver) | ✅ Done |
| 7.4 | **LLMResearchAgent** — AI-generated research summaries (OpenAI + template fallback) | ✅ Done |
| 7.5 | **AlternativeDataAgent** — SEC filings, insider trades, institutional holders, social sentiment | ✅ Done |
| 7.6 | Comprehensive pytest suite (124 tests across all agents & utilities) | ✅ Done |
| 7.7 | CI/CD pipeline — GitHub Actions (lint + test on Python 3.11–3.13) | ✅ Done |

## Phase 8 ✅  Cost Controls & Configuration

| Step | Deliverable | Status |
|------|-------------|--------|
| 8.1 | `.env` file for all API keys and configuration (gitignored) | ✅ Done |
| 8.2 | OpenAI token limits — `LLM_MAX_TOKENS` (per-call cap, default 500) | ✅ Done |
| 8.3 | Daily token budget — `LLM_DAILY_TOKEN_LIMIT` (default 10,000) | ✅ Done |
| 8.4 | Model selection — `LLM_MODEL` (default gpt-4o-mini, cheapest) | ✅ Done |
| 8.5 | Automatic fallback to free template summary when budget exhausted | ✅ Done |
| 8.6 | Deprecation warnings fixed (datetime, fillna) | ✅ Done |

## Phase 9 ✅  Portfolio Optimisation & Advanced Backtesting

| Step | Deliverable | Status |
|------|-------------|--------|
| 9.1 | **PortfolioOptimizer** — Mean-Variance (Markowitz) efficient frontier | ✅ Done |
| 9.2 | Risk-Parity allocation (equal risk contribution) | ✅ Done |
| 9.3 | Black-Litterman model (market-cap priors + investor views) | ✅ Done |
| 9.4 | Rebalancing engine with transaction cost awareness | ✅ Done |
| 9.5 | **AdvancedBacktest** — Walk-forward analysis (rolling train/test) | ✅ Done |
| 9.6 | Monte Carlo simulation (percentile bands, probability of profit) | ✅ Done |
| 9.7 | Parameter optimisation via grid search | ✅ Done |
| 9.8 | Multi-strategy portfolio backtesting with allocation control | ✅ Done |
| 9.9 | 47 new tests (171 total, all passing) | ✅ Done |

---

## Environment Variables (.env)

The `.env` file controls all API keys and cost limits. It is **gitignored**
to keep secrets out of version control.

```env
# ─── API Keys ─────────────────────────────────────────────────
OPENAI_API_KEY=sk-...         # Required for LLM summaries

# ─── OpenAI Cost Controls ────────────────────────────────────
LLM_MAX_TOKENS=500            # Max tokens per API call
LLM_DAILY_TOKEN_LIMIT=10000   # Max tokens per day (all calls)
LLM_MODEL=gpt-4o-mini         # Cheapest model (~$0.15/1M input)

# ─── Optional ────────────────────────────────────────────────
# LLM_BASE_URL=               # Custom OpenAI-compatible endpoint
# LLM_API_KEY=                # Key for custom endpoint
# ALPACA_API_KEY=             # Live trading (paper mode needs none)
# ALPACA_SECRET=
# FRED_API_KEY=               # Macroeconomic data
```

**Cost safety net:** When the daily token limit is reached, the LLM agent
automatically falls back to the free deterministic template summary — no
further API calls are made for the rest of the day.

---

## How Agents Connect — Full Pipeline

```
User selects symbol + dates + strategy
             │
             ▼
  ┌──────────────────────┐
  │   MarketDataAgent    │  ← fetches OHLCV (stocks, crypto, forex, indices)
  └──────────┬───────────┘
             │ raw DataFrame
             ▼
  ┌──────────────────────┐
  │  Indicator Utilities │  ← RSI, MACD, BB, Stoch, ATR, ADX
  └──────────┬───────────┘
             │ enriched DataFrame
             ▼
  ┌──────────────────────┐
  │    StrategyAgent     │  ← generates BUY/SELL/HOLD + Confidence
  └──────────┬───────────┘
             │ signals DataFrame
             ├──────────────────────────────┐
             ▼                              ▼
  ┌──────────────────────┐      ┌──────────────────────┐
  │  CompanyResearchAgent│      │      RiskAgent        │
  │  (fundamentals,      │      │  (VaR, sizing, trade  │
  │   score, peers, news)│      │   validation)         │
  └──────────┬───────────┘      └──────────┬───────────┘
             │                              │
             ▼                              ▼
  ┌──────────────────────┐      ┌──────────────────────┐
  │  NewsSentimentAgent  │      │   ExecutionAgent      │
  │  (FinBERT headlines) │      │   (paper/live orders) │
  └──────────┬───────────┘      └──────────┬───────────┘
             │                              │
             ▼                              ▼
  ┌──────────────────────┐      ┌──────────────────────┐
  │  LLMResearchAgent   │      │    BacktestAgent      │
  │  (AI summaries)      │      │   (equity curve,      │
  └──────────┬───────────┘      │    trade log, metrics)│
             │                  └──────────┬───────────┘
             ▼                              │
  ┌──────────────────────┐                  │
  │ AlternativeDataAgent │                  │
  │ (SEC, insider, social)│                 │
  └──────────┬───────────┘                  │
             │                              │
             └──────────────┬───────────────┘
                            ▼
                  Unified result dict
                            │
                            ▼
                  Streamlit Dashboard
```

Each arrow passes data forward — no agent calls another agent directly.
The controller is the single point of orchestration.
