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

## Phase 7 🔮  Future Enhancements

| Step | Deliverable | Status |
|------|-------------|--------|
| 7.1 | **ExecutionAgent** — paper / live trading via broker API | 🔮 Planned |
| 7.2 | **NewsSentimentAgent** — NLP sentiment on financial news | 🔮 Planned |
| 7.3 | Multi-asset support (options, crypto, forex) | 🔮 Planned |
| 7.4 | LLM-powered research summaries | 🔮 Planned |
| 7.5 | Alternative data agents (social media, SEC filings) | 🔮 Planned |
| 7.6 | Unit & integration tests | ✅ Done |
| 7.7 | CI/CD pipeline | 🔮 Planned |

---

## How Agents Connect — Full Pipeline

```
User selects symbol + dates + strategy
             │
             ▼
  ┌──────────────────────┐
  │   MarketDataAgent    │  ← fetches OHLCV from Yahoo Finance
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
  │    BacktestAgent     │      │  Portfolio Summary    │
  │  (equity curve,      │      │  (positions, P&L)     │
  │   trade log, metrics)│      │                       │
  └──────────┬───────────┘      └──────────┬───────────┘
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
