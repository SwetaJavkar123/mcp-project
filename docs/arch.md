# Architecture Overview — Hedge Fund Multi-Agent Platform

## High-Level Design

The platform follows a **controller → agents** pattern. A single orchestrator
(`controller_hedge_fund.py`) drives every stage of analysis by delegating to
specialised agent modules. Each agent is stateless and depends only on its
inputs, making the system easy to test, extend, and reason about.

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Streamlit Dashboard (app_hedge_fund.py)          │
│  Overview │ Technical │ Strategy │ Research │ Risk │ Backtest │ Compare│
└──────────────────────────────┬───────────────────────────────────────┘
                               │  calls
                ┌──────────────▼──────────────┐
                │  controller_hedge_fund.py   │
                │       (Orchestrator)        │
                └──┬────┬────┬────┬────┬──────┘
                   │    │    │    │    │
      ┌────────────┘    │    │    │    └─────────────┐
      ▼                 ▼    ▼    ▼                  ▼
 MarketData       Strategy  Risk  Backtest   CompanyResearch
   Agent           Agent   Agent  Agent         Agent
      │                                          │
      │         ┌──────────────────┐             │
      └────────►│ utils_marketdata │◄────────────┘
                │ (indicator lib)  │
                └──────────────────┘
```

## Pipeline Stages

1. **MarketDataAgent** — downloads OHLCV data from Yahoo Finance.
2. **Indicator Calculation** (`utils_marketdata.py`) — RSI, MACD, Bollinger,
   Stochastic, ATR, ADX.
3. **StrategyAgent** — generates BUY / SELL / HOLD signals with confidence
   scores using one of four strategy families.
4. **CompanyResearchAgent** — pulls fundamentals, calculates a 0-100 score,
   compares to sector peers, and collects recent news.
5. **RiskAgent** — computes VaR, CVaR, Sharpe, Sortino, max drawdown; validates
   trade proposals against portfolio constraints and position-sizing rules.
6. **BacktestAgent** — replays the signals on historical data, simulating
   commission / slippage / stop-loss / take-profit, and produces an equity curve
   with performance metrics.

## Data Flow

```
Symbol + dates
      │
      ▼
  MarketDataAgent.get_data()  →  raw OHLCV DataFrame
      │
      ▼
  utils_marketdata.*           →  DataFrame + indicator columns
      │
      ▼
  StrategyAgent.generate_signals()  →  DataFrame + Signal & Confidence
      │           │
      ▼           ▼
  RiskAgent    BacktestAgent
  (validate)   (simulate)
      │           │
      ▼           ▼
  CompanyResearchAgent  →  research report dict
      │
      ▼
  Unified result dict  →  Streamlit UI  /  CLI
```

## Key Design Principles

- **Single Responsibility** — each agent does one thing well.
- **Stateless Agents** — no hidden state; all context passed as arguments.
- **Shared Utility Layer** — `utils_marketdata.py` avoids duplicating indicator
  logic across agents.
- **Incremental Enrichment** — the DataFrame grows as it passes through stages;
  downstream agents simply read the columns they need.
- **Fail-Soft** — if any agent raises, the controller catches it and continues
  with partial results; the UI shows whatever data is available.

## Deployment Notes

- Runs locally under a Python virtual environment.
- No external API keys required (uses free yfinance data).
- For GPU-accelerated NLP agents, ensure MPS / CUDA is configured.

## Extensibility

- **New strategies** — add a `_my_strategy()` function in `strategy_agent.py`.
- **New agents** — create `agents/my_agent.py`, wire it into the controller.
- **New data sources** — wrap any API behind the same `get_data()` signature.

## File Map

```
mcp-project/
├── agents/
│   ├── __init__.py
│   ├── market_data_agent.py
│   ├── strategy_agent.py
│   ├── company_research_agent.py
│   ├── risk_agent.py
│   ├── backtest_agent.py
│   ├── search_agent.py
│   ├── summarizer_agent.py
│   └── sentiment_agent.py
├── utils_marketdata.py
├── controller_hedge_fund.py
├── controller_strategy.py
├── controller.py
├── app_hedge_fund.py
├── app_marketdata.py
├── app.py
├── requirements.txt
├── readme.md
└── docs/
    ├── arch.md
    ├── plan.md
    └── futurework.md
```
