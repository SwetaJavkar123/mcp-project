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
│  Overview │ Technical │ Strategy │ Research │ Risk │ Backtest │Compare│
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
      │              │      │       │             │
      ▼              ▼      ▼       ▼             ▼
 NewsSentiment  Execution  LLMResearch    AlternativeData
   Agent          Agent      Agent            Agent
      │                                        │
      │         ┌──────────────────┐           │
      └────────►│ utils_marketdata │◄──────────┘
                │ (indicator lib)  │
                └──────────────────┘
```

## Pipeline Stages

1. **MarketDataAgent** — downloads OHLCV data from Yahoo Finance (stocks,
   crypto, forex, indices). Includes symbol resolver for common names.
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
7. **ExecutionAgent** — executes paper or live trades (market, limit, stop-limit)
   with portfolio state tracking.
8. **NewsSentimentAgent** — runs FinBERT on financial news headlines, returning
   bullish / bearish / neutral scores with confidence.
9. **LLMResearchAgent** — generates AI-powered research summaries via OpenAI
   (or any compatible API). Falls back to a free template when no API key is set
   or the daily token budget is exhausted.
10. **AlternativeDataAgent** — fetches SEC filings (EDGAR), insider trades,
    institutional holders, and social sentiment data.
11. **PortfolioOptimizer** — optimises multi-asset allocation using
    Mean-Variance (Markowitz), Risk-Parity, and Black-Litterman models.
    Generates rebalancing recommendations with transaction cost awareness.
12. **AdvancedBacktest** — extends backtesting with walk-forward analysis,
    Monte Carlo simulation, parameter optimisation (grid search), and
    multi-strategy portfolio backtesting.

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
      ├─► NewsSentimentAgent  →  headline sentiment scores
      ├─► LLMResearchAgent    →  AI research summary (or template)
      ├─► AlternativeDataAgent→  SEC filings, insider data
      └─► ExecutionAgent      →  paper / live order placement
      │
      ▼
  Unified result dict  →  Streamlit UI  /  CLI
```

## Cost Controls (LLM Agent)

The LLM agent enforces strict cost limits, all configurable via `.env`:

| Variable | Purpose | Default |
|----------|---------|---------|
| `LLM_MAX_TOKENS` | Max tokens per single API call | `500` |
| `LLM_DAILY_TOKEN_LIMIT` | Daily token budget across all calls | `10,000` |
| `LLM_MODEL` | Model to use | `gpt-4o-mini` (~$0.15/1M input) |

When the daily limit is reached, all subsequent calls fall back to the **free
template summary** automatically — no further API spend for the rest of the day.

## Key Design Principles

- **Single Responsibility** — each agent does one thing well.
- **Stateless Agents** — no hidden state; all context passed as arguments.
- **Shared Utility Layer** — `utils_marketdata.py` avoids duplicating indicator
  logic across agents.
- **Incremental Enrichment** — the DataFrame grows as it passes through stages;
  downstream agents simply read the columns they need.
- **Fail-Soft** — if any agent raises, the controller catches it and continues
  with partial results; the UI shows whatever data is available.
- **Cost-Safe** — LLM usage is capped per-call and per-day; free fallback
  ensures the app works without any API key at all.

## Deployment Notes

- Runs locally under a Python virtual environment.
- No external API keys required for core features (uses free yfinance data).
- Set `OPENAI_API_KEY` in `.env` for AI research summaries (optional).
- For GPU-accelerated NLP agents, ensure MPS / CUDA is configured.
- CI/CD via GitHub Actions: lint + test on Python 3.11–3.13.

## Extensibility

- **New strategies** — add a `_my_strategy()` function in `strategy_agent.py`.
- **New agents** — create `agents/my_agent.py`, wire it into the controller.
- **New data sources** — wrap any API behind the same `get_data()` signature.
- **New asset classes** — add mappings to the symbol resolver in `market_data_agent.py`.

## File Map

```
mcp-project/
├── agents/
│   ├── __init__.py
│   ├── market_data_agent.py       # OHLCV fetcher (stocks, crypto, forex)
│   ├── strategy_agent.py          # Signal generation
│   ├── company_research_agent.py  # Fundamental analysis
│   ├── risk_agent.py              # Risk metrics & trade validation
│   ├── backtest_agent.py          # Historical simulation engine
│   ├── execution_agent.py         # Paper / live trading
│   ├── news_sentiment_agent.py    # FinBERT news sentiment
│   ├── llm_research_agent.py      # AI research summaries
│   ├── alternative_data_agent.py  # SEC filings, insider trades
│   ├── portfolio_optimizer.py     # Markowitz, risk-parity, Black-Litterman
│   ├── advanced_backtest.py       # Walk-forward, Monte Carlo, param opt
│   ├── search_agent.py            # Web scraping (legacy)
│   ├── summarizer_agent.py        # NLP summariser (legacy)
│   └── sentiment_agent.py         # NLP sentiment (legacy)
├── tests/
│   ├── conftest.py
│   ├── test_advanced_backtest.py
│   ├── test_backtest_agent.py
│   ├── test_company_research_agent.py
│   ├── test_execution_agent.py
│   ├── test_llm_research_agent.py
│   ├── test_market_data_agent.py
│   ├── test_news_sentiment_agent.py
│   ├── test_portfolio_optimizer.py
│   ├── test_risk_agent.py
│   ├── test_strategy_agent.py
│   └── test_utils_marketdata.py
├── utils_marketdata.py
├── controller_hedge_fund.py
├── controller_strategy.py
├── controller.py
├── app_hedge_fund.py
├── app_marketdata.py
├── app.py
├── .env                           # API keys & cost limits (gitignored)
├── .github/workflows/ci.yml      # CI/CD pipeline
├── requirements.txt
├── readme.md
└── docs/
    ├── arch.md
    ├── plan.md
    └── futurework.md
```
