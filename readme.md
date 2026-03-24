# 🏦 Hedge Fund Multi-Agent Platform

A modular, multi-agent Python platform for hedge-fund-style analysis and trading.
Each "agent" is a self-contained module that handles one concern; a central
controller orchestrates the full pipeline.

---

## ✨ Platform Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Streamlit Dashboard                             │
│  Overview │ Technical │ Strategy │ Research │ Risk │ Backtest │ Compare │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                ┌─────────────▼──────────────┐
                │   controller_hedge_fund.py  │
                │      (Orchestrator)         │
                └─┬───┬───┬───┬───┬───┬──────┘
                  │   │   │   │   │   │
     ┌────────────┘   │   │   │   │   └─────────────┐
     ▼                ▼   ▼   ▼   ▼                  ▼
 MarketData    Strategy  Risk  Backtest  CompanyResearch
   Agent        Agent   Agent  Agent        Agent
     │            │       │      │            │
     ▼            ▼       ▼      ▼            ▼
 NewsSentiment  Execution  LLMResearch  AlternativeData
   Agent          Agent      Agent          Agent
     │                                        │
     │            ┌───────────────┐            │
     └───────────►│ utils_market  │◄───────────┘
                  │  data.py      │
                  └───────────────┘
```

## 🧩 Agents

| Agent | File | Responsibility |
|-------|------|----------------|
| **MarketDataAgent** | `agents/market_data_agent.py` | Fetch OHLCV data (stocks, crypto, forex, indices) via yfinance |
| **StrategyAgent** | `agents/strategy_agent.py` | Generate BUY/SELL/HOLD signals (combined, momentum, mean-reversion, trend-following) |
| **CompanyResearchAgent** | `agents/company_research_agent.py` | Fundamentals, scoring, peer comparison, news |
| **RiskAgent** | `agents/risk_agent.py` | VaR, Sharpe, position sizing, trade validation |
| **BacktestAgent** | `agents/backtest_agent.py` | Historical simulation with equity curve, win rate, drawdown |
| **ExecutionAgent** | `agents/execution_agent.py` | Paper / live trading with market, limit, stop-limit orders |
| **NewsSentimentAgent** | `agents/news_sentiment_agent.py` | FinBERT-powered sentiment on financial news headlines |
| **LLMResearchAgent** | `agents/llm_research_agent.py` | AI-generated research summaries (OpenAI / template fallback) |
| **AlternativeDataAgent** | `agents/alternative_data_agent.py` | SEC filings, insider trades, institutional holders, social sentiment |
| **PortfolioOptimizer** | `agents/portfolio_optimizer.py` | Markowitz, risk-parity, Black-Litterman, rebalancing |
| **AdvancedBacktest** | `agents/advanced_backtest.py` | Walk-forward, Monte Carlo, param optimisation, multi-strategy |
| **SearchAgent** | `agents/search_agent.py` | Web scraping (original NLP demo) |
| **SummarizerAgent** | `agents/summarizer_agent.py` | Text summarisation (original NLP demo) |
| **SentimentAgent** | `agents/sentiment_agent.py` | Sentiment analysis (original NLP demo) |

## 🚀 Quick Start

```bash
# 1. Clone & enter project
cd mcp-project

# 2. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment (copy and edit with your keys)
cp .env.example .env   # or edit .env directly

# 5. Launch the dashboard
streamlit run app_hedge_fund.py

# — or run the CLI pipeline —
python controller_hedge_fund.py
```

## ⚙️ Environment Variables

All configuration is in `.env` (gitignored). Copy `.env` and fill in your keys:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM research summaries | _(none — uses free template)_ |
| `LLM_MAX_TOKENS` | Max tokens per single LLM API call | `500` |
| `LLM_DAILY_TOKEN_LIMIT` | Max total tokens per day (all calls) | `10000` |
| `LLM_MODEL` | OpenAI model to use | `gpt-4o-mini` (cheapest) |
| `LLM_BASE_URL` | Custom OpenAI-compatible endpoint | _(optional)_ |
| `ALPACA_API_KEY` | Alpaca broker key (live trading only) | _(optional)_ |
| `ALPACA_SECRET` | Alpaca secret (live trading only) | _(optional)_ |
| `FRED_API_KEY` | FRED API key (macro indicators) | _(optional)_ |

> **💡 Cost safety:** When the daily token limit is hit, the LLM agent
> automatically falls back to the free template summary — no surprise bills.

## 🗂 Project Structure

```
mcp-project/
├── agents/
│   ├── __init__.py
│   ├── market_data_agent.py      # OHLCV data fetcher (stocks, crypto, forex)
│   ├── strategy_agent.py         # Signal generation
│   ├── company_research_agent.py # Fundamental analysis
│   ├── risk_agent.py             # Risk metrics & trade validation
│   ├── backtest_agent.py         # Historical simulation engine
│   ├── execution_agent.py        # Paper / live trading engine
│   ├── news_sentiment_agent.py   # FinBERT news sentiment
│   ├── llm_research_agent.py     # AI-powered research summaries
│   ├── alternative_data_agent.py # SEC filings, insider trades, holders
│   ├── portfolio_optimizer.py    # Markowitz, risk-parity, Black-Litterman
│   ├── advanced_backtest.py      # Walk-forward, Monte Carlo, param opt
│   ├── search_agent.py           # Web scraping agent (legacy)
│   ├── summarizer_agent.py       # NLP summariser (legacy)
│   └── sentiment_agent.py        # NLP sentiment (legacy)
├── tests/
│   ├── conftest.py               # Shared fixtures
│   ├── test_backtest_agent.py
│   ├── test_advanced_backtest.py
│   ├── test_company_research_agent.py
│   ├── test_execution_agent.py
│   ├── test_llm_research_agent.py
│   ├── test_market_data_agent.py
│   ├── test_news_sentiment_agent.py
│   ├── test_portfolio_optimizer.py
│   ├── test_risk_agent.py
│   ├── test_strategy_agent.py
│   └── test_utils_marketdata.py
├── utils_marketdata.py           # Technical indicator library
├── controller_hedge_fund.py      # Unified hedge-fund orchestrator
├── controller_strategy.py        # Strategy-only controller (legacy)
├── controller.py                 # Original NLP controller (legacy)
├── app_hedge_fund.py             # Full Streamlit dashboard
├── app_marketdata.py             # Market-data-only dashboard (legacy)
├── app.py                        # Original NLP demo app (legacy)
├── .env                          # API keys & cost limits (gitignored)
├── .github/workflows/ci.yml     # CI/CD: lint + test on Python 3.11–3.13
├── requirements.txt
├── readme.md                     # ← this file
└── docs/
    ├── plan.md                   # Implementation roadmap
    ├── arch.md                   # Architecture reference
    └── futurework.md             # Ideas for next iterations
```

## 📈 Features

### Technical Analysis
- Moving Averages, RSI, Bollinger Bands, MACD, Stochastic, ATR, ADX

### Strategy Signals
- **Combined** — RSI + Bollinger + MACD (volatile markets)
- **Momentum** — MACD + RSI (trending markets)
- **Mean Reversion** — Bollinger + RSI (range-bound markets)
- **Trend Following** — MACD + ADX + Stochastic (strong trends)

### Company Research
- Key fundamentals (P/E, margins, debt, ROE, earnings growth)
- Proprietary 0–100 scoring with auto-generated strengths/risks
- Sector peer comparison table
- Recent news feed

### Risk Management
- Historical VaR & CVaR (95%)
- Sharpe & Sortino ratios
- Max drawdown
- Position sizing (Kelly-inspired with risk-per-trade limits)
- Trade validation against portfolio constraints

### Backtesting
- Full historical simulation with commission & slippage
- Equity curve, monthly returns, win rate, profit factor
- Stop-loss & take-profit enforcement
- Detailed trade log

### Execution (Paper Trading)
- Market, limit, and stop-limit order types
- Paper trading engine with portfolio state tracking
- Extensible to live trading via broker API (Alpaca)

### News Sentiment
- FinBERT-powered sentiment analysis on financial headlines
- Bullish / bearish / neutral classification with confidence scores
- Automatic fallback to DistilBERT if FinBERT unavailable

### LLM Research Summaries
- AI-generated equity research reports from structured data
- Cost controls: per-call token cap, daily budget, cheapest model
- Automatic fallback to free template when API key missing or budget exhausted

### Alternative Data
- SEC filings (10-K, 10-Q, 8-K) via EDGAR
- Insider trading activity
- Institutional holder breakdown
- Social sentiment placeholders

### Portfolio Optimisation
- **Mean-Variance (Markowitz)** — Monte Carlo efficient frontier with max-Sharpe and min-volatility portfolios
- **Risk-Parity** — equal risk contribution allocation
- **Black-Litterman** — market-cap equilibrium + investor views
- **Rebalancing** — trade recommendations with transaction cost awareness (skips tiny drifts)

### Advanced Backtesting
- **Walk-Forward Analysis** — rolling train/test windows for out-of-sample validation
- **Monte Carlo Simulation** — 1,000 future scenarios with percentile bands and probability of profit/loss
- **Parameter Optimisation** — grid search over stop-loss, take-profit, position size
- **Multi-Strategy** — run all strategies in parallel, compare metrics, find the best performer

### Multi-Asset Support
- Stocks, crypto (BTC-USD, ETH-USD), forex (EURUSD=X), indices (^GSPC)
- Symbol resolver for common names → yfinance tickers
- Intraday intervals (1m, 5m, 1h) and daily data

### Multi-Symbol Comparison
- Side-by-side price charts, normalised returns, correlation matrix, comparative stats

## 🧪 Testing

```bash
# Run the full test suite (171 tests)
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_execution_agent.py
```

## 🔄 CI/CD

GitHub Actions runs on every push and PR:
- **Linting** with `flake8`
- **Tests** with `pytest` on Python 3.11, 3.12, and 3.13

## 🛣 Roadmap

See [`docs/plan.md`](docs/plan.md) for the full implementation roadmap and
[`docs/futurework.md`](docs/futurework.md) for remaining ideas:
- Alerting (Slack / Telegram / email)
- Docker deployment & auth
- Persistent storage (SQLite / PostgreSQL)

---

_Built as a learning project to demonstrate multi-agent orchestration in Python._
