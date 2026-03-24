# 🏦 Hedge Fund Multi-Agent Platform

A modular, multi-agent Python platform for hedge-fund-style analysis and trading.
Each "agent" is a self-contained module that handles one concern; a central
controller orchestrates the full pipeline.

---

## ✨ Platform Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Streamlit Dashboard                       │
│  Overview │ Technical │ Strategy │ Research │ Risk │ Backtest │  │
└─────────────────────────────┬───────────────────────────────────┘
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
     │                                        │
     │            ┌───────────────┐            │
     └───────────►│ utils_market  │◄───────────┘
                  │  data.py      │
                  └───────────────┘
```

## 🧩 Agents

| Agent | File | Responsibility |
|-------|------|----------------|
| **MarketDataAgent** | `agents/market_data_agent.py` | Fetch historical OHLCV data via yfinance |
| **StrategyAgent** | `agents/strategy_agent.py` | Generate BUY/SELL/HOLD signals (combined, momentum, mean-reversion, trend-following) |
| **CompanyResearchAgent** | `agents/company_research_agent.py` | Fundamentals, scoring, peer comparison, news |
| **RiskAgent** | `agents/risk_agent.py` | VaR, Sharpe, position sizing, trade validation |
| **BacktestAgent** | `agents/backtest_agent.py` | Historical simulation with equity curve, win rate, drawdown |
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

# 4. Launch the dashboard
streamlit run app_hedge_fund.py

# — or run the CLI pipeline —
python controller_hedge_fund.py
```

## � Project Structure

```
mcp-project/
├── agents/
│   ├── __init__.py
│   ├── market_data_agent.py      # OHLCV data fetcher
│   ├── strategy_agent.py         # Signal generation
│   ├── company_research_agent.py # Fundamental analysis
│   ├── risk_agent.py             # Risk metrics & trade validation
│   ├── backtest_agent.py         # Historical simulation engine
│   ├── search_agent.py           # Web scraping agent
│   ├── summarizer_agent.py       # NLP summariser
│   └── sentiment_agent.py        # NLP sentiment
├── utils_marketdata.py           # Technical indicator library
├── controller_hedge_fund.py      # Unified hedge-fund orchestrator
├── controller_strategy.py        # Strategy-only controller (legacy)
├── controller.py                 # Original NLP controller (legacy)
├── app_hedge_fund.py             # Full Streamlit dashboard
├── app_marketdata.py             # Market-data-only dashboard (legacy)
├── app.py                        # Original NLP demo app (legacy)
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

### Multi-Symbol Comparison
- Side-by-side price charts, normalised returns, correlation matrix, comparative stats

## 🛣 Roadmap

See [`docs/futurework.md`](docs/futurework.md) for planned enhancements:
- ExecutionAgent (paper / live trading via broker API)
- NewsSentimentAgent (NLP-driven sentiment on financial news)
- Alternative-data agents (social media, SEC filings)
- Multi-asset support (options, crypto, forex)
- LLM-powered research summaries

---

_Built as a learning project to demonstrate multi-agent orchestration in Python._
