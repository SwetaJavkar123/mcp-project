# Future Work: Hedge Fund Multi-Agent Platform

This document tracks enhancements that would move the platform closer to
production-grade hedge fund software.

> **Note:** Items marked ✅ have been implemented. See `plan.md` for details.

---

## ✅ 1. ExecutionAgent — Paper & Live Trading

- ✅ Paper-trading mode with portfolio state tracking.
- ✅ Order types: market, limit, stop-limit.
- Connect to a live broker API (Alpaca, Interactive Brokers).
- Execution quality monitoring (slippage, fill rates).
- Integration point: RiskAgent approves → ExecutionAgent places trade.

## ✅ 2. NewsSentimentAgent — Financial NLP

- ✅ FinBERT-powered sentiment on yfinance news headlines.
- ✅ Bullish / bearish / neutral classification with confidence scores.
- ✅ Automatic fallback to DistilBERT if FinBERT unavailable.
- Feed sentiment into StrategyAgent as an additional signal factor.
- Aggregate daily sentiment scores and visualise on the dashboard.

## ✅ 3. Multi-Asset & Multi-Timeframe Support

- ✅ Crypto (BTC-USD, ETH-USD), forex (EURUSD=X), indices (^GSPC).
- ✅ Symbol resolver for common names → yfinance tickers.
- ✅ Intraday intervals (1m, 5m, 1h) alongside daily data.
- Allow strategy parameters to vary by timeframe.
- Options chains support.

## ✅ 4. LLM-Powered Research Summaries

- ✅ OpenAI-compatible API with template fallback.
- ✅ Cost controls: LLM_MAX_TOKENS, LLM_DAILY_TOKEN_LIMIT, LLM_MODEL (all in .env).
- ✅ Automatic fallback to free template when budget exhausted.
- Auto-generate morning briefings or weekly strategy summaries.

## ✅ 5. Alternative Data Agents

- ✅ SEC filings (10-K, 10-Q, 8-K) via EDGAR.
- ✅ Insider trading activity.
- ✅ Institutional holder breakdown.
- ✅ Social sentiment placeholders.
- **MacroAgent** — ingest economic indicators (CPI, Fed rate, PMI) via FRED.

## ✅ 6. Portfolio Optimisation

- ✅ Mean-variance (Markowitz) optimisation via Monte Carlo efficient frontier.
- ✅ Risk-parity allocation (equal risk contribution).
- ✅ Black-Litterman model (market-cap priors + investor views).
- ✅ Rebalancing recommendations with transaction cost awareness (skip tiny trades).

## ✅ 7. Advanced Backtesting

- ✅ Walk-forward analysis (rolling train / test windows, out-of-sample validation).
- ✅ Monte Carlo simulation of strategy returns (percentile bands, prob of profit).
- ✅ Parameter optimisation with grid search (stop-loss, take-profit, position size).
- ✅ Multi-strategy portfolio backtesting (equal or custom allocation, best-strategy ranking).

## ✅ 8. Testing & CI/CD

- ✅ Comprehensive pytest suite (124 tests across all agents & utilities).
- ✅ GitHub Actions workflow: lint → test on Python 3.11–3.13.
- Integration tests for the full pipeline end-to-end.

## 9. Alerting & Monitoring

- Email / Slack / Telegram alerts on BUY/SELL signals.
- Daily portfolio P&L report.
- Risk-breach notifications.

## 10. Deployment

- Dockerise the Streamlit app for cloud hosting.
- Add authentication for multi-user access.
- Persistent storage (SQLite / PostgreSQL) for trade logs and portfolio state.

---

_Each item above is designed to slot into the existing agent-based architecture
with minimal changes to the controller — just import the new agent and wire it
into the pipeline._
