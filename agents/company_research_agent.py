"""
CompanyResearchAgent
Fetches fundamental company data, financial metrics, news, and peer comparison
using yfinance. Generates a structured research report.
"""

from __future__ import annotations

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Sector peer mapping (expandable)
# ---------------------------------------------------------------------------
SECTOR_PEERS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMZN", "CRM", "ORCL", "INTC", "AMD"],
    "Financial Services": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW"],
    "Healthcare": ["UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT"],
    "Consumer Cyclical": ["TSLA", "AMZN", "HD", "NKE", "MCD", "SBUX", "TGT"],
    "Communication Services": ["GOOGL", "META", "DIS", "NFLX", "CMCSA", "T", "VZ"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC"],
}


def get_fundamentals(symbol: str) -> dict:
    """
    Fetch key fundamental data for a stock.

    Returns dict with:
      - company_name, sector, industry, market_cap
      - pe_ratio, forward_pe, peg_ratio
      - revenue, net_income, profit_margin
      - debt_to_equity, current_ratio, return_on_equity
      - dividend_yield, beta, 52w_high, 52w_low
      - analyst_recommendation, target_mean_price
    """
    ticker = yf.Ticker(symbol)
    info = ticker.info or {}

    fundamentals = {
        "symbol": symbol,
        "company_name": info.get("longName", symbol),
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "peg_ratio": info.get("pegRatio"),
        "revenue": info.get("totalRevenue"),
        "net_income": info.get("netIncomeToCommon"),
        "profit_margin": info.get("profitMargins"),
        "debt_to_equity": info.get("debtToEquity"),
        "current_ratio": info.get("currentRatio"),
        "return_on_equity": info.get("returnOnEquity"),
        "dividend_yield": info.get("dividendYield"),
        "beta": info.get("beta"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "analyst_recommendation": info.get("recommendationKey"),
        "target_mean_price": info.get("targetMeanPrice"),
        "number_of_analysts": info.get("numberOfAnalystOpinions"),
        "earnings_growth": info.get("earningsGrowth"),
        "revenue_growth": info.get("revenueGrowth"),
        "description": info.get("longBusinessSummary", ""),
    }
    return fundamentals


def get_recent_news(symbol: str, max_items: int = 10) -> list[dict]:
    """
    Fetch recent news headlines for a ticker via yfinance.
    Returns list of {title, publisher, link, publish_time}.
    """
    ticker = yf.Ticker(symbol)
    news_items = []
    for item in (ticker.news or [])[:max_items]:
        news_items.append({
            "title": item.get("title", ""),
            "publisher": item.get("publisher", ""),
            "link": item.get("link", ""),
            "publish_time": datetime.fromtimestamp(item["providerPublishTime"]).strftime("%Y-%m-%d %H:%M")
            if item.get("providerPublishTime") else "",
        })
    return news_items


def get_earnings_history(symbol: str) -> pd.DataFrame:
    """Return recent quarterly earnings (actual vs estimate)."""
    ticker = yf.Ticker(symbol)
    try:
        earnings = ticker.quarterly_earnings
        if earnings is not None and not earnings.empty:
            return earnings.reset_index()
    except Exception:
        pass
    return pd.DataFrame()


def get_peer_symbols(symbol: str, max_peers: int = 5) -> list[str]:
    """
    Find peer symbols in the same sector.
    Returns up to *max_peers* tickers (excluding the symbol itself).
    """
    fundamentals = get_fundamentals(symbol)
    sector = fundamentals.get("sector", "Unknown")
    peers = SECTOR_PEERS.get(sector, [])
    return [p for p in peers if p != symbol][:max_peers]


def compare_peers(symbol: str, peer_symbols: list[str] | None = None) -> pd.DataFrame:
    """
    Build a comparison table of key metrics across *symbol* and its peers.
    """
    if peer_symbols is None:
        peer_symbols = get_peer_symbols(symbol)
    all_symbols = [symbol] + peer_symbols

    rows = []
    for sym in all_symbols:
        f = get_fundamentals(sym)
        rows.append({
            "Symbol": sym,
            "Company": f.get("company_name"),
            "Market Cap": f.get("market_cap"),
            "P/E": f.get("pe_ratio"),
            "Forward P/E": f.get("forward_pe"),
            "PEG": f.get("peg_ratio"),
            "Profit Margin": f.get("profit_margin"),
            "D/E": f.get("debt_to_equity"),
            "ROE": f.get("return_on_equity"),
            "Beta": f.get("beta"),
            "Analyst Rec.": f.get("analyst_recommendation"),
        })
    return pd.DataFrame(rows)


def _score_fundamentals(f: dict) -> dict:
    """
    Simple scoring heuristic (0-100) based on key metrics.
    Returns {score, strengths, risks}.
    """
    score = 50  # neutral starting point
    strengths: list[str] = []
    risks: list[str] = []

    pe = f.get("pe_ratio")
    if pe is not None:
        if pe < 15:
            score += 10
            strengths.append(f"Low P/E ({pe:.1f}) – potentially undervalued")
        elif pe > 30:
            score -= 5
            risks.append(f"High P/E ({pe:.1f}) – potentially overvalued")

    margin = f.get("profit_margin")
    if margin is not None:
        if margin > 0.20:
            score += 10
            strengths.append(f"High profit margin ({margin:.1%})")
        elif margin < 0.05:
            score -= 5
            risks.append(f"Low profit margin ({margin:.1%})")

    roe = f.get("return_on_equity")
    if roe is not None:
        if roe > 0.20:
            score += 10
            strengths.append(f"Strong ROE ({roe:.1%})")
        elif roe < 0.05:
            score -= 5
            risks.append(f"Weak ROE ({roe:.1%})")

    de = f.get("debt_to_equity")
    if de is not None:
        if de > 200:
            score -= 10
            risks.append(f"High debt-to-equity ({de:.0f})")
        elif de < 50:
            score += 5
            strengths.append(f"Low debt-to-equity ({de:.0f})")

    eg = f.get("earnings_growth")
    if eg is not None:
        if eg > 0.10:
            score += 10
            strengths.append(f"Earnings growth {eg:.1%}")
        elif eg < 0:
            score -= 5
            risks.append(f"Negative earnings growth ({eg:.1%})")

    rec = f.get("analyst_recommendation")
    if rec in ("strong_buy", "buy"):
        score += 10
        strengths.append(f"Analyst consensus: {rec}")
    elif rec in ("sell", "strong_sell"):
        score -= 10
        risks.append(f"Analyst consensus: {rec}")

    score = max(0, min(100, score))
    return {"score": score, "strengths": strengths, "risks": risks}


def generate_research_report(symbol: str) -> dict:
    """
    Full research report combining fundamentals, scoring, news, and peers.

    Returns:
        {
            fundamentals: dict,
            score: int (0-100),
            recommendation: str (STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL),
            strengths: list[str],
            risks: list[str],
            news: list[dict],
            peer_comparison: pd.DataFrame,
        }
    """
    fundamentals = get_fundamentals(symbol)
    analysis = _score_fundamentals(fundamentals)
    news = get_recent_news(symbol)
    peers_df = compare_peers(symbol)

    # Derive recommendation from score
    s = analysis["score"]
    if s >= 80:
        rec = "STRONG_BUY"
    elif s >= 65:
        rec = "BUY"
    elif s >= 40:
        rec = "HOLD"
    elif s >= 25:
        rec = "SELL"
    else:
        rec = "STRONG_SELL"

    return {
        "symbol": symbol,
        "fundamentals": fundamentals,
        "score": s,
        "recommendation": rec,
        "strengths": analysis["strengths"],
        "risks": analysis["risks"],
        "news": news,
        "peer_comparison": peers_df,
    }
