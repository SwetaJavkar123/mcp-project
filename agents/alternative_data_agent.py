"""
Alternative Data Agents
========================
A collection of agents that fetch non-price data for enriched analysis.

Agents:
  - SECFilingAgent   — recent SEC filings (10-K, 10-Q, 8-K) via EDGAR
  - InsiderAgent     — insider trading activity via yfinance
  - SocialSentiment  — placeholder for Reddit / StockTwits integration
  - MacroAgent       — key economic indicators via FRED (placeholder)
"""

from __future__ import annotations

import requests
import pandas as pd
import yfinance as yf
from datetime import datetime


# ---------------------------------------------------------------------------
# SEC Filing Agent (EDGAR)
# ---------------------------------------------------------------------------

EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index?q=%22{symbol}%22&dateRange=custom&startdt={start}&enddt={end}&forms={forms}"
EDGAR_FULL_TEXT_URL = "https://efts.sec.gov/LATEST/search-index"


def get_sec_filings(
    symbol: str,
    filing_types: list[str] | None = None,
    max_results: int = 10,
) -> list[dict]:
    """
    Fetch recent SEC filings for a company via the EDGAR full-text search API.

    Returns list of {filing_type, date, description, url}.
    """
    if filing_types is None:
        filing_types = ["10-K", "10-Q", "8-K"]

    headers = {
        "User-Agent": "HedgeFundPlatform/1.0 (educational project)",
        "Accept": "application/json",
    }

    try:
        params = {
            "q": f'"{symbol}"',
            "forms": ",".join(filing_types),
            "dateRange": "custom",
            "startdt": "2024-01-01",
            "enddt": datetime.now().strftime("%Y-%m-%d"),
        }
        resp = requests.get(
            "https://efts.sec.gov/LATEST/search-index",
            params=params,
            headers=headers,
            timeout=10,
        )
        if resp.status_code != 200:
            return _fallback_filings(symbol, filing_types, max_results)

        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])

        filings = []
        for hit in hits[:max_results]:
            src = hit.get("_source", {})
            filings.append({
                "filing_type": src.get("forms", ""),
                "date": src.get("file_date", ""),
                "description": src.get("display_names", [""])[0] if src.get("display_names") else "",
                "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={symbol}&type=&dateb=&owner=include&count=40",
            })
        return filings

    except Exception:
        return _fallback_filings(symbol, filing_types, max_results)


def _fallback_filings(symbol: str, filing_types: list[str], max_results: int) -> list[dict]:
    """Fallback: return a link to EDGAR search for the user to browse."""
    return [{
        "filing_type": ", ".join(filing_types),
        "date": "",
        "description": f"Search EDGAR for {symbol} filings",
        "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={symbol}&type=&dateb=&owner=include&count=40",
    }]


# ---------------------------------------------------------------------------
# Insider Trading Agent (via yfinance)
# ---------------------------------------------------------------------------

def get_insider_trades(symbol: str) -> pd.DataFrame:
    """
    Fetch insider transactions for a ticker using yfinance.

    Returns DataFrame with columns like: Insider, Relation, Date, Transaction, Shares, Value.
    """
    try:
        ticker = yf.Ticker(symbol)
        insiders = ticker.insider_transactions
        if insiders is not None and not insiders.empty:
            return insiders.head(20)
    except Exception:
        pass
    return pd.DataFrame()


def get_institutional_holders(symbol: str) -> pd.DataFrame:
    """Fetch top institutional holders."""
    try:
        ticker = yf.Ticker(symbol)
        holders = ticker.institutional_holders
        if holders is not None and not holders.empty:
            return holders.head(15)
    except Exception:
        pass
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Social Sentiment (placeholder — extensible)
# ---------------------------------------------------------------------------

def get_social_sentiment(symbol: str) -> dict:
    """
    Placeholder for social-media sentiment integration.

    In production this would connect to Reddit (PRAW), StockTwits API,
    or Twitter/X API. Currently returns a stub.
    """
    return {
        "symbol": symbol,
        "source": "placeholder",
        "mentions": 0,
        "bullish_pct": 0.0,
        "bearish_pct": 0.0,
        "note": "Social sentiment integration planned — connect Reddit/StockTwits API.",
    }


# ---------------------------------------------------------------------------
# Macro Economic Agent (placeholder — extensible)
# ---------------------------------------------------------------------------

def get_macro_indicators() -> dict:
    """
    Placeholder for macroeconomic data (FRED API).

    In production set FRED_API_KEY and fetch CPI, Fed rate, PMI, etc.
    Currently returns static reference data.
    """
    return {
        "source": "placeholder",
        "indicators": {
            "fed_funds_rate": "See https://fred.stlouisfed.org/series/FEDFUNDS",
            "cpi_yoy": "See https://fred.stlouisfed.org/series/CPIAUCSL",
            "unemployment": "See https://fred.stlouisfed.org/series/UNRATE",
            "gdp_growth": "See https://fred.stlouisfed.org/series/GDP",
            "pmi": "See https://fred.stlouisfed.org/series/MANEMP",
        },
        "note": "Set FRED_API_KEY to enable live macro data.",
    }


# ---------------------------------------------------------------------------
# Unified alternative data report
# ---------------------------------------------------------------------------

def get_alternative_data_report(symbol: str) -> dict:
    """Bundle all alternative data into one report."""
    return {
        "symbol": symbol,
        "sec_filings": get_sec_filings(symbol),
        "insider_trades": get_insider_trades(symbol),
        "institutional_holders": get_institutional_holders(symbol),
        "social_sentiment": get_social_sentiment(symbol),
        "macro_indicators": get_macro_indicators(),
    }
