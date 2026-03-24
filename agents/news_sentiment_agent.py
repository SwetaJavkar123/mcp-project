"""
NewsSentimentAgent — Financial NLP
====================================
Fetches news headlines from yfinance and scores them using a
finance-tuned sentiment model (FinBERT by default, falls back
to the generic distilbert model already in the project).

Each headline gets: label (bullish / bearish / neutral), score (0-1).
Daily aggregate sentiment is also computed.
"""

from __future__ import annotations

import pandas as pd
from datetime import datetime
from dataclasses import dataclass

import yfinance as yf

# Lazy-load the model so import is fast (heavy models loaded on first call)
_pipeline = None


def _get_pipeline():
    """Load FinBERT if available, otherwise fall back to distilbert."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from transformers import pipeline as hf_pipeline

    try:
        _pipeline = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            top_k=None,
        )
    except Exception:
        # Fallback to the generic model already in the project
        _pipeline = hf_pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            revision="714eb0f",
        )
    return _pipeline


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class HeadlineSentiment:
    title: str
    publisher: str
    publish_time: str
    label: str          # bullish / bearish / neutral  (or POSITIVE/NEGATIVE)
    score: float        # 0-1 confidence
    raw_scores: dict    # full label→score mapping


def _normalise_label(label: str) -> str:
    """Map model-specific labels to bullish/bearish/neutral."""
    label = label.lower()
    if label in ("positive", "bullish"):
        return "bullish"
    if label in ("negative", "bearish"):
        return "bearish"
    return "neutral"


def analyze_headlines(symbol: str, max_items: int = 15) -> list[HeadlineSentiment]:
    """
    Fetch recent news for *symbol* and score every headline.

    Returns a list of HeadlineSentiment objects sorted newest-first.
    """
    ticker = yf.Ticker(symbol)
    news = ticker.news or []
    news = news[:max_items]

    if not news:
        return []

    pipe = _get_pipeline()
    titles = [item.get("title", "") for item in news]

    # Batch inference
    results = pipe(titles, truncation=True, max_length=512)

    sentiments: list[HeadlineSentiment] = []
    for item, scores in zip(news, results):
        # scores is a list of dicts: [{'label': '...', 'score': ...}, ...]
        if isinstance(scores, dict):
            scores = [scores]
        best = max(scores, key=lambda s: s["score"])
        raw = {s["label"]: round(s["score"], 4) for s in scores}

        pub_time = ""
        if item.get("providerPublishTime"):
            pub_time = datetime.fromtimestamp(item["providerPublishTime"]).strftime("%Y-%m-%d %H:%M")

        sentiments.append(HeadlineSentiment(
            title=item.get("title", ""),
            publisher=item.get("publisher", ""),
            publish_time=pub_time,
            label=_normalise_label(best["label"]),
            score=round(best["score"], 4),
            raw_scores=raw,
        ))

    return sentiments


def aggregate_sentiment(sentiments: list[HeadlineSentiment]) -> dict:
    """
    Aggregate headline sentiments into a single summary.

    Returns:
        {
            total: int,
            bullish: int,
            bearish: int,
            neutral: int,
            avg_score: float,
            overall: str,        # "bullish" / "bearish" / "neutral"
            sentiment_score: float,  # -1 (very bearish) to +1 (very bullish)
        }
    """
    if not sentiments:
        return {
            "total": 0, "bullish": 0, "bearish": 0, "neutral": 0,
            "avg_score": 0.0, "overall": "neutral", "sentiment_score": 0.0,
        }

    bull = sum(1 for s in sentiments if s.label == "bullish")
    bear = sum(1 for s in sentiments if s.label == "bearish")
    neut = sum(1 for s in sentiments if s.label == "neutral")

    # Sentiment score: +1 per bullish, -1 per bearish, 0 neutral, normalised
    raw_score = (bull - bear) / len(sentiments)

    if bull > bear:
        overall = "bullish"
    elif bear > bull:
        overall = "bearish"
    else:
        overall = "neutral"

    return {
        "total": len(sentiments),
        "bullish": bull,
        "bearish": bear,
        "neutral": neut,
        "avg_score": round(sum(s.score for s in sentiments) / len(sentiments), 4),
        "overall": overall,
        "sentiment_score": round(raw_score, 4),
    }


def get_sentiment_report(symbol: str, max_items: int = 15) -> dict:
    """
    Full sentiment report: individual headlines + aggregate summary.

    Returns:
        {
            symbol: str,
            headlines: list[dict],
            summary: dict,
        }
    """
    sentiments = analyze_headlines(symbol, max_items)
    summary = aggregate_sentiment(sentiments)

    headlines = [
        {
            "title": s.title,
            "publisher": s.publisher,
            "publish_time": s.publish_time,
            "label": s.label,
            "score": s.score,
        }
        for s in sentiments
    ]

    return {
        "symbol": symbol,
        "headlines": headlines,
        "summary": summary,
    }
