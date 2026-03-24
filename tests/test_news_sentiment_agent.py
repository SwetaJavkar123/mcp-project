"""Tests for agents/news_sentiment_agent.py — sentiment scoring."""

import pytest
from unittest.mock import patch, MagicMock

from agents.news_sentiment_agent import (
    HeadlineSentiment,
    _normalise_label,
    aggregate_sentiment,
)


class TestNormaliseLabel:
    def test_positive_to_bullish(self):
        assert _normalise_label("POSITIVE") == "bullish"
        assert _normalise_label("positive") == "bullish"
        assert _normalise_label("bullish") == "bullish"

    def test_negative_to_bearish(self):
        assert _normalise_label("NEGATIVE") == "bearish"
        assert _normalise_label("negative") == "bearish"
        assert _normalise_label("bearish") == "bearish"

    def test_neutral(self):
        assert _normalise_label("neutral") == "neutral"
        assert _normalise_label("NEUTRAL") == "neutral"

    def test_unknown_defaults_to_neutral(self):
        assert _normalise_label("something_else") == "neutral"


class TestAggregateSentiment:
    def _make(self, label, score=0.9):
        return HeadlineSentiment(
            title="Test", publisher="Pub", publish_time="",
            label=label, score=score, raw_scores={},
        )

    def test_empty(self):
        result = aggregate_sentiment([])
        assert result["total"] == 0
        assert result["overall"] == "neutral"

    def test_all_bullish(self):
        items = [self._make("bullish") for _ in range(5)]
        result = aggregate_sentiment(items)
        assert result["overall"] == "bullish"
        assert result["bullish"] == 5
        assert result["sentiment_score"] == 1.0

    def test_all_bearish(self):
        items = [self._make("bearish") for _ in range(3)]
        result = aggregate_sentiment(items)
        assert result["overall"] == "bearish"
        assert result["sentiment_score"] == -1.0

    def test_mixed(self):
        items = [
            self._make("bullish"),
            self._make("bearish"),
            self._make("neutral"),
        ]
        result = aggregate_sentiment(items)
        assert result["total"] == 3
        assert result["bullish"] == 1
        assert result["bearish"] == 1
        assert result["neutral"] == 1
        assert result["overall"] == "neutral"

    def test_sentiment_score_range(self):
        items = [self._make("bullish"), self._make("bearish")]
        result = aggregate_sentiment(items)
        assert -1.0 <= result["sentiment_score"] <= 1.0
