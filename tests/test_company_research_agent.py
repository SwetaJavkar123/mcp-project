"""Tests for agents/company_research_agent.py — fundamentals & scoring (mocked)."""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from agents.company_research_agent import (
    _score_fundamentals,
    get_peer_symbols,
    SECTOR_PEERS,
)


# ── Scoring (pure logic, no network) ─────────────────────────────────────


class TestScoreFundamentals:
    def test_neutral_baseline(self):
        """Empty fundamentals should return score ~50."""
        result = _score_fundamentals({})
        assert result["score"] == 50
        assert isinstance(result["strengths"], list)
        assert isinstance(result["risks"], list)

    def test_strong_stock(self):
        data = {
            "pe_ratio": 10,
            "profit_margin": 0.30,
            "return_on_equity": 0.25,
            "debt_to_equity": 30,
            "earnings_growth": 0.20,
            "analyst_recommendation": "strong_buy",
        }
        result = _score_fundamentals(data)
        assert result["score"] > 70
        assert len(result["strengths"]) >= 4

    def test_weak_stock(self):
        data = {
            "pe_ratio": 50,
            "profit_margin": 0.02,
            "return_on_equity": 0.02,
            "debt_to_equity": 300,
            "earnings_growth": -0.10,
            "analyst_recommendation": "sell",
        }
        result = _score_fundamentals(data)
        assert result["score"] < 30
        assert len(result["risks"]) >= 4

    def test_score_clamped_0_100(self):
        # Even with extreme values, score stays in [0, 100]
        extreme_good = {
            "pe_ratio": 1, "profit_margin": 0.99, "return_on_equity": 0.99,
            "debt_to_equity": 1, "earnings_growth": 0.99,
            "analyst_recommendation": "strong_buy",
        }
        extreme_bad = {
            "pe_ratio": 999, "profit_margin": -0.5, "return_on_equity": -0.5,
            "debt_to_equity": 9999, "earnings_growth": -0.99,
            "analyst_recommendation": "strong_sell",
        }
        assert 0 <= _score_fundamentals(extreme_good)["score"] <= 100
        assert 0 <= _score_fundamentals(extreme_bad)["score"] <= 100


# ── Sector peer mapping ──────────────────────────────────────────────────


class TestSectorPeers:
    def test_tech_peers_exist(self):
        assert "Technology" in SECTOR_PEERS
        assert len(SECTOR_PEERS["Technology"]) >= 5

    @patch("agents.company_research_agent.get_fundamentals")
    def test_get_peer_symbols_excludes_self(self, mock_fund):
        mock_fund.return_value = {"sector": "Technology"}
        peers = get_peer_symbols("AAPL", max_peers=5)
        assert "AAPL" not in peers
        assert len(peers) <= 5

    @patch("agents.company_research_agent.get_fundamentals")
    def test_unknown_sector_returns_empty(self, mock_fund):
        mock_fund.return_value = {"sector": "Alien Tech"}
        peers = get_peer_symbols("XYZ", max_peers=5)
        assert peers == []
