"""Tests for agents/llm_research_agent.py — LLM summary generation."""

import pytest
from unittest.mock import patch

from agents.llm_research_agent import (
    _template_summary,
    _build_prompt,
    generate_llm_summary,
)


@pytest.fixture
def sample_report():
    return {
        "symbol": "AAPL",
        "score": 75,
        "recommendation": "BUY",
        "strengths": ["Strong ROE", "High profit margin"],
        "risks": ["High P/E ratio"],
        "fundamentals": {
            "company_name": "Apple Inc.",
            "sector": "Technology",
            "market_cap": 3_000_000_000_000,
            "pe_ratio": 28.5,
            "forward_pe": 25.0,
            "profit_margin": 0.26,
            "return_on_equity": 0.45,
            "debt_to_equity": 180,
            "beta": 1.2,
            "earnings_growth": 0.08,
            "revenue_growth": 0.05,
            "analyst_recommendation": "buy",
        },
        "news": [
            {"title": "Apple launches new product", "publisher": "Reuters"},
            {"title": "AAPL beats earnings estimates", "publisher": "Bloomberg"},
        ],
    }


class TestTemplateSummary:
    def test_contains_company_name(self, sample_report):
        summary = _template_summary(sample_report)
        assert "Apple Inc." in summary

    def test_contains_score(self, sample_report):
        summary = _template_summary(sample_report)
        assert "75/100" in summary

    def test_contains_strengths(self, sample_report):
        summary = _template_summary(sample_report)
        assert "Strong ROE" in summary

    def test_contains_risks(self, sample_report):
        summary = _template_summary(sample_report)
        assert "High P/E ratio" in summary

    def test_contains_news(self, sample_report):
        summary = _template_summary(sample_report)
        assert "Apple launches new product" in summary

    def test_empty_report(self):
        summary = _template_summary({})
        assert "Research Summary" in summary


class TestBuildPrompt:
    def test_returns_string(self, sample_report):
        prompt = _build_prompt(sample_report)
        assert isinstance(prompt, str)
        assert "Apple Inc." in prompt
        assert "AAPL" in prompt

    def test_includes_instructions(self, sample_report):
        prompt = _build_prompt(sample_report)
        assert "equity research analyst" in prompt.lower()


class TestGenerateLlmSummary:
    @patch.dict("os.environ", {}, clear=True)
    def test_falls_back_to_template(self, sample_report):
        """Without an API key, should return template summary."""
        summary = generate_llm_summary(sample_report)
        assert "Apple Inc." in summary
        assert "75/100" in summary

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_falls_back_on_import_error(self, sample_report):
        """If openai package not installed, should fall back to template."""
        import sys
        # Temporarily remove openai from available modules
        original = sys.modules.get("openai")
        sys.modules["openai"] = None  # type: ignore
        try:
            summary = generate_llm_summary(sample_report)
            assert "Apple Inc." in summary
        finally:
            if original is not None:
                sys.modules["openai"] = original
            else:
                sys.modules.pop("openai", None)
