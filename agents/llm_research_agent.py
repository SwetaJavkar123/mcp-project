"""
LLM Research Agent — AI-Powered Research Summaries
=====================================================
Generates natural-language research reports from structured
CompanyResearchAgent data using an LLM.

Supports:
  - OpenAI (GPT-4o / GPT-3.5)
  - Anthropic (Claude)
  - Local / self-hosted via any OpenAI-compatible endpoint

Set OPENAI_API_KEY in the environment for OpenAI, or
LLM_BASE_URL + LLM_API_KEY for a custom endpoint.

Falls back to a deterministic template if no API key is set.
"""

from __future__ import annotations

import os
from datetime import date

# ---------------------------------------------------------------------------
# Token / cost budget — prevents runaway API spend
# ---------------------------------------------------------------------------
# Override via env vars or pass directly to generate_llm_summary()
DEFAULT_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", 500))
DAILY_TOKEN_LIMIT  = int(os.environ.get("LLM_DAILY_TOKEN_LIMIT", 10_000))
DEFAULT_MODEL      = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# Simple in-memory daily counter (resets on app restart)
_daily_usage: dict[str, int] = {"tokens": 0, "date": ""}


# ---------------------------------------------------------------------------
# Template-based fallback (no LLM needed)
# ---------------------------------------------------------------------------

def _template_summary(report: dict) -> str:
    """Deterministic research summary — no API key required."""
    f = report.get("fundamentals", {})
    name = f.get("company_name", report.get("symbol", "Company"))
    symbol = report.get("symbol", "")
    sector = f.get("sector", "N/A")
    score = report.get("score", "N/A")
    rec = report.get("recommendation", "N/A")

    strengths = report.get("strengths", [])
    risks = report.get("risks", [])

    lines = [
        f"# Research Summary — {name} ({symbol})",
        f"**Date:** {date.today().isoformat()}",
        f"**Sector:** {sector}",
        f"**Fundamental Score:** {score}/100",
        f"**Recommendation:** {rec}",
        "",
        "## Key Metrics",
    ]

    metrics = [
        ("Market Cap", f.get("market_cap"), "${:,.0f}"),
        ("P/E Ratio", f.get("pe_ratio"), "{:.1f}"),
        ("Forward P/E", f.get("forward_pe"), "{:.1f}"),
        ("Profit Margin", f.get("profit_margin"), "{:.1%}"),
        ("Return on Equity", f.get("return_on_equity"), "{:.1%}"),
        ("Debt/Equity", f.get("debt_to_equity"), "{:.0f}"),
        ("Beta", f.get("beta"), "{:.2f}"),
        ("Earnings Growth", f.get("earnings_growth"), "{:.1%}"),
        ("Revenue Growth", f.get("revenue_growth"), "{:.1%}"),
    ]
    for label, val, fmt in metrics:
        if val is not None:
            lines.append(f"- **{label}:** {fmt.format(val)}")

    if strengths:
        lines += ["", "## Strengths"]
        for s in strengths:
            lines.append(f"- ✅ {s}")

    if risks:
        lines += ["", "## Risks"]
        for r in risks:
            lines.append(f"- ⚠️ {r}")

    # News summary
    news = report.get("news", [])
    if news:
        lines += ["", "## Recent Headlines"]
        for n in news[:5]:
            lines.append(f"- {n.get('title', '')} — _{n.get('publisher', '')}_")

    lines += [
        "",
        "---",
        f"_Auto-generated research summary for {name}. "
        "This is not financial advice._",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM-powered summary (OpenAI-compatible API)
# ---------------------------------------------------------------------------

def _build_prompt(report: dict) -> str:
    """Build a prompt for the LLM from the research report data."""
    f = report.get("fundamentals", {})
    name = f.get("company_name", report.get("symbol", ""))
    symbol = report.get("symbol", "")

    data_block = (
        f"Company: {name} ({symbol})\n"
        f"Sector: {f.get('sector', 'N/A')}\n"
        f"Score: {report.get('score', 'N/A')}/100\n"
        f"Recommendation: {report.get('recommendation', 'N/A')}\n"
        f"Market Cap: {f.get('market_cap')}\n"
        f"P/E: {f.get('pe_ratio')}  Forward P/E: {f.get('forward_pe')}\n"
        f"Profit Margin: {f.get('profit_margin')}  ROE: {f.get('return_on_equity')}\n"
        f"Debt/Equity: {f.get('debt_to_equity')}  Beta: {f.get('beta')}\n"
        f"Earnings Growth: {f.get('earnings_growth')}  Revenue Growth: {f.get('revenue_growth')}\n"
        f"Analyst Recommendation: {f.get('analyst_recommendation')}\n"
    )

    strengths = report.get("strengths", [])
    risks = report.get("risks", [])
    if strengths:
        data_block += "Strengths: " + "; ".join(strengths) + "\n"
    if risks:
        data_block += "Risks: " + "; ".join(risks) + "\n"

    news = report.get("news", [])
    if news:
        data_block += "Recent news:\n"
        for n in news[:5]:
            data_block += f"  - {n.get('title', '')}\n"

    prompt = (
        "You are a senior equity research analyst at a hedge fund. "
        "Write a concise research summary (300-500 words) based on the "
        "following structured data. Include:\n"
        "1. Executive overview\n"
        "2. Key fundamental strengths and risks\n"
        "3. Valuation assessment\n"
        "4. Sentiment from recent news\n"
        "5. Actionable recommendation\n\n"
        f"Data:\n{data_block}\n"
        "Write in professional tone with Markdown formatting."
    )
    return prompt


def generate_llm_summary(
    report: dict,
    model: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Generate an LLM-powered research summary.

    Cost controls (all configurable in .env):
      - LLM_MAX_TOKENS    → caps each response  (default 500)
      - LLM_DAILY_TOKEN_LIMIT → daily budget     (default 10,000)
      - LLM_MODEL          → model to use        (default gpt-4o-mini, cheapest)

    Falls back to template summary if no API key is configured or budget is exhausted.
    """
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL")

    if not api_key:
        return _template_summary(report)

    # ── Budget check ────────────────────────────────────────────────────
    today = date.today().isoformat()
    if _daily_usage["date"] != today:
        _daily_usage["tokens"] = 0
        _daily_usage["date"] = today

    if _daily_usage["tokens"] >= DAILY_TOKEN_LIMIT:
        return _template_summary(report) + (
            f"\n\n_⚠️ Daily token limit reached ({DAILY_TOKEN_LIMIT:,} tokens). "
            "Using template summary to avoid extra cost._"
        )

    token_cap = max_tokens or DEFAULT_MAX_TOKENS
    resolved_model = model or DEFAULT_MODEL

    try:
        from openai import OpenAI

        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": "You are a senior equity research analyst."},
                {"role": "user", "content": _build_prompt(report)},
            ],
            max_tokens=token_cap,
            temperature=0.4,
        )

        # Track usage
        usage = response.usage
        if usage:
            _daily_usage["tokens"] += usage.total_tokens

        return response.choices[0].message.content

    except ImportError:
        return _template_summary(report)
    except Exception as exc:
        return _template_summary(report) + f"\n\n_LLM call failed: {exc}_"
