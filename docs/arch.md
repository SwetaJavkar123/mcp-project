# Architecture Overview

This document describes the high-level architecture of the Multi-Agent Control Platform (MCP).

## Purpose

The MCP demonstrates how a controller orchestrates multiple specialized agents to accomplish a task. It is intended as an educational example and not a production-ready system.

## Components

- Controller
  - Orchestrates workflows between agents.
  - Handles task decomposition and result aggregation.

- Agents
  - SearchAgent: fetches raw text from a web source.
  - SummarizerAgent: summarizes long text using a Transformer model.
  - SentimentAgent: analyzes sentiment of the summary.

- Frontend
  - Optional Streamlit or CLI entrypoint to trigger the controller.

## Data Flow

1. The controller receives an input prompt or URL.
2. The SearchAgent retrieves the content.
3. The SummarizerAgent compresses the content into a concise summary.
4. The SentimentAgent analyzes the emotional tone of the summary.
5. Controller aggregates outputs and returns the final result.

## Deployment Notes

- This project uses Hugging Face Transformers and should run locally under a Python virtual environment.
- For GPU acceleration on macOS, ensure MPS is configured correctly.

## Extensibility

- Add more agents (e.g., QA Agent, Topic Extraction Agent).
- Swap local transformer models for remote API-based models.

## File Map

mcp-project/
├── controller.py
├── agents/
│   ├── __init__.py
│   ├── search_agent.py
│   ├── summarizer_agent.py
│   └── sentiment_agent.py
├── app.py
├── docs/
│   └── arch.md
├── requirements.txt
└── README.md
