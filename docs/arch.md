# Architecture Overview

## Beginner-Friendly Explanation

This project is a simple example of how you can use multiple small AI programs (called “agents”) to work together and solve a bigger problem. It’s written in Python and uses popular AI tools from Hugging Face.

### What does it do?
Given a web page (like a Wikipedia article), the project:
1. Fetches the text from the web page.
2. Summarizes the main points of that text.
3. Analyzes the sentiment (emotional tone) of the summary (e.g., is it positive or negative?).

### How does it work?
- There is a controller (main script) that acts like a manager. It tells each agent what to do, in order.
- There are three main agents (small Python modules):
  - SearchAgent: Downloads and extracts the text from a web page.
  - SummarizerAgent: Uses an AI model to create a short summary of the text.
  - SentimentAgent: Uses another AI model to decide if the summary is positive, negative, or neutral.

### Why is this useful?
- It shows how to break a big task into smaller, reusable parts (agents).
- It demonstrates how to use pre-trained AI models for real-world tasks.
- It’s a good starting point for learning about multi-agent systems and AI orchestration.

### What do you need to know to understand this project?
- Basic Python: Functions, imports, and running scripts.
- How to install Python packages (like requests and transformers).
- Very basic web concepts: What a URL is, what a web page is.

### How do you run it?
1. Install the required Python packages (see requirements.txt).
2. Run the main script (controller.py).
3. The script will print out the summary and sentiment for a sample web page.

### Architecture Diagram

```
+-------------+        +----------------+        +-------------------+        +-------------------+
|             |        |                |        |                   |        |                   |
|   User /    +------->+   Controller   +------->+  SearchAgent      +------->+  SummarizerAgent  +
| Frontend    |        |  (Orchestrator)|        | (Web Scraper)     |        | (Text Summarizer) |
| (CLI/UI)    |        |                |        |                   |        |                   |
+-------------+        +----------------+        +-------------------+        +-------------------+
                                                                                         |
                                                                                         v
                                                                              +-------------------+
                                                                              |                   |
                                                                              | SentimentAgent    |
                                                                              | (Sentiment Model) |
                                                                              +-------------------+
```

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
