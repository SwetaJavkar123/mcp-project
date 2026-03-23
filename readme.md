# 🧠 Multi-Agent Control Platform (MCP)

A beginner-friendly Python project that simulates a **multi-agent orchestration system** using NLP tasks like search, summarization, and sentiment analysis — built entirely with Hugging Face Transformers, Python, and modular design.

---

## ✨ Project Overview

This project demonstrates how a **controller script** can delegate tasks to multiple specialized **agents**:

- 🔍 `SearchAgent` — fetches raw text from a webpage.
- 🧠 `SummarizerAgent` — summarizes the content using a pre-trained Transformer.
- ❤️ `SentimentAgent` — analyzes the emotional tone of the summary.

This system mimics a **Multi-Agent Control Platform (MCP)** where the controller orchestrates individual AI agents to collaborate on a task.

---

## 🧪 Sample Output

```bash
No model was supplied, defaulted to sshleifer/distilbart-cnn-12-6 and revision a4f8f3e.
Using a pipeline without specifying a model name and revision in production is not recommended.
Device set to use mps:0
No model was supplied, defaulted to distilbert/distilbert-base-uncased-finetuned-sst-2-english and revision 714eb0f.
Using a pipeline without specifying a model name and revision in production is not recommended.
Device set to use mps:0
Searching...
Summarizing...
Analyzing Sentiment...
{'summary': " The OpenAI project is the result of a collaboration between OpenAI and Wikipedia . The project is based on OpenAI's OpenAI initiative . It is the first time OpenAI has been used by OpenAI in the U.S. Wikipedia has been involved in the project . OpenAI is an openAI project .", 'sentiment': {'label': 'POSITIVE', 'score': 0.9904592037200928}
}
```


## 📦 Project Structure

```text
mcp-project/
├── controller.py                # Entry point: orchestrates agents
├── agents/                      # Agent modules
│   ├── search_agent.py          # Fetches raw text from a webpage
│   ├── summarizer_agent.py      # Summarizes input text
│   └── sentiment_agent.py       # Analyzes sentiment of input text
├── docs/                        # Documentation
│   ├── arch.md                  # Architecture overview
│   └── futurework.md            # Future work and ideas
├── app.py                       # Streamlit frontend
├── requirements.txt             # Python dependencies
├── readme.md                    # Project overview (this file)
└── .venv/                       # Python virtual environment (not pushed to Git)
```
