# import tensorflow as tf
from agents.search_agent import search_and_extract
from agents.summarizer_agent import summarize
from agents.sentiment_agent import analyze_sentiment

def run_mcp(url):
    print("Searching...")
    raw_text = search_and_extract(url)

    print("Summarizing...")
    summary = summarize(raw_text)

    print("Analyzing Sentiment...")
    sentiment = analyze_sentiment(summary)

    return {
        "summary": summary,
        "sentiment": sentiment
    }

if __name__ == "__main__":
    result = run_mcp("https://en.wikipedia.org/wiki/OpenAI")
    print(result)
