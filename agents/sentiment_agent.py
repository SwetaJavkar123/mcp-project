from transformers import pipeline

sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f"
)

def analyze_sentiment(text):
    return sentiment(text[:512])[0]

