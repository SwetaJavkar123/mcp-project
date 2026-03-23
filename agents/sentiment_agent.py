from transformers import pipeline

sentiment = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    return sentiment(text[:512])[0]

