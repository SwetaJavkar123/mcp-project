from transformers import pipeline

summarizer = pipeline("summarization")

def summarize(text):
    return summarizer(text[:1024])[0]['summary_text']
