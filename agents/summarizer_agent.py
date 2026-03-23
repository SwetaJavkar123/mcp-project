from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    revision="a4f8f3e"
)

def summarize(text):
    return summarizer(text[:1024])[0]['summary_text']
