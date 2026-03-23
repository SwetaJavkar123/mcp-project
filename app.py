import streamlit as st
from agents.search_agent import search_and_extract
from agents.summarizer_agent import summarize
from agents.sentiment_agent import analyze_sentiment

st.set_page_config(page_title="Multi-Agent NLP Platform", layout="centered")

st.title("🧠 Multi-Agent Control Platform (MCP)")
st.subheader("Summarization + Sentiment Analysis from a URL")

# User input
url = st.text_input("Enter a website URL (Wikipedia works well):")

if st.button("Run MCP") and url:
    with st.spinner("Extracting text..."):
        raw_text = search_and_extract(url)
        st.success("Text extracted!")

    st.markdown("### Raw Extract (first 800 characters)")
    st.write(raw_text[:800] + "...")

    with st.spinner("Summarizing..."):
        summary = summarize(raw_text)
        st.success("Summary ready!")

    st.markdown("### ✨ Summary")
    st.write(summary)

    with st.spinner("Analyzing sentiment..."):
        sentiment = analyze_sentiment(summary)
        st.success("Sentiment analyzed!")

    st.markdown("### ❤️ Sentiment")
    st.write(f"**Label:** {sentiment['label']}  \n**Confidence:** {round(sentiment['score'] * 100, 2)}%")
