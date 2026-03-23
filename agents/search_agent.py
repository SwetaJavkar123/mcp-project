import requests
from bs4 import BeautifulSoup

def search_and_extract(url):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; MCPBot/1.0; +https://github.com/SwetaJavkar123/mcp-project)"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    # Try to extract only the main article content for Wikipedia
    content_div = soup.find("div", id="mw-content-text")
    if content_div:
        text = content_div.get_text(separator=" ", strip=True)
    else:
        text = soup.get_text(separator=" ", strip=True)
    return text[:1000]  # return first 1000 chars
