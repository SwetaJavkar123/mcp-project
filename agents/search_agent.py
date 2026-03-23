import requests
from bs4 import BeautifulSoup

def search_and_extract(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text()[:1000]  # return first 1000 chars
