import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from chatbot_ui import logger

def fetch_support_pages(base_url="https://www.angelone.in"):
    try:
        resp = requests.get(base_url)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.info(f"Error fetching {base_url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")

    angelone_links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "//www.angelone.in" in href:
            angelone_links.add(href)

    logger.info(f"Total AngelOne links found: {len(angelone_links)}")
    return sorted(angelone_links)


def load_webpage_as_document(url):
    """Fetch a URL and extract meaningful text into a Document."""
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html5lib") 

    for tag in soup(["script", "style", "nav", "footer", "header", "svg", "noscript"]):
        tag.decompose()

    # Try to find main content area
    main_content = soup.select_one("main") or soup.select_one("article") or soup.body
    paragraphs = main_content.find_all("p") if main_content else soup.find_all("p")

    # Combine and clean the paragraph texts
    text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

    logger.info(f"Fetched from: {url}\nSample Text:\n{text[:300]}...\n---")
    return Document(page_content=text, metadata={"source": url})



if __name__ == "__main__":
    logger.info(fetch_support_pages())