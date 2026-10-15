import requests
from bs4 import BeautifulSoup
import csv
import logging
import time
from pathlib import Path

BASE_URL = "https://en.wiktionary.org"
START_CATEGORY = "/wiki/Category:Hiligaynon_lemmas"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_ROOT / "data" / "lexicons" / "raw" / "hiligaynon_lexicon_scrapedFromWiktionary.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def fetch_url(url):
    """Fetch URL and return soup, or None if error."""
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        res.raise_for_status()
        return BeautifulSoup(res.text, "html.parser")
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        return None

def scrape_word_page(word_url):
    """Scrape a single Hiligaynon word entry page."""
    soup = fetch_url(word_url)
    if not soup:
        return None

    word = soup.find("h1", {"id": "firstHeading"}).text.strip()

    # Try to detect part of speech (POS)
    pos = None
    pos_header = soup.find("span", class_="mw-headline")
    if pos_header:
        pos = pos_header.text.strip()

    # Grab the first definition if available
    meaning = None
    ul = soup.find("ol")
    if ul:
        li = ul.find("li")
        if li:
            meaning = li.text.strip()

    # Build row (with placeholders for affixation, example, English example, source)
    return {
        "Word": word,
        "Part of speech": pos if pos else "",
        "affixation": "",
        "meaning": meaning if meaning else "",
        "example": "",
        "English example": "",
        "source": word_url
    }

def scrape_category(category_url, writer, limit=1200):
    """Scrape all words in a given Wiktionary category (with pagination)."""
    count = 0
    next_page = category_url

    while next_page and count < limit:
        soup = fetch_url(next_page)
        if not soup:
            break

        # Collect word links
        for link in soup.select("div.mw-category-group ul li a"):
            href = link.get("href")
            if not href:
                continue

            word_url = BASE_URL + href
            entry = scrape_word_page(word_url)
            if entry:
                writer.writerow(entry)
                count += 1
                logging.info(f"Saved word: {entry['Word']} ({count})")

            if count >= limit:
                break

            time.sleep(1)  # be polite

        # Pagination link
        next_link = soup.find("a", string="next page")
        if next_link:
            next_page = BASE_URL + next_link["href"]
        else:
            next_page = None

    logging.info(f"Scraped {count} words from {category_url}")

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Word", "Part of speech", "affixation", "meaning", "example", "English example", "source"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Start with Hiligaynon lemmas category
        start_url = BASE_URL + START_CATEGORY
        soup = fetch_url(start_url)
        if not soup:
            return

        # Scrape each lemma subcategory (e.g., nouns, verbs, adjectives)
        for subcat in soup.select("div#mw-subcategories a"):
            href = subcat.get("href")
            if not href:
                continue

            subcat_url = BASE_URL + href
            logging.info(f"Scraping subcategory: {subcat.text}")
            scrape_category(subcat_url, writer, limit=1200)

if __name__ == "__main__":
    main()
