import requests
from bs4 import BeautifulSoup
import csv
import concurrent.futures
from urllib.parse import quote
from pathlib import Path

VALID_POS = {
    "Noun", "Verb", "Adjective", "Adverb",
    "Pronoun", "Conjunction", "Interjection",
    "Preposition", "Determiner", "Particle"
}

BASE_URL = "https://en.wiktionary.org"
MOBILE_URL = "https://en.m.wiktionary.org/wiki/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Educational Lexicon Scraper)"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_ROOT / "data" / "lexicons" / "preprocessed" / "wiktionary_final.csv"

def get_category_words(category):
    words = []
    url = f"{BASE_URL}/wiki/Category:{category}"
    while url:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            break
        soup = BeautifulSoup(r.text, "html.parser")
        for li in soup.select("div.mw-category-group li a"):
            word = li.get("title")
            if word:
                words.append(word)
        next_link = soup.find("a", string="next page")
        url = BASE_URL + next_link["href"] if next_link else None
    return words

def fetch_entry(word):
    url = MOBILE_URL + quote(word)
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return []
    except Exception:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    hiligaynon_header = soup.find("h2", id="Hiligaynon")
    if not hiligaynon_header:
        return []

    section_content = []
    for sib in hiligaynon_header.find_all_next():
        if sib.name == "h2":
            break
        section_content.append(sib)

    section_soup = BeautifulSoup("".join(str(s) for s in section_content), "html.parser")
    results = []
    for h3 in section_soup.find_all("h3"):
        pos_name = h3.get_text(strip=True)
        if pos_name not in VALID_POS:
            continue
        ol = h3.find_next_sibling("ol")
        if not ol:
            continue
        for li in ol.find_all("li", recursive=False):
            definition = li.get_text(" ", strip=True)
            if definition:
                results.append([word, pos_name.lower(), definition, url])
    return results

if __name__ == "__main__":
    categories = [
        "Hiligaynon_adjectives",
        "Hiligaynon_nouns",
        "Hiligaynon_verbs",
        "Hiligaynon_adverbs",
    ]

    all_words = set()
    for cat in categories:
        all_words.update(get_category_words(cat))

    print(f" Collected {len(all_words)} words")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Word", "Part of speech", "meaning", "source"])

        # Run with 15 parallel workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = {executor.submit(fetch_entry, w): w for w in all_words}
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                word = futures[future]
                try:
                    entries = future.result()
                    if not entries:
                        print(f"⚠ {word}: no Hiligaynon section")
                    else:
                        for row in entries:
                            writer.writerow(row)
                        print(f"✔ {word}: {len(entries)} entries")
                except Exception as e:
                    print(f" {word}: {e}")
