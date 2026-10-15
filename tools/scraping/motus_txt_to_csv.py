import re
import csv
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
txt_path = project_root / "data" / "lexicons" / "raw" / "output.txt"
csv_path = project_root / "data" / "lexicons" / "raw" / "output_parsed.csv"
start_page = 28

# Create the output directory if it is missing.
csv_path.parent.mkdir(parents=True, exist_ok=True)

# DIACRITICS to help detect Hiligaynon text
DIACRITICS = "âêîôûáéíóúàèìòùÁÉÍÓÚÀÈÌÒÙ’'"

# POS map (from your file)
pos_map = {
    "n": "noun", "v": "verb", "pt": "particle", "va": "verbal affix",
    "aa": "adjective formative", "na": "noun formative", "con": "conjunction",
    "num": "numeral", "id": "idiom", "intj": "interjection", "adj": "adjective",
    "pr": "pronoun", "adv": "adverb", "d": "deictic"
}

# prepare POS-token regex (longer tokens first)
pos_tokens = sorted(pos_map.keys(), key=len, reverse=True)
pos_regex = "|".join(pos_tokens)  # e.g. "intj|con|...|n|v"
HEADWORD_RE = re.compile(rf"^(\S+)\s+({pos_regex})\b\s*(.*)$", flags=re.IGNORECASE)
POSONLY_RE = re.compile(rf"^({pos_regex})\b\s*(.*)$", flags=re.IGNORECASE)
AFFIX_RE = re.compile(r"/[^/]+/")  # /mag-,-un/, /-un/, etc.

# CSV columns
columns = ["word", "POS symbol", "POS word", "placement", "affix",
           "meaning", "Example", "English Sample", "Page"]


def looks_like_hiligaynon_sentence(s: str) -> bool:
    """Heuristic: detect Hiligaynon sentence by markers or diacritics."""
    low = " " + s.lower() + " "
    markers = (" ang ", " si ", " nga ", " mga ", " sang ", " sa ")
    if any(m in low for m in markers):
        return True
    # diacritics or glottal/apostrophe often indicate Hiligaynon forms
    if any(ch in s for ch in DIACRITICS) or re.search(r"[’']", s):
        return True
    # Hiligaynon often uses affix/given prefixes like Nag', Gin' etc.
    if re.search(r"\b(Nag|Gin|Nag')", s):
        return True
    return False


def classify_sentences(remaining: str):
    """
    Split remaining text into meaning, example (Hiligaynon), english sample.
    Uses heuristics to decide example vs english sample.
    """
    # first, split into sentences (keep punctuation)
    sentences = [p.strip() for p in re.split(r"(?<=[.!?])\s+", remaining) if p.strip()]
    meaning = ""
    example = ""
    eng = ""

    if not sentences:
        return meaning, example, eng

    meaning = sentences[0].strip()
    rest_sents = sentences[1:]

    # iterate rest to find Hiligaynon vs English samples
    for s in rest_sents:
        if not s:
            continue
        if not example and looks_like_hiligaynon_sentence(s):
            example = s if s.endswith('.') else s + '.'
            continue
        # if not Hiligaynon, treat as English (or fallback)
        if not eng and not looks_like_hiligaynon_sentence(s):
            eng = s if s.endswith('.') else s + '.'
            continue
        # if we've already assigned example and eng, append leftover to eng
        if example and eng:
            eng = (eng.rstrip('.') + '. ' + s).strip()
        elif example and not eng:
            # if example assigned but not eng and s not Hiligaynon, assign as eng
            if not looks_like_hiligaynon_sentence(s):
                eng = s if s.endswith('.') else s + '.'
            else:
                # rare: multiple Hiligaynon examples; append to example
                example = (example.rstrip('.') + '. ' + s).strip()
        elif not example and eng:
            # eng was assigned but not example -> try to find Hiligaynon next
            if looks_like_hiligaynon_sentence(s):
                example = s if s.endswith('.') else s + '.'
            else:
                eng = (eng.rstrip('.') + '. ' + s).strip()

    # ensure punctuation
    if meaning and not meaning.endswith('.'):
        meaning += '.'
    return meaning, example, eng


def parse_chunk(word_field: str, pos_symbol: str, chunk_rest: str, page: int):
    """
    From chunk_rest (possibly multi-line), extract affix, placement, meaning, example, english.
    Returns one dict row.
    """
    # Normalize spaces
    rest = " ".join(chunk_rest.split())

    # Extract first affix block if present
    affix_m = AFFIX_RE.search(rest)
    affix = affix_m.group(0) if affix_m else ""
    if affix:
        # remove first affix occurrence from rest (only)
        rest = rest[:affix_m.start()] + rest[affix_m.end():]
        rest = rest.strip()

    # Sometimes placement is before affix or before meaning (e.g., "imperative /-un/ affix.")
    placement = ""
    # After removing affix, check if rest begins with known placement words like 'imperative'
    placement_keywords = {"imperative", "adjective", "adverb", "transitive", "intransitive", "plural", "singular"}
    # take first token
    first_token_match = re.match(r"^([A-Za-z\-]+)\b(?:\s+(.*))?", rest)
    if first_token_match:
        first_tok = first_token_match.group(1).lower()
        remainder_after_first = first_token_match.group(2) or ""
        # if the first token is a placement keyword or if remainder starts with 'affix' word
        if first_tok in placement_keywords:
            placement = first_tok
            rest = remainder_after_first.strip()
        else:
            # Also handle "imperative affix." pattern where first token followed by "affix"
            if re.match(rf"^{first_tok}\s+affix\b", rest, flags=re.IGNORECASE):
                placement = first_tok
                # remove that portion
                rest = re.sub(rf"^{first_tok}\s+affix\b\.?\s*", "", rest, flags=re.IGNORECASE).strip()

    # Now split into meaning/example/english
    meaning, example, eng = classify_sentences(rest)

    # If meaning is very short and equals 'affix' or similar, keep as-is.
    return {
        "word": word_field,
        "POS symbol": pos_symbol,
        "POS word": pos_map.get(pos_symbol, ""),
        "placement": placement,
        "affix": affix,
        "meaning": meaning,
        "Example": example,
        "English Sample": eng,
        "Page": page
    }


# ---------- Main parsing ----------
entries = []
with open(txt_path, "r", encoding="utf-8") as f:
    raw_lines = [ln.rstrip("\n") for ln in f]

i = 0
current_page = None
while i < len(raw_lines):
    line = raw_lines[i].strip()
    # page marker?
    if line.startswith("--- Page"):
        m = re.search(r"Page\s+(\d+)", line)
        if m:
            current_page = int(m.group(1))
        i += 1
        continue

    # only start when page >= start_page
    if current_page is None or current_page < start_page:
        i += 1
        continue

    # Try headword pattern first
    m_head = HEADWORD_RE.match(line)
    m_posonly = None
    if not m_head:
        m_posonly = POSONLY_RE.match(line)

    if m_head:
        headword = m_head.group(1).strip()
        pos_symbol = m_head.group(2).lower().strip()
        rest = m_head.group(3).strip()
        # gather continuation lines until next headword/posonly/page marker
        j = i + 1
        cont_parts = [rest] if rest else []
        while j < len(raw_lines):
            nxt = raw_lines[j].strip()
            if not nxt:
                j += 1
                continue
            if nxt.startswith("--- Page"):
                break
            if HEADWORD_RE.match(nxt) or POSONLY_RE.match(nxt):
                break
            # append as continuation text
            cont_parts.append(nxt)
            j += 1
        chunk_rest = " ".join(cont_parts).strip()
        entries.append(parse_chunk(headword, pos_symbol, chunk_rest, current_page))
        i = j
        continue

    elif m_posonly:
        # continuation entry for the previous headword (word cell left blank)
        pos_symbol = m_posonly.group(1).lower().strip()
        rest = m_posonly.group(2).strip()
        # gather continuation lines similarly
        j = i + 1
        cont_parts = [rest] if rest else []
        while j < len(raw_lines):
            nxt = raw_lines[j].strip()
            if not nxt:
                j += 1
                continue
            if nxt.startswith("--- Page"):
                break
            if HEADWORD_RE.match(nxt) or POSONLY_RE.match(nxt):
                break
            cont_parts.append(nxt)
            j += 1
        chunk_rest = " ".join(cont_parts).strip()
        # Word cell intentionally blank for POS-only continuation rows
        entries.append(parse_chunk("", pos_symbol, chunk_rest, current_page))
        i = j
        continue

    else:
        # not a recognizable entry line — skip
        i += 1
        continue

# --------- Write CSV ----------
with open(csv_path, "w", encoding="utf-8", newline="") as outf:
    writer = csv.DictWriter(outf, fieldnames=columns)
    writer.writeheader()
    for row in entries:
        writer.writerow(row)

print(f"✅ Done — parsed {len(entries)} rows. CSV saved to: {csv_path}")
