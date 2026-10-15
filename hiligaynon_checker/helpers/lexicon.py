"""Lexicon loading and CFG terminal-category construction.

Maps the merged Hiligaynon lexicon (6,477 rows, Penn Treebank POS tags)
onto the terminal categories a-o of the grammar in legacy/AutomataSentenceValidator.py:

  a common noun | b proper noun | c pronoun
  d past verb (nag-) | e present verb (naga-) | f future verb (ma-)
  g past-time word | h non-past-time word | i manner adverb
  j "sang" | k "kag" | l "ang" | m "si"/"kay" | n "." | o adjective
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MERGED_CSV = DATA_DIR / "lexicons" / "Ceciel_Wiki_merged_scores.csv"
PINOY_CSV = DATA_DIR / "dictionaries" / "pinoydictionary-hiligaynon_lexicon.csv"
MOTUS_CSV = DATA_DIR / "dictionaries" / "motusdictionary_lexicon.csv"

TOKEN_RE = re.compile(r"^[a-z][a-z'-]*$")

# Time-adverb detection on RB meanings (whole-word match)
PAST_TIME_KEYS = r"yesterday|yesterday's|ago|last|previous|before|earlier|former|once|beforehand"
NONPAST_TIME_KEYS = r"tomorrow|tomorrow's|now|today|tonight|later|soon|again|already|still|yet|currently|presently|nowadays"

HANDPICKED = {
    "a": [
        "tawo", "balay", "syudad", "pagkaon", "kasapatan", "kakahuyan", "baybay",
        "hubin", "pamatyag", "bulak", "ilaw", "asawa", "gab-i", "adlaw", "tubig",
        "kapid", "isda", "prutas", "bana", "utod", "balasahon", "kagamitan",
        "mangga", "tinapay", "lamesa", "kasilyas", "kahoy", "saging", "nanay",
        "abyan", "telepono", "kuwarto", "miga", "migo",
    ],
    "b": [
        "hans", "melmar", "yves", "renuel", "liam", "junsel", "milton", "rene",
        "jeremey", "fredrick", "anne", "frank", "leslie", "ralph", "fred",
        "vince", "joel", "ram", "randy", "james",
    ],
    "c": ["sila", "kita", "kamo", "ako", "ka", "kami", "siya"],
    "g": ["kagina", "kahapon", "kagab-i", "sang-una"],
    "h": ["bwas", "subong", "karon", "sadason"],
    "i": [
        "mahinay", "madugay", "madasig", "magahod", "mahipos", "malayo",
        "maluya", "makasug", "magrabi", "maabtik", "mapagsik",
    ],
    "j": ["sang"],
    "k": ["kag"],
    "l": ["ang"],
    "m": ["si", "kay"],
    "n": ["."],
    "o": [
        "guapa", "law-ay", "nami", "mayo", "maluya", "matahom", "maayo", "mata",
        "malimpyo", "dako", "mahamot", "gamay", "madulom", "masanag", "mabaho",
        "ka-ilinit", "mahigko", "kakadlaw", "tahom", "limpyo",
    ],
}

CATEGORY_META = {
    "a": ("Common noun", "A -> a N", "Nouns from the lexicon (NN), plus a handpicked core list."),
    "b": ("Proper noun", "B -> b N", "Handpicked names + names you add (the lexicon has no proper nouns)."),
    "c": ("Pronoun", "C -> c N", "Pronouns from the lexicon (PRP), plus handpicked forms."),
    "d": ("Past-tense verb", "D -> d G | d K | d L | d M | d C", "Generated: nag- + verb root (e.g. hampang -> naghampang)."),
    "e": ("Present-tense verb", "E -> e J | e K | e L | e M | e C", "Generated: naga- + verb root (e.g. hampang -> nagahampang)."),
    "f": ("Future-tense verb", "F -> f H | f K | f L | f M | f C", "Generated: ma- + verb root (e.g. hampang -> mahampang)."),
    "g": ("Past time word", "G -> g K | g L | g M", "Handpicked (kahapon, kagina...) + lexicon adverbs whose meaning is past-time."),
    "h": ("Non-past time word", "H -> h K | h L | h M", "Handpicked (karon, bwas...) + lexicon adverbs whose meaning is now/future-time."),
    "i": ("Manner adverb", "I -> i K | i L | i M", "Lexicon adverbs (RB) that are not time words, plus handpicked forms."),
    "j": ("Linker 'sang'", "J -> j I", "Fixed word."),
    "k": ("Conjunction 'kag'", "K -> k D | k E | k F | k O", "Fixed word."),
    "l": ("Topic marker 'ang'", "L -> l A", "Fixed word."),
    "m": ("Name marker 'si'/'kay'", "M -> m B", "Fixed words."),
    "n": ("Period", "N -> n", "Sentence-final period."),
    "o": ("Adjective", "O -> o K | o L | o M | o C", "Adjectives from the lexicon (JJ), plus handpicked forms."),
}


def normalize(text: str) -> str:
    """Lowercase + strip accents (tubig == tubíg) for accent-insensitive matching."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", text)


class Category:
    def __init__(self, letter: str):
        self.letter = letter
        self.label, self.rule, self.description = CATEGORY_META[letter]
        self.words: list[str] = []          # normalized forms, deduped, ordered
        self.display: dict[str, str] = {}   # normalized -> original form
        self.source: dict[str, str] = {}    # normalized -> lexicon | handpicked | generated | you

    def add(self, word: str, display: str | None = None, source: str = "handpicked") -> None:
        norm = normalize(word) if word != "." else word
        if norm not in self.display:
            self.words.append(norm)
            self.display[norm] = display if display is not None else word
            self.source[norm] = source

    def rows(self) -> list[dict]:
        return [
            {"word": self.display[w], "normalized": w, "source": self.source[w]}
            for w in self.words
        ]


class Lexicon:
    def __init__(self):
        self.merged = pd.read_csv(MERGED_CSV)
        self.pinoy = pd.read_csv(PINOY_CSV)
        self.motus = pd.read_csv(MOTUS_CSV)
        for df, col in ((self.merged, "word"), (self.pinoy, "word"), (self.motus, "word")):
            df["_norm"] = df[col].astype(str).map(normalize)
        self.merged["_meaning_norm"] = self.merged["meaning"].fillna("").astype(str).map(normalize)
        self.pinoy["_definition_norm"] = self.pinoy["definition"].fillna("").astype(str).map(normalize)
        self.motus["_definitions_norm"] = self.motus["definitions"].fillna("").astype(str).map(normalize)
        self._verb_roots = self._collect_verb_roots()
        self._past_time, self._nonpast_time = self._split_time_adverbs()
        self.all_words: list[str] = sorted(set(self.merged["_norm"]))

    def _collect_verb_roots(self) -> list[str]:
        vb = self.merged[
            (self.merged["pos_symbol"] == "VB") & (self.merged["pos_word"] == "verb")
        ]
        roots = [w for w in dict.fromkeys(vb["_norm"]) if TOKEN_RE.match(w) and len(w) >= 2]
        return roots

    def _split_time_adverbs(self) -> tuple[list[str], list[str]]:
        past, nonpast = [], []
        rb = self.merged[self.merged["pos_symbol"] == "RB"]
        for _, row in rb.iterrows():
            meaning = str(row["meaning"] or "")
            word = row["_norm"]
            if not TOKEN_RE.match(word):
                continue
            if re.search(PAST_TIME_KEYS, meaning, re.IGNORECASE):
                past.append(word)
            elif re.search(NONPAST_TIME_KEYS, meaning, re.IGNORECASE):
                nonpast.append(word)
        return past, nonpast

    def build_categories(self, extra_names: list[str] | None = None) -> dict[str, Category]:
        cats = {letter: Category(letter) for letter in CATEGORY_META}

        for letter, words in HANDPICKED.items():
            for w in words:
                cats[letter].add(w, source="handpicked")

        # a - common nouns (NN)
        for _, row in self.merged[self.merged["pos_symbol"] == "NN"].iterrows():
            if TOKEN_RE.match(row["_norm"]):
                cats["a"].add(row["_norm"], display=str(row["word"]), source="lexicon")

        # b - proper nouns (handpicked + user added)
        for name in extra_names or []:
            if name.strip():
                cats["b"].add(normalize(name.strip()), display=name.strip(), source="you")

        # c - pronouns (PRP)
        for _, row in self.merged[self.merged["pos_symbol"] == "PRP"].iterrows():
            if TOKEN_RE.match(row["_norm"]):
                cats["c"].add(row["_norm"], display=str(row["word"]), source="lexicon")

        # d / e / f - generated tense forms from verb roots
        for root in self._verb_roots:
            cats["d"].add(f"nag{root}", display=f"nag{root}", source="generated")
            cats["e"].add(f"naga{root}", display=f"naga{root}", source="generated")
            cats["f"].add(f"ma{root}", display=f"ma{root}", source="generated")

        # g / h - time words
        for w in self._past_time:
            cats["g"].add(w, source="lexicon")
        for w in self._nonpast_time:
            cats["h"].add(w, source="lexicon")

        # i - manner adverbs = remaining RB, minus time words
        time_words = set(cats["g"].words) | set(cats["h"].words)
        for _, row in self.merged[self.merged["pos_symbol"] == "RB"].iterrows():
            if TOKEN_RE.match(row["_norm"]) and row["_norm"] not in time_words:
                cats["i"].add(row["_norm"], display=str(row["word"]), source="lexicon")

        # o - adjectives (JJ)
        for _, row in self.merged[self.merged["pos_symbol"] == "JJ"].iterrows():
            if TOKEN_RE.match(row["_norm"]):
                cats["o"].add(row["_norm"], display=str(row["word"]), source="lexicon")

        return cats

    def lookup(self, query: str) -> dict[str, pd.DataFrame]:
        q = normalize(query.strip())
        return {
            "lexicon": self.merged[self.merged["_norm"] == q],
            "pinoy": self.pinoy[self.pinoy["_norm"] == q],
            "motus": self.motus[self.motus["_norm"] == q],
        }

    def search(self, query: str, direction: str = "word", limit: int = 80) -> list[str]:
        """Return distinct headwords matching either a Hiligaynon word or a meaning.

        ``direction`` is ``"word"`` for Hiligaynon headwords and ``"meaning"``
        for English definitions. Search text is normalized once while loading the
        dictionaries, so this stays responsive as the user types.
        """
        q = normalize(query.strip())
        if not q:
            return []

        if direction == "meaning":
            sources = (
                (self.merged, "_meaning_norm"),
                (self.pinoy, "_definition_norm"),
                (self.motus, "_definitions_norm"),
            )
        else:
            sources = (
                (self.merged, "_norm"),
                (self.pinoy, "_norm"),
                (self.motus, "_norm"),
            )

        matches: dict[str, int] = {}
        for frame, column in sources:
            mask = frame[column].str.contains(q, regex=False, na=False)
            matched_words = frame.loc[mask, "_norm"].head(limit * 3).tolist()
            if direction != "meaning":
                exact_words = frame.loc[frame["_norm"] == q, "_norm"].tolist()
                matched_words = [*exact_words, *matched_words]
            for word in matched_words:
                if not word or word == "nan":
                    continue
                if direction == "meaning":
                    rank = 3
                elif word == q:
                    rank = 0
                elif word.startswith(q):
                    rank = 1
                else:
                    rank = 2
                matches[word] = min(matches.get(word, rank), rank)

        return [word for word, _ in sorted(matches.items(), key=lambda item: (item[1], item[0]))[:limit]]

    def meaning_of(self, word: str) -> str:
        """Short meaning for one word; generated tense forms fall back to their root."""
        w = normalize(word)
        for candidate in (w, *[re.sub(p, "", w) for p in (r"^naga", r"^nag", r"^ma")]):
            if not candidate:
                continue
            rows = self.merged[self.merged["_norm"] == candidate]
            if len(rows):
                text = str(rows.iloc[0]["meaning"] or "").strip()
                if text:
                    return text if len(text) <= 160 else text[:157] + "..."
        return ""

    def suggest(self, query: str, n: int = 5) -> list[str]:
        return difflib.get_close_matches(normalize(query), self.all_words, n=n, cutoff=0.6)


_lexicon: Lexicon | None = None


def load_lexicon() -> Lexicon:
    global _lexicon
    if _lexicon is None:
        _lexicon = Lexicon()
    return _lexicon
