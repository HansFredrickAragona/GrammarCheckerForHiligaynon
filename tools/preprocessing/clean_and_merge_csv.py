import pandas as pd
import unicodedata
import csv
from pathlib import Path

# === File paths ===
project_root = Path(__file__).resolve().parents[2]
raw_dir = project_root / "data" / "lexicons" / "raw"
base_dir = project_root / "data" / "lexicons" / "preprocessed"
cec_file = raw_dir / "NLP Lexicon - Cecielmotus.csv"
wiki_file = base_dir / "wiktionary_final.csv"

cec_cleaned = base_dir / "Cecielmotus_cleaned.csv"
wiki_cleaned = base_dir / "Wikitionary_cleaned.csv"
merged_file = base_dir / "Ceciel_Wiki_merged.csv"
excluded_file = base_dir / "excluded_expressions.csv"

# === Penn Treebank POS Mapping ===
pos_map = {
    "noun": "NN",
    "noun formative": "NN",
    "verb": "VB",
    "verbal affix": "VB",
    "adjective": "JJ",
    "adjective formative": "JJ",
    "adverb": "RB",
    "pronoun": "PRP",
    "particle": "RP",
    "conjunction": "CC",
    "interjection": "UH",
    "numeral": "CD",
    "determiner": "DT",
    "deictic": "DT",
}

drop_categories = {"phrase", "idiom", "phrasebook"}


# --- Helpers ---
def _norm_colname(c: str) -> str:
    return c.strip().lower().replace(" ", "_").replace("-", "_")


def normalize_word(word: str) -> str:
    if not isinstance(word, str):
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn"
    ).lower()


# === Function to clean a dataset ===
def clean_dataset(path, rename_map=None, default_symbol="UNK"):
    # detect delimiter (best-effort)
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
        try:
            dialect = csv.Sniffer().sniff(sample)
            delimiter = dialect.delimiter
        except Exception:
            delimiter = ","

    # read file robustly
    df = pd.read_csv(path, dtype=str, delimiter=delimiter, on_bad_lines="skip", engine="python")

    # normalize column names
    df.columns = [_norm_colname(c) for c in df.columns]

    # apply rename_map if provided (normalize its keys/values too)
    if rename_map:
        ren = {_norm_colname(k): _norm_colname(v) for k, v in rename_map.items()}
        df = df.rename(columns=ren)

    # keep only relevant columns if they exist
    keep_cols = ["word", "pos_word", "meaning", "pos_symbol"]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # normalize cell values (strip + lower)
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # forward fill 'word'
    if "word" in df.columns:
        df["word"] = df["word"].replace("nan", pd.NA).ffill()

    # exclude unwanted POS categories (works because pos_word is lowercased)
    excluded = pd.DataFrame()
    if "pos_word" in df.columns:
        mask_excluded = df["pos_word"].isin(drop_categories)
        excluded = df[mask_excluded].copy()
        df = df[~mask_excluded].copy()

    # ensure pos_symbol exists and map pos_word -> pos_symbol where possible
    if "pos_symbol" not in df.columns:
        df["pos_symbol"] = default_symbol
    if "pos_word" in df.columns:
        df["pos_symbol"] = df["pos_word"].map(pos_map).fillna(df["pos_symbol"])

    # reorder columns if present
    col_order = ["word", "pos_symbol", "pos_word", "meaning"]
    df = df[[c for c in col_order if c in df.columns]]

    return df, excluded


# === Run cleaning for both sources ===
cec_df, cec_excluded = clean_dataset(cec_file)
cec_df.to_csv(cec_cleaned, index=False)

wiki_df, wiki_excluded = clean_dataset(
    wiki_file, rename_map={"Part of speech": "POS word", "Word": "word"}
)
wiki_df.to_csv(wiki_cleaned, index=False)

# save excluded rows (phrase/idiom/phrasebook)
excluded_all = pd.concat([cec_excluded, wiki_excluded], ignore_index=True)
excluded_all.to_csv(excluded_file, index=False)

# === Merge and group (keep original spelling, group variants together) ===
merged = pd.concat([cec_df, wiki_df], ignore_index=True)

# add helper normalized form and sort by it so variants (túbig / tubig) are adjacent
merged["group"] = merged["word"].apply(normalize_word)
merged = merged.sort_values(by=["group", "word"]).reset_index(drop=True)

# drop rows with blank meaning
merged = merged[merged["meaning"].notna() & (merged["meaning"].str.strip() != "")].copy()

# remove helper column and save
merged.drop(columns=["group"], inplace=True)
merged.to_csv(merged_file, index=False)

# === Report ===
print("=== Cleaning Report ===")
print(f"Cecielmotus rows: {cec_df.shape[0]}")
print(f"Wiktionary rows: {wiki_df.shape[0]}")
print(f"Excluded rows saved: {excluded_all.shape[0]} (phrase/idiom/phrasebook)")
print(f"Final merged rows: {merged.shape[0]}")
