import pandas as pd
from pathlib import Path

# Input/output paths
project_root = Path(__file__).resolve().parents[2]
csv_input = project_root / "data" / "lexicons" / "preprocessed" / "Ceciel_Wiki_merged.csv"
txt_input = project_root / "data" / "sentiment" / "vader_hiligaynon_cleaned.txt"
output_path = project_root / "data" / "lexicons" / "Ceciel_Wiki_merged_scores.csv"

# Load CSV
lexicon = pd.read_csv(csv_input)

# Load TXT (row order matters here)
scores = pd.read_csv(
    txt_input,
    sep="\t",
    header=None,
    names=["word", "score", "col3", "col4"]
)[["word", "score"]]

# Rename "score" -> "sentiment score"
scores = scores.rename(columns={"score": "sentiment score"})

# Reset indexes to align rows properly
lexicon = lexicon.reset_index(drop=True)
scores = scores.reset_index(drop=True)

# Merge row by row (ignore word matching, just align by row order)
lexicon_with_scores = pd.concat([lexicon, scores["sentiment score"]], axis=1)

# Add sentiment label based on VADER thresholds
def get_sentiment(score: float) -> str:
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

lexicon_with_scores["sentiment"] = lexicon_with_scores["sentiment score"].apply(get_sentiment)

# Save
lexicon_with_scores.to_csv(output_path, index=False)

print(f"✅ Merged with VADER scores and sentiment labels: {output_path}")
