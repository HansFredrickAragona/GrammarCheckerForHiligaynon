import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path

# Input/output paths
project_root = Path(__file__).resolve().parents[2]
csv_input = project_root / "data" / "lexicons" / "preprocessed" / "Ceciel_Wiki_merged.csv"
lexicon_output = project_root / "data" / "sentiment" / "vader_hiligaynon_cleaned.txt"

# Load the parsed CSV
df = pd.read_csv(csv_input)

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

lexicon_entries = []

for _, row in df.iterrows():
    word = str(row["word"]).strip()
    meaning = str(row.get("meaning", "")).strip()

    if not word or not meaning:
        continue  # skip empty rows

    # Get sentiment score from the meaning column
    score = analyzer.polarity_scores(meaning)["compound"]

    # Format: word <tab> score <tab> 0.0 <tab> 0.0
    lexicon_entries.append(f"{word}\t{score:.3f}\t0.0\t0.0")

# Save custom lexicon
lexicon_output.parent.mkdir(parents=True, exist_ok=True)
with open(lexicon_output, "w", encoding="utf-8") as f:
    f.write("\n".join(lexicon_entries))

print(f"✅ Custom Hiligaynon VADER lexicon created: {lexicon_output}")
