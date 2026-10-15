import pandas as pd
from pathlib import Path

# Get input/output paths
project_root = Path(__file__).resolve().parents[2]
input_csv = project_root / "data" / "lexicons" / "Ceciel_Wiki_merged_scores.csv"

# Load dataset
df = pd.read_csv(input_csv)

# Sentiment column
sentiment_col = "sentiment"

# Target sample sizes per sentiment
target_samples = {
    "Neutral": 280,
    "Negative": 57,
    "Positive": 48
}

# Sample each sentiment separately
neutral_sample = df[df[sentiment_col] == "Neutral"].sample(n=target_samples["Neutral"], random_state=42)
negative_sample = df[df[sentiment_col] == "Negative"].sample(n=target_samples["Negative"], random_state=42)
positive_sample = df[df[sentiment_col] == "Positive"].sample(n=target_samples["Positive"], random_state=42)

# Combine into one DataFrame
final_sample = pd.concat([neutral_sample, negative_sample, positive_sample], axis=0)

# Save to single CSV
output_file = project_root / "data" / "lexicons" / "samples" / "lexicon_sample.csv"
final_sample.to_csv(output_file, index=False)

print("Combined random sample saved as sample.csv")
print(final_sample[sentiment_col].value_counts())
