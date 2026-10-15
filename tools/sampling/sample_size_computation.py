import math
import pandas as pd
from pathlib import Path

# ===========================================
# Step 1: Load dataset
# ===========================================
project_root = Path(__file__).resolve().parents[2]
file_path = project_root / "data" / "lexicons" / "Ceciel_Wiki_merged_scores.csv"
df = pd.read_csv(file_path)

# === Count per class ===
sentiment_col = "sentiment"
counts = df[sentiment_col].value_counts().to_dict()
total_population = sum(counts.values())

# === Cochran baseline (95% CI, ±5% MOE) ===
Z = 1.96
p = 0.5
e = 0.05
n0 = math.ceil((Z**2 * p * (1 - p)) / (e**2))

# === Finite Population Correction (FPC) per class ===
def fpc(N, n0):
    return math.ceil((n0 * N) / (n0 + N - 1))

results = {}
total_sample = 0
for cls, N in counts.items():
    n = fpc(N, n0)
    results[cls] = {"population": N, "ideal_n": n}
    total_sample += n

# === Compute proportions in the verification sample (FPC) ===
for cls, info in results.items():
    info["proportion"] = info["ideal_n"] / total_sample
    info["percent"] = info["proportion"] * 100

# === Print tidy summary for FPC ===
print(f"Baseline Cochran n0 (95% CI, ±5% MOE): {n0}\n")
print("Class     | Population | Ideal sample (FPC) | Proportion of verification sample | Percent")
print("----------|------------|--------------------|----------------------------------|--------")
for cls, info in results.items():
    print(f"{cls:9} | {info['population']:10d} | {info['ideal_n']:18d} | "
          f"{info['proportion']:.4f}                          | {info['percent']:.2f}%")

print(f"\nTotal verification sample (sum of ideal per-class, FPC) = {total_sample}")

# ===========================================
# Step 2: Proportional breakdown of Cochran n0 (385)
# ===========================================
allocation = {}
allocated_total = 0
for sentiment, N in counts.items():
    ideal = round((N / total_population) * n0)
    allocation[sentiment] = ideal
    allocated_total += ideal

# Fix rounding difference to ensure exact sum = 385
diff = n0 - allocated_total
if diff != 0:
    # Adjust the largest group (Neutral most likely)
    largest_group = max(allocation, key=allocation.get)
    allocation[largest_group] += diff

print("\n=== Cochran Baseline (Proportional Allocation) ===")
for sentiment, n in allocation.items():
    print(f"{sentiment}: {n}")
print(f"\nCheck: Sum of proportional allocation = {sum(allocation.values())} (should equal {n0})")
