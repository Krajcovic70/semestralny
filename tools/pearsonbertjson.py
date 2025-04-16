import pandas as pd
from scipy.stats import pearsonr

# Load original scores from JSON
original_file = "../sts_benchmark_test.json"
data = pd.read_json(original_file)
original_scores = data['similarity_score'].astype(float)

# Load predicted scores from .txt file
predicted_file = "../druhypokustrenovaniabenchmark.txt"
with open(predicted_file, "r", encoding="utf-8") as f:
    predicted_scores = [float(line.strip()) for line in f]

# Ensure same number of entries
if len(original_scores) != len(predicted_scores):
    raise ValueError("Počet pôvodných a predikovaných skóre sa nezhoduje!")

# Calculate Pearson correlation
correlation, p_value = pearsonr(original_scores, predicted_scores)

print(f"Pearsonov korelačný koeficient: {correlation:.4f}")
