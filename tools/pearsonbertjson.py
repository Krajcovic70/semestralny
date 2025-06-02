import pandas as pd
from scipy.stats import pearsonr

original_file = "../sts_benchmark_test.json"
data = pd.read_json(original_file)
original_scores = data['similarity_score'].astype(float)

predicted_file = "../druhypokustrenovaniabenchmark.txt"
with open(predicted_file, "r", encoding="utf-8") as f:
    predicted_scores = [float(line.strip()) for line in f]

if len(original_scores) != len(predicted_scores):
    raise ValueError("Počet pôvodných a predikovaných skóre sa nezhoduje!")

correlation, p_value = pearsonr(original_scores, predicted_scores)

print(f"Pearsonov korelačný koeficient: {correlation:.4f}")
