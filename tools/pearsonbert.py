import pandas as pd
from scipy.stats import pearsonr, spearmanr

original_file = "../stsbenchmark_sk.txt"
predicted_file = "../slovakbert_stsbenchmark_trenovany.txt"

df_original = pd.read_csv(original_file, sep='\t', header=None, names=['score', 'sentence1', 'sentence2'])
original_scores = df_original['score'].astype(float)

with open(predicted_file, "r", encoding="utf-8") as f:
    predicted_scores = [float(line.strip()) for line in f]

if len(original_scores) != len(predicted_scores):
    raise ValueError("Počet pôvodných a predikovaných skóre sa nezhoduje!")

pearson_corr, pearson_p = pearsonr(original_scores, predicted_scores)
spearman_corr, spearman_p = spearmanr(original_scores, predicted_scores)

print(f"Pearsonov korelačný koeficient: {pearson_corr:.4f} (p-hodnota: {pearson_p:.4g})")
print(f"Spearmanov korelačný koeficient: {spearman_corr:.4f} (p-hodnota: {spearman_p:.4g})")
