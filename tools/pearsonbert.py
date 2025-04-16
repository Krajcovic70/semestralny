import pandas as pd
from scipy.stats import pearsonr

original_file = "../mojtext.txt"
predicted_file = ("../tretipokustrenovania.txt")

df_original = pd.read_csv(original_file, sep='\t', header=None, names=['score', 'sentence1', 'sentence2'])
original_scores = df_original['score'].astype(float)  # Konverzia na float

with open(predicted_file, "r", encoding="utf-8") as f:
    predicted_scores = [float(line.strip()) for line in f]

if len(original_scores) != len(predicted_scores):
    raise ValueError("Počet pôvodných a predikovaných skóre sa nezhoduje!")

correlation, p_value = pearsonr(original_scores, predicted_scores)

print(f"Pearsonov korelačný koeficient: {correlation:.4f}")
