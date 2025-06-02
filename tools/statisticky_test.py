import scipy.stats as stats
import itertools
import numpy as np

models = {
    "Text-embedding-ada-002": [0.68150, 0.72240, 0.79980],
    "SlovakBERT": [0.58030, 0.60550, 0.78830],
    "SlovakBERT (dotrénovaný)": [0.75150, 0.75370, 0.83290],
    "Paraphrase-MPNet": [0.77821, 0.82417, 0.94750],
    "ChatGPT": [0.73963, 0.78012, 0.82808]
}

pairs = [
    ("Text-embedding-ada-002", "SlovakBERT (dotrénovaný)"),
    ("Text-embedding-ada-002", "Paraphrase-MPNet"),
    ("Text-embedding-ada-002", "ChatGPT"),
    ("SlovakBERT (dotrénovaný)", "Paraphrase-MPNet"),
    ("SlovakBERT (dotrénovaný)", "ChatGPT"),
    ("Paraphrase-MPNet", "ChatGPT"),
    ("SlovakBERT", "SlovakBERT (dotrénovaný)")
]

print("\n=== Štatistické porovnanie modelov ===\n")

for model1, model2 in pairs:
    values1 = models[model1]
    values2 = models[model2]

    diffs = [b - a for a, b in zip(values1, values2)]
    shapiro_stat, shapiro_p = stats.shapiro(diffs)

    mean1 = np.mean(values1)
    mean2 = np.mean(values2)

    print(f"Porovnanie: {model1} vs {model2}")
    print(f" - Priemer {model1}: {mean1:.5f}")
    print(f" - Priemer {model2}: {mean2:.5f}")
    print(f" - Nulová hypotéza: Výkon oboch modelov je rovnaký (žiadny rozdiel v priemere).")

    if shapiro_p > 0.05:
        t_stat, p_value = stats.ttest_rel(values1, values2)
        test_used = "Studentov párový t-test"
    else:
        stat, p_value = stats.mannwhitneyu(values1, values2)
        test_used = "Wilcoxon–Mann–Whitney test"

    print(f" - Shapiro-Wilk p-hodnota (normalita): {shapiro_p:.4f}")
    print(f" - Použitý test: {test_used}")
    print(f" - p-hodnota testu: {p_value:.4f}")

    if p_value < 0.05:
        print(f" -> Rozdiel je štatisticky významný (zamietame nulovú hypotézu).")
    else:
        print(f" -> Rozdiel nie je štatisticky významný (nulovú hypotézu nezamietame).")
    print("-" * 60)