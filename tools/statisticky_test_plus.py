import json
import numpy as np
from scipy.stats import ttest_rel, wilcoxon, normaltest

anotacie = []
with open("../sick_sk.txt", encoding="utf-8") as f:
    for line in f:
        score = float(line.split("\t")[0])
        anotacie.append(score)

predikcie_1 = []
with open("../statisticke testy/ai_sick_sk_correct.txt") as f:
    for line in f:
        predikcie_1.append(float(line.strip()))

# predikcie_2 = []
# with open("../statisticke testy/ai_sick_sk_correct.txt") as f:
#     for line in f:
#         predikcie_2.append(float(line.strip()))

with open("../statisticke testy/sick_sk_results.json", encoding="utf-8") as f:
    json_data = json.load(f)
    predikcie_2 = [item["numerical_response"] for item in json_data["responses"]] ##pre json
    #predikcie_1 = [item["similarity_score"] for item in json_data] #pre NLPjson

chyby_1 = [abs(pred - gold) for pred, gold in zip(predikcie_1, anotacie)]
chyby_2 = [abs(pred - gold) for pred, gold in zip(predikcie_2, anotacie)]

print(f" - Priemerná absolútna chyba model1: {np.mean(chyby_1):.4f}")
print(f" - Priemerná absolútna chyba model2: {np.mean(chyby_2):.4f}")

# rozdiely = np.array(chyby_1) - np.array(chyby_2)

stat_normal, p_normal = normaltest(chyby_1)
stat_normal2, p_normal2 = normaltest(chyby_2)
print(f" - Normaltest (D’Agostino-Pearson) p-hodnota: {p_normal:.4f}")
print(f" - Normaltest (D’Agostino-Pearson) p-hodnota: {p_normal2:.4f}")

if (p_normal > 0.05 and p_normal2 > 0.05):
    stat, p_value = ttest_rel(chyby_1, chyby_2)
    print(" - Použitý test: Párový t-test")
else:
    stat, p_value = wilcoxon(chyby_1, chyby_2)
    print(" - Použitý test: Wilcoxonov párový test")

print(f" - p-hodnota testu: {p_value:.4f}")
if p_value < 0.05:
    print(" -> Rozdiel je štatisticky významný (zamietame nulovú hypotézu).")
else:
    print(" -> Rozdiel nie je štatisticky významný (nulovú hypotézu nezamietame).")
