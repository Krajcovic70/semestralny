import json

input_file = "../statisticke testy/stsbenchmark_sk_results_NLP.json"
output_file = "../statisticke testy/nlp_stsbenchmark_fix.txt"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

scores_scaled = [item["similarity_score"] * 5 for item in data]

with open(output_file, "w", encoding="utf-8") as f:
    for score in scores_scaled:
        f.write(f"{score:.4f}\n")

print(f"Uložené do súboru: {output_file}")
