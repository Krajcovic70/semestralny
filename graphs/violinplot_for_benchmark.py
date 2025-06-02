import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

with open("../stsbenchmark_sk.txt", "r", encoding="utf-8") as f:
    annotations = [float(line.split('\t')[0]) for line in f.readlines()]

with open("../sts_benchmark_test.json", "r", encoding="utf-8") as f:
    slovakbert_json = json.load(f)

annotations_slovakbert = slovakbert_json["similarity_score"]
print(len(annotations_slovakbert))

with open("../druhypokustrenovaniabenchmark.txt", "r", encoding="utf-8") as f:
    scores_slovakbert = [float(line.strip()) for line in f.readlines()]

with open("../statisticke testy/nlp_stsbenchmark_fix.txt", "r", encoding="utf-8") as f:
    scores_nlp = [float(line.strip()) for line in f.readlines()]

with open("../statisticke testy/ai_stsbenchmark_sk_correct.txt", "r", encoding="utf-8") as f:
    scores_ai = [float(line.strip()) for line in f.readlines()]

with open("../statisticke testy/stsbenchmark_sk_results.json", "r", encoding="utf-8") as f:
    predictions_chatgpt = json.load(f)
chatgpt_scores = [entry['numerical_response'] for entry in predictions_chatgpt['responses']]


def compute_abs_errors(preds, gold):
    return [abs(p - g) for p, g in zip(preds, gold)]

models = {
    "SlovakBERT": scores_slovakbert,
    "Ada002": scores_ai,
    "MPNet": scores_nlp,
    "ChatGPT": chatgpt_scores
}

data = []
for model_name, preds in models.items():
    if model_name == "SlovakBERT":
        relevant_annotations = annotations_slovakbert
    else:
        relevant_annotations = annotations

    abs_errors = compute_abs_errors(preds, relevant_annotations)
    for err in abs_errors:
        data.append({"Model": model_name, "Absolute Error": err})
df = pd.DataFrame(data)


plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x="Model", y="Absolute Error", inner="box")
plt.title("Rozdelenie absolútnej chyby modelov na datasete STS Benchmark_sk", fontsize=14)
plt.ylabel("Absolútna chyba", fontsize=14)
plt.xlabel("Model", fontsize=14)
plt.tight_layout()
plt.savefig("violin_plot_stsbenchmark.png", dpi=300)
plt.show()
