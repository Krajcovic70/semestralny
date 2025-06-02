import pandas as pd
import json

file_path = "stsbenchmark_sk.txt"
df = pd.read_csv(file_path, sep="\t", header=None, names=["similarity_score", "sentence1", "sentence2"])

grouped_records = {}
for _, row in df.iterrows():
    key = row["similarity_score"]
    if key not in grouped_records:
        grouped_records[key] = []
    grouped_records[key].append(row)

train_ratio = 0.80
train_sentences1, train_sentences2, train_scores = [], [], []
test_sentences1, test_sentences2, test_scores = [], [], []

for score, records in grouped_records.items():
    split_index = int(len(records) * train_ratio)
    train_records = records[:split_index]
    test_records = records[split_index:]

    for record in train_records:
        train_sentences1.append(record["sentence1"])
        train_sentences2.append(record["sentence2"])
        train_scores.append(record["similarity_score"])

    for record in test_records:
        test_sentences1.append(record["sentence1"])
        test_sentences2.append(record["sentence2"])
        test_scores.append(record["similarity_score"])

train_json = {
    "sentence1": train_sentences1,
    "sentence2": train_sentences2,
    "similarity_score": train_scores
}

test_json = {
    "sentence1": test_sentences1,
    "sentence2": test_sentences2,
    "similarity_score": test_scores
}

train_file_path = "sts_benchmark_train.json"
test_file_path = "sts_benchmark_test.json"

with open(train_file_path, "w", encoding="utf-8") as train_file:
    json.dump(train_json, train_file, indent=4, ensure_ascii=False)

with open(test_file_path, "w", encoding="utf-8") as test_file:
    json.dump(test_json, test_file, indent=4, ensure_ascii=False)

train_file_path, test_file_path
