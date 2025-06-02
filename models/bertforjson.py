
import torch
import numpy as np
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity


def get_sentence_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    sentence_embedding = torch.mean(embeddings, dim=1)
    return sentence_embedding.numpy()


tokenizer = RobertaTokenizer.from_pretrained('../slovakbert-sts-model')
model = RobertaModel.from_pretrained('../slovakbert-sts-model')

dataset_path = '../sts_benchmark_test.json'
data = pd.read_json(dataset_path)

results = []
for index in range(len(data['sentence1'])):
    s1 = data['sentence1'][index]
    s2 = data['sentence2'][index]

    embedding1 = get_sentence_embedding(s1, tokenizer, model)
    embedding2 = get_sentence_embedding(s2, tokenizer, model)

    similarity = cosine_similarity(embedding1, embedding2)[0][0] * 5
    results.append(f"{similarity:.4f}\n")

    print(f"Processed pair {index + 1}/{len(data)}")

with open('../druhypokustrenovaniabenchmark.txt', 'w') as f:
    f.writelines(results)

print("Processing complete, results saved to trained_sts_benchmark.txt.")
